"""
Layer-wise Hessian Analysis using LessWrong Implementation
Computes top eigenvalues for attention, MLP, and output layers separately.

Based on: https://www.lesswrong.com/posts/mwBaS2qE9RNNfqYBC/
"""

import torch
import json
import argparse
import signal
import sys
from pathlib import Path
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ============================================================================
# Signal Handler - Catch Segfaults
# ============================================================================

def signal_handler(signum, frame):
    """Catch segfaults and other fatal signals from C libraries"""
    signal_names = {
        signal.SIGSEGV: "SIGSEGV (Segmentation Fault)",
        signal.SIGABRT: "SIGABRT (Abort)",  
        signal.SIGFPE: "SIGFPE (Floating Point Exception)",
        signal.SIGILL: "SIGILL (Illegal Instruction)",
    }
    signal_name = signal_names.get(signum, f"Signal {signum}")
    print(f"\n❌❌❌ CAUGHT {signal_name} ❌❌❌")
    print(f"This is a C library crash (scipy/BLAS/LAPACK or CUDA)")
    print(f"Last known state will be in the partial save file")
    sys.exit(1)

# Register handlers (not all signals can be caught, but try)
try:
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)
    signal.signal(signal.SIGFPE, signal_handler)
    signal.signal(signal.SIGILL, signal_handler)
except (ValueError, OSError):
    # Some signals can't be caught on all platforms
    pass

# ============================================================================
# Core Hessian Eigenvalue Computation (from LessWrong)
# ============================================================================

def get_hessian_eigenvectors(model, loss_fn, train_data_loader, num_batches, 
                             device, n_top_vectors, param_extract_fn):
    """
    Compute top Hessian eigenvalues/eigenvectors for a subset of parameters.
    
    Memory-efficient version: Iterates through batches without accumulating.
    
    Args:
        model: PyTorch model
        loss_fn: Loss function (e.g., F.cross_entropy)
        train_data_loader: DataLoader for computing loss
        num_batches: Number of batches to iterate through for HVP
        device: Device to use
        n_top_vectors: Number of top eigenvalues to compute
        param_extract_fn: Function that takes model and returns list of parameters
                         to compute Hessian for (None = all parameters)
    
    Returns:
        eigenvalues: numpy array of top eigenvalues (increasing order)
        eigenvectors: numpy array of eigenvectors, shape (n_top_vectors, num_params)
    """
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())
    params_list = list(param_extract_fn(model))
    num_params = sum(p.numel() for p in params_list)
    
    print(f"  Computing Hessian for {num_params:,} parameters...")
    
    # Monitor initial memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU memory at start: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")
    
    def hessian_vector_product(vector):
        """
        Compute H*v by iterating through batches.
        This is memory-efficient - only one batch in memory at a time.
        """
        hvp_accumulator = None
        n_accumulated = 0
        
        for batch_idx, batch in enumerate(train_data_loader):
            if batch_idx >= num_batches:
                break
            
            try:
                model.zero_grad()
                
                # Get batch
                inputs = batch['input_ids'].to(device)
                labels = batch['input_ids'].to(device)
                
                # Forward pass
                output = model(inputs)
                logits = output.logits if hasattr(output, 'logits') else output
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"NaN/Inf loss detected in batch {batch_idx}")
                
                # First derivative (gradient)
                grad_params = grad(loss, params_list, create_graph=True, retain_graph=True)
                flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_params])
                
                # Check for NaN gradient
                if torch.isnan(flat_grad).any() or torch.isinf(flat_grad).any():
                    raise ValueError(f"NaN/Inf gradient detected in batch {batch_idx}")
                
                # Dot product with vector
                grad_vector_product = torch.sum(flat_grad * vector)
                
                # Second derivative (HVP for this batch)
                hvp_batch = grad(grad_vector_product, params_list, retain_graph=False)
                hvp_batch_flat = torch.cat([g.contiguous().view(-1) for g in hvp_batch])
                
                # Check for NaN HVP
                if torch.isnan(hvp_batch_flat).any() or torch.isinf(hvp_batch_flat).any():
                    raise ValueError(f"NaN/Inf HVP detected in batch {batch_idx}")
                
                # Accumulate
                if hvp_accumulator is None:
                    hvp_accumulator = hvp_batch_flat
                else:
                    hvp_accumulator += hvp_batch_flat
                
                n_accumulated += 1
                
            except RuntimeError as e:
                print(f"      ⚠️  RuntimeError in HVP batch {batch_idx}: {e}")
                raise
            except Exception as e:
                print(f"      ⚠️  Exception in HVP batch {batch_idx}: {e}")
                raise
        
        # Average over batches
        if n_accumulated > 0:
            hvp_accumulator = hvp_accumulator / n_accumulated
        
        return hvp_accumulator
    
    # Track matvec calls
    import time
    matvec_calls = [0]
    start_time = [time.time()]
    
    def matvec(v):
        """Wrapper for scipy LinearOperator"""
        matvec_calls[0] += 1
        call_num = matvec_calls[0]
        elapsed = time.time() - start_time[0]
        
        # Log EVERY call with timing
        avg_time_per_call = elapsed / call_num if call_num > 0 else 0
        print(f"    [matvec #{call_num:3d}] {elapsed:6.1f}s elapsed, {avg_time_per_call:.2f}s/call", end="")
        
        if torch.cuda.is_available() and call_num % 10 == 0:
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f" | GPU: {mem_allocated:.1f}GB/{mem_reserved:.1f}GB", end="")
        
        print()  # Newline
        
        try:
            v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
            result = hessian_vector_product(v_tensor).cpu().detach().numpy()
            
            # Validate result
            if np.isnan(result).any():
                raise ValueError(f"NaN in matvec result (call #{call_num})")
            if np.isinf(result).any():
                raise ValueError(f"Inf in matvec result (call #{call_num})")
            
            return result
        except RuntimeError as e:
            # CUDA errors often show as RuntimeError
            print(f"    ❌ RUNTIME ERROR in matvec call #{call_num}: {e}")
            if "out of memory" in str(e).lower():
                print(f"    ❌ GPU OUT OF MEMORY!")
                if torch.cuda.is_available():
                    print(f"       Allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                    print(f"       Reserved: {torch.cuda.memory_reserved()/1024**3:.1f}GB")
            raise
        except Exception as e:
            print(f"    ❌ ERROR in matvec call #{call_num}: {e}")
            raise
    
    # Progress tracking
    iteration_count = [0]
    
    def progress_callback(x):
        """Called by eigsh after each iteration"""
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            print(f"    Lanczos iteration {iteration_count[0]}...")
    
    # Create linear operator and compute eigenvalues
    print(f"  Running Lanczos iteration...")
    print(f"  Creating LinearOperator for {num_params:,} parameters...")
    linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
    
    print(f"  Calling scipy.sparse.linalg.eigsh...")
    print(f"    Requested eigenvalues: {n_top_vectors}")
    print(f"    Max iterations: {num_params * 10}")
    
    try:
        eigenvalues, eigenvectors = eigsh(
            linear_operator, 
            k=n_top_vectors, 
            tol=0.001, 
            which='LM',  # Largest magnitude
            maxiter=num_params * 10,  # Prevent infinite loops
            return_eigenvectors=True,
            # Note: scipy eigsh doesn't support callback in older versions
        )
        print(f"  ✅ eigsh returned successfully!")
    except RuntimeError as e:
        print(f"  ❌ eigsh failed with RuntimeError: {e}")
        if "out of memory" in str(e).lower():
            print(f"  ❌ This was a CUDA OOM error")
        raise
    except Exception as e:
        print(f"  ❌ eigsh failed with {type(e).__name__}: {e}")
        raise
    except BaseException as e:
        print(f"  ❌ eigsh failed with BaseException {type(e).__name__}: {e}")
        raise
    
    print(f"  ✅ Converged after {iteration_count[0]} iterations")
    print(f"  Eigenvalues computed: {len(eigenvalues)}")
    print(f"  Transposing eigenvectors...")
    
    eigenvectors = np.transpose(eigenvectors)
    
    print(f"  ✅ get_hessian_eigenvectors returning successfully")
    return eigenvalues, eigenvectors


# ============================================================================
# Architecture Detection & Layer Mapping
# ============================================================================

def get_layer_groups(model):
    """
    Detect architecture and return layer group keywords.
    
    Returns:
        dict: {layer_type: [keywords]}
    """
    model_type = model.config.model_type.lower()
    
    # Architecture-specific mappings
    if model_type in ['gpt2', 'gpt_neo', 'gpt_neox']:
        return {
            'attention': ['attn.c_attn', 'attn.c_proj'],
            'mlp': ['mlp.c_fc', 'mlp.c_proj'],
            'output': ['lm_head', 'wte']  # Tied weights
        }
    
    elif model_type in ['llama', 'mistral']:
        return {
            'attention': ['self_attn.q_proj', 'self_attn.k_proj', 
                         'self_attn.v_proj', 'self_attn.o_proj'],
            'mlp': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
            'output': ['embed_tokens'] if getattr(model.config, 'tie_word_embeddings', False) else ['lm_head']
        }
    
    elif model_type == 'qwen2':
        return {
            'attention': ['self_attn.q_proj', 'self_attn.k_proj',
                         'self_attn.v_proj', 'self_attn.o_proj'],
            'mlp': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
            'output': ['lm_head']
        }
    
    elif model_type == 'opt':
        return {
            'attention': ['self_attn.q_proj', 'self_attn.k_proj',
                         'self_attn.v_proj', 'self_attn.out_proj'],
            'mlp': ['fc1', 'fc2'],
            'output': ['lm_head', 'embed_tokens']
        }
    
    else:
        # Generic fallback
        print(f"⚠️  Unknown architecture '{model_type}', using generic mapping")
        return {
            'attention': ['attn', 'attention'],
            'mlp': ['mlp', 'ffn', 'fc'],
            'output': ['lm_head']
        }


def make_param_filter(model, keywords):
    """Create parameter filter function for specific layer types"""
    def filter_fn(m):
        params = []
        for name, param in m.named_parameters():
            if any(keyword in name for keyword in keywords):
                params.append(param)
        return params
    return filter_fn


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_layerwise(model_path, output_dir, num_batches=5, num_eigenthings=20, device='auto'):
    """
    Run layer-wise Hessian analysis.
    
    Args:
        model_path: Path to model (HF hub or local)
        output_dir: Directory to save results
        num_batches: Number of batches for Hessian computation (default: 5)
        num_eigenthings: Number of top eigenvalues to compute
        device: Device to use - 'auto', 'cuda', or 'cpu' (default: 'auto')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cpu':
        print("⚠️  CPU MODE - This will be SLOW but stable")
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    print(f"📚 Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation[:200]")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors=None
        )
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Get layer groups
    layer_groups = get_layer_groups(model)
    print(f"\n🏗️  Architecture: {model.config.model_type}")
    
    # Analyze each layer group
    results = {}
    
    for layer_name, keywords in layer_groups.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {layer_name}")
        print(f"{'='*60}")
        
        # Clear CUDA cache before each layer group
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create parameter filter
        param_filter = make_param_filter(model, keywords)
        filtered_params = param_filter(model)
        
        if not filtered_params:
            print(f"⚠️  No parameters found for {layer_name}, skipping...")
            results[layer_name] = None
            
            # Save after each layer (incremental)
            output_file = output_dir / "layerwise_hessian.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Progress saved to: {output_file}")
            
            continue
        
        num_params = sum(p.numel() for p in filtered_params)
        print(f"Parameters: {num_params:,}")
        
        try:
            # Compute eigenvalues
            eigenvalues, _ = get_hessian_eigenvectors(
                model=model,
                loss_fn=loss_fn,
                train_data_loader=dataloader,
                num_batches=num_batches,
                device=device,
                n_top_vectors=num_eigenthings,
                param_extract_fn=param_filter
            )
            
            # Clear cache after computation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Compute metrics
            # Sort by actual value (descending)
            eigenvalues_sorted = np.sort(eigenvalues)[::-1]
            
            # NOTE: Hessian eigenvalues can be negative (indicating saddle points).
            # Positive eigenvalues = convex directions (local minima)
            # Negative eigenvalues = concave directions (saddle points)
            # This is NORMAL and important information about the loss landscape.
            
            # Get top and bottom eigenvalues
            top_eigenvalue = float(eigenvalues_sorted[0])
            bottom_eigenvalue = float(eigenvalues_sorted[-1])
            median_eigenvalue = float(np.median(eigenvalues_sorted))
            
            # Count positive/negative eigenvalues
            num_positive = int(np.sum(eigenvalues_sorted > 0))
            num_negative = int(np.sum(eigenvalues_sorted < 0))
            
            # Spectral ratio: use absolute values
            # This measures the "spread" of eigenvalues
            spectral_ratio_abs = abs(top_eigenvalue) / abs(median_eigenvalue) if median_eigenvalue != 0 else float('inf')
            
            # Trace (sum of eigenvalues) - approximation from top-k
            # Negative trace = more saddle point directions than convex directions
            trace_approx = float(eigenvalues_sorted.sum())
            
            results[layer_name] = {
                'num_params': num_params,
                'top_eigenvalues': eigenvalues_sorted.tolist(),
                'top_eigenvalue': top_eigenvalue,
                'bottom_eigenvalue': bottom_eigenvalue,
                'median_eigenvalue': median_eigenvalue,
                'num_positive': num_positive,
                'num_negative': num_negative,
                'spectral_ratio_abs': spectral_ratio_abs,
                'trace_approx': trace_approx
            }
            
            print(f"✅ Top eigenvalue: {top_eigenvalue:.2f}")
            print(f"✅ Bottom eigenvalue: {bottom_eigenvalue:.2f}")
            print(f"✅ Median eigenvalue: {median_eigenvalue:.4f}")
            print(f"✅ Eigenvalue signs: {num_positive} positive, {num_negative} negative")
            print(f"✅ Spectral ratio (|max|/|median|): {spectral_ratio_abs:.2f}")
            print(f"✅ Trace (approx): {trace_approx:.2f}")
            
            # Save after each layer (incremental)
            output_file = output_dir / "layerwise_hessian.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Progress saved to: {output_file}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            # Save partial results before exiting
            output_file = output_dir / "layerwise_hessian.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Partial results saved to: {output_file}")
            raise
            
        except BaseException as e:
            print(f"❌ Error analyzing {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            results[layer_name] = {'error': str(e), 'error_type': type(e).__name__}
            
            # Save partial results even on error
            output_file = output_dir / "layerwise_hessian.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Partial results saved to: {output_file}")
    
    # Final save (redundant but ensures consistency)
    output_file = output_dir / "layerwise_hessian.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Final results saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for layer_name, data in results.items():
        if data is None:
            print(f"{layer_name:15s}: No parameters")
        elif 'error' in data:
            print(f"{layer_name:15s}: Error")
        else:
            print(f"{layer_name:15s}: top_λ={data['top_eigenvalue']:8.2f}  "
                  f"median_λ={data['median_eigenvalue']:8.2f}  "
                  f"ratio={data['spectral_ratio_abs']:8.2f}  "
                  f"(+{data['num_positive']}/-{data['num_negative']})")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer-wise Hessian Analysis for Language Models"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model (HuggingFace hub ID or local path)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save results'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=5,
        help='Number of batches for Hessian computation (default: 5)'
    )
    parser.add_argument(
        '--num_eigenthings',
        type=int,
        default=20,
        help='Number of top eigenvalues to compute (default: 20)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use: auto (cuda if available), cuda, or cpu (default: auto)'
    )
    
    args = parser.parse_args()
    
    analyze_layerwise(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        num_eigenthings=args.num_eigenthings,
        device=args.device
    )