import psutil
import torch.profiler

def log_memory():
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    gpu_cached = torch.cuda.memory_reserved() / 1024**3
    ram_mem = psutil.Process().memory_info().rss / 1024**3
    print(f"🔍 GPU: {gpu_mem:.2f}GB allocated | {gpu_cached:.2f}GB cached | RAM: {ram_mem:.2f}GB")

# Add memory logging at critical points:
print("📊 Memory before model load:")
log_memory()

model = AutoModelForCausalLM.from_pretrained(...)
print("📊 Memory after model load:")
log_memory()

print("📊 Memory before density:")
log_memory()
density_eigen, density_weight = hessian_comp.density(iter=20)
print("📊 Memory after density:")
log_memory()