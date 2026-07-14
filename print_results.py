import json
import numpy as np

d = json.load(open('results/circuit_analysis/causal_restoration_results.json'))

for model, res in d.items():
    print(f'\n{model}:')
    print(f'  Gen0 PPL: {res["baseline_gen0_ppl"]:.3f}')
    print(f'  Gen5 PPL: {res["baseline_gen5_ppl"]:.3f}')
    
    layers = {int(k): v for k, v in res['layers'].items()}
    top = sorted(layers.items(), key=lambda x: x[1]['recovery'], reverse=True)[:5]
    
    print('  Top 5 recovery layers:')
    for l, v in top:
        print(f'    Layer {l:2d}: recovery={v["recovery"]:+.4f}  patched_ppl={v["patched_ppl"]:.3f}')