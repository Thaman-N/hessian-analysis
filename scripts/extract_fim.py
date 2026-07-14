import json, sys

def extract_fim(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data.get('blocks', []):
        if not isinstance(entry, dict): continue
        b = entry.get('block_idx', 0)
        attn = entry.get('attention', {})
        mlp  = entry.get('mlp', {})
        at = float(attn.get('top', 0)) if isinstance(attn, dict) and 'error' not in attn else 0.0
        mt = float(mlp.get('top', 0))  if isinstance(mlp,  dict) and 'error' not in mlp  else 0.0
        if at > 0 or mt > 0:
            out[b] = at + mt
    return out

path = sys.argv[1]
fim = extract_fim(path)
for b in sorted(fim):
    print(f"{b},{fim[b]:.4f}")