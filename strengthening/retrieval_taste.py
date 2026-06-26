"""Product-mechanic validation:
 A 'shop the vibe' retrieval precision@k (does NN retrieval return same-vibe items?)
 B few-shot taste learning (learn a user's vibe from k 'likes', rank the rest -> how fast
   does the flywheel spin?)"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"

def L2(X): return X / np.linalg.norm(X, axis=1, keepdims=True)
def load(npy, lab): return L2(np.load(f'{OUT}/{npy}')), lab

curated_y = pd.read_csv(f'{OUT}/labels.csv')['_modifier'].to_numpy(dtype=str)
scraped_y = pd.read_csv(f'{OUT}/scraped_meta.csv')['adj'].to_numpy(dtype=str)
SETS = {
  'curated/B32':   load('Xclip_curated.npy', curated_y),
  'curated/SigLIP':load('oc_SigLIP-SO400M-384_curated.npy', curated_y),
  'scraped/B32':   load('Xscraped_b32.npy', scraped_y),
  'scraped/SigLIP':load('Xscraped_siglip.npy', scraped_y),
}

def patk(X, y, ks=(1,5,10)):
    nn = NearestNeighbors(n_neighbors=max(ks)+1, metric='cosine').fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]                       # drop self
    same = (y[idx] == y[:, None])
    return {k: same[:, :k].mean() for k in ks}

def patk_per_class(X, y, k=10):
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
    _, idx = nn.kneighbors(X); idx = idx[:, 1:]
    same = (y[idx] == y[:, None]).mean(1)
    return {c: same[y == c].mean() for c in sorted(set(y))}

print('=== A: "shop the vibe" retrieval precision@k (random = class prior) ===')
for name, (X, y) in SETS.items():
    prior = max(Counter(y).values())/len(y); chance = sum((v/len(y))**2 for v in Counter(y).values())
    p = patk(X, y)
    print(f'  {name:16s} P@1={p[1]:.3f} P@5={p[5]:.3f} P@10={p[10]:.3f}  (random P@k≈{chance:.3f})')

print('\n=== A2: concrete vs abstract vibes (scraped/SigLIP, P@10) ===')
X, y = SETS['scraped/SigLIP']; pc = patk_per_class(X, y, 10)
rank = sorted(pc.items(), key=lambda t: -t[1])
print('  cleanest-retrieving:', ', '.join(f'{c} {v:.2f}' for c, v in rank[:6]))
print('  weakest-retrieving :', ', '.join(f'{c} {v:.2f}' for c, v in rank[-6:]))

print('\n=== A3: per-modifier retrieval (curated/SigLIP, P@10) ===')
X, y = SETS['curated/SigLIP']
for c, v in sorted(patk_per_class(X, y, 10).items(), key=lambda t: -t[1]):
    print(f'    {c:13s} {v:.3f}')

print('\n=== B: few-shot taste learning — learn a vibe from k likes, rank the rest (mean AP) ===')
def fewshot(X, y, ks=(1,3,5,10), reps=20, seed=0):
    rng = np.random.RandomState(seed); classes = sorted(set(y)); out = {}
    for k in ks:
        aps = []
        for c in classes:
            pos = np.where(y == c)[0]
            if len(pos) <= k+5: continue
            for _ in range(reps):
                liked = rng.choice(pos, k, replace=False)
                proto = L2(X[liked].mean(0, keepdims=True))[0]
                pool = np.setdiff1d(np.arange(len(y)), liked)
                scores = X[pool] @ proto
                aps.append(average_precision_score((y[pool] == c).astype(int), scores))
        out[k] = np.mean(aps)
    return out, sum((v/len(y))**2 for v in Counter(y).values())

for name in ('curated/SigLIP', 'scraped/SigLIP'):
    X, y = SETS[name]; res, base = fewshot(X, y)
    print(f'  {name:16s} ' + '  '.join(f'k={k}:AP={v:.3f}' for k, v in res.items()) + f'   (random AP≈{base:.3f})')
print('DONE')
