"""Raise the ceiling: multi-backbone CLIP image probe on curated 4,994 (9 modifiers),
+ per-modifier precision/recall for the best. Caches each embedding set.
Run: /Users/jessicafan/tvm/venv/bin/python strengthening/ceiling_backbones.py"""
import json, glob, os, warnings, numpy as np, pandas as pd, torch, clip
warnings.filterwarnings('ignore')
from PIL import Image
from collections import Counter
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

CUR = "/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'

def flatten(feats, prefix='', out=None):
    if out is None: out = {}
    for k, v in feats.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict): flatten(v, key+'_', out)
        elif isinstance(v, (bool, int, float)): out[key] = float(v)
    return out

rows, ys, paths = [], [], []
for f in sorted(glob.glob(os.path.join(CUR, 'TVM JSON', '*.json'))):
    d = json.load(open(f)); rp = d['meta'].get('_relpath', ''); p = os.path.join(CUR, rp)
    if not (rp and os.path.exists(p)): continue
    rows.append(flatten(d['features']))
    ys.append(os.path.basename(f).split('_', 1)[1].rsplit('.json', 1)[0].split()[0]); paths.append(p)
Xobj = RobustScaler().fit_transform(pd.DataFrame(rows).fillna(0).values)
y = np.array(ys); n = len(y)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
print(f'n={n}, chance={1/len(set(y)):.3f}', flush=True)

def embed(model_name, bs):
    safe = model_name.replace('/', '-').replace('@', '_')
    cache = f'{OUT}/clip_{safe}_curated.npy'
    if os.path.exists(cache):
        print(f'[{model_name}] cached', flush=True); return np.load(cache)
    model, prep = clip.load(model_name, device=dev)
    print(f'[{model_name}] embedding on {dev}...', flush=True)
    embs, batch = [], []
    with torch.no_grad():
        for i, p in enumerate(paths):
            try: batch.append(prep(Image.open(p).convert('RGB')))
            except Exception: batch.append(torch.zeros(3, prep.transforms[1].size[0], prep.transforms[1].size[0]))
            if len(batch) == bs or i == n-1:
                embs.append(model.encode_image(torch.stack(batch).to(dev)).float().cpu().numpy()); batch = []
                if (i+1) % 1024 < bs: print(f'   {i+1}/{n}', flush=True)
    X = np.vstack(embs); np.save(cache, X); del model; return X

def probe(X):
    Xs = StandardScaler().fit_transform(X)
    pred = cross_val_predict(LogisticRegression(max_iter=4000, class_weight='balanced'), Xs, y, cv=cv, n_jobs=-1)
    return pred, balanced_accuracy_score(y, pred), f1_score(y, pred, average='macro')

results = {}
print(f'\n=== ceiling by backbone (linear probe, chance {1/len(set(y)):.3f}) ===', flush=True)
print(f'  {"F_obj (baseline)":22s} bacc={probe(Xobj)[1]:.3f}', flush=True)
for mn, bs in [('ViT-B/32', 64), ('ViT-B/16', 48), ('ViT-L/14', 24), ('ViT-L/14@336px', 12)]:
    X = embed(mn, bs); pred, ba, f1 = probe(X)
    results[mn] = (X, pred, ba, f1)
    print(f'  {mn:22s} bacc={ba:.3f}  macroF1={f1:.3f}', flush=True)

best = max(results, key=lambda k: results[k][2])
Xb, predb, _, _ = results[best]
comb = np.hstack([StandardScaler().fit_transform(Xobj), StandardScaler().fit_transform(Xb)])
predc, bac, f1c = probe(comb)
print(f'\n  {"F_obj + "+best:22s} bacc={bac:.3f}  macroF1={f1c:.3f}', flush=True)

print(f'\n=== per-modifier precision/recall — best backbone: {best} ===', flush=True)
print(classification_report(y, predb, digits=3), flush=True)
print(f'=== per-modifier precision/recall — F_obj only (floor) ===', flush=True)
print(classification_report(y, cross_val_predict(LogisticRegression(max_iter=4000, class_weight='balanced'),
      StandardScaler().fit_transform(Xobj), y, cv=cv, n_jobs=-1), digits=3), flush=True)
print('DONE', flush=True)
