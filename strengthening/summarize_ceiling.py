"""Summarize CLIP ceiling from cached embeddings (no re-embedding).
Per-modifier precision/recall for best backbone + F_obj floor + combined."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

CUR = "/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"

def flatten(feats, prefix='', out=None):
    if out is None: out = {}
    for k, v in feats.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict): flatten(v, key+'_', out)
        elif isinstance(v, (bool, int, float)): out[key] = float(v)
    return out

rows, ys = [], []
for f in sorted(glob.glob(os.path.join(CUR, 'TVM JSON', '*.json'))):
    d = json.load(open(f)); rp = d['meta'].get('_relpath', '')
    if not (rp and os.path.exists(os.path.join(CUR, rp))): continue
    rows.append(flatten(d['features']))
    ys.append(os.path.basename(f).split('_', 1)[1].rsplit('.json', 1)[0].split()[0])
Xobj = RobustScaler().fit_transform(pd.DataFrame(rows).fillna(0).values)
y = np.array(ys); cv = StratifiedKFold(5, shuffle=True, random_state=42)
clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')

def probe(X):
    p = cross_val_predict(clf(), StandardScaler().fit_transform(X), y, cv=cv, n_jobs=-1)
    return p, balanced_accuracy_score(y, p), f1_score(y, p, average='macro')

backbones = {'ViT-B/32': 'clip_ViT-B-32_curated.npy', 'ViT-B/16': 'clip_ViT-B-16_curated.npy',
             'ViT-L/14': 'clip_ViT-L-14_curated.npy'}
print(f'n={len(y)} chance={1/len(set(y)):.3f}\n=== ceiling by backbone (linear probe) ===')
print(f'  {"F_obj (floor)":14s} bacc={probe(Xobj)[1]:.3f}')
res = {}
for name, fn in backbones.items():
    path = os.path.join(OUT, fn)
    if not os.path.exists(path): print(f'  {name:14s} (no cache yet)'); continue
    X = np.load(path); p, ba, f1 = probe(X); res[name] = (X, p, ba, f1)
    print(f'  {name:14s} bacc={ba:.3f}  macroF1={f1:.3f}')

best = max(res, key=lambda k: res[k][2]); Xb, pb, _, _ = res[best]
comb = np.hstack([StandardScaler().fit_transform(Xobj), StandardScaler().fit_transform(Xb)])
pc, bac, f1c = probe(comb)
print(f'  {"F_obj + "+best:14s} bacc={bac:.3f}  macroF1={f1c:.3f}')

print(f'\n=== per-modifier precision/recall — best backbone: {best} ===')
print(classification_report(y, pb, digits=3))
print('=== per-modifier precision/recall — F_obj floor ===')
pf, _, _ = probe(Xobj)
print(classification_report(y, pf, digits=3))
