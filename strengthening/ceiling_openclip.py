"""Push the ceiling with stronger backbones via open_clip (HuggingFace CDN).
ViT-L/14 (scale) and SigLIP-SO400M-384 (stronger architecture) on curated 4,994.
Run: /Users/jessicafan/tvm/venv/bin/python strengthening/ceiling_openclip.py"""
import json, glob, os, warnings, numpy as np, pandas as pd, torch, open_clip
warnings.filterwarnings('ignore')
from PIL import Image
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
y = np.array(ys); n = len(y); cv = StratifiedKFold(5, shuffle=True, random_state=42)
clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')
print(f'n={n} chance={1/len(set(y)):.3f}', flush=True)

def probe(X):
    p = cross_val_predict(clf(), StandardScaler().fit_transform(X), y, cv=cv, n_jobs=-1)
    return p, balanced_accuracy_score(y, p), f1_score(y, p, average='macro')

def embed(name, pretrained, bs, tag):
    cache = f'{OUT}/oc_{tag}_curated.npy'
    if os.path.exists(cache): print(f'[{tag}] cached', flush=True); return np.load(cache)
    model, _, prep = open_clip.create_model_and_transforms(name, pretrained=pretrained, cache_dir=f'{OUT}/oc_models')
    model = model.to(dev).eval()
    print(f'[{tag}] embedding on {dev}...', flush=True)
    embs, batch = [], []
    with torch.no_grad():
        for i, p in enumerate(paths):
            try: batch.append(prep(Image.open(p).convert('RGB')))
            except Exception: batch.append(torch.zeros_like(prep(Image.new('RGB',(64,64)))))
            if len(batch) == bs or i == n-1:
                embs.append(model.encode_image(torch.stack(batch).to(dev)).float().cpu().numpy()); batch=[]
                if (i+1) % 1024 < bs: print(f'   {i+1}/{n}', flush=True)
    X = np.vstack(embs); np.save(cache, X); del model
    import gc; gc.collect()
    return X

print(f'\n=== open_clip ceiling (linear probe, chance {1/len(set(y)):.3f}) ===', flush=True)
print(f'  {"F_obj (floor)":26s} bacc={probe(Xobj)[1]:.3f}', flush=True)
res = {}
for name, pre, bs, tag in [('ViT-L-14', 'openai', 24, 'ViT-L-14-openai'),
                           ('ViT-SO400M-14-SigLIP-384', 'webli', 8, 'SigLIP-SO400M-384')]:
    try:
        X = embed(name, pre, bs, tag); p, ba, f1 = probe(X); res[tag] = (X, p, ba, f1)
        print(f'  {tag:26s} bacc={ba:.3f}  macroF1={f1:.3f}', flush=True)
    except Exception as e:
        print(f'  {tag:26s} FAILED: {e}', flush=True)

if res:
    best = max(res, key=lambda k: res[k][2]); Xb, pb, _, _ = res[best]
    comb = np.hstack([StandardScaler().fit_transform(Xobj), StandardScaler().fit_transform(Xb)])
    print(f'  {"F_obj + "+best:26s} bacc={probe(comb)[1]:.3f}', flush=True)
    print(f'\n=== per-modifier P/R — best: {best} ===', flush=True)
    print(classification_report(y, pb, digits=3), flush=True)
print('DONE', flush=True)
