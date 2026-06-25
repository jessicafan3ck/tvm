"""Curated 9-modifier CLIP image ceiling vs F_obj floor, on the SAME 4,994 images.
Run with: /Users/jessicafan/tvm/venv/bin/python strengthening/clip_ceiling_curated.py"""
import json, glob, os, warnings, numpy as np, pandas as pd, torch, clip
warnings.filterwarnings('ignore')
from PIL import Image
from collections import Counter
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

CUR = "/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"

def flatten(feats, prefix='', out=None):
    if out is None: out = {}
    for k, v in feats.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict): flatten(v, key+'_', out)
        elif isinstance(v, bool): out[key] = float(v)
        elif isinstance(v, (int, float)): out[key] = float(v)
    return out

rows, ys, subj, paths = [], [], [], []
for f in sorted(glob.glob(os.path.join(CUR, 'TVM JSON', '*.json'))):
    d = json.load(open(f)); rp = d['meta'].get('_relpath', '')
    p = os.path.join(CUR, rp)
    if not (rp and os.path.exists(p)): continue
    label = os.path.basename(f).split('_', 1)[1].rsplit('.json', 1)[0]
    toks = label.split()
    rows.append(flatten(d['features'])); ys.append(toks[0])
    subj.append(' '.join(toks[1:]) if len(toks) > 1 else '(none)'); paths.append(p)

df = pd.DataFrame(rows).fillna(0)
Xobj = RobustScaler().fit_transform(df.values)
y = np.array(ys); groups = np.array(subj)
print(f'n={len(y)} modifiers={dict(Counter(y))}')

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model, prep = clip.load('ViT-B/32', device=dev)
print(f'embedding {len(paths)} images on {dev}...')
embs, batch = [], []
with torch.no_grad():
    for i, p in enumerate(paths):
        try: batch.append(prep(Image.open(p).convert('RGB')))
        except Exception: batch.append(torch.zeros(3, 224, 224))
        if len(batch) == 64 or i == len(paths)-1:
            embs.append(model.encode_image(torch.stack(batch).to(dev)).cpu().numpy()); batch = []
            if (i+1) % 1024 < 64: print(f'  {i+1}/{len(paths)}')
Xclip = np.vstack(embs)
np.save(f'{OUT}/Xclip_curated.npy', Xclip)
print('CLIP', Xclip.shape)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
def probe(X, name, clf=None, cvv=cv, grp=None):
    clf = clf or HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)
    pred = cross_val_predict(clf, X, y, cv=cvv, groups=grp, n_jobs=-1)
    print(f'  {name:38s} balanced_acc={balanced_accuracy_score(y,pred):.3f}  macro_F1={f1_score(y,pred,average="macro"):.3f}')
    return pred

print(f'\n=== Curated floor vs ceiling, SAME {len(y)} images (chance={1/len(set(y)):.3f}) ===')
probe(Xobj, 'F_obj (hand-crafted composition)')
probe(StandardScaler().fit_transform(Xclip), 'CLIP-image (linear probe)',
      LogisticRegression(max_iter=3000, class_weight='balanced'))
pred_clip = probe(Xclip, 'CLIP-image (GBM)')
comb = np.hstack([StandardScaler().fit_transform(Xobj), StandardScaler().fit_transform(Xclip)])
probe(comb, 'F_obj + CLIP (combined, linear)', LogisticRegression(max_iter=3000, class_weight='balanced'))
print('  -- subject-independent (GroupKFold by subject) --')
probe(StandardScaler().fit_transform(Xclip), 'CLIP-image, grouped by subject',
      LogisticRegression(max_iter=3000, class_weight='balanced'), GroupKFold(5), groups)

labs = sorted(set(y))
cm = confusion_matrix(y, pred_clip, labels=labs, normalize='true')
print('\nCLIP confusion (row=true, normalized), most-confused pairs:')
pairs = [(labs[i], labs[j], cm[i, j]) for i in range(len(labs)) for j in range(len(labs)) if i != j]
for a, b, v in sorted(pairs, key=lambda t: -t[2])[:8]:
    print(f'  {a:13s} -> {b:13s} {v:.2f}')
