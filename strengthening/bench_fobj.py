"""Benchmark (EmotionROI/FI subset, 8 emotions): F_obj supervised probe + resolve image paths
for a PAIRED CLIP comparison. Replicates the new curated finding on an independent dataset."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score

PF = "/Users/jessicafan/Library/CloudStorage/GoogleDrive-tvm.tiervibemap@gmail.com/My Drive/tvm/per_file_results"
IMG = "/Users/jessicafan/tvm_data/emotion_dataset"
EMAP = {'sad': 'sadness'}  # json label -> image folder

def flatten(feats, prefix='', out=None):
    if out is None: out = {}
    for k, v in feats.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict): flatten(v, key+'_', out)
        elif isinstance(v, bool): out[key] = float(v)
        elif isinstance(v, (int, float)): out[key] = float(v)
    return out

rows, ys, imgpaths = [], [], []
missing = 0
for f in sorted(glob.glob(os.path.join(PF, '*', '*.json'))):
    emo = os.path.basename(os.path.dirname(f))
    stem = os.path.basename(f).rsplit('.json', 1)[0]      # excitement_0000_6345
    parts = stem.split('_')
    idx = parts[1] if len(parts) > 1 else None             # 0000
    folder = EMAP.get(emo, emo)
    img = os.path.join(IMG, folder, f'{folder}_{idx}.jpg') if idx else ''
    if not os.path.exists(img):
        # try json-label prefix too
        alt = os.path.join(IMG, folder, f'{emo}_{idx}.jpg')
        img = alt if os.path.exists(alt) else ''
    try:
        d = json.load(open(f))
        r = flatten(d['features'])
    except Exception:
        continue
    rows.append(r); ys.append(emo); imgpaths.append(img)
    if not img: missing += 1

df = pd.DataFrame(rows).fillna(0)
feat_cols = [c for c in df.columns]
X = RobustScaler().fit_transform(df[feat_cols].values)
y = np.array(ys)
n = len(y)
print(f'benchmark: n={n}, n_features={len(feat_cols)}, classes={df.shape and dict(Counter(y))}')
print(f'images resolved: {n-missing}/{n} (missing {missing})')
print(f'majority={Counter(y).most_common(1)[0][1]/n:.3f}  random-chance={1/len(set(y)):.3f}')

cv = StratifiedKFold(5, shuffle=True, random_state=42)
hgb = HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)
pred = cross_val_predict(hgb, X, y, cv=cv, n_jobs=-1)
print(f'\nF_obj -> EMOTION: balanced_acc={balanced_accuracy_score(y,pred):.3f}  macro_F1={f1_score(y,pred,average="macro"):.3f}')
rng = np.random.RandomState(0)
perm = [balanced_accuracy_score(yp:=rng.permutation(y), cross_val_predict(hgb,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=i),n_jobs=-1)) for i in range(3)]
print(f'permuted (true chance): {np.mean(perm):.3f}')

# save paired arrays (only rows with resolved images) for CLIP step
mask = np.array([bool(p) for p in imgpaths])
np.save('/private/tmp/claude-501/-Users-jessicafan/33041f73-b2cb-4f34-a907-05c227e6ccb0/scratchpad/Xbench.npy', X[mask])
pd.DataFrame({'emotion': y[mask], 'img': [p for p,m in zip(imgpaths,mask) if m]}).to_csv(
    '/private/tmp/claude-501/-Users-jessicafan/33041f73-b2cb-4f34-a907-05c227e6ccb0/scratchpad/bench_paired.csv', index=False)
print(f'\nsaved {mask.sum()} paired (F_obj, image) rows for CLIP ceiling test')
