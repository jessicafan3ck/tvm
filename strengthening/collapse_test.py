"""Capacity vs subjectivity: using cached SigLIP-SO400M embeddings, (1) full per-modifier
report, (2) confusion matrix, (3) greedily merge most-confused label pairs and re-probe,
showing accuracy vs #classes. Fast recovery => the 'ceiling' is taxonomy overlap, not capacity."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report, confusion_matrix

CUR = "/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
X = StandardScaler().fit_transform(np.load(f'{OUT}/oc_SigLIP-SO400M-384_curated.npy'))

ys = []
for f in sorted(glob.glob(os.path.join(CUR, 'TVM JSON', '*.json'))):
    d = json.load(open(f)); rp = d['meta'].get('_relpath', '')
    if not (rp and os.path.exists(os.path.join(CUR, rp))): continue
    ys.append(os.path.basename(f).split('_', 1)[1].rsplit('.json', 1)[0].split()[0])
y = np.array(ys)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')

def run(yy):
    p = cross_val_predict(clf(), X, yy, cv=cv, n_jobs=-1)
    return p, balanced_accuracy_score(yy, p), accuracy_score(yy, p), f1_score(yy, p, average='macro')

print('=== Full SigLIP per-modifier report (9 classes) ===')
p0, ba0, ac0, f0 = run(y)
print(classification_report(y, p0, digits=3))

labs = sorted(set(y))
cm = confusion_matrix(y, p0, labels=labs, normalize='true')
print('Most-confused label pairs (symmetric avg of off-diagonals):')
pairs = []
for i in range(len(labs)):
    for j in range(i+1, len(labs)):
        pairs.append((labs[i], labs[j], (cm[i, j]+cm[j, i])/2))
for a, b, v in sorted(pairs, key=lambda t: -t[2])[:6]:
    print(f'  {a:13s} <-> {b:13s} {v:.3f}')

print('\n=== Progressive merge of most-confused pair, re-probe (capacity vs subjectivity) ===')
print(f'  {len(set(y))} classes: balanced_acc={ba0:.3f}  raw_acc={ac0:.3f}  macroF1={f0:.3f}')
ycur = y.copy()
while len(set(ycur)) > 4:
    labs = sorted(set(ycur))
    p = cross_val_predict(clf(), X, ycur, cv=cv, n_jobs=-1)
    cm = confusion_matrix(ycur, p, labels=labs, normalize='true')
    best = max(((labs[i], labs[j], (cm[i, j]+cm[j, i])/2)
                for i in range(len(labs)) for j in range(i+1, len(labs))), key=lambda t: t[2])
    a, b, _ = best
    merged = f'{a}+{b}'
    ycur = np.array([merged if v in (a, b) else v for v in ycur])
    p2, ba, ac, f1 = run(ycur)
    print(f'  {len(set(ycur))} classes (merged {a}<->{b}): balanced_acc={ba:.3f}  raw_acc={ac:.3f}  macroF1={f1:.3f}')
print('DONE')
