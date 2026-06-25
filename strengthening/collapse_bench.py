"""Benchmark (8 emotions) replication: does the 'residual = subjectivity' finding generalize?
Reuses cached CLIP-B/32 benchmark embeddings. Full report + progressive merge of confused pairs."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report, confusion_matrix

SP = '/Users/jessicafan/tvm/strengthening/artifacts/'
X = StandardScaler().fit_transform(np.load(SP + 'Xclip_bench.npy'))
y = pd.read_csv(SP + 'bench_paired.csv')['emotion'].to_numpy(dtype=str)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')

def run(yy):
    p = cross_val_predict(clf(), X, yy, cv=cv, n_jobs=-1)
    return p, balanced_accuracy_score(yy, p), accuracy_score(yy, p), f1_score(yy, p, average='macro')

print(f'=== Benchmark CLIP-B/32 per-emotion report (n={len(y)}, chance={1/len(set(y)):.3f}) ===')
p0, ba0, ac0, f0 = run(y)
print(classification_report(y, p0, digits=3))

print('=== Progressive merge of most-confused emotion pair ===')
print(f'  {len(set(y))} classes: balanced_acc={ba0:.3f}  raw_acc={ac0:.3f}  macroF1={f0:.3f}')
ycur = y.copy()
while len(set(ycur)) > 3:
    labs = sorted(set(ycur))
    p = cross_val_predict(clf(), X, ycur, cv=cv, n_jobs=-1)
    cm = confusion_matrix(ycur, p, labels=labs, normalize='true')
    a, b, _ = max(((labs[i], labs[j], (cm[i, j]+cm[j, i])/2)
                   for i in range(len(labs)) for j in range(i+1, len(labs))), key=lambda t: t[2])
    ycur = np.array([f'{a}+{b}' if v in (a, b) else v for v in ycur])
    _, ba, ac, f1 = run(ycur)
    print(f'  {len(set(ycur))} classes (merged {a}<->{b}): balanced_acc={ba:.3f}  raw_acc={ac:.3f}  macroF1={f1:.3f}')
print('DONE')
