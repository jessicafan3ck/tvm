"""
TVM strengthening battery — supervised probes + confound controls + signal quantification.
The paper's claim ("sentiment not encoded in F_obj") rests on unsupervised k-means failing.
That is weak. Here we test it the rigorous way.
"""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import f_oneway

DATA = '/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)/TVM JSON'

def flatten(feats, prefix='', out=None):
    if out is None: out = {}
    for k, v in feats.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict): flatten(v, key+'_', out)
        elif isinstance(v, bool): out[key] = float(v)
        elif isinstance(v, (int, float)): out[key] = float(v)
    return out

rows = []
for f in sorted(glob.glob(os.path.join(DATA, '*.json'))):
    d = json.load(open(f))
    label = os.path.basename(f).split('_', 1)[1].rsplit('.json', 1)[0]
    toks = label.split()
    r = flatten(d['features'])
    r['_modifier'] = toks[0]
    r['_subject'] = ' '.join(toks[1:]) if len(toks) > 1 else '(none)'
    rows.append(r)

df = pd.DataFrame(rows).fillna(0)
feat_cols = [c for c in df.columns if not c.startswith('_')]
X = RobustScaler().fit_transform(df[feat_cols].values)
y = df['_modifier'].to_numpy(dtype=str)
subj = df['_subject'].to_numpy(dtype=str)
n = len(df)
print(f'n={n}, n_features={len(feat_cols)}, n_modifiers={df._modifier.nunique()}, n_subjects={df._subject.nunique()}')
maj = Counter(y).most_common(1)[0]
print(f'majority class: {maj[0]} = {maj[1]/n:.3f}  | random-balanced chance = {1/df._modifier.nunique():.3f}')

cv = StratifiedKFold(5, shuffle=True, random_state=42)
def evaluate(clf, X, y, cv, name):
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    ba = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    print(f'  {name:28s} balanced_acc={ba:.3f}  macro_F1={f1:.3f}')
    return pred, ba, f1

print('\n=== TEST 1: Can F_obj predict MODIFIER (vibe)? (stratified 5-fold) ===')
logit = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
hgb = HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)
evaluate(logit, X, y, cv, 'LogReg (balanced)')
pred_hgb,_,_ = evaluate(hgb, X, y, cv, 'GradientBoosting')

print('\n  Permutation test (shuffle labels, GBM) — establishes the real noise floor:')
rng = np.random.RandomState(0)
perm_ba = []
for i in range(5):
    yp = rng.permutation(y)
    p = cross_val_predict(hgb, X, yp, cv=StratifiedKFold(5, shuffle=True, random_state=i), n_jobs=-1)
    perm_ba.append(balanced_accuracy_score(yp, p))
print(f'    permuted balanced_acc = {np.mean(perm_ba):.3f} +/- {np.std(perm_ba):.3f}  (this is true chance)')

print('\n=== TEST 2: Subject confound — does F_obj predict SUBJECT better than MODIFIER? ===')
# restrict to subjects with enough samples for a fair multiclass probe
sc = Counter(subj); keep = {s for s,c in sc.items() if c >= 20}
m = np.array([s in keep for s in subj])
print(f'  subjects with >=20 samples: {len(keep)} covering {m.sum()} images')
evaluate(hgb, X[m], subj[m], StratifiedKFold(5, shuffle=True, random_state=42), 'F_obj -> SUBJECT')

print('\n=== TEST 3: Does modifier signal SURVIVE when subjects cannot leak? (GroupKFold by subject) ===')
gkf = GroupKFold(n_splits=5)
pred = cross_val_predict(hgb, X, y, cv=gkf, groups=subj, n_jobs=-1)
print(f'  F_obj -> MODIFIER, grouped by subject: balanced_acc={balanced_accuracy_score(y,pred):.3f}  macro_F1={f1_score(y,pred,average="macro"):.3f}')

print('\n=== TEST 4: Where is the signal? per-feature ANOVA across modifiers (top 12) ===')
fvals = {}
for j, c in enumerate(feat_cols):
    groups = [X[y == m_, j] for m_ in np.unique(y)]
    fvals[c] = f_oneway(*groups).statistic
top = sorted(fvals.items(), key=lambda kv: -kv[1])[:12]
for c, fv in top: print(f'  {c:48s} F={fv:7.1f}')

print('\n=== TEST 5: Mutual information F_obj features vs modifier (total bits of signal) ===')
mi = mutual_info_classif(X, y, random_state=42)
print(f'  sum MI = {mi.sum():.3f} nats | top features:')
for c, v in sorted(zip(feat_cols, mi), key=lambda kv:-kv[1])[:8]:
    print(f'    {c:48s} {v:.4f}')

print('\n=== TEST 6: Outlier-driven k=2? quantify the 99.4/0.6 split ===')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
km = KMeans(2, random_state=42, n_init=10).fit(X)
sizes = Counter(km.labels_)
print(f'  k=2 cluster sizes: {dict(sizes)}  silhouette(full)={silhouette_score(X, km.labels_):.4f}')
big = max(sizes, key=sizes.get)
mask = km.labels_ == big
Xin = X[mask]
for k in (2,3,5,9):
    lab = KMeans(k, random_state=42, n_init=10).fit_predict(Xin)
    print(f'  WITHIN main blob (n={mask.sum()}), k={k}: silhouette={silhouette_score(Xin,lab):.4f}')

np.save('/private/tmp/claude-501/-Users-jessicafan/33041f73-b2cb-4f34-a907-05c227e6ccb0/scratchpad/X.npy', X)
df[['_modifier','_subject']].to_csv('/private/tmp/claude-501/-Users-jessicafan/33041f73-b2cb-4f34-a907-05c227e6ccb0/scratchpad/labels.csv', index=False)
print('\nsaved X.npy + labels.csv for follow-ups')
