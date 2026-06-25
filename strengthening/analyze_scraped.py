"""Scraped 30-vibe corpus analyses (CLIP-B/32):
 S1 recoverability at scale (30-way probe), S2 data-driven vibe ontology (centroid
 hierarchy + collapse), S3 weak-label noise + cross-source validity + weighted
 augmentation on the 3 overlapping vibes. Scraped is treated as WEAK (down-weighted)."""
import os, glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
Xs = np.load(f'{OUT}/Xscraped_b32.npy')
meta = pd.read_csv(f'{OUT}/scraped_meta.csv')
ys = meta['adj'].to_numpy(dtype=str)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')
Xss = StandardScaler().fit_transform(Xs)
print(f'scraped n={len(ys)}  adjectives={len(set(ys))}  chance={1/len(set(ys)):.3f}\n')

# ---- S1: recoverability at scale (30-way) ----
print('=== S1: 30-vibe recoverability (CLIP-B/32 linear probe) ===')
p = cross_val_predict(clf(), Xss, ys, cv=cv, n_jobs=-1)
print(f'  balanced_acc={balanced_accuracy_score(ys,p):.3f}  macroF1={f1_score(ys,p,average="macro"):.3f}')
rng = np.random.RandomState(0)
perm = balanced_accuracy_score(yp:=rng.permutation(ys), cross_val_predict(clf(),Xss,yp,cv=cv,n_jobs=-1))
print(f'  permuted chance={perm:.3f}')

# ---- S2a: data-driven vibe ontology via centroid hierarchy ----
print('\n=== S2: data-driven vibe ontology (agglomerative on class centroids, cosine) ===')
labs = sorted(set(ys))
C = np.vstack([Xss[ys==l].mean(0) for l in labs])
Cn = C/np.linalg.norm(C,axis=1,keepdims=True)
D = 1 - Cn@Cn.T; np.fill_diagonal(D,0)
Z = linkage(squareform(D,checks=False), method='average')
for k in (12, 8, 6):
    fc = fcluster(Z, k, criterion='maxclust')
    groups = {}
    for l,c in zip(labs,fc): groups.setdefault(c,[]).append(l)
    print(f'  {k} clusters: ' + ' | '.join(sorted(','.join(sorted(g)) for g in groups.values())))

# ---- S2b: collapse accuracy recovery (confusion-driven) ----
print('\n=== S2b: progressive merge of most-confused pair (capacity vs subjectivity) ===')
ycur = ys.copy(); print(f'  {len(set(ycur))} classes raw_acc={accuracy_score(ys,p):.3f}')
while len(set(ycur)) > 8:
    L = sorted(set(ycur)); pp = cross_val_predict(clf(), Xss, ycur, cv=cv, n_jobs=-1)
    cm = confusion_matrix(ycur, pp, labels=L, normalize='true')
    a,b,_ = max(((L[i],L[j],(cm[i,j]+cm[j,i])/2) for i in range(len(L)) for j in range(i+1,len(L))), key=lambda t:t[2])
    ycur = np.array([f'{a}+{b}' if v in (a,b) else v for v in ycur])
    pm = cross_val_predict(clf(), Xss, ycur, cv=cv, n_jobs=-1)
    print(f'  {len(set(ycur))} (merged {a}<->{b}): raw_acc={accuracy_score(ycur,pm):.3f}')

# ---- S3: overlapping 3 vibes — curated(gold) vs scraped(weak) ----
print('\n=== S3: cross-source validity + weak-label noise + weighted augmentation ===')
OVL = ['ethereal','melancholic','whimsical']
# curated B/32 embeddings + labels (same sorted-JSON order)
ycur_all = pd.read_csv(f'{OUT}/labels.csv')['_modifier'].to_numpy(dtype=str)
Xc_all = np.load(f'{OUT}/Xclip_curated.npy')
assert len(ycur_all)==len(Xc_all), (len(ycur_all), len(Xc_all))
cm_ = np.isin(ycur_all, OVL); Xc = Xc_all[cm_]; yc = ycur_all[cm_]
sm_ = np.isin(ys, OVL); Xsc = Xs[sm_]; ysc = ys[sm_]
print(f'  curated(3)={Counter(yc)}  scraped(3)={Counter(ysc)}')
# joint-standardize for fair cross-source
sc = StandardScaler().fit(np.vstack([Xc, Xsc]))
Xc_s, Xsc_s = sc.transform(Xc), sc.transform(Xsc)

def ba(tr_X,tr_y,te_X,te_y,w=None):
    m=clf().fit(tr_X,tr_y,sample_weight=w); return balanced_accuracy_score(te_y,m.predict(te_X))
# within-source CV (label quality)
print(f'  within-source CV bacc:  curated={balanced_accuracy_score(yc,cross_val_predict(clf(),Xc_s,yc,cv=cv,n_jobs=-1)):.3f}'
      f'  scraped={balanced_accuracy_score(ysc,cross_val_predict(clf(),Xsc_s,ysc,cv=cv,n_jobs=-1)):.3f}  (lower=noisier labels)')
# cross-source transfer
print(f'  cross-source: train CURATED -> test SCRAPED bacc={ba(Xc_s,yc,Xsc_s,ysc):.3f}')
print(f'  cross-source: train SCRAPED -> test CURATED bacc={ba(Xsc_s,ysc,Xc_s,yc):.3f}')
# weighted augmentation: gold=curated train, add scraped at weight alpha, test on held-out curated
Xtr,Xte,ytr,yte = train_test_split(Xc_s,yc,test_size=0.3,stratify=yc,random_state=42)
base = ba(Xtr,ytr,Xte,yte)
print(f'  augmentation (test=held-out curated): curated-only bacc={base:.3f}')
for a in (0.1,0.25,0.5,1.0):
    Xaug=np.vstack([Xtr,Xsc_s]); yaug=np.concatenate([ytr,ysc])
    w=np.concatenate([np.ones(len(ytr)), np.full(len(ysc),a)])
    print(f'    + scraped (weight={a}): bacc={ba(Xaug,yaug,Xte,yte,w):.3f}')
print('DONE')
