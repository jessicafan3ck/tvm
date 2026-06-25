"""SigLIP-SO400M check that 30-vibe recoverability + ontology are not a CLIP-B/32 artifact."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
m = pd.read_csv(f'{OUT}/scraped_meta.csv'); y = m['adj'].to_numpy(dtype=str); subj = m['subject'].to_numpy(dtype=str)
X = StandardScaler().fit_transform(np.load(f'{OUT}/Xscraped_siglip.npy'))
cv = StratifiedKFold(5, shuffle=True, random_state=42); clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')
print(f'SigLIP scraped n={len(y)} chance={1/len(set(y)):.3f}')
p = cross_val_predict(clf(), X, y, cv=cv, n_jobs=-1)
pg = cross_val_predict(clf(), X, y, cv=GroupKFold(5), groups=subj, n_jobs=-1)
print(f'30-vibe: bacc={balanced_accuracy_score(y,p):.3f} (B/32 was 0.211) | group-by-subj={balanced_accuracy_score(y,pg):.3f} | macroF1={f1_score(y,p,average="macro"):.3f}')
labs = sorted(set(y)); C = np.vstack([X[y==l].mean(0) for l in labs]); Cn = C/np.linalg.norm(C,axis=1,keepdims=True)
D = 1-Cn@Cn.T; np.fill_diagonal(D,0); Z = linkage(squareform(D,checks=False), 'average')
fc = fcluster(Z,6,'maxclust'); g={}
for l,c in zip(labs,fc): g.setdefault(c,[]).append(l)
print('6-cluster ontology (SigLIP):')
for grp in sorted(g.values()): print('  ', ','.join(sorted(grp)))
print('DONE')
