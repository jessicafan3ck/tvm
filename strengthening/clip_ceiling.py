"""CLIP image-embedding ceiling vs F_obj floor on the SAME 812 benchmark images.
Run with the repo venv python (has torch+clip)."""
import warnings, numpy as np, pandas as pd, torch, clip
warnings.filterwarnings('ignore')
from PIL import Image
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score

SP = '/private/tmp/claude-501/-Users-jessicafan/33041f73-b2cb-4f34-a907-05c227e6ccb0/scratchpad/'
meta = pd.read_csv(SP + 'bench_paired.csv')
Xobj = np.load(SP + 'Xbench.npy')
y = meta['emotion'].to_numpy(dtype=str)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model, prep = clip.load('ViT-B/32', device=dev)
print(f'CLIP ViT-B/32 on {dev}; embedding {len(meta)} images...')

embs = []
B = 64
with torch.no_grad():
    batch = []
    for i, p in enumerate(meta['img']):
        try: batch.append(prep(Image.open(p).convert('RGB')))
        except Exception: batch.append(torch.zeros(3,224,224))
        if len(batch) == B or i == len(meta)-1:
            t = torch.stack(batch).to(dev)
            e = model.encode_image(t).cpu().numpy()
            embs.append(e); batch = []
            if (i+1) % 256 < B: print(f'  {i+1}/{len(meta)}')
Xclip = np.vstack(embs)
print('CLIP emb shape', Xclip.shape)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
def probe(X, name, clf=None):
    clf = clf or HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    ba, f1 = balanced_accuracy_score(y, pred), f1_score(y, pred, average='macro')
    print(f'  {name:34s} balanced_acc={ba:.3f}  macro_F1={f1:.3f}')
    return ba

print(f'\n=== Floor vs Ceiling on SAME 812 images (chance={1/len(set(y)):.3f}) ===')
probe(Xobj, 'F_obj (41 hand-crafted comp feats)')
# CLIP: linear probe is the standard readout; also GBM
probe(StandardScaler().fit_transform(Xclip), 'CLIP-image (linear probe)',
      LogisticRegression(max_iter=3000, class_weight='balanced'))
probe(Xclip, 'CLIP-image (GBM)')
comb = np.hstack([StandardScaler().fit_transform(Xobj), StandardScaler().fit_transform(Xclip)])
probe(comb, 'F_obj + CLIP (combined, linear)', LogisticRegression(max_iter=3000, class_weight='balanced'))
np.save(SP + 'Xclip_bench.npy', Xclip)
print('\nsaved Xclip_bench.npy')
