"""Can scraped data help if FILTERED instead of naively added?
Curated-trained model pseudo-validates scraped images; keep only confident agreements,
then augment. Reports scraped 'clean yield' and clean-test bacc vs raw augmentation."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
OVL = ['ethereal', 'melancholic', 'whimsical']
yc_all = pd.read_csv(f'{OUT}/labels.csv')['_modifier'].to_numpy(dtype=str)
Xc_all = np.load(f'{OUT}/Xclip_curated.npy')
ys = pd.read_csv(f'{OUT}/scraped_meta.csv')['adj'].to_numpy(dtype=str)
Xs = np.load(f'{OUT}/Xscraped_b32.npy')
cm = np.isin(yc_all, OVL); sm = np.isin(ys, OVL)
Xc, yc = Xc_all[cm], yc_all[cm]; Xsc, ysc = Xs[sm], ys[sm]
sc = StandardScaler().fit(np.vstack([Xc, Xsc]))
Xc, Xsc = sc.transform(Xc), sc.transform(Xsc)
clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')

Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=42)
base = balanceds = balanced_accuracy_score(yte, clf().fit(Xtr, ytr).predict(Xte))
print(f'curated-only baseline bacc={base:.3f}  (raw-augment@1.0 was 0.798 — hurts)\n')

gate = clf().fit(Xtr, ytr)              # curated-trained validator
proba = gate.predict_proba(Xsc); pred = gate.classes_[proba.argmax(1)]; conf = proba.max(1)
agree = pred == ysc                      # scraped label matches curated model's call
print(f'scraped agreement with curated model (any conf): {agree.mean():.3f}  '
      f'(=fraction of scraped imgs whose search-term vibe a validated model confirms)\n')
print('=== filtered augmentation (keep agree & conf>=tau), test=held-out curated ===')
for tau in (0.5, 0.6, 0.7, 0.8):
    keep = agree & (conf >= tau)
    if keep.sum() < 10: print(f'  tau={tau}: only {keep.sum()} kept'); continue
    Xk, yk = Xsc[keep], ysc[keep]
    for w in (1.0, 0.5):
        Xa = np.vstack([Xtr, Xk]); ya = np.concatenate([ytr, yk])
        sw = np.concatenate([np.ones(len(ytr)), np.full(len(yk), w)])
        ba = balanced_accuracy_score(yte, clf().fit(Xa, ya, sample_weight=sw).predict(Xte))
        print(f'  tau={tau} kept={keep.sum():4d} ({keep.sum()/len(ysc):.0%}) weight={w}: bacc={ba:.3f}'
              + ('  <-- beats baseline' if ba > base else ''))
print('DONE')
