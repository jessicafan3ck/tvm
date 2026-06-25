"""Fast rigor checks on cached scraped B/32 embeddings:
 A subject-confound (GroupKFold by subject) on 30-vibe probe
 B per-adjective recall — which scraped vibes are clean vs noisy search terms
 C crosswalk: map curated-9 modifiers into the scraped 30-vibe ontology"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
m = pd.read_csv(f'{OUT}/scraped_meta.csv')
Xs = np.load(f'{OUT}/Xscraped_b32.npy')
y = m['adj'].to_numpy(dtype=str); subj = m['subject'].to_numpy(dtype=str)
Xss = StandardScaler().fit_transform(Xs)
clf = lambda: LogisticRegression(max_iter=4000, class_weight='balanced')

print('=== A: subject-confound — 30-vibe probe with GroupKFold by subject ===')
ps = cross_val_predict(clf(), Xss, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
pg = cross_val_predict(clf(), Xss, y, cv=GroupKFold(5), groups=subj, n_jobs=-1)
print(f'  stratified   : bacc={balanced_accuracy_score(y,ps):.3f}')
print(f'  group-by-subj: bacc={balanced_accuracy_score(y,pg):.3f}  (held-out scenes; pure vibe)')

print('\n=== B: which scraped vibes are clean vs noisy (per-adjective recall, stratified) ===')
rep = classification_report(y, ps, output_dict=True, zero_division=0)
rows = sorted(((k, v['recall'], v['f1-score']) for k, v in rep.items()
               if k not in ('accuracy', 'macro avg', 'weighted avg')), key=lambda t: -t[1])
print('  cleanest:', ', '.join(f'{k} {r:.2f}' for k, r, _ in rows[:6]))
print('  noisiest:', ', '.join(f'{k} {r:.2f}' for k, r, _ in rows[-6:]))

print('\n=== C: crosswalk — where do curated-9 modifiers land in the 30-vibe map? ===')
yc = pd.read_csv(f'{OUT}/labels.csv')['_modifier'].to_numpy(dtype=str)
Xc = np.load(f'{OUT}/Xclip_curated.npy')
sc = StandardScaler().fit(np.vstack([Xc, Xs]))
Xc2, Xs2 = sc.transform(Xc), sc.transform(Xs)
def unit(a): return a / np.linalg.norm(a, axis=-1, keepdims=True)
adjs = sorted(set(y))
Cs = unit(np.vstack([Xs2[y == a].mean(0) for a in adjs]))
SUPER = {'decay':'abandoned cracked forgotten gothic overgrown','magical':'angelic celestial dreamy enchanted ethereal pastel surreal whimsical',
         'structural':'brutalist futuristic','cozy':'cozy elegant rustic vintage',
         'dark-moody':'dark eerie foggy melancholic nocturnal silent stormy','glow':'glimmering glowing sun-drenched twinkling'}
adj2super = {a: s for s, lst in SUPER.items() for a in lst.split()}
for mod in sorted(set(yc)):
    c = unit(Xc2[yc == mod].mean(0))
    sims = Cs @ c
    top = np.argsort(-sims)[:3]
    near = ', '.join(f'{adjs[i]}({sims[i]:.2f})' for i in top)
    print(f'  {mod:13s} -> super={adj2super[adjs[top[0]]]:11s} | nearest: {near}')
print('DONE')
