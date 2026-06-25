"""Embed a balanced sample of the scraped 30-vibe corpus with CLIP-B/32 (matches curated cache).
<=2 images/folder for diversity, ~1000/adjective. Saves Xscraped + meta."""
import glob, os, warnings, numpy as np, pandas as pd, torch, clip
warnings.filterwarnings('ignore')
from PIL import Image
from collections import defaultdict
PI = "/Users/jessicafan/Library/CloudStorage/GoogleDrive-tvm.tiervibemap@gmail.com/My Drive/tvm/pinterest_images"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
PER_ADJ, PER_FOLDER = 1000, 2

folders = [d for d in sorted(glob.glob(PI + '/*')) if os.path.isdir(d)]
by_adj = defaultdict(list)
for d in folders:
    name = os.path.basename(d); adj = name.split()[0]
    if adj in ('prompts', 'pinterest_images', 'objective_output', 'normalized_images'): continue
    by_adj[adj].append(d)

paths, adjs, subjects = [], [], []
rng = np.random.RandomState(42)
for adj, ds in by_adj.items():
    count = 0
    order = rng.permutation(len(ds))
    for i in order:
        if count >= PER_ADJ: break
        d = ds[i]
        imgs = sorted(glob.glob(d + '/*.jpg') + glob.glob(d + '/*.png') + glob.glob(d + '/*.webp') + glob.glob(d + '/*.jpeg'))
        for p in imgs[:PER_FOLDER]:
            if count >= PER_ADJ: break
            paths.append(p); adjs.append(adj); subjects.append(' '.join(os.path.basename(d).split()[1:])); count += 1
print(f'sampled {len(paths)} images across {len(by_adj)} adjectives', flush=True)

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model, prep = clip.load('ViT-B/32', device=dev)
embs, batch = [], []
with torch.no_grad():
    for i, p in enumerate(paths):
        try: batch.append(prep(Image.open(p).convert('RGB')))
        except Exception: batch.append(torch.zeros(3, 224, 224))
        if len(batch) == 64 or i == len(paths)-1:
            embs.append(model.encode_image(torch.stack(batch).to(dev)).float().cpu().numpy()); batch = []
            if (i+1) % 4096 < 64: print(f'  {i+1}/{len(paths)}', flush=True)
X = np.vstack(embs)
np.save(f'{OUT}/Xscraped_b32.npy', X)
pd.DataFrame({'adj': adjs, 'subject': subjects, 'path': paths}).to_csv(f'{OUT}/scraped_meta.csv', index=False)
print(f'saved Xscraped_b32 {X.shape} + scraped_meta.csv', flush=True)
print('DONE', flush=True)
