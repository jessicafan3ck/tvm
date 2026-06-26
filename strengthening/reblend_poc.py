"""V1 proof: can pure color/tone grading move a 'product' photo's vibe toward a target
context vibe, and how much of the gap does it close vs where native target images sit?
Source=radiant (bright/product-like), Target=melancholic (moody feed). CLIP-B/32."""
import json, glob, os, warnings, numpy as np, pandas as pd, torch, clip
warnings.filterwarnings('ignore')
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
CUR = "/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
SRC, TGT = 'radiant', 'melancholic'

def L2(X): return X / np.linalg.norm(X, axis=-1, keepdims=True)
# centroids + native band from cached B/32 embeddings
yc = pd.read_csv(f'{OUT}/labels.csv')['_modifier'].to_numpy(dtype=str)
Xc = L2(np.load(f'{OUT}/Xclip_curated.npy'))
tgt_c = L2(Xc[yc == TGT].mean(0, keepdims=True))[0]
def cdist(E): return 1 - E @ tgt_c
native = cdist(Xc[yc == TGT]).mean()                       # where target images sit (blend zone)
src_baseline_cached = cdist(Xc[yc == SRC]).mean()           # where source images sit (jarring)
print(f'native {TGT} band (cos-dist to {TGT} centroid): {native:.3f}')
print(f'{SRC} images baseline distance to {TGT}: {src_baseline_cached:.3f}  (the jarring gap)\n')

# source image paths
paths = []
for f in sorted(glob.glob(f'{CUR}/TVM JSON/*.json')):
    d = json.load(open(f)); rp = d['meta'].get('_relpath', '')
    if rp and os.path.exists(os.path.join(CUR, rp)) and os.path.basename(f).split('_',1)[1].split()[0] == SRC:
        paths.append(os.path.join(CUR, rp))
rng = np.random.RandomState(0); paths = [paths[i] for i in rng.permutation(len(paths))[:100]]

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model, prep = clip.load('ViT-B/32', device=dev)
def embed(imgs):
    with torch.no_grad():
        t = torch.stack([prep(im) for im in imgs]).to(dev)
        return L2(model.encode_image(t).float().cpu().numpy())

def grade(img, b, c, s, rw, bw):
    im = ImageEnhance.Brightness(img).enhance(b)
    im = ImageEnhance.Contrast(im).enhance(c)
    im = ImageEnhance.Color(im).enhance(s)
    a = np.asarray(im).astype(float); a[..., 0] *= rw; a[..., 2] *= bw
    return Image.fromarray(np.clip(a, 0, 255).astype('uint8'))

GRID = [(b, c, s, rw, bw) for b in (0.6, 0.8, 1.0) for c in (1.0, 1.25)
        for s in (0.7, 1.0) for (rw, bw) in ((0.85, 1.18), (1.0, 1.0), (1.18, 0.85))]
ID = GRID.index((1.0, 1.0, 1.0, 1.0, 1.0))

d0s, dstars, best_imgs, base_embs, best_embs = [], [], [], [], []
for p in paths:
    img = Image.open(p).convert('RGB')
    graded = [grade(img, *g) for g in GRID]
    E = embed(graded); d = cdist(E)
    d0s.append(d[ID]); j = int(d.argmin()); dstars.append(d[j])
    best_imgs.append(graded[j]); base_embs.append(E[ID]); best_embs.append(E[j])
d0, dstar = np.array(d0s), np.array(dstars)
closed = (d0 - dstar) / (d0 - native)
print(f'=== color/tone grading: {SRC} -> {TGT} (n={len(paths)}, {len(GRID)} grades/img) ===')
print(f'  baseline dist to {TGT}:        {d0.mean():.3f}')
print(f'  best-graded dist to {TGT}:     {dstar.mean():.3f}')
print(f'  native {TGT} band:             {native:.3f}')
print(f'  fraction of gap closed:        {np.clip(closed,0,2).mean():.0%}')
print(f'  reached the native blend band: {(dstar <= native).mean():.0%} of images')

# probe: does P(target) rise after grading?
clf = LogisticRegression(max_iter=4000, class_weight='balanced').fit(Xc, yc)
classes = list(clf.classes_); ti = classes.index(TGT)
pb = clf.predict_proba(np.array(base_embs))[:, ti]; pa = clf.predict_proba(np.array(best_embs))[:, ti]
print(f'  P({TGT}) before grade: {pb.mean():.2f}  ->  after: {pa.mean():.2f}')
print(f'  classifier now calls it "{TGT}": {(clf.predict(np.array(best_embs))==TGT).mean():.0%} (was {(clf.predict(np.array(base_embs))==TGT).mean():.0%})')

# before/after montage (5 examples)
k = 5; W = 256
canvas = Image.new('RGB', (2*W, k*W), 'white')
for i in range(k):
    o = Image.open(paths[i]).convert('RGB').resize((W, W)); g = best_imgs[i].resize((W, W))
    canvas.paste(o, (0, i*W)); canvas.paste(g, (W, i*W))
canvas.save(f'{OUT}/reblend_examples.png')
print(f'\nsaved before/after montage -> {OUT}/reblend_examples.png (left=original radiant, right=graded toward {TGT})')
print('DONE')
