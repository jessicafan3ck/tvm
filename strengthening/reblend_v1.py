"""V1 sharpened: (A) multi-target gap-closing, (B) structure-vs-color fidelity guardrail,
(C) foreground-lock demo (re-vibe context, keep 'product' pixel-true). CLIP-B/32."""
import json, glob, os, warnings, numpy as np, pandas as pd, torch, clip
warnings.filterwarnings('ignore')
from PIL import Image, ImageEnhance
import cv2
from skimage.color import rgb2lab, deltaE_cie76
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import LogisticRegression
CUR = "/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
def L2(X): return X / np.linalg.norm(X, axis=-1, keepdims=True)
yc = pd.read_csv(f'{OUT}/labels.csv')['_modifier'].to_numpy(dtype=str)
Xc = L2(np.load(f'{OUT}/Xclip_curated.npy'))
cent = {m: L2(Xc[yc == m].mean(0, keepdims=True))[0] for m in set(yc)}
native = {m: (1 - Xc[yc == m] @ cent[m]).mean() for m in cent}

def paths_for(mod, n, seed=0):
    ps = [os.path.join(CUR, json.load(open(f))['meta']['_relpath']) for f in sorted(glob.glob(f'{CUR}/TVM JSON/*.json'))
          if os.path.basename(f).split('_', 1)[1].split()[0] == mod]
    ps = [p for p in ps if os.path.exists(p)]
    return [ps[i] for i in np.random.RandomState(seed).permutation(len(ps))[:n]]

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model, prep = clip.load('ViT-B/32', device=dev)
def embed(imgs):
    with torch.no_grad():
        return L2(model.encode_image(torch.stack([prep(im) for im in imgs]).to(dev)).float().cpu().numpy())
def grade(img, b, c, s, rw, bw):
    im = ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(b)).enhance(c)).enhance(s)
    a = np.asarray(im).astype(float); a[..., 0] *= rw; a[..., 2] *= bw
    return Image.fromarray(np.clip(a, 0, 255).astype('uint8'))
GRID = [(b,c,s,rw,bw) for b in (0.6,0.8,1.0) for c in (1.0,1.25) for s in (0.7,1.0) for (rw,bw) in ((0.85,1.18),(1.0,1.0),(1.18,0.85))]
ID = GRID.index((1.0,1.0,1.0,1.0,1.0))
clf = LogisticRegression(max_iter=4000, class_weight='balanced').fit(Xc, yc); cls = list(clf.classes_)

print('=== A: multi-target gap-closing (source=radiant, 60 imgs) ===')
srcs = paths_for('radiant', 60)
imgs = [Image.open(p).convert('RGB') for p in srcs]
for TGT in ('ethereal', 'melancholic', 'haunted'):
    tc = cent[TGT]; flips = 0; d0s = []; dss = []
    for im in imgs:
        E = embed([grade(im, *g) for g in GRID]); d = 1 - E @ tc
        d0s.append(d[ID]); j = int(d.argmin()); dss.append(d[j])
        flips += int(clf.predict(E[j:j+1])[0] == TGT)
    d0, ds = np.mean(d0s), np.mean(dss); closed = (np.array(d0s)-np.array(dss))/(np.array(d0s)-native[TGT])
    print(f'  radiant->{TGT:12s} base={d0:.3f} best={ds:.3f} native={native[TGT]:.3f} | gap_closed={np.clip(closed,0,2).mean():.0%} | flip={flips/len(imgs):.0%}')

print('\n=== B: fidelity guardrail — structure preserved vs color shifted (intensity sweep, target=melancholic) ===')
tc = cent['melancholic']; sub = imgs[:30]
print('  intensity | gap_closed | structure(grad-SSIM) | color(meanLAB dE)')
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    b = 1-0.4*a; c = 1+0.3*a; s = 1-0.4*a; rw = 1-0.18*a; bw = 1+0.18*a
    gs, struct, de = [], [], []
    G = [grade(im, b, c, s, rw, bw) for im in sub]; E = embed(G); d0 = embed(sub)
    for o, g in zip(sub, G):
        oa = np.asarray(o.resize((224,224))); ga = np.asarray(g.resize((224,224)))
        struct.append(ssim(sobel(np.asarray(o.convert('L').resize((224,224)))/255.),
                           sobel(np.asarray(g.convert('L').resize((224,224)))/255.), data_range=1.0))
        de.append(deltaE_cie76(rgb2lab(oa/255.), rgb2lab(ga/255.)).mean())
    dd = ((1-d0@tc)-(1-E@tc)); gap = np.clip(dd/((1-d0@tc)-native['melancholic']),0,2).mean()
    print(f'  {a:4.2f}      | {gap:6.0%}     | {np.mean(struct):.3f}                | {np.mean(de):.1f}')

print('\n=== C: foreground-lock — re-vibe context, keep product pixel-true (GrabCut) ===')
def fg_mask(im):
    a = np.asarray(im); h, w = a.shape[:2]; m = np.zeros((h, w), np.uint8)
    rect = (int(w*0.12), int(h*0.12), int(w*0.76), int(h*0.76))
    bg, fg = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)
    try: cv2.grabCut(a, m, rect, bg, fg, 3, cv2.GC_INIT_WITH_RECT)
    except Exception: return np.zeros((h,w),bool)
    return np.isin(m, [cv2.GC_FGD, cv2.GC_PR_FGD])
demo = imgs[:4]; W = 240; canvas = Image.new('RGB', (3*W, 4*W), 'white'); fg_de = []
for i, im in enumerate(demo):
    g = grade(im, 0.6, 1.25, 0.7, 0.85, 1.18)
    m = fg_mask(im)
    comp = np.asarray(im).copy(); ga = np.asarray(g)
    comp[~m] = ga[~m]                                   # background graded, foreground original
    compim = Image.fromarray(comp)
    fg = np.asarray(im)[m]/255.; fgc = comp[m]/255.
    fg_de.append(deltaE_cie76(rgb2lab(fg.reshape(-1,1,3)), rgb2lab(fgc.reshape(-1,1,3))).mean() if m.sum() else 0)
    for j, x in enumerate((im, g, compim)):
        canvas.paste(x.resize((W, W)), (j*W, i*W))
canvas.save(f'{OUT}/reblend_v1_fglock.png')
print(f'  foreground color change after lock (mean LAB dE): {np.mean(fg_de):.2f}  (0 = product pixel-true)')
print(f'  montage [original | full-grade | fg-locked grade] -> reblend_v1_fglock.png')
print('DONE')
