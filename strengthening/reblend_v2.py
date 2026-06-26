"""V2: generative re-vibe (SD img2img) vs V1 color-only, same images/targets.
Tests whether generation breaks the semantic wall (higher classifier flip) and at what
fidelity cost (structure + color). CLIP-B/32 for vibe scoring."""
import json, glob, os, warnings, numpy as np, pandas as pd, torch, clip
warnings.filterwarnings('ignore')
from PIL import Image, ImageEnhance
from skimage.color import rgb2lab, deltaE_cie76
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import LogisticRegression
from diffusers import StableDiffusionImg2ImgPipeline
CUR = "/Users/jessicafan/Downloads/tvm board - Nov 22nd 2025 (5060 images)"
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
def L2(X): return X / np.linalg.norm(X, axis=-1, keepdims=True)
yc = pd.read_csv(f'{OUT}/labels.csv')['_modifier'].to_numpy(dtype=str)
Xc = L2(np.load(f'{OUT}/Xclip_curated.npy'))
cent = {m: L2(Xc[yc == m].mean(0, keepdims=True))[0] for m in set(yc)}
native = {m: (1 - Xc[yc == m] @ cent[m]).mean() for m in cent}
clf = LogisticRegression(max_iter=4000, class_weight='balanced').fit(Xc, yc)
cmodel, prep = clip.load('ViT-B/32', device=dev)
def cembed(imgs):
    with torch.no_grad(): return L2(cmodel.encode_image(torch.stack([prep(im) for im in imgs]).to(dev)).float().cpu().numpy())
def grade(img,b,c,s,rw,bw):
    im=ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(b)).enhance(c)).enhance(s)
    a=np.asarray(im).astype(float); a[...,0]*=rw; a[...,2]*=bw; return Image.fromarray(np.clip(a,0,255).astype('uint8'))
GRID=[(b,c,s,rw,bw) for b in(0.6,0.8,1.0) for c in(1.0,1.25) for s in(0.7,1.0) for(rw,bw)in((0.85,1.18),(1.0,1.0),(1.18,0.85))]
def gradSSIM(o,g): return ssim(sobel(np.asarray(o.convert('L').resize((224,224)))/255.), sobel(np.asarray(g.convert('L').resize((224,224)))/255.), data_range=1.0)
def dE(o,g): return deltaE_cie76(rgb2lab(np.asarray(o.resize((224,224)))/255.), rgb2lab(np.asarray(g.resize((224,224)))/255.)).mean()

srcs=[os.path.join(CUR,json.load(open(f))['meta']['_relpath']) for f in sorted(glob.glob(f'{CUR}/TVM JSON/*.json'))
      if os.path.basename(f).split('_',1)[1].split()[0]=='radiant']
srcs=[p for p in srcs if os.path.exists(p)]
srcs=[srcs[i] for i in np.random.RandomState(0).permutation(len(srcs))[:20]]
imgs=[Image.open(p).convert('RGB').resize((512,512)) for p in srcs]

print('loading SD img2img (downloads ~4GB first run)...', flush=True)
pipe=StableDiffusionImg2ImgPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float32, safety_checker=None)
pipe=pipe.to(dev); pipe.set_progress_bar_config(disable=True)
PROMPTS={'melancholic':'a melancholic somber muted moody overcast desaturated photograph',
         'haunted':'a haunted eerie dark ominous foggy unsettling photograph'}
NEG='bright cheerful sunny vibrant saturated golden'

def v1_best(im,tc):
    E=cembed([grade(im,*g) for g in GRID]); j=int((1-E@tc).argmin()); return grade(im,*GRID[j]),E[j]

for TGT in ('melancholic','haunted'):
    tc=cent[TGT]
    v1d,v1flip,v1ss,v1de=[],0,[],[]
    v2d,v2flip,v2ss,v2de=[],0,[],[]
    gen=torch.Generator(dev).manual_seed(0)
    for im in imgs:
        gimg,ge=v1_best(im,tc); v1d.append(1-ge@tc); v1flip+=int(clf.predict(ge[None])[0]==TGT); v1ss.append(gradSSIM(im,gimg)); v1de.append(dE(im,gimg))
        out=pipe(prompt=PROMPTS[TGT],negative_prompt=NEG,image=im,strength=0.5,guidance_scale=7.5,num_inference_steps=30,generator=gen).images[0]
        e=cembed([out])[0]; v2d.append(1-e@tc); v2flip+=int(clf.predict(e[None])[0]==TGT); v2ss.append(gradSSIM(im,out)); v2de.append(dE(im,out))
    n=len(imgs)
    print(f'\n=== radiant -> {TGT} (n={n}, native band {native[TGT]:.3f}) ===')
    print(f'  V1 color : dist={np.mean(v1d):.3f}  flip={v1flip/n:.0%}  struct={np.mean(v1ss):.3f}  colordE={np.mean(v1de):.1f}')
    print(f'  V2 genimg: dist={np.mean(v2d):.3f}  flip={v2flip/n:.0%}  struct={np.mean(v2ss):.3f}  colordE={np.mean(v2de):.1f}')

# montage for melancholic
W=240; canvas=Image.new('RGB',(3*W,4*W),'white'); tc=cent['melancholic']; gen=torch.Generator(dev).manual_seed(0)
for i,im in enumerate(imgs[:4]):
    g,_=v1_best(im,tc); o2=pipe(prompt=PROMPTS['melancholic'],negative_prompt=NEG,image=im,strength=0.5,guidance_scale=7.5,num_inference_steps=30,generator=gen).images[0]
    for j,x in enumerate((im,g,o2)): canvas.paste(x.resize((W,W)),(j*W,i*W))
canvas.save(f'{OUT}/reblend_v2.png'); print(f'\nmontage [original | V1 color | V2 generative] -> reblend_v2.png'); print('DONE')
