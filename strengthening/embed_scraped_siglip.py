"""Re-embed the SAME scraped sample (from scraped_meta.csv) with SigLIP-SO400M-384,
to confirm the 30-vibe recoverability + ontology are not a CLIP-B/32 artifact."""
import os, warnings, numpy as np, pandas as pd, torch, open_clip
warnings.filterwarnings('ignore')
from PIL import Image
OUT = "/Users/jessicafan/tvm/strengthening/artifacts"
paths = pd.read_csv(f'{OUT}/scraped_meta.csv')['path'].tolist()
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model, _, prep = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli', cache_dir=f'{OUT}/oc_models')
model = model.to(dev).eval()
print(f'embedding {len(paths)} scraped images with SigLIP-SO400M-384 on {dev}...', flush=True)
embs, batch = [], []
with torch.no_grad():
    for i, p in enumerate(paths):
        try: batch.append(prep(Image.open(p).convert('RGB')))
        except Exception: batch.append(torch.zeros_like(prep(Image.new('RGB', (64, 64)))))
        if len(batch) == 8 or i == len(paths)-1:
            embs.append(model.encode_image(torch.stack(batch).to(dev)).float().cpu().numpy()); batch = []
            if (i+1) % 2048 < 8: print(f'  {i+1}/{len(paths)}', flush=True)
X = np.vstack(embs); np.save(f'{OUT}/Xscraped_siglip.npy', X)
print(f'saved Xscraped_siglip {X.shape}', flush=True); print('DONE', flush=True)
