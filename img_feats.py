"""Extrait les vecteurs d'images d'un encodeur gelé (open_clip), pour train et test.

  python img_feats.py ViT-SO400M-14-SigLIP-384 webli
écrit feats/<modèle>_train.npy et feats/<modèle>_test.npy (vecteurs normalisés, float16).
"""
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import HERE, load

arch, pretrained = sys.argv[1], sys.argv[2]
out = HERE / "feats"
out.mkdir(exist_ok=True)
model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
model = model.cuda().eval()


class Imgs(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return preprocess(Image.open(self.paths[i]).convert("RGB"))


CHUNK = 6400   # sauvegarde par morceaux : une coupure (veille, plantage) ne fait perdre que le morceau en cours
tr, te, _ = load()
for split, df in (("train", tr), ("test", te)):
    paths = list(df.image)
    parts = []
    for start in range(0, len(paths), CHUNK):
        part = out / f"tmp_{arch}_{split}_{start:06d}.npy"
        parts.append(part)
        if part.exists():
            continue
        dl = DataLoader(Imgs(paths[start:start + CHUNK]), batch_size=64, num_workers=8)
        with torch.no_grad():
            vecs = [torch.nn.functional.normalize(model.encode_image(x.cuda()), dim=-1)
                    .cpu().numpy().astype(np.float16) for x in dl]
        np.save(part, np.concatenate(vecs))
        print(split, start + CHUNK, "/", len(df), flush=True)
    np.save(out / f"{arch}_{split}.npy", np.concatenate([np.load(p) for p in parts]))
    for p in parts:
        p.unlink()
print("ok")
