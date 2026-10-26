"""Classifieur MLP sur des vecteurs d'images gelés, en 5 plis -> oof/img_<modèle>_*.npy.

  python img_head.py ViT-SO400M-14-SigLIP-384
"""
import sys

import numpy as np
import torch
import torch.nn as nn

from common import HERE, folds, load, save_preds, wf1

arch = sys.argv[1]
tr, te, codes = load()
y = tr.label.values
Xtr = torch.tensor(np.load(HERE / "feats" / f"{arch}_train.npy"), dtype=torch.float32)
Xte = torch.tensor(np.load(HERE / "feats" / f"{arch}_test.npy"), dtype=torch.float32)
dev = "cuda"


def fit_predict(xa, ya, xs, epochs=30, seed=0):
    torch.manual_seed(seed)
    d = xa.shape[1]
    net = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.5),
                        nn.Linear(1024, len(codes))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = epochs * ((len(xa) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    xa, ya = xa.to(dev), torch.tensor(ya).to(dev)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(xa), device=dev).split(256):
            loss = nn.functional.cross_entropy(net(xa[idx] * 10), ya[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        return [net(x.to(dev) * 10).softmax(-1).cpu().numpy() for x in xs]


oof = np.zeros((len(tr), len(codes)))
test = np.zeros((len(te), len(codes)))
for k, (a, b) in enumerate(folds(y)):
    pv, pt = fit_predict(Xtr[a], y[a], [Xtr[b], Xte], seed=k)
    oof[b], test = pv, test + pt / 5
    print(f"pli {k} : {wf1(y[b], pv.argmax(1)):.4f}", flush=True)
print(f"img {arch} OOF : {wf1(y, oof.argmax(1)):.4f}")
save_preds(f"img_{arch}", oof, test)
