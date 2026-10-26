"""Méta-modèle multimodal : MLP sur [log-probas OOF des modèles, vecteurs SigLIP], en validation croisée.

  python stack_mlp.py tfidf_svm cbv2 cbav2 img_ViT-SO400M-14-SigLIP [--sub nom]
Contrairement à stack.py (régression logistique sur les probas seules), le méta-modèle voit aussi l'image
elle-même et peut apprendre quand faire confiance au texte ou à l'image. Moyenne de plusieurs graines.
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from common import HERE, OOF, load, wf1

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--img", default="ViT-SO400M-14-SigLIP")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--sub", help="écrit submissions/<sub>_<date>.csv")
args = ap.parse_args()

tr, te, codes = load()
y = tr.label.values
log = lambda p: np.log(np.clip(p, 1e-6, 1))
Po = np.hstack([log(np.load(OOF / f"{n}_oof.npy")) for n in args.names]) / 5
Pt = np.hstack([log(np.load(OOF / f"{n}_test.npy")) for n in args.names]) / 5
Fo = np.load(HERE / "feats" / f"{args.img}_train.npy").astype(np.float32) * 10
Ft = np.load(HERE / "feats" / f"{args.img}_test.npy").astype(np.float32) * 10
Xo = torch.tensor(np.hstack([Po, Fo]), dtype=torch.float32)
Xt = torch.tensor(np.hstack([Pt, Ft]), dtype=torch.float32)
n_prob = Po.shape[1]


class Net(nn.Module):
    """Les probas passent par un raccourci linéaire (comme stack.py) ; le MLP ajoute une correction."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(n_prob, len(codes))
        self.mlp = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(512, len(codes)))

    def forward(self, x):
        return self.lin(x[:, :n_prob]) + self.mlp(x)


def fit_predict(xa, ya, xs, seed):
    torch.manual_seed(seed)
    net = Net(xa.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = args.epochs * ((len(xa) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    xa, ya = xa.cuda(), torch.tensor(ya).cuda()
    for _ in range(args.epochs):
        net.train()
        for idx in torch.randperm(len(xa), device="cuda").split(256):
            loss = nn.functional.cross_entropy(net(xa[idx]), ya[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        return [net(x.cuda()).softmax(-1).cpu().numpy() for x in xs]


meta = np.zeros((len(y), len(codes)))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Xo, y):
    for s in range(args.seeds):
        meta[b] += fit_predict(Xo[a], y[a], [Xo[b]], s)[0] / args.seeds
print(f"stacking MLP multimodal (CV) : {wf1(y, meta.argmax(1)):.4f}")

if args.sub:
    pt = sum(fit_predict(Xo, y, [Xt], s)[0] for s in range(args.seeds)) / args.seeds
    out = HERE / "submissions" / f"{args.sub}_{date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"prdtypecode": np.array(codes)[pt.argmax(1)]}, index=te.index).to_csv(out, index_label="")
    np.save(out.with_suffix(".npy"), pt.astype(np.float32))
    print("écrit", out)
