"""Stacking des probas hors-pli de plusieurs modèles + soumission.

  python stack.py tfidf_svm cbv2 img_ViT-SO400M-14-SigLIP qwen7b [--sub nom]
Méta-modèle : régression logistique sur log-probas concaténées, évaluée en validation croisée
sur les OOF (plis différents de ceux des modèles de base), puis réentraînée sur tout le train.
Un modèle qui n'a que le pli 0 (oof/<nom>_f0.npz, ex. Qwen sur Colab) est accepté : le méta-modèle
n'est alors appris que sur les ~17 000 lignes du pli 0, et ce modèle donne ses probas test du pli 0.
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import HERE, OOF, folds, load, wf1

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--C", type=float, default=0.05)
ap.add_argument("--sub", help="écrit submissions/<sub>.csv")
args = ap.parse_args()

tr, te, codes = load()
y = tr.label.values
fold0 = folds(y)[0][1]


def preds(n):
    """(probas OOF sur tout le train ou None, probas OOF du pli 0, probas test)."""
    if (OOF / f"{n}_oof.npy").exists():
        o = np.load(OOF / f"{n}_oof.npy")
        return o, o[fold0], np.load(OOF / f"{n}_test.npy")
    z = np.load(OOF / f"{n}_f0.npz")
    return None, z["val"], z["test"]


P = {n: preds(n) for n in args.names}
full = all(p[0] is not None for p in P.values())
rows = np.arange(len(y)) if full else fold0
print("méta-modèle appris sur", "tout le train" if full else f"le pli 0 seulement ({len(rows)} lignes)")
log = lambda p: np.log(np.clip(p, 1e-6, 1))
Xo = np.hstack([log(p[0] if full else p[1]) for p in P.values()])
Xt = np.hstack([log(p[2]) for p in P.values()])
y = y[rows]
for n, p in P.items():
    print(f"  {n:40s} {wf1(y, (p[0] if full else p[1]).argmax(1)):.4f}")
mean = np.mean([p[0] if full else p[1] for p in P.values()], axis=0)
print(f"  moyenne simple                           {wf1(y, mean.argmax(1)):.4f}")

meta = np.zeros((len(y), len(codes)))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Xo, y):
    clf = LogisticRegression(C=args.C, max_iter=2000).fit(Xo[a], y[a])
    meta[b] = clf.predict_proba(Xo[b])
print(f"  stacking (CV)                            {wf1(y, meta.argmax(1)):.4f}")

if args.sub:
    clf = LogisticRegression(C=args.C, max_iter=2000).fit(Xo, y)
    pt = clf.predict_proba(Xt)
    out = HERE / "submissions" / f"{args.sub}_{date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"prdtypecode": np.array(codes)[pt.argmax(1)]}, index=te.index).to_csv(out, index_label="")
    np.save(out.with_suffix(".npy"), pt.astype(np.float32))
    print("écrit", out)
    # repère (facultatif) : la soumission 2020 (0,9037) ; un désaccord de d % borne l'écart de score à ~d points
    ref_file = HERE / "submissions" / "ref_2020_0.9037.csv"
    if ref_file.exists():
        ref = pd.read_csv(ref_file, index_col=0)["prdtypecode"]
        agree = (ref.loc[te.index].values == np.array(codes)[pt.argmax(1)]).mean()
        print(f"accord avec la soumission 2020 (0,9037) : {agree:.2%}")
