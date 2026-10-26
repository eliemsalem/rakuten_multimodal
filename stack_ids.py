"""Stacking + voisinage dans le catalogue : LightGBM sur [probas du stacking, catégories des voisins par identifiant].

  python stack_ids.py tfidf_svm cbv2 cbav2 img_ViT-SO400M-14-SigLIP [--sub nom]
Constat : trié par productid, un produit du train a la même catégorie que son voisin dans 61,5 % des cas
(hasard : 5,3 %) — les produits arrivent par lots (vendeur, gamme). Les identifiants du test sont mêlés à
ceux du train, donc le signal se transpose. Pour chaque produit, on donne au modèle la catégorie et l'écart
des K voisins du train de chaque côté (par productid et par imageid), en excluant le produit lui-même.
"""
import argparse
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import HERE, OOF, load, wf1

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--k", type=int, default=3, help="voisins de chaque côté")
ap.add_argument("--sub", help="écrit submissions/<sub>_<date>.csv")
args = ap.parse_args()

tr, te, codes = load()
y = tr.label.values
log = lambda p: np.log(np.clip(p, 1e-6, 1))

# 1) probas du stacking : hors-pli sur le train, méta-modèle appris sur tout le train pour le test
Xo = np.hstack([log(np.load(OOF / f"{n}_oof.npy")) for n in args.names])
Xt = np.hstack([log(np.load(OOF / f"{n}_test.npy")) for n in args.names])
meta_o = np.zeros((len(y), len(codes)))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Xo, y):
    meta_o[b] = LogisticRegression(C=0.05, max_iter=2000).fit(Xo[a], y[a]).predict_proba(Xo[b])
meta_t = LogisticRegression(C=0.05, max_iter=2000).fit(Xo, y).predict_proba(Xt)
print(f"stacking seul (CV) : {wf1(y, meta_o.argmax(1)):.4f}", flush=True)


# 2) voisins dans le train, triés par identifiant
def neighbours(col):
    ids_tr = tr[col].values.astype(np.int64)
    order = np.argsort(ids_tr, kind="stable")
    s, ys = ids_tr[order], y[order]

    def feats(q, self_rank=None):
        # self_rank : position du produit lui-même dans s (train), à sauter
        pos = np.searchsorted(s, q)
        out = []
        for side in (-1, 1):
            for j in range(1, args.k + 1):
                if side < 0:
                    idx = pos - j
                    if self_rank is not None:
                        idx = self_rank - j
                else:
                    idx = pos + j - 1
                    if self_rank is not None:
                        idx = self_rank + j
                ok = (idx >= 0) & (idx < len(s))
                idx = idx.clip(0, len(s) - 1)
                out.append(np.where(ok, ys[idx], -1))                                   # catégorie du voisin
                out.append(np.where(ok, np.log1p(np.abs(s[idx] - q)), 99))              # écart (log)
        return np.stack(out, 1)

    rank = np.empty(len(s), dtype=np.int64)
    rank[order] = np.arange(len(s))
    return feats(ids_tr, rank), feats(te[col].values.astype(np.int64))


No, Nt = zip(*(neighbours(c) for c in ("productid", "imageid")))
Fo = np.hstack([meta_o, *No])
Ft = np.hstack([meta_t, *Nt])
cat = [len(codes) + i for i in range(Fo.shape[1] - len(codes)) if i % 2 == 0]   # colonnes « catégorie du voisin »

params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
oof = np.zeros_like(meta_o)
for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(Fo, y):
    m = lgb.LGBMClassifier(**params).fit(Fo[a], y[a], categorical_feature=cat)
    oof[b] = m.predict_proba(Fo[b])
print(f"stacking + voisins par identifiant (CV) : {wf1(y, oof.argmax(1)):.4f}", flush=True)

if args.sub:
    pt = lgb.LGBMClassifier(**params).fit(Fo, y, categorical_feature=cat).predict_proba(Ft)
    out = HERE / "submissions" / f"{args.sub}_{date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"prdtypecode": np.array(codes)[pt.argmax(1)]}, index=te.index).to_csv(out, index_label="")
    np.save(out.with_suffix(".npy"), pt.astype(np.float32))
    print("écrit", out)
    ref_file = HERE / "submissions" / "ref_2020_0.9037.csv"   # repère facultatif : la soumission 2020
    if ref_file.exists():
        ref = pd.read_csv(ref_file, index_col=0)["prdtypecode"]
        print(f"accord avec la soumission 2020 (0,9037) : {(ref.loc[te.index].values == np.array(codes)[pt.argmax(1)]).mean():.2%}")
