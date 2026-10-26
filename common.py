"""Chargement des données, plis de validation croisée et nettoyage communs à tous les modèles.

Convention : chaque modèle écrit dans oof/<nom>_oof.npy (84 916 x 27, probas hors-pli sur le train)
et oof/<nom>_test.npy (13 812 x 27, moyenne des modèles des plis sur le test). stack.py les combine.
"""
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).parent
DATA = HERE / "data"
OOF = HERE / "oof"
OOF.mkdir(exist_ok=True)
N_FOLDS = 5
SEED = 42

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(s):
    """Retire les balises et entités HTML des descriptions (garde casse et chiffres : utiles aux transformers)."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(html.unescape(s))
    s = _TAG.sub(" ", s)
    return _WS.sub(" ", s).strip()


def load():
    """Renvoie train (avec 'label' 0-26), test, et la liste des prdtypecode (indice -> code)."""
    tr = pd.read_csv(DATA / "X_train_update.csv", index_col=0)
    te = pd.read_csv(DATA / "X_test_update.csv", index_col=0)
    y = pd.read_csv(DATA / "Y_train_CVw08PX.csv", index_col=0)["prdtypecode"]
    codes = sorted(y.unique())
    tr["label"] = y.map({c: i for i, c in enumerate(codes)}).values
    for df, split in ((tr, "train"), (te, "test")):
        df["designation"] = df["designation"].map(clean_html)
        df["description"] = df["description"].map(clean_html)
        df["text"] = (df["designation"] + " . " + df["description"]).str.strip(" .")
        df["image"] = [str(DATA / f"images/image_{split}/image_{i}_product_{p}.jpg")
                       for i, p in zip(df["imageid"], df["productid"])]
    return tr, te, codes


def folds(labels):
    """Liste de (train_idx, val_idx), identique pour tous les modèles."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(labels)), labels))


def wf1(y, p):
    """Métrique du challenge : F1 pondéré."""
    return f1_score(y, p, average="weighted")


def save_preds(name, oof, test):
    np.save(OOF / f"{name}_oof.npy", oof.astype(np.float32))
    np.save(OOF / f"{name}_test.npy", test.astype(np.float32))
