"""Base rapide : TF-IDF mots + caractères, SVM linéaire (scores passés en softmax pour le stacking)."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import softmax
from sklearn.svm import LinearSVC

from common import folds, load, save_preds, wf1

tr, te, codes = load()
words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=400_000, sublinear_tf=True,
                        lowercase=True, strip_accents="unicode")
chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=600_000,
                        sublinear_tf=True, lowercase=True)
# titre seul en plus du texte complet : le titre est l'information la plus fiable
title = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
all_text = list(tr.text) + list(te.text)
all_title = list(tr.designation) + list(te.designation)
X = sp.hstack([words.fit_transform(all_text), chars.fit_transform(all_text),
               title.fit_transform(all_title)]).tocsr()
Xtr, Xte = X[:len(tr)], X[len(tr):]
y = tr.label.values

oof = np.zeros((len(tr), len(codes)))
test = np.zeros((len(te), len(codes)))
for k, (a, b) in enumerate(folds(y)):
    clf = LinearSVC(C=0.5).fit(Xtr[a], y[a])
    # facteur 5 : met les marges du SVM à une échelle proche de log-probas
    oof[b] = softmax(5 * clf.decision_function(Xtr[b]), axis=1)
    test += softmax(5 * clf.decision_function(Xte), axis=1) / 5
    print(f"pli {k} : {wf1(y[b], oof[b].argmax(1)):.4f}", flush=True)
print(f"tfidf OOF : {wf1(y, oof.argmax(1)):.4f}")
save_preds("tfidf_svm", oof, test)
