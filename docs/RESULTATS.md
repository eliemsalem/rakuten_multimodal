# Résultats — Rakuten multimodal (challenge ENS n°35)

Métrique : F1 pondéré. « CV » = validation croisée 5 plis (`common.py`), « pli 0 » = 16 984 produits.

## Modèles seuls (probas hors-pli)

| Modèle | CV 5 plis | pli 0 |
| --- | --- | --- |
| TF-IDF mots + caractères + titre, LinearSVC (`tfidf.py`) | 0,8663 | 0,8698 |
| SigLIP so400m 224 gelé + MLP, image seule (`img_feats.py`, `img_head.py`) | 0,8232 | 0,8266 |
| CamemBERTv2-base, 3 epochs (`train_text.py`) | 0,8918 | 0,8899 |
| CamemBERTav2-base (DeBERTa), 3 epochs, batch 16 × 2 | **0,8954** | 0,8954 |
| XLM-RoBERTa-base, 3 epochs, lr 3e-5 | 0,8758 | 0,8735 |
| Qwen2.5-7B QLoRA, 1 epoch, Colab L4 (`colab/rakuten_qwen_lora.ipynb`) | — | 0,9001 |
| **Mistral-7B-v0.3 QLoRA**, 1 epoch, Colab L4 (`colab/rakuten_mistral_lora.ipynb`) | — | **0,9022** |

## Stacking (régression logistique sur log-probas, `stack.py`)

| Combinaison | CV 5 plis | pli 0 |
| --- | --- | --- |
| TF-IDF + image | 0,9016 | 0,9008 |
| TF-IDF + CamemBERTv2 | — | 0,9020 |
| TF-IDF + CamemBERTv2 + image | **0,9234** | 0,9194 |
| **TF-IDF + CamemBERTv2 + CamemBERTav2 + image** | **0,9254** | — |
| idem + 2e tête image CLIP ViT-L/14 (0,8077 seule) | 0,9255 | — |
| idem, tête image unique SigLIP ⊕ CLIP (0,8249 seule) | 0,9254 | — |
| **idem (4 modèles) + voisins par identifiant, LightGBM (`stack_ids.py`)** | **0,9499** | — |
| 5 modèles (+ XLM-R) : sans identifiants / avec voisins par identifiant | 0,9252 / 0,9501 | — |
| idem, méta-modèle MLP multimodal (probas + vecteurs SigLIP, `stack_mlp.py`) | 0,9243 | — |
| TF-IDF + CamemBERTv2 + image + Qwen 7B (méta appris sur le pli 0) | — | 0,9212 à 0,9236 |

## Registre des soumissions

| Date | Nom | Validation | **Plateforme** | Remarque |
| --- | --- | --- | --- | --- |
| 2020-12-10 | all_19model (ancien ensemble Keras + CamemBERT) | — | 0,9037 | 8e au 2026-10-23 |
| 2026-10-23 | stack_tfidf_cbv2_siglip | CV 0,9234 | **0,92598** | 3e (2e : 0,9273 ; 1er : 0,9346) |
| 2026-10-23 | stack_tfidf_cbv2_siglip_qwen7b | pli 0 0,9212 | 0,91741 | méta appris sur le pli 0 seul (17 k lignes) |
| 2026-10-24 | stack_tfidf_cbv2_cbav2_siglip | CV 0,9254 | non soumise | |
| 2026-10-24 | **stack4_ids** | CV 0,9499 | **0,95397** | **1er** (ancien 1er : 0,9346) ; voisins par productid / imageid (voir plus bas) |

Écart validation → plateforme : **+0,26 pt** (stack texte + image) et **+0,41 pt** (stack4_ids) pour la CV 5 plis
(fiable, légèrement pessimiste) ; −0,38 pt pour une validation sur le pli 0 seul.

**Leçon (soumission Qwen, −0,86 pt)** : les prédictions test de Qwen sont saines (même accord avec la
soumission 0,926 que CamemBERTv2, même confiance en validation et en test). La perte vient du méta-modèle
appris sur le pli 0 seul : les 3 mêmes modèles sans Qwen, appris sur ces 17 k lignes, changent déjà 2,9 % des
prédictions. → Un modèle n'entre dans le stacking qu'avec ses **5 plis** hors-pli.

Essais sans gain : méta-modèle MLP multimodal (0,9243 contre 0,9254) ; 2e encodeur d'images CLIP ViT-L/14
(+0,0001) ; XLM-RoBERTa-base comme 5e modèle (±0,0002).

## Voisinage dans le catalogue (`stack_ids.py`, 2026-10-24)
Trié par `productid`, un produit du train a la même catégorie que son voisin dans **61,5 %** des cas
(hasard : 5,3 %) ; par `imageid` : 56,6 %. Les produits arrivent par lots (vendeur, gamme). Les identifiants
du test sont dans la même plage que ceux du train, intercalés (écart médian au plus proche id train : 103,
contre 434 entre ids train consécutifs), donc le signal se transpose au test. Ajout au stacking des catégories
et écarts des 3 voisins train de chaque côté (le produit lui-même exclu) → **0,9254 → 0,9499** en CV.
C'est une propriété de la constitution du jeu de données, pas du contenu du produit : à mentionner tel quel.

Analyse d'erreurs (stacking 4 modèles, sans identifiants) : les points perdus viennent de catégories voisines,
1280 ↔ 1281 (491 erreurs), 1140 ↔ 1280 (375), 2280 ↔ 2403 (353), 10 ↔ 2403 (335), 1560 ↔ 2060 (320) ;
146 titres en double dans le train ont des étiquettes différentes (bruit d'étiquetage).

## LLM sur le pli 0 (2026-10-25)
Stacking évalué sur le pli 0 seul (16 984 produits, CV interne 5 plis) : TF-IDF + CamemBERTv2 + CamemBERTav2 + SigLIP
**0,9201** ; + Qwen 0,9225 ; + Mistral 0,9222 ; + Qwen + Mistral 0,9226. Mistral est le meilleur modèle seul
(0,9022) mais Qwen et Mistral sont d'accord à 94 % : ils apportent la même chose (≈ +0,2 pt), pas plus à deux.
Pour en profiter, il faudrait le LLM sur les 5 plis (≈ 15 h de Colab L4, ≈ 75 unités).
