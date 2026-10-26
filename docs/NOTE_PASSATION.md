# Note de passation — Rakuten « Multimodal Product Data Classification »

> À lire en premier pour reprendre le projet. Créé le 2026-10-23 depuis la session Kili.
> Projets frères : `~/Documents/Hugging-Face/posos_camembert` (CamemBERT, Qwen 7B QLoRA, ensembles)
> et `~/Documents/Hugging-Face/kili_data_centric`.

## Le challenge (ENS Challenge Data n°35, Rakuten Institute of Technology)
https://challengedata.ens.fr/participants/challenges/35/

- **Tâche** : prédire le `prdtypecode` (27 classes) d'un produit du catalogue Rakuten France à
  partir de son titre (`designation`), de sa `description` (souvent vide) et de son image.
- **Métrique** : F1 pondéré (`f1_score(..., average="weighted")`).
- **Données** : 84 916 produits en train, 13 812 en test, textes en français, une image par produit
  (`image_<imageid>_product_<productid>.jpg`, dossiers `image_train/` et `image_test/`).
- **Benchmark** : CNN texte sur le titre seul, F1 0,8113 ; ResNet50 image seule, F1 0,5534.
- **Format de soumission** : `,prdtypecode`, une ligne par identifiant de `X_test`.
- **Résultat** : **l'utilisateur a gagné ce challenge (1er au classement)**, avec un ensemble
  texte + image en 2020.

## ⚠ Confidentialité (mention légale du challenge)
Les données sont **strictement confidentielles** : ne pas les divulguer, ne les utiliser que pour
le challenge, et cesser de les utiliser à la fin du challenge, sauf accord de Rakuten. Donc :
pas de dépôt public, pas d'envoi à un service tiers (y compris un Colab ou un Drive partagé),
et `.gitignore` sur `data/`, `submissions/`, `images/`, modèles. **Vérifier si le challenge est
encore ouvert** avant de réutiliser les données (la page indique un début le 6 janvier 2020).

## Où est l'ancien code (disque externe `/media/elie/265C-8015`)
Rien sur le disque interne. Deux copies quasi identiques du projet 2020 :
- `/media/elie/265C-8015/rakuten2020/` : la copie la plus complète (avec les `.npy` d'embeddings
  et les modèles `.h5`) ;
- `/media/elie/265C-8015/document_2022_10_22/rakuten2020/` : la même sans la plupart des
  `.npy`, plus `Spyder/`, `pytorch-V1.ipynb`, `trunc text from train and test.ipynb`.
- `/media/elie/265C-8015/document_2022_10_22/rakuten2021/` : une 2e édition (janvier 2021) ;
  données `data/` (X_train 330 Mo, images en `.tar` 2,9 Go) et une seule soumission
  (`sub/TEXT_bert2021-02-07_sub_70.csv`).

Les noms des notebooks donnent le score obtenu (score local / score plateforme « on sub ») :
- débuts (avril-mai 2020) : `correctionword*.ipynb` (correction orthographique des titres,
  `287_mots_corriger.csv`), `anlayse decriptive.ipynb`, `PUBLIC 87.6.ipynb`, `PUBLIC 0.8801.ipynb` ;
- milieu (septembre-novembre 2020) : `88_*.ipynb`, `89_09 on sub !!!-ALL TRAIN.ipynb`,
  `89_09 image crop.ipynb`, `89_70 on sub !!!-ALL TRAIN-Copy1.ipynb` ;
- **le plus récent : `89_09 on sub !!!-ALL TRAIN-18models sub 89_86.ipynb`** (11 décembre 2020,
  278 cellules, Keras/TensorFlow). On y trouve : correction orthographique, nettoyage HTML
  (BeautifulSoup), OCR des images (`pytesseract`, colonne `image_text`), embeddings fastText
  entraînés sur train + test, CNN / Bi-LSTM / GRU texte, CNN image (ResNet50, InceptionV3,
  EfficientNetB0/B3, augmentations), un bloc Transformer maison, et un ensemble de modèles.
- `data/X_train_20200907.csv` (105 Mo) : train enrichi (OCR des images, etc.).

Les soumissions (`rakuten2020/sub/`) retracent l'ensemble final : 8 modèles (F1 0,896), puis 14
à 16 modèles (0,8986 à 0,8994), 18 avec CamemBERT (**0,9011**), 19 avec
`gilf/french-camembert-postag-model` (**0,9037, la meilleure**), le 10 décembre 2020.
Nom du fichier : `all_19model__90prc+10prc_gilf_french-camembert-postag-model_V22020-12-10_sub_90.37.csv`.
« 90prc+10prc » : probablement modèles entraînés sur 90 % puis complétés sur 10 % du train.

## ⚠ Fichiers abîmés sur le disque externe
Plusieurs `.npy` ont exactement **4 294 967 295 octets** (`IMAGE_ARRAY.npy`,
`IMAGE_ARRAY_80_80.npy`, `x_train_img_*.npy`) : c'est la limite de 4 Go du système de fichiers
du disque. Ces fichiers sont **tronqués et inutilisables** ; il faudra les régénérer à partir des
images. Les fichiers `*.1%` sont des copies interrompues.

## État (2026-10-23, fin d'après-midi)
- [x] Classement public au 2026-10-23 : 1er **0,9346** (Enguerrandln, oct. 2026), 2e 0,9273, 3e 0,9196 ;
      nous sommes **8e avec 0,9037** (2020). Le challenge accepte encore des soumissions.
- [x] Données copiées dans `data/` (CSV + `images/image_train` 84 916, `images/image_test` 13 812,
      `X_*_20200907.csv` = textes avec OCR de 2020). Repère : `submissions/ref_2020_0.9037.csv`.
- [x] Env conda **`rakuten`** (clone de `posos` + timm, open_clip, lightgbm ; torch 2.4.1+cu121,
      ⚠ ne pas laisser pip installer un autre torch : la 2.14 ne gère plus la 1080 Ti).
- [x] Nouvelle chaîne en **validation croisée 5 plis** (plis fixés dans `common.py`) :
  - `tfidf.py` : TF-IDF mots + caractères + titre, LinearSVC → **0,8663** OOF (stacking seul 0,8692) ;
  - `img_feats.py` + `img_head.py` : vecteurs SigLIP so400m (224) gelés + MLP ;
  - `train_text.py` : CamemBERTv2 affiné par pli (~35 min/pli sur la 1080 Ti, 3 epochs) ;
  - `stack.py` : régression logistique sur les log-probas OOF, écrit la soumission et affiche
    l'accord avec la soumission 2020 ;
  - `queue.sh` : enchaîne tout ; **à lancer sous `gnome-session-inhibit --inhibit suspend:idle`** (la veille automatique de GNOME, 20 min sur secteur, fige CUDA ; `systemd-inhibit` ne la bloque PAS : nuit du 23 au 24 perdue).
  - `colab/rakuten_qwen_lora.ipynb` : Qwen2.5 QLoRA par pli (même format que `train_text.py`).
- [x] **Soumission 2026-10-23 : `stack_tfidf_cbv2_siglip` → 0,92598 sur la plateforme (3e)**, CV 0,9234.
      Qwen 7B (pli 0 seulement, Colab) : 0,9001 seul ; soumission avec Qwen prête
      (`submissions/stack_tfidf_cbv2_siglip_qwen7b_2026-10-23.csv`). Tous les chiffres : `RESULTATS.md`.
- [x] **2026-10-24 : `stack4_ids` → 0,95397 sur la plateforme = 1er** (ancien 1er 0,9346). Stacking
      TF-IDF + CamemBERTv2 + CamemBERTav2 + SigLIP, plus les catégories des voisins train par productid /
      imageid (`stack_ids.py`, LightGBM). ⚠ À sélectionner (« Selected ») sur la plateforme.
      Vrai modèle texte + image, sans identifiants : 0,9260 (meilleur réutilisable).
      Prêt, non soumis : `stack5_ids` (+ XLM-R, CV 0,9501). Notebook Mistral : `colab/rakuten_mistral_lora.ipynb`.
- Format des prédictions : `oof/<nom>_oof.npy` (84 916 × 27) et `oof/<nom>_test.npy` (13 812 × 27).

## Pistes
1. **Reproduire d'abord** : mesurer en local (validation croisée) le meilleur modèle texte seul du
   notebook, pour avoir une base comparable aux scores de 2020.
2. **Ce qui a marché sur Posos** (textes français courts, comme ces titres) : CamemBERT affiné,
   ensembles, et surtout **Qwen2.5-7B en QLoRA** (+1 à +1,7 pt dans l'ensemble). Voir
   `posos_camembert/MEMO` et `colab/posos_qwen7b_lora.ipynb`. ⚠ Colab = service tiers : pas avec
   des données confidentielles sans l'accord de Rakuten.
3. **Image** : un encodeur moderne (CLIP / SigLIP / ConvNeXt) gelé à la place de ResNet50, et
   ses vecteurs donnés à l'ensemble.
4. Méthode (leçons Posos) : une évaluation fiable d'abord, une modification à la fois, analyse
   des erreurs tôt, registre des soumissions.
5. Matériel : GTX 1080 Ti (Pascal) : pas de fp16, `group_by_length`, `save_total_limit=1`,
   `set_seed()` avant le modèle ; Ollama et entraînement jamais en même temps sur le GPU.
