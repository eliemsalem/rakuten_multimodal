# Rakuten France multimodal product classification

Predict the product type (`prdtypecode`, 27 classes) of a Rakuten France catalogue item from its French title,
its optional description and its image. Metric: weighted F1.
([ENS Challenge Data #35](https://challengedata.ens.fr/participants/challenges/35/), Rakuten Institute of Technology.)

**Result.** I first took part in 2020 with a 19-model Keras ensemble: **0.9037** on the public leaderboard
(8th in October 2026). Coming back in October 2026 with a cleaner method and modern encoders:

| Submission | Validation (5-fold CV) | Public leaderboard |
| --- | --- | --- |
| 2020: 19-model Keras ensemble + CamemBERT, hand-tuned weights | — | 0.9037 |
| Text + image stacking (TF-IDF, CamemBERTv2, SigLIP) | 0.9234 | **0.9260** |
| Same + CamemBERTav2 + catalogue neighbours by product id (see [Limits](#limits)) | 0.9499 | **0.9540 (1st)** |

The previous leader stood at 0.9346. The **content-only model (0.9260) is the reusable one**. The last 2.8
points come from a property of how the dataset was built, not from understanding the product.

## Data

The data is **not included**: it is strictly confidential under the challenge's legal notice and may only be
used for the challenge. Register on the [challenge page](https://challengedata.ens.fr/participants/challenges/35/)
to download it, then arrange it like this:

| File | Where | Content |
| --- | --- | --- |
| `X_train_update.csv` | `data/` | 84,916 products: `designation` (title), `description`, `productid`, `imageid` |
| `Y_train_CVw08PX.csv` | `data/` | `prdtypecode` for each training product |
| `X_test_update.csv` | `data/` | 13,812 test products, same columns |
| `images.zip` | unzip into `data/` | `data/images/image_train/`, `data/images/image_test/`, one JPEG per product |

Pretrained models are downloaded from the Hugging Face Hub on first use:
[`almanach/camembertv2-base`](https://huggingface.co/almanach/camembertv2-base),
[`almanach/camembertav2-base`](https://huggingface.co/almanach/camembertav2-base),
[`timm/ViT-SO400M-14-SigLIP`](https://huggingface.co/timm/ViT-SO400M-14-SigLIP) (via open_clip).

## Approach

1. **Reliable evaluation first.** One stratified 5-fold split, fixed in `common.py` and shared by every model.
   Each model writes its out-of-fold probabilities on the whole training set and the average of its five
   fold models on the test set. In 2020, I validated on the last 10 % of the file (not shuffled) and tuned
   the ensemble weights by hand on that same split. My last 2020 submissions differed by less than 0.001,
   which is noise.
2. **Modern encoders, one change at a time.** TF-IDF + linear SVM as a fast baseline, then French
   transformers fine-tuned per fold, then a frozen image encoder (SigLIP) with a small MLP head.
3. **Learned stacking instead of hand-tuned weights.** A logistic regression on the concatenated
   out-of-fold log-probabilities, evaluated with its own cross-validation.
4. **Error analysis**, which led to the catalogue-neighbour features (below).

Hardware: one GTX 1080 Ti (fp32, PyTorch 2.4.1, the last release that supports Pascal GPUs). The 7B LLM
experiments ran on a Colab L4.

## Results, step by step (including what did not help)

Single models, weighted F1 on out-of-fold predictions:

| Model | 5-fold CV | Fold 0 |
| --- | --- | --- |
| TF-IDF words + chars + title, LinearSVC (`tfidf.py`) | 0.8663 | 0.8698 |
| SigLIP so400m (224 px) frozen + MLP, image only (`img_feats.py`, `img_head.py`) | 0.8232 | 0.8266 |
| CLIP ViT-L/14 (DataComp) frozen + MLP, image only | 0.8077 | — |
| CamemBERTv2-base, 3 epochs (`train_text.py`) | 0.8918 | 0.8899 |
| CamemBERTav2-base (DeBERTa), 3 epochs | 0.8954 | 0.8954 |
| XLM-RoBERTa-base, 3 epochs | 0.8758 | 0.8735 |
| Qwen2.5-7B, QLoRA, 1 epoch (`colab/rakuten_qwen_lora.ipynb`) | — | 0.9001 |
| Mistral-7B-v0.3, QLoRA, 1 epoch (`colab/rakuten_mistral_lora.ipynb`) | — | 0.9022 |

For reference, the challenge benchmarks were 0.8113 (text CNN on titles) and 0.5534 (ResNet50, images only).
The image models of my 2020 ensemble scored around 0.70.

Stacking (`stack.py`, logistic regression):

| Combination | 5-fold CV | Public LB |
| --- | --- | --- |
| TF-IDF + SigLIP | 0.9016 | |
| TF-IDF + CamemBERTv2 + SigLIP | 0.9234 | 0.9260 |
| + CamemBERTav2 | 0.9254 | |
| + CLIP ViT-L/14 as a second image model | 0.9255 | |
| + XLM-RoBERTa | 0.9252 | |
| MLP meta-model on probabilities + SigLIP embeddings (`stack_mlp.py`) | 0.9243 | |
| Qwen 7B (fold 0 only) with a meta-model trained on fold 0 only | 0.9212 (fold 0) | 0.9174 |
| **4-model stack + catalogue neighbours, LightGBM (`stack_ids.py`)** | **0.9499** | **0.9540** |

What I learned from the misses:

- **The image side saturates with SigLIP.** A second encoder, or the raw embeddings given to the
  meta-model, adds nothing.
- **A model enters the stack only with all five folds.** Training the meta-model on the 17k rows of a
  single fold scored 0.86 points lower on the leaderboard, even though the Qwen predictions themselves were sound.
- **Qwen and Mistral agree on 94 % of products.** On fold 0, either one adds about +0.2 points to the
  stack, and both together add no more. Running them on all five folds (about 15 GPU-hours) was not
  worth it.
- **5-fold CV tracks the leaderboard well** (+0.26 and +0.41 points), whereas validation on a single fold
  did not (−0.38).

## Limits

**Catalogue neighbours are a dataset artifact.** Sorted by `productid`, a training product has the same
category as its neighbour in 61.5 % of cases (5.3 % by chance). Products are uploaded in batches by the same
seller or range. Test ids are interleaved with training ids, so `stack_ids.py` gives the meta-model the
categories of the 3 nearest training products on each side (by `productid` and by `imageid`, the product
itself excluded), plus the id gaps. This is allowed by the challenge, since the ids are provided for the
test set too. It would **not** help on a genuinely new product, or on another catalogue. The content-only
stack (0.9260) is the honest measure of the model.

Other limits:

- Scores are on the **public** leaderboard only.
- 146 duplicated titles carry different labels in the training set, so part of the remaining error is
  label noise. Most remaining confusions are between neighbouring categories: toys vs. children's games,
  magazines vs. book lots, furniture vs. decoration.

## Repository

| File | Role |
| --- | --- |
| `common.py` | data loading, HTML cleaning, the shared 5-fold split, metric, prediction file convention |
| `tfidf.py` | TF-IDF (words, characters, title) + LinearSVC, 5 folds |
| `img_feats.py` | frozen open_clip image embeddings, saved in resumable chunks |
| `img_head.py` | MLP classifier on the image embeddings, 5 folds |
| `train_text.py` | fine-tunes one Hugging Face transformer on one fold; `--merge` assembles the 5 folds |
| `stack.py` | logistic-regression stacking and submission file |
| `stack_mlp.py` | MLP meta-model on probabilities + image embeddings (tried, not kept) |
| `stack_ids.py` | stacking + catalogue-neighbour features, LightGBM |
| `run_all.sh` | the whole pipeline in order, resumable |
| `colab/` | Qwen2.5-7B and Mistral-7B QLoRA notebooks (one fold per Colab session, same folds) |
| `docs/` | lab notebook in French: handoff note, full results log, the job queues actually run |
| `requirements.txt` | versions that actually ran |

## Reproduce

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# put the challenge files in data/ (see Data), then:
./run_all.sh        # about 12 h on a GTX 1080 Ti; each finished step is skipped on re-run
```

Outputs: out-of-fold and test probabilities in `oof/`, submissions in `submissions/`.
