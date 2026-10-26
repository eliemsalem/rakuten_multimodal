#!/bin/bash
# Full pipeline, in order. Each step is skipped if its output already exists, so the script can be re-run
# after an interruption. Long GPU jobs: keep the machine awake, e.g.
#   gnome-session-inhibit --inhibit suspend:idle ./run_all.sh
# (on GNOME, systemd-inhibit alone does not stop the automatic suspend, which freezes CUDA jobs).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs
IMG=ViT-SO400M-14-SigLIP

# 1. text baseline: TF-IDF + LinearSVC, 5 folds (CPU, a few minutes)
[ -f oof/tfidf_svm_oof.npy ] || $PY tfidf.py > logs/tfidf.log 2>&1

# 2. image: frozen SigLIP so400m embeddings (~2 h 45 on a GTX 1080 Ti), then an MLP head, 5 folds
[ -f feats/${IMG}_test.npy ] || $PY img_feats.py $IMG webli > logs/siglip.log 2>&1
[ -f oof/img_${IMG}_oof.npy ] || $PY img_head.py $IMG > logs/img_head.log 2>&1

# 3. text transformers, 5 folds each (~35-75 min per fold on a GTX 1080 Ti)
for k in 0 1 2 3 4; do
  [ -f oof/cbv2_f$k.npz ] || $PY train_text.py --model almanach/camembertv2-base --name cbv2 --fold $k > logs/cbv2_f$k.log 2>&1
  [ -f oof/cbav2_f$k.npz ] || $PY train_text.py --model almanach/camembertav2-base --name cbav2 --fold $k --bs 16 --accum 2 > logs/cbav2_f$k.log 2>&1
done
$PY train_text.py --model almanach/camembertv2-base --name cbv2 --merge > logs/cbv2_merge.log 2>&1
$PY train_text.py --model almanach/camembertav2-base --name cbav2 --merge > logs/cbav2_merge.log 2>&1

# 4. stacking (logistic regression on out-of-fold log-probabilities) -> content-only submission
$PY stack.py tfidf_svm cbv2 cbav2 img_$IMG --sub stack_tfidf_cbv2_cbav2_siglip > logs/stack.log 2>&1

# 5. stacking + categories of the nearest training products by productid / imageid (LightGBM)
$PY stack_ids.py tfidf_svm cbv2 cbav2 img_$IMG --sub stack4_ids > logs/stack_ids.log 2>&1
echo "done: see submissions/ and logs/"
