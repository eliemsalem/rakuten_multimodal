#!/bin/bash
# File d'attente GPU, à lancer sous systemd-inhibit (sinon la mise en veille fige CUDA) :
#   systemd-inhibit --what=sleep:idle --why="entraînement Rakuten" ./queue.sh
# Chaque étape déjà faite est sautée : on peut relancer la file après une coupure.
cd "$(dirname "$0")"
PY=${PY:-python}
IMG=ViT-SO400M-14-SigLIP

cbv2() {
  [ -f oof/cbv2_f$1.npz ] || $PY train_text.py --model almanach/camembertv2-base --name cbv2 --fold $1 > logs/cbv2_f$1.log 2>&1
}

cbv2 0
[ -f feats/${IMG}_test.npy ] || $PY img_feats.py $IMG webli > logs/siglip.log 2>&1
[ -f oof/img_${IMG}_oof.npy ] || $PY img_head.py $IMG > logs/img_head.log 2>&1
$PY stack.py tfidf_svm img_$IMG > logs/stack_1.log 2>&1
for k in 1 2 3 4; do cbv2 $k; done
$PY train_text.py --model almanach/camembertv2-base --name cbv2 --merge > logs/cbv2_merge.log 2>&1
$PY stack.py tfidf_svm cbv2 img_$IMG --sub stack_tfidf_cbv2_siglip > logs/stack_2.log 2>&1
echo fini > logs/queue.done
