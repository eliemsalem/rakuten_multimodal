#!/bin/bash
# 2e encodeur d'images (CLIP ViT-L/14 DataComp) + tête image + stacking à 5 modèles.
#   gnome-session-inhibit --inhibit suspend:idle --reason "entraînement Rakuten" ./queue3.sh
cd "$(dirname "$0")"
PY=${PY:-python}
IMG=ViT-L-14

[ -f feats/${IMG}_test.npy ] || $PY img_feats.py $IMG datacomp_xl_s13b_b90k > logs/clip_l14.log 2>&1
[ -f oof/img_${IMG}_oof.npy ] || $PY img_head.py $IMG > logs/img_head_l14.log 2>&1
$PY stack.py tfidf_svm cbv2 cbav2 img_ViT-SO400M-14-SigLIP img_$IMG --sub stack_tfidf_cbv2_cbav2_siglip_clipl14 > logs/stack_4.log 2>&1
echo fini > logs/queue3.done
