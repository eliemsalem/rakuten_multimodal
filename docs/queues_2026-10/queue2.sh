#!/bin/bash
# Nuit du 2026-10-23 : 2e transformer texte (CamemBERTav2, architecture DeBERTa) sur les 5 plis, puis stacking.
#   gnome-session-inhibit --inhibit suspend:idle --reason "entraînement Rakuten" ./queue2.sh
# (systemd-inhibit ne suffit pas : la veille automatique de GNOME, 20 min sur secteur, l'ignore)
cd "$(dirname "$0")"
PY=${PY:-python}

for k in 0 1 2 3 4; do
  [ -f oof/cbav2_f$k.npz ] || $PY train_text.py --model almanach/camembertav2-base --name cbav2 --fold $k --bs 16 --accum 2 > logs/cbav2_f$k.log 2>&1
done
$PY train_text.py --model almanach/camembertav2-base --name cbav2 --merge > logs/cbav2_merge.log 2>&1
$PY stack.py tfidf_svm cbv2 cbav2 img_ViT-SO400M-14-SigLIP --sub stack_tfidf_cbv2_cbav2_siglip > logs/stack_3.log 2>&1
echo fini > logs/queue2.done
