#!/bin/bash
# Après queue3 : XLM-RoBERTa-base (multilingue : titres FR, DE, EN) sur les 5 plis, puis stacking de tout.
#   gnome-session-inhibit --inhibit suspend:idle --reason "entraînement Rakuten" ./queue4.sh
cd "$(dirname "$0")"
PY=${PY:-python}

while [ ! -f logs/queue3.done ]; do sleep 30; done
for k in 0 1 2 3 4; do
  [ -f oof/xlmr_f$k.npz ] || $PY train_text.py --model FacebookAI/xlm-roberta-base --name xlmr --fold $k --lr 3e-5 > logs/xlmr_f$k.log 2>&1
done
$PY train_text.py --model FacebookAI/xlm-roberta-base --name xlmr --merge > logs/xlmr_merge.log 2>&1
$PY stack.py tfidf_svm cbv2 cbav2 xlmr img_ViT-SO400M-14-SigLIP img_ViT-L-14 --sub stack_all6 > logs/stack_5.log 2>&1
echo fini > logs/queue4.done
