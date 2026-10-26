"""Affinage d'un transformer texte sur un pli (ou tous) ; écrit les probas hors-pli et test.

  python train_text.py --model almanach/camembertv2-base --name cbv2 --fold 0
  python train_text.py ... --merge     # assemble oof/<name>_f*.npz en oof/<name>_oof.npy / _test.npy
fp32 : sur la 1080 Ti (Pascal), le fp16 est plus lent.
"""
import argparse

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from common import HERE, N_FOLDS, OOF, folds, load, save_preds, wf1

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--max-len", type=int, default=160)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()

tr, te, codes = load()
y = tr.label.values
splits = folds(y)

if args.merge:
    oof = np.zeros((len(tr), len(codes)))
    test = np.zeros((len(te), len(codes)))
    for k, (_, b) in enumerate(splits):
        z = np.load(OOF / f"{args.name}_f{k}.npz")
        oof[b] = z["val"]
        test += z["test"] / N_FOLDS
    print(f"{args.name} OOF : {wf1(y, oof.argmax(1)):.4f}")
    save_preds(args.name, oof, test)
    raise SystemExit

set_seed(42 + args.fold)
tok = AutoTokenizer.from_pretrained(args.model)


def ds(df, labels=None):
    d = {"text": list(df.text)}
    if labels is not None:
        d["label"] = labels
    return Dataset.from_dict(d).map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                                    batched=True, remove_columns=["text"])


a, b = splits[args.fold]
train_ds, val_ds, test_ds = ds(tr.iloc[a], y[a]), ds(tr.iloc[b], y[b]), ds(te)

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(codes))
out = HERE / "runs" / f"{args.name}_f{args.fold}"
targs = TrainingArguments(
    output_dir=str(out), learning_rate=args.lr, num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
    per_device_eval_batch_size=128, weight_decay=0.01, warmup_ratio=0.06, lr_scheduler_type="linear",
    eval_strategy="epoch", save_strategy="no", logging_steps=200, report_to="none",
    group_by_length=True, dataloader_num_workers=2, seed=42 + args.fold,
)
trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                  tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=lambda p: {"wf1": wf1(p.label_ids, p.predictions.argmax(1))})
trainer.train()
trainer.args.group_by_length = False   # sinon l'ordre des prédictions est permuté


def probs(d):
    return torch.from_numpy(trainer.predict(d).predictions).float().softmax(-1).numpy()


pv, pt = probs(val_ds), probs(test_ds)
print(f"{args.name} pli {args.fold} : {wf1(y[b], pv.argmax(1)):.4f}")
np.savez(OOF / f"{args.name}_f{args.fold}.npz", val=pv, test=pt)
