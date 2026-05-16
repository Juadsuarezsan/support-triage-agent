"""LoRA fine-tune script for DistilBERT on Bitext 27-intent classification.

Run:  python notebooks/01_classifier_training.py
Output: models/intent-classifier-lora/ (adapter + tokenizer + metrics)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

OUT_DIR = Path(__file__).parent.parent / "models" / "intent-classifier-lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "distilbert-base-uncased"
N_SAMPLES = 4500
SEED = 20260516


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Loading Bitext from HF Hub ...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    intents = sorted(set(ds["intent"]))
    label2id = {l: i for i, l in enumerate(intents)}
    id2label = {i: l for l, i in label2id.items()}

    # Stratified subsample for CPU-friendly training
    rng = np.random.default_rng(SEED)
    subset_idx: list[int] = []
    for intent in intents:
        candidates = [i for i, t in enumerate(ds["intent"]) if t == intent]
        per_class = min(N_SAMPLES // len(intents), len(candidates))
        pick = rng.choice(candidates, size=per_class, replace=False)
        subset_idx.extend(pick.tolist())
    rng.shuffle(subset_idx)
    ds = ds.select(subset_idx)
    print(f"  stratified subset rows: {len(ds)}")

    ds = ds.shuffle(seed=SEED)
    n = len(ds)
    train = ds.select(range(int(n * 0.8)))
    val   = ds.select(range(int(n * 0.8), int(n * 0.9)))
    test  = ds.select(range(int(n * 0.9), n))
    print(f"  splits: train={len(train)} val={len(val)} test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def preprocess(batch):
        enc = tokenizer(batch["instruction"], truncation=True, max_length=128)
        enc["labels"] = [label2id[i] for i in batch["intent"]]
        return enc

    train_tok = train.map(preprocess, batched=True, remove_columns=train.column_names)
    val_tok   = val.map(preprocess,   batched=True, remove_columns=val.column_names)
    test_tok  = test.map(preprocess,  batched=True, remove_columns=test.column_names)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(intents), id2label=id2label, label2id=label2id,
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.argmax(axis=-1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "trainer-output"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-4,
        logging_steps=50,
        report_to=[],
        seed=SEED,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"[{time.strftime('%H:%M:%S')}] Training (CUDA: {torch.cuda.is_available()})...")
    t0 = time.perf_counter()
    trainer.train()
    train_secs = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Training done in {train_secs:.1f}s")

    test_out = trainer.predict(test_tok)
    y_pred = test_out.predictions.argmax(axis=-1)
    y_true = test_out.label_ids
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, target_names=intents,
                                     output_dict=True, zero_division=0)

    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    with open(OUT_DIR / "label_mapping.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    with open(OUT_DIR / "training_metrics.json", "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_labels": len(intents),
            "epochs": 2, "batch_size": 16, "lr": 5e-4,
            "train_seconds": train_secs,
            "test_macro_f1": test_macro_f1,
            "per_class_f1": {k: v["f1-score"] for k, v in report.items() if k in intents},
        }, f, indent=2, default=str)

    print(f"\nDONE — test macro-F1 = {test_macro_f1:.4f}")
    print(f"Artifacts at: {OUT_DIR}")


if __name__ == "__main__":
    main()
