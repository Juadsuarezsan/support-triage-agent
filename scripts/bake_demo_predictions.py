"""Run the LoRA classifier on 10 realistic tickets and write demo/predictions.json.

This file is consumed by demo/index.html so the static demo page shows
genuine model outputs (no backend required at view time).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "models" / "intent-classifier-lora"
OUT = ROOT / "demo" / "predictions.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

TICKETS = [
    "I'd like a refund for the order I placed yesterday — it never arrived.",
    "How do I cancel my subscription before the next billing cycle?",
    "Can you help me reset my password? The reset email never came.",
    "I want to talk to a real person, the chatbot keeps misunderstanding me.",
    "I forgot to add an item to my cart, can I edit my order?",
    "Where is my package? It was supposed to arrive 3 days ago.",
    "Please delete my account and erase all my personal data per GDPR.",
    "I want to change the shipping address for order #4521.",
    "What payment methods do you accept for international orders?",
    "I was charged twice for the same purchase, please look into it.",
]


def main() -> None:
    with open(MODEL / "label_mapping.json") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    n_labels = len(id2label)

    print(f"Loading base + LoRA ({n_labels} labels)...")
    base = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=n_labels,
        id2label=id2label,
        label2id=mapping["label2id"],
    )
    model = PeftModel.from_pretrained(base, str(MODEL))
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(MODEL))

    results = []
    for text in TICKETS:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**enc)
        dt_ms = (time.perf_counter() - t0) * 1000
        probs = torch.softmax(out.logits, dim=-1)[0]
        top3 = torch.topk(probs, k=3)
        top = [
            {"intent": id2label[int(i)], "confidence": float(p)}
            for i, p in zip(top3.indices, top3.values)
        ]
        results.append({
            "ticket": text,
            "top_intent": top[0]["intent"],
            "confidence": top[0]["confidence"],
            "alternatives": top[1:],
            "inference_ms": round(dt_ms, 2),
        })
        print(f"  '{text[:50]}...' -> {top[0]['intent']} ({top[0]['confidence']:.3f}) in {dt_ms:.1f}ms")

    payload = {
        "model": "distilbert-base-uncased + LoRA r=8",
        "n_labels": n_labels,
        "test_macro_f1": 0.9864,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions": results,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
