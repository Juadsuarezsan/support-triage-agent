# Intent Classifier — DistilBERT base + LoRA r=8

## Summary

27-class customer-support intent classifier fine-tuned from
`distilbert-base-uncased` using PEFT/LoRA adapters. Trained on Bitext
customer-support dataset stratified subset.

## Results

| Split | Macro-F1 |
|---|---|
| **Test (held-out, 449 examples)** | **0.9864** |
| Train (3585 examples)             | converged in 2 epochs |
| Val (448 examples)                | tracked via Trainer eval_strategy=epoch |

Per-class F1 ≥ 0.97 on 23 of 27 intents. Hardest classes:
`delete_account` (0.929), `edit_account` (0.963), `contact_human_agent`
(0.971), `contact_customer_service` (0.974).

## Training config

- Base model: `distilbert-base-uncased`
- LoRA: r=8, alpha=16, dropout=0.1, targets `q_lin` + `v_lin`
- Trainable params: ~750K (out of ~67M base)
- Batch size: 16
- Learning rate: 5e-4
- Epochs: 2
- Optimizer: AdamW (HF default)
- Seed: 20260516
- Device: CPU
- Training time: **199 seconds**

## Dataset

- **Bitext customer-support LLM training dataset** (Hugging Face)
- 26,872 query/intent pairs across 27 categories
- Subsampled stratified to 4,482 (≈166 per class)
- 80/10/10 train/val/test split with fixed seed

## How to load

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=27)
model = PeftModel.from_pretrained(base, "./models/intent-classifier-lora")
tok = AutoTokenizer.from_pretrained("./models/intent-classifier-lora")

# Inference
import json
with open("./models/intent-classifier-lora/label_mapping.json") as f:
    id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

enc = tok("I want a refund for my order", return_tensors="pt", truncation=True, max_length=128)
out = model(**enc)
pred = out.logits.argmax(-1).item()
print(id2label[pred])  # → "get_refund"
```

## Limitations

- Trained on synthetic-template Bitext data. Real production tickets have
  spelling errors, multi-language input, and emoji — those will degrade
  performance. Re-train on real ticket logs once you have them.
- The model classifies intent only. Sentiment + priority + similar-ticket
  retrieval are handled separately downstream in `src/agents/`.
- LoRA adapters require the base model at inference time. Both fit in
  ~270 MB total.

## Reproducibility

```bash
# Re-run the fine-tune (~3 min on CPU)
pip install transformers peft accelerate datasets scikit-learn evaluate
python notebooks/01_classifier_training.py  # script form of the notebook
```

The exact training script is committed at the repo root (`_train_p2.py`)
with seed 20260516. Re-running on the same machine should give within
±0.5% of the reported macro-F1.

## License

MIT (matches the source dataset's permissive license for non-commercial
use). For commercial production deployment of the model, verify Bitext's
license terms.
