---
base_model: distilbert-base-uncased
library_name: peft
license: mit
language: [en]
tags: [lora, text-classification, customer-support, intent-classification, bitext]
datasets: [bitext/Bitext-customer-support-llm-chatbot-training-dataset]
metrics: [f1]
model-index:
  - name: distilbert-support-intent-lora
    results:
      - task: { type: text-classification, name: Customer-support intent classification }
        dataset:
          name: Bitext Customer Support (stratified subset, 4,482 rows)
          type: bitext/Bitext-customer-support-llm-chatbot-training-dataset
          split: test (449 examples)
        metrics:
          - { type: f1, name: Macro-F1, value: 0.9864 }
---

# distilbert-support-intent-lora

LoRA adapter (r=8, alpha=16, dropout 0.1 on `q_lin` and `v_lin`) on top of
`distilbert-base-uncased` for 27-way customer-support intent classification.
Trained for the [Customer Support Triage Agent](https://github.com/Juadsuarezsan/support-triage-agent),
where it is the first node of a LangGraph triage flow.

**Publication status:** the adapter weights (`adapter_model.safetensors`) are
not yet on the Hugging Face Hub; this card, the adapter config, the tokenizer
and `label_mapping.json` are versioned in the repository. Publishing needs a
write token for `Juadsuarezsan/distilbert-support-intent-lora`.

## Intended use

Route English customer-support tickets to one of the 27 Bitext intents
(`cancel_order`, `get_refund`, `track_order`, `contact_human_agent`, ...).
Downstream, a rule-based router combines the top score with retrieval
similarity and sentiment to decide auto-resolve / suggest / escalate.

Out of scope: languages other than English, multi-intent tickets, tickets
longer than 128 tokens (truncated), and anything outside e-commerce/account
support.

## Training data

Bitext Customer Support LLM Chatbot Training Dataset (CC BY 4.0), 26,872
`instruction`/`intent` pairs. A stratified subset of 4,482 rows (~166 per
intent) was drawn with seed 20260516 and split 80/10/10:
train 3,585 / val 448 / test 449.

## Training procedure

- Script: `notebooks/01_classifier_training.py` (CPU, seed 20260516)
- 2 epochs, batch 16, lr 5e-4, AdamW, max length 128, fp32
- Trainable parameters: ~750 K of ~67 M
- Wall time: 199 s on CPU

## Evaluation

| Split | n | Macro-F1 |
|---|---|---|
| test (stratified subset) | 449 | **0.9864** |

Per-class F1 is in `training_metrics.json`; 18 of 27 classes reach 1.0. The
weakest classes are `delete_account` (0.929), `track_refund` (0.941),
`get_refund` (0.952) and `review` (0.952): semantically neighbouring labels
(refund ×3, account ×3, contact ×2). See `docs/error_analysis.md` in the
repository. With ~17 test examples per class, one error moves a class F1 by
~0.06, so rankings below rank 5 are within noise. Evaluation on the full 10 %
test split (2,687 examples) is pending.

## How to use

```python
import json
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "models/intent-classifier-lora"  # or the Hub ID once published
mapping = json.load(open(f"{path}/label_mapping.json"))
id2label = {int(k): v for k, v in mapping["id2label"].items()}
base = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=27, id2label=id2label, label2id=mapping["label2id"]
)
model = PeftModel.from_pretrained(base, path).eval()
tok = AutoTokenizer.from_pretrained(path)
enc = tok("I want a refund for my order", return_tensors="pt", truncation=True, max_length=128)
print(id2label[int(model(**enc).logits.argmax(-1))])  # get_refund
```

## Limitations and risks

- Trained on templated synthetic data: spelling errors, emoji, code-switching
  and long multi-issue tickets will degrade accuracy.
- The label set is fixed; unseen intents are forced into the nearest class
  with a possibly high score. The triage router mitigates this with the
  similarity term and hard escalation rules, not the classifier.
- No demographic or fairness evaluation was performed; ticket text may
  contain personal data that must be handled by the calling system.

## License

Adapter: MIT. Base model: Apache 2.0 (`distilbert-base-uncased`). Dataset:
CC BY 4.0 (Bitext); verify Bitext's terms for commercial deployment.

## Contact

Juan David Suárez Sánchez · juadsuarezsan@unal.edu.co
