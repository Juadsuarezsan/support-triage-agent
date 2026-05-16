# Project 02 — Customer Support Triage Agent

> DistilBERT fine-tuned with LoRA for intent classification + Claude for reasoning + similar-ticket retrieval. Tickets arrive via email/Slack/chat, the system classifies, drafts a solution, and decides auto-resolve / suggest / escalate.

[![Status](https://img.shields.io/badge/status-planned-fbbf24)]()
[![LLM](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.5-7c5cff)]()
[![Fine-tune](https://img.shields.io/badge/fine--tune-DistilBERT%20%2B%20LoRA-22d3ee)]()

**Industrial use case:** Intercom, Zendesk, Freshdesk, HubSpot.

## What this project does

Automatic triage of customer support tickets. Combines a fine-tuned BERT classifier (cheap, fast) with Claude reasoning (expensive, smart) plus retrieval of similar resolved tickets. Routes each ticket to auto-resolve, suggest-to-human, or escalate based on a confidence score.

## Architecture

```
Ticket entra (email/chat/Slack)
   │
   ▼
[Intent Classifier] DistilBERT + LoRA fine-tuned on Bitext (27 intents)
   │ → intent + confidence
   │
   ▼
[Sentiment + Priority] Claude structured output (Pydantic)
   │ → sentiment, priority P0-P3, urgency_score
   │
   ▼
[Similar Tickets] Qdrant semantic search
   │ → top 5 resolved tickets with their resolutions
   │
   ▼
[Solution Drafter] Claude
   │ → proposed response grounded in past resolutions
   │
   ▼
[Confidence Estimator]
   │ score = f(intent_conf, sentiment, similarity, priority)
   │
   ├─ > 0.85 → AUTO-RESOLVE
   ├─ 0.6-0.85 → SUGGEST (human reviews)
   └─ < 0.6 → ESCALATE
   │
   ▼
[Audit Logger] Postgres
```

## Roadmap to v1.0.0

1. [ ] Load Bitext dataset (27K queries × 27 intents), 80/10/10 stratified split
2. [ ] LoRA fine-tune DistilBERT on Bitext train split, evaluate macro-F1
3. [ ] Publish fine-tuned model to **HuggingFace Hub** with full model card
4. [ ] Ingest historical resolved tickets into Qdrant (Twitter CS dataset for evaluation)
5. [ ] LangGraph agent: classifier → sentiment → similar tickets → drafter → confidence scorer
6. [ ] Eval set on Twitter dataset (200 end-to-end conversations)
7. [ ] Compare: zero-shot Claude vs classifier-only vs hybrid pipeline
8. [ ] Next.js Kanban UI showing tickets through states (pending / drafted / suggested / resolved / escalated)
9. [ ] LangSmith public trace gallery (≥30 examples)
10. [ ] Confusion matrix visualized in README

## Stack

| Layer | Technology |
|---|---|
| Classifier | `distilbert-base-uncased` or `roberta-base` + PEFT/LoRA |
| Model hosting | HuggingFace Hub (with full model card) |
| LLM | Claude Sonnet 4.5 |
| Embeddings (for retrieval) | `sentence-transformers/all-mpnet-base-v2` |
| Vector store | Qdrant |
| Orchestration | LangGraph |
| State | PostgreSQL |
| Frontend | Next.js 14 + shadcn/ui (Kanban) |
| Observability | LangSmith (public traces) |

## Definition of Done — project-specific

- [ ] DistilBERT fine-tuned and published to HF Hub with model card (benchmarks, use cases, limitations, license)
- [ ] Comparative table: zero-shot Claude vs fine-tuned-only vs hybrid system, with macro-F1 / auto-resolve / latency / cost columns
- [ ] Confusion matrix in README
- [ ] Demo: Kanban with pre-loaded tickets, can drag through states
- [ ] LangSmith gallery linked from README
- [ ] Error analysis on the 10 worst-performing intents documented in `docs/error_analysis.md`

Plus the 12 universal blocks of the DoD.

## License

MIT.
