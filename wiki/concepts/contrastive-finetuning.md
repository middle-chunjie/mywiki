---
type: concept
title: Contrastive Finetuning
slug: contrastive-finetuning
date: 2026-04-20
updated: 2026-04-20
aliases: [contrastive fine-tuning, 对比微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Finetuning** (对比微调) — a supervised representation-learning stage that refines an encoder on labeled positive pairs and hard negatives to improve downstream retrieval or semantic similarity behavior.

## Key Points

- In the paper, contrastive finetuning is the final stage that converts the pretrained encoder into the released embedding model.
- The loss extends the base contrastive objective by inserting `H` hard negative documents into the partition function.
- The authors train on about `1.6M` supervised examples from MSMarco, NQ, NLI, HotpotQA, FEVER, MEDI subsets, WikiAnswers, and Reddit.
- Retrieval datasets use mined hard negatives from `gte-base`, while non-retrieval tasks use random negatives when mining does not help.
- The reported best setting trains for `1` epoch with `7` hard negatives per pair, batch size `256`, and learning rate `2e-5`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[nussbaum-2025-nomic-2402-01613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[nussbaum-2025-nomic-2402-01613]].
