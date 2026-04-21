---
type: concept
title: Parameter-Efficient Transfer Learning
slug: parameter-efficient-transfer-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [PETL, parameter efficient transfer learning, 参数高效迁移学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parameter-Efficient Transfer Learning** (参数高效迁移学习) — adapting a pretrained model to a new task by training only a small subset of added or selected parameters while keeping most backbone weights frozen.

## Key Points

- The paper formulates parameter-efficient video-text retrieval as a PETL problem built on frozen CLIP encoders.
- Its goal is to avoid storing a full fine-tuned model copy for each downstream retrieval dataset.
- MV-Adapter reaches competitive or better retrieval quality with only about `2.4%` trainable parameters.
- The reported storage budget for five tasks drops from roughly `500%` under full fine-tuning to about `112%` with shared backbone reuse.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-mvadapter-2301-07868]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-mvadapter-2301-07868]].
