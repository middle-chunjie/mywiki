---
type: concept
title: SimCSE
slug: simcse
date: 2026-04-20
updated: 2026-04-20
aliases: [SimCSE, Simple Contrastive Sentence Embeddings]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**SimCSE** — a contrastive sentence representation method that creates two views of the same input by using different dropout masks and trains the model to pull those views together while pushing away other examples in the batch.

## Key Points

- LLM2Vec uses unsupervised SimCSE as its third stage to improve sequence-level embeddings after bidirectional adaptation.
- The paper reports that standard `0.1` dropout is too low for larger decoder-only LLMs and increases it to `0.3`.
- MNTP LoRA weights are merged into the base model before starting the SimCSE stage, and new LoRA parameters are initialized for contrastive training.
- SimCSE is crucial for the best unsupervised MTEB results, but is less important for final supervised training quality on some backbones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
