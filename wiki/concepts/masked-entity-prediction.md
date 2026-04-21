---
type: concept
title: Masked Entity Prediction
slug: masked-entity-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [MEP, 实体掩码预测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Masked Entity Prediction** (实体掩码预测) — a pretraining objective that masks detected entities in structured data and trains the model to autoregressively recover the missing entities from context.

## Key Points

- SANTA replaces detected code identifiers or product noun phrases with special sentinel tokens and asks the decoder to regenerate them.
- The same entity mention can be replaced with the same special token, encouraging the model to capture coherent structural semantics.
- Compared with random span masking, the entity-focused masking strategy yields stronger retrieval performance in the paper's ablations.
- MEP alone is weaker than [[structured-data-alignment]], but it adds complementary gains when combined with SDA in the full SANTA objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structureaware-2305-19912]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structureaware-2305-19912]].
