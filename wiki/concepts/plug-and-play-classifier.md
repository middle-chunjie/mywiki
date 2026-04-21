---
type: concept
title: Plug-and-Play Classifier
slug: plug-and-play-classifier
date: 2026-04-20
updated: 2026-04-20
aliases: [plug-and-play classifier, 即插即用分类器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Plug-and-Play Classifier** (即插即用分类器) — a lightweight classifier attached to frozen model representations so that a new control decision can be added without fine-tuning the base model.

## Key Points

- UAR implements every retrieval criterion with a separate plug-and-play single-layer MLP over the last-token hidden state.
- Because the LLM is frozen, the same input encoding can be shared between generation and retrieval-timing classification.
- This design gives UAR negligible extra inference cost relative to methods that fine-tune or repeatedly query the full model.
- The paper treats plug-and-play classifiers as the systems mechanism that makes multifaceted active retrieval practical.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-unified-2406-12534]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-unified-2406-12534]].
