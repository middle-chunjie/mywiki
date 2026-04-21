---
type: concept
title: Multi-Granular Contrastive Loss
slug: multi-granular-contrastive-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-granularity contrastive loss, 多粒度对比损失]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Granular Contrastive Loss** (多粒度对比损失) — a training objective that jointly supervises ranking behavior at a coarse retrieval level and a finer sub-unit level within the same encoded text.

## Key Points

- AGRaME augments ColBERTv2's passage-level distillation loss with an additional sentence-level KL objective inside each passage.
- The sentence-level loss is weighted by the cross-encoder passage relevance score so more relevant passages contribute more to fine-grained supervision.
- A separate sentence-marking cross-encoder `CE'` provides soft labels over sentences in a passage rather than only over passages.
- The full training objective is `L = L_psg + L_sent`, allowing fine-grained improvements without sacrificing passage-level quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[reddy-2024-agrame-2405-15028]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[reddy-2024-agrame-2405-15028]].
