---
type: concept
title: Lightweight Alignment Projector
slug: lightweight-alignment-projector
date: 2026-04-20
updated: 2026-04-20
aliases: [alignment projector, 轻量对齐投影器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Lightweight Alignment Projector** (轻量对齐投影器) — a small trainable module that maps non-language representations into a language model's token space while leaving the backbone encoder and LLM frozen.

## Key Points

- GraphGPT uses the projector `f_P` as the only trainable interface between graph embeddings and the Vicuna token space.
- The paper notes that `f_P` can be as simple as a single linear layer, emphasizing efficiency over expressive cross-modal fusion.
- Stage 2 is initialized from the projector learned in stage 1, so the projector accumulates both structural alignment and downstream task adaptation.
- Freezing the LLM and graph encoder while tuning only this projector cuts trainable parameters from `6.61B` to `131.6M` and avoids OOM during training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-graphgpt-2310-13023]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-graphgpt-2310-13023]].
