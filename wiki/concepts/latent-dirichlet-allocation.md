---
type: concept
title: Latent Dirichlet Allocation
slug: latent-dirichlet-allocation
date: 2026-04-20
updated: 2026-04-20
aliases: [LDA, 潜在狄利克雷分配]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Latent Dirichlet Allocation** (潜在狄利克雷分配) — a probabilistic topic model that represents documents and words with mixtures over latent topics.

## Key Points

- [[fang-2021-guided]] uses LDA to obtain document topic vector `z_d` and word-level topic vectors `z_w` before neural encoding.
- These topic distributions are concatenated with token embeddings so the Bi-LSTM receives topic-enhanced inputs for both titles and body text.
- The number of topics is dataset-specific: `50` for CSEN, `100` for KP-20K, and `50` for MTB.
- LDA is treated as an external preprocessing step that provides global semantic structure for topic-aware attention.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2021-guided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2021-guided]].
