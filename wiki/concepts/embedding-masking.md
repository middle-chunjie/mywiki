---
type: concept
title: Embedding Masking
slug: embedding-masking
date: 2026-04-20
updated: 2026-04-20
aliases: [EM, 嵌入掩码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Embedding Masking** (嵌入掩码) — a data-augmentation strategy that masks random dimensions of the concatenated embedding vector while preserving every original feature field in the input.

## Key Points

- CL4CVR uses embedding masking instead of feature masking because CVR inputs mix user, ad, context, and interaction fields that cannot be safely removed wholesale.
- If there are `F` fields and embedding size `K`, the masking vector operates in `F K` dimensions rather than `F`, enabling finer-grained perturbations.
- Two independently masked embedding views of the same sample are fed to the same encoder to form a positive pair for contrastive learning.
- The paper shows EM alone outperforms random and correlated feature masking on both experimental datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ouyang-2023-contrastive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ouyang-2023-contrastive]].
