---
type: concept
title: Minimal Description Length
slug: minimal-description-length
date: 2026-04-20
updated: 2026-04-20
aliases: [MDL, minimum description length]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Minimal Description Length** — a principle that prefers hypotheses or configurations that minimize the total code length needed to describe a model and the data it explains.

## Key Points

- The paper uses MDL as the ranking principle for choosing among candidate in-context example organizations.
- For a fixed pretrained model, the model-description term is constant across organizations, so ranking reduces to minimizing label codelength.
- The practical score is the expected codelength `-\mathbb{E}_{p(y_i|c,x)} log_2 p(y_i|c,x)`, which behaves like an entropy-based confidence criterion.
- Lower MDL values correlate with higher task accuracy across SST-2, SNLI, CommonsenseQA, and TREC in the reported experiments.
- MDL gives an unsupervised alternative to validation-set-based prompt selection for instance-level ICL.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-selfadaptive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-selfadaptive]].
