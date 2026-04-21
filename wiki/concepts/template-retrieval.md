---
type: concept
title: Template Retrieval
slug: template-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [thought-template retrieval, template matching]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Template Retrieval** — selecting the most relevant reusable reasoning template for a distilled task by matching task and template representations in an embedding space.

## Key Points

- BoT retrieves the best template by maximizing `Sim(f(x_d), {f(D_{T_i})})` over the template descriptions stored in the meta-buffer.
- A threshold `δ` decides whether the task is close enough to existing templates or should be treated as novel.
- The paper recommends `δ` in the range `0.5-0.7` to balance reuse against introducing new templates.
- Retrieval happens over abstract template descriptions rather than over raw exemplars or textual evidence passages.
- Better retrieval reduces the need for expensive multi-query search because the model can start from a reusable reasoning scaffold.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-buffer-2406-04271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-buffer-2406-04271]].
