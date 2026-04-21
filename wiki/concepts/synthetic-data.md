---
type: concept
title: Synthetic Data
slug: synthetic-data
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic training data, 合成数据]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Data** (合成数据) — model-generated or programmatically constructed training data designed to target desired capabilities, formats, or reasoning behaviors more directly than naturally occurring corpora.

## Key Points

- Phi-4 allocates `40%` of pretraining tokens to synthetic data and reports that repeated epochs over synthetic corpora can outperform adding more unique web tokens on reasoning-heavy benchmarks.
- The synthetic pipeline uses multi-agent prompting, self-revision, question extraction from reasoning chains, and rewrite workflows built from curated seeds.
- Purely synthetic training improves coding and reasoning but hurts knowledge-heavy evaluation such as TriviaQA, so the final recipe mixes synthetic and organic sources.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
