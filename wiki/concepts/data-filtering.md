---
type: concept
title: Data Filtering
slug: data-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [quality filtering, 数据过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Filtering** (数据过滤) — the process of selecting a higher-quality subset from a larger corpus using rules, learned models, or both.

## Key Points

- [[cui-2022-codeexp-2211-15395]] combines heuristic screening with a learned BERT filter to refine noisy GitHub code-docstring pairs.
- The heuristics require `6-30` code lines, docstrings longer than `3` lines, and cyclomatic complexity above `3`.
- The learned filter predicts adequacy and coverage scores, then keeps examples whose predicted scores exceed `1.0` after rescaling.
- The resulting refined set is much smaller than the raw set but substantially more useful for downstream generation quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cui-2022-codeexp-2211-15395]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cui-2022-codeexp-2211-15395]].
