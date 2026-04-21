---
type: concept
title: Selective Context
slug: selective-context
date: 2026-04-20
updated: 2026-04-20
aliases: [selective context]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Selective Context** — a model-agnostic context-compression method that scores lexical units by self-information and retains only the higher-information subset before LLM inference.

## Key Points

- [[li-2023-compressing]] introduces Selective Context as an input-side alternative to architectural changes such as sparse attention.
- The method computes token-level self-information with a smaller causal LM, then aggregates scores to phrases or sentences before pruning.
- Retention is controlled by a percentile threshold rather than a fixed top-`k`, allowing compression to adapt to each input's score distribution.
- Phrase-level filtering is reported as the most reliable granularity among token, phrase, and sentence variants.
- At `50%` context reduction, the paper reports materially lower GPU memory and latency with only modest quality degradation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-compressing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-compressing]].
