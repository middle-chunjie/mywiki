---
type: concept
title: Adversarial Filtering
slug: adversarial-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [adversarial selection, 对抗式过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adversarial Filtering** (对抗式过滤) — a data-selection procedure that keeps only examples that are difficult for a reference model or evaluator, thereby concentrating benchmark difficulty.

## Key Points

- The paper applies adversarial filtering to retain candidate instruction-following comparisons that mislead existing ChatGPT-based evaluators.
- Each candidate is judged by `4` ChatGPT-based evaluators under `2` output orders, producing `8` labels before filtering.
- Instances are removed when the majority already agrees with the expected gold label, so the surviving set concentrates hard comparisons instead of easy ones.
- Manual inspection follows adversarial filtering to ensure that the remaining instances still have objective preference labels and clean instructions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-evaluating-2310-07641]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-evaluating-2310-07641]].
