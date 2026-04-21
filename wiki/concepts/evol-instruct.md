---
type: concept
title: Evol-Instruct
slug: evol-instruct
date: 2026-04-20
updated: 2026-04-20
aliases: [Evol-Instruct]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Evol-Instruct** — a data-synthesis procedure that rewrites existing instructions into more complex variants to produce harder supervision for instruction fine-tuning.

## Key Points

- [[luo-2023-wizardcoder-2306-08568]] ports Evol-Instruct from WizardLM into the code domain instead of using it only for general chat data.
- The paper simplifies the evolution prompt format and explicitly removes some generic transformations while adding code-oriented ones.
- Starting from [[code-alpaca]], iterative evolution expands the training set from `20k` seeds to as many as `98k` instructions across four rounds.
- Model selection is tied to downstream HumanEval `pass@1`, so data evolution is treated as an empirical optimization loop rather than a fixed recipe.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[luo-2023-wizardcoder-2306-08568]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[luo-2023-wizardcoder-2306-08568]].
