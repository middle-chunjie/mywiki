---
type: concept
title: Vulnerable Token
slug: vulnerable-token
date: 2026-04-20
updated: 2026-04-20
aliases: [salient token for attack]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Vulnerable Token** — an input token whose perturbation has disproportionately large influence on a model's output quality or predictions.

## Key Points

- [[jha-2023-codeattack-2206-00052]] defines vulnerable tokens through the influence score ``I_{x_i} = sum_t o_t - sum_t o'_t`` obtained by masking token `x_i`.
- Tokens are ranked by influence and only the top candidates are attacked, which reduces search cost in the black-box setting.
- The paper hypothesizes and empirically supports that keywords and identifiers often act as especially vulnerable tokens for PL models.
- The ablation study indicates that prioritizing vulnerable tokens is materially better than random token selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jha-2023-codeattack-2206-00052]].
