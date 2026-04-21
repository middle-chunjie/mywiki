---
type: concept
title: Token Efficiency
slug: token-efficiency
date: 2026-04-20
updated: 2026-04-20
aliases: [token efficiency, 词元效率]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Token Efficiency** (词元效率) — the proportion of token budget spent on useful generated outputs rather than on repeated prompt overhead.

## Key Points

- The paper defines token efficiency as the fraction of tokens in one call that are allocated to generated outputs.
- Under the paper's simplified few-shot accounting, standard prompting has `η = 1 / (K + 1)` while batch prompting has `η = b / (K + b)`.
- Because shared demonstrations dominate token count, increasing `b` improves efficiency nearly inverse-linearly for practical few-shot prompts.
- The authors argue token efficiency is a more stable comparison target than wall-clock time because API implementations and rate limits can change over time.
- The empirical curves on multiple datasets match the predicted reduction trend as batch size grows.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2023-batch-2301-08721]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2023-batch-2301-08721]].
