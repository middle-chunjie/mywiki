---
type: concept
title: Natural Language Reasoning
slug: natural-language-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [NL reasoning, 自然语言推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Natural Language Reasoning** (自然语言推理) — the ability of a model to answer, infer, or complete language-based tasks that require semantic understanding, commonsense, inference, or multi-step judgment.

## Key Points

- [[unknown-nd-code]] defines a composite reasoning suite of `11` benchmarks spanning QA, NLI, sentence completion, coreference resolution, and ARC-Easy.
- In the initialization study, `code -> text` and `balanced -> text` improve over the text-only baseline by `8.8%` and `8.2%` respectively.
- In the proportion sweep, a `25%`-code mixture is best, improving `3.4%` over a text-only model, while `100%` code hurts reasoning substantially.
- Adding `10%` synthetic code to code-only pretraining yields a further `9.0%` relative gain in reasoning over the web-code baseline.
- Cooldown that includes code improves reasoning by `3.6%` relative to the no-cooldown model built from the same balanced-text recipe.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code]].
