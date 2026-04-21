---
type: concept
title: Natural Language Inference
slug: natural-language-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [NLI, textual entailment, 自然语言推理]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Natural Language Inference** (自然语言推理) — the task of deciding whether a premise semantically entails, contradicts, or is neutral with respect to a hypothesis.

## Key Points

- ALCE uses the TRUE `T5-11B` NLI model as the backbone for both ELI5 claim recall and citation verification.
- Citation recall is computed by checking whether the concatenated cited passages entail the generated statement.
- Citation precision uses NLI twice: once on an individual citation and once on the remaining citations after removing it.
- The paper formats each passage as `Title: {TITLE}\n{TEXT}` before entailment checking, making NLI central to its automatic evaluation pipeline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2023-enabling-2305-14627]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2023-enabling-2305-14627]].
