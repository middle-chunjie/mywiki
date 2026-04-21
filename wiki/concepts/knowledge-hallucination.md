---
type: concept
title: Knowledge Hallucination
slug: knowledge-hallucination
date: 2026-04-20
updated: 2026-04-20
aliases: [hallucination]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Hallucination** (知识幻觉) — generated content that introduces knowledge claims or event relations inconsistent with grounded evidence or real-world facts.

## Key Points

- KoLA treats hallucination as the main risk inside knowledge-creating evaluation, especially when a model must imagine plausible events from narrative context.
- The benchmark's self-contrast setup is designed to separate knowledge hallucination from stylistic variation by comparing a free continuation with a knowledge-grounded continuation from the same model.
- The paper highlights hallucinated event relations, such as incorrect causal links in narrative generation, as a central failure mode that ordinary reference-overlap metrics do not isolate well.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-kola-2306-09296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-kola-2306-09296]].
