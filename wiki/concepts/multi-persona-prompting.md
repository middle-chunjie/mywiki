---
type: concept
title: Multi-Persona Prompting
slug: multi-persona-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-persona prompting, solo-performance prompting, SPP]
tags: [llm, prompting, personas]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Persona Prompting** — a prompting method in which a language model instantiates several personas, lets them discuss a problem, and then synthesizes a final answer from their dialogue.

## Key Points

- The paper uses multi-persona prompting as a zero-shot baseline that explicitly allows collaborative dialogue among generated personas.
- Meta-prompting differs by centering a conductor model that mediates all expert communication instead of allowing direct persona-to-persona exchange.
- In the reported results, multi-persona prompting attains a macro average of `57.7`, outperforming some baselines but remaining well below meta-prompting with Python (`72.9`).
- The gap is especially large on `Game of 24`, where multi-persona prompting reaches `25.0` while meta-prompting with Python reaches `67.0`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[suzgun-2024-metaprompting-2401-12954]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[suzgun-2024-metaprompting-2401-12954]].
