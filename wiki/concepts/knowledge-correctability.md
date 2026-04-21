---
type: concept
title: Knowledge Correctability
slug: knowledge-correctability
date: 2026-04-20
updated: 2026-04-20
aliases: [correctable reasoning, editable evidence, 知识可纠正性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Correctability** (知识可纠正性) — the ability to fix a model's answer by editing or replacing the explicit evidence it relied on, without retraining the model itself.

## Key Points

- ToG treats KG triples as editable external evidence, so users can revise faulty facts and rerun the reasoning process.
- The paper's stadium-name example shows how tracing an answer back to an outdated triple can guide manual correction.
- Correctability is presented as a practical advantage over parametric-only LLM knowledge, whose updates are expensive and slow.
- The concept depends on the combination of explicit paths, user feedback, and an external KG that can be amended.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-thinkongraph-2307-07697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-thinkongraph-2307-07697]].
