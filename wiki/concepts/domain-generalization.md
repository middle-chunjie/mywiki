---
type: concept
title: Domain Generalization
slug: domain-generalization
date: 2026-04-20
updated: 2026-04-20
aliases: [OOD generalization, 领域泛化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Domain Generalization** (领域泛化) — the ability of a model trained on one source domain to remain effective on unseen target domains without target-specific supervised retraining.

## Key Points

- The paper evaluates domain generalization for end-to-end ODQA, not just retrieval or reading in isolation.
- A Wikipedia-trained source model is tested on seven datasets spanning PubMed, StackOverflow, Reddit, news, legal text, and Wikipedia.
- The authors show that zero-shot transfer can fail badly even when retrieval metrics remain high, because answer-containing passages often do not justify the answer.
- Performance differences are tied to the severity and type of dataset shift, motivating shift-aware intervention selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dua-2023-adapt]]
- [[unknown-nd-rare-2410-20088]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dua-2023-adapt]].
