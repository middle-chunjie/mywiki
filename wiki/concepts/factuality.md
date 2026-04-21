---
type: concept
title: Factuality
slug: factuality
date: 2026-04-20
updated: 2026-04-20
aliases: [truthfulness, factual accuracy, 事实性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Factuality** (事实性) — the extent to which a model's generated statements are correct, grounded, and non-hallucinatory with respect to the task or world knowledge.

## Key Points

- The paper's central finding is that broad ChatGPT imitation improves style far more than factual correctness.
- On Natural Questions, broad ShareGPT-Mix fine-tuning reduces exact match from `17` to `10` for `7B` and from `20` to `15` for `13B`.
- The authors argue that crowd workers are often misled because factual errors are hidden behind fluent and authoritative language.
- Figure 2 illustrates an imitation answer that closely matches ChatGPT's structure while remaining substantively incorrect.
- The results support the view that lightweight fine-tuning does not inject large amounts of new factual knowledge into a weak base LM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gudibande-2023-false-2305-15717]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gudibande-2023-false-2305-15717]].
