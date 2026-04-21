---
type: concept
title: Semi-Adversarial Benchmark Construction
slug: semi-adversarial-benchmark-construction
date: 2026-04-20
updated: 2026-04-20
aliases: [adversarial benchmark design, model-in-the-loop benchmark design, 半对抗式基准构造]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semi-Adversarial Benchmark Construction** (半对抗式基准构造) — an iterative benchmark-design process that uses strong models as critics to increase task difficulty without fully overfitting to a single system.

## Key Points

- The authors iteratively test candidate tasks against Gemini `1.5 Flash` and Gemini Thinking Experimental while increasing difficulty until both score below `70%`.
- Most of the time the reference models are treated as black boxes that only provide difficulty feedback.
- In some cases, model behavior is inspected directly to block shortcuts, such as rewriting Boolean Expressions so models cannot trivially execute Python on them.
- The paper notes that this strategy produces a stronger benchmark, but also introduces bias toward the failure modes of the chosen reference models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kazemi-2025-bigbench-2502-19187]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kazemi-2025-bigbench-2502-19187]].
