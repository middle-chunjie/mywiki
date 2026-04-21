---
type: concept
title: Unsupervised Evaluation
slug: unsupervised-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [Unsupervised Evaluation, 无监督评估]
tags: [evaluation, methodology, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Unsupervised Evaluation** (无监督评估) — model assessment that avoids manually labeled reference outputs and instead relies on automatic signals or structure already present in the data.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] proposes RTC as an unsupervised alternative to hand-curated code benchmarks.
- The method leverages unit tests, exact match, and round-trip reconstruction instead of human-written prompts or target descriptions.
- It is motivated by the cost and limited coverage of benchmarks such as HumanEval and ARCADE.
- The paper shows unsupervised RTC can still correlate strongly with standard supervised benchmark metrics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
