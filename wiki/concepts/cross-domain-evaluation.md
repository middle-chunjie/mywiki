---
type: concept
title: Cross-Domain Evaluation
slug: cross-domain-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [Cross-Domain Evaluation, 跨领域评估]
tags: [evaluation, generalization, benchmark]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Domain Evaluation** (跨领域评估) — measuring model performance across multiple task or repository domains rather than on a single narrow benchmark distribution.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] argues that narrow curated code benchmarks do not capture the full diversity of real software domains.
- The paper constructs a broader benchmark from open-source Python repositories with executable test suites.
- Results show substantial project-to-project performance variation that standard narrow-domain benchmarks hide.
- The moderate correlation between Gemini Pro and Gemini Nano 2 across projects suggests domain-specific strengths and weaknesses.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
