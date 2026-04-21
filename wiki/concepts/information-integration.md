---
type: concept
title: Information Integration
slug: information-integration
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-document integration]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information Integration** — the ability of a retrieval-augmented model to combine evidence from multiple retrieved documents in order to answer a composite question correctly.

## Key Points

- RGB builds this testbed by rewriting base questions into multi-aspect questions whose answers are distributed across different retrieved documents.
- Even without noise, the best accuracy is only `60%` in English and `67%` in Chinese, making integration far harder than simple answer extraction.
- With noise ratio `0.4`, the best scores fall to `43%` in English and `55%` in Chinese, showing that complex multi-document reasoning is especially fragile under noisy retrieval.
- The paper identifies three distinctive integration failures: merging errors, ignoring errors, and misalignment errors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-benchmarking-2309-01431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-benchmarking-2309-01431]].
