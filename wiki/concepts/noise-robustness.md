---
type: concept
title: Noise Robustness
slug: noise-robustness
date: 2026-04-20
updated: 2026-04-20
aliases: [robustness to noisy retrieval]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Noise Robustness** — the ability of a retrieval-augmented model to extract answer-bearing evidence when retrieved context includes relevant but non-answer-bearing distractor documents.

## Key Points

- RGB measures noise robustness by mixing positive and negative retrieved documents at noise ratios `0`, `0.2`, `0.4`, `0.6`, and `0.8`.
- Each evaluation instance provides `5` retrieved documents, so the capability is tested under a fixed context budget rather than unlimited retrieval.
- The paper finds that strong models remain accurate under mild noise but degrade sharply at high noise, e.g. ChatGPT falls from `96.33%` to `76.00%` in English between noise ratios `0` and `0.8`.
- Error analysis attributes failures to long-distance information, evidence uncertainty, and concept confusion within retrieved documents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-benchmarking-2309-01431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-benchmarking-2309-01431]].
