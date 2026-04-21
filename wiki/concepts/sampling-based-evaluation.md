---
type: concept
title: Sampling-Based Evaluation
slug: sampling-based-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [sample-based evaluation, 基于采样的评估]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sampling-Based Evaluation** (基于采样的评估) — assessing a generative model by drawing full outputs under a decoding policy and measuring downstream task success on those samples.

## Key Points

- The paper argues that sampling benchmarks reveal FIM quality differences that are nearly invisible in FIM loss or perplexity.
- All infilling benchmarks are evaluated with nucleus sampling, and the number of samples per task is increased to reduce variance.
- This evaluation lens is essential for comparing FIM rates, PSM/SPM formatting, and span-selection strategies because practical usefulness depends on generated completions.
- The paper recommends preferring samples that end with `<EOT>` and reranking them as a mitigation for infilling failures.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
