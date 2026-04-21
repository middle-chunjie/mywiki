---
type: concept
title: Meta-Evaluation Benchmark
slug: meta-evaluation-benchmark
date: 2026-04-20
updated: 2026-04-20
aliases: [meta evaluation benchmark, 元评测基准]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Meta-Evaluation Benchmark** (元评测基准) — a benchmark used to assess the reliability of an evaluation method itself, rather than directly measuring task performance.

## Key Points

- LLMBar is designed as a meta-evaluation benchmark for testing whether LLM evaluators recover objectively correct instruction-following preferences.
- The paper argues that previous benchmarks such as FairEval, LLMEval2, and MT-Bench are limited by subjective or noisy preference labels.
- LLMBar combines a filtered Natural split with an adversarially constructed split so that evaluator failures on objective instruction following become visible.
- The benchmark reaches `94%` expert agreement, supporting the claim that its gold preferences are much less subjective than those in prior meta-evaluation sets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-evaluating-2310-07641]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-evaluating-2310-07641]].
