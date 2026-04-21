---
type: concept
title: Input-Output Prediction
slug: input-output-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [I/O prediction, 输入-输出预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Input-Output Prediction** (输入-输出预测) — a supervision format in which a model predicts feasible inputs or outputs for an executable procedure, usually explaining the prediction in natural language rather than emitting code.

## Key Points

- CodeI/O turns raw Python programs into paired input-prediction and output-prediction tasks built from executable test cases.
- The dataset is roughly balanced at `50%/50%` between the two directions, which the paper argues exposes complementary reasoning skills.
- Each prompt includes cleaned reference code, a natural-language query, and either the given input or the given output.
- The response is a chain-of-thought rationale in natural language, designed to decouple reasoning flow from code syntax.
- Training on the full mixed objective outperforms input-only (`56.1` average) and output-only (`56.4`) ablations, reaching `57.2`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-codeio-2502-07316]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-codeio-2502-07316]].
