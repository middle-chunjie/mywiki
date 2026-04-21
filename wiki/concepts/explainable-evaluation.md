---
type: concept
title: Explainable Evaluation
slug: explainable-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [explainable evaluation, 可解释评估]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Explainable Evaluation** (可解释评估) — an evaluation setup that exposes the intermediate reasoning steps, module outputs, and error evidence behind a score instead of returning only a scalar metric.

## Key Points

- `VPEval` implements explainable evaluation by routing prompts through specialized modules for object, count, spatial, scale, and text-rendering skills.
- The framework returns textual explanations and visual evidence such as bounding boxes or OCR outputs alongside binary or averaged scores.
- For open-ended prompts, the paper generates evaluation programs dynamically with an LLM instead of using one fixed scoring function.
- The reported human-correlation gains over CLIP, captioning metrics, and single VQA models are central evidence that module decomposition improves evaluation quality.
- The paper also argues that explainability helps identify which skills a T2I model fails on rather than collapsing all errors into one uninterpretable number.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cho-2023-visual-2305-15328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cho-2023-visual-2305-15328]].
