---
type: concept
title: Training-Free Optimization
slug: training-free-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [training free optimization, prompt-only optimization, 免训练优化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Training-Free Optimization** (免训练优化) — improving a model's outputs through prompts, feedback, or search procedures without updating the model parameters.

## Key Points

- The paper replaces fine-tuning with iterative prompt updates that expose historical questions and observed CTR values to the LLM.
- Optimization happens at inference time through in-context learning, so the same base model can serve many topics without retraining.
- The method is motivated by the cost of continually fine-tuning large language models for recommendation workloads with many contexts.
- The paper treats observed CTR as the optimization signal that guides prompt-based generation toward higher-engagement outputs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[senel-2024-generative-2406-05255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[senel-2024-generative-2406-05255]].
