---
type: concept
title: Self-Critical Sequence Training
slug: self-critical-sequence-training
date: 2026-04-20
updated: 2026-04-20
aliases: [SCST]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Self-Critical Sequence Training** — a policy-gradient technique that reduces variance by comparing sampled sequence reward against the reward of the model's own greedy output.

## Key Points

- [[gou-2023-diversify]] applies SCST to the style transfer model so that RL updates use `r(y^s, z^s) - b` instead of raw reward alone.
- The baseline `b` is computed from the greedy output of the current style transfer model.
- SCST is used only for the generator-side RL update; the retriever still follows a standard REINFORCE-style reward signal.
- The paper combines SCST with KL regularization to limit policy drift during RL fine-tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2023-diversify]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2023-diversify]].
