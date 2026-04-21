---
type: concept
title: Group Sequence Policy Optimization
slug: group-sequence-policy-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [GSPO]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Group Sequence Policy Optimization** — a policy-optimization method that computes sequence-level importance weighting and advantage estimates over grouped samples generated for the same prompt or question.

## Key Points

- IterResearch uses GSPO as the base RL algorithm underneath its efficiency-aware reward design.
- All rounds from the `G = 16` rollouts of one question are treated as a training group for normalized advantage computation.
- The objective clips importance ratios across sequence likelihood terms while averaging over the retained multi-round corpus.
- In the paper's ablation, plain GSPO is a strong baseline, but EAPO attains slightly better efficiency at comparable accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
