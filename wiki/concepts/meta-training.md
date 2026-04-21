---
type: concept
title: Meta-Training
slug: meta-training
date: 2026-04-20
updated: 2026-04-20
aliases: [meta training, 元训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Meta-Training** (元训练) — training a model across many tasks or episodes so that it can adapt to new tasks more effectively at inference time, often by learning reusable task-conditioning behavior.

## Key Points

- This paper studies MetaICL, which is initialized from GPT-2 Large and then meta-trained with an explicit in-context-learning objective.
- Meta-training makes the model markedly less sensitive to whether demonstration labels are correct, with only `0.1-0.9%` degradation from gold to random labels.
- The authors argue that meta-training encourages exploitation of simpler prompt cues such as format and label-space hints rather than true input-label correspondence.
- In the paper's analysis, MetaICL shows near-zero reliance on some prompt components that non-meta-trained models still use, especially in direct and channel variants.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[min-2022-rethinking-2202-12837]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[min-2022-rethinking-2202-12837]].
