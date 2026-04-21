---
type: concept
title: Zero-Shot Ranking
slug: zero-shot-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [zero-shot reranking, 零样本排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Ranking** (零样本排序) — ranking or reranking performed without task-specific fine-tuning, using a pretrained model directly at inference time.

## Key Points

- The paper explicitly studies zero-shot ranking with off-the-shelf LLMs and no additional training on ranking labels.
- Prior zero-shot pointwise and listwise prompting baselines underperform strong supervised rankers or require very large closed models such as GPT-4.
- PRP shows that moderate open models such as FLAN-T5-XL, FLAN-T5-XXL, and FLAN-UL2 can become competitive zero-shot rankers when the prompt is pairwise.
- The authors frame accessibility as a practical advantage: open models reduce dependence on expensive commercial APIs for ranking research.
- The work treats zero-shot ranking as a benchmark for inherent comparative relevance reasoning rather than learned task specialization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qin-2024-large-2306-17563]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qin-2024-large-2306-17563]].
