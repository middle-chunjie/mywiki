---
type: concept
title: Information Gain
slug: information-gain
date: 2026-04-20
updated: 2026-04-20
aliases: [marginal information gain, 信息增益]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information Gain** (信息增益) — the incremental value of newly retrieved evidence measured by how much it improves coverage of the required supporting information beyond what previous retrieval rounds have already found.

## Key Points

- StepSearch computes information gain against gold supporting documents using cosine similarity over TF-IDF document representations.
- For each gold document `d_i^g`, the method tracks a memory value `m_i^t` storing the best similarity seen so far.
- The round-level gain is `G^t = (1 / n) * sum_i max(c_i^t - m_i^t, 0)`, so only genuinely new evidence contributes positive reward.
- This design encourages search policies that progressively uncover new sub-evidence instead of repeatedly retrieving near-duplicate passages.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-stepsearch-2505-15107]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-stepsearch-2505-15107]].
