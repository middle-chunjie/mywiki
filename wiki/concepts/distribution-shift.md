---
type: concept
title: Distribution Shift
slug: distribution-shift
date: 2026-04-20
updated: 2026-04-20
aliases: [domain shift, 分布偏移]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Distribution Shift** (分布偏移) — a mismatch between the data distribution used for training or fine-tuning and the distribution encountered at evaluation or deployment time.

## Key Points

- The authors observe that adding more ShareGPT-Mix imitation data can reduce accuracy on evaluation sets relative to the unfine-tuned base model.
- They attribute these regressions to tension between conversational imitation data and downstream benchmarks such as factual QA or coding.
- The paper raises regularization and mixing with pretraining data as possible ways to mitigate this shift, but does not test them.
- Distribution shift is presented as one reason broad imitation data may look good to humans while still harming benchmark accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gudibande-2023-false-2305-15717]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gudibande-2023-false-2305-15717]].
