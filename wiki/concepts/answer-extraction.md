---
type: concept
title: Answer Extraction
slug: answer-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [response parsing, 答案抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Answer Extraction** (答案抽取) — the procedure for parsing a model response into the normalized answer representation used for scoring or downstream evaluation.

## Key Points

- MMLU-Pro requires models to end with a discrete option choice, so evaluation depends on reliably extracting one letter from generated reasoning traces.
- The paper first matches the pattern `answer is ([A-J])`, then falls back to `Answer: ([A-J])` when formatting deviates from the expected template.
- If both regex-based parsers fail, the evaluation script samples a random option so every response still receives a score.
- The authors treat answer extraction as part of benchmark reliability because fragile parsing can distort leaderboard comparisons even when the underlying reasoning is sound.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-mmlupro-2406-01574]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-mmlupro-2406-01574]].
