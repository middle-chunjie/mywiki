---
type: concept
title: Tag Classification
slug: tag-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [tag classification, algorithm tag prediction, 标签分类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tag Classification** (标签分类) — a multi-label task that predicts problem or algorithm tags from source code and optionally from the accompanying natural-language problem description.

## Key Points

- [[unknown-nd-xcodeevalan-2303-03004]] includes Tag Classification as one of the two classification tasks in xCODEEval.
- It is the only task in the benchmark that does not depend on execution-based evaluation.
- The tag labels come from Codeforces problem metadata and are also reused in the paper's validation/test balancing procedure.
- Adding the problem description improves `gpt-3.5-turbo-0301` from `27.29` to `33.60` macro-F1 on average.
- The task covers `5,494,008` training instances across `11` languages, making it one of the largest components of the benchmark.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-xcodeevalan-2303-03004]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-xcodeevalan-2303-03004]].
