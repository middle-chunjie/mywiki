---
type: concept
title: Code Duplication
slug: code-duplication
date: 2026-04-20
updated: 2026-04-20
aliases: [duplicate code, exact duplicate, 代码重复]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Duplication** (代码重复) — the presence of identical or highly similar code samples across a dataset, especially across train and test partitions.

## Key Points

- The paper reports that TL-CodeSum contains about `20%` exact code duplication across partitions.
- Increasing duplication ratio in the test set raises BLEU scores for all evaluated models.
- Retrieval-based Rencos benefits especially strongly because duplicated test examples can be retrieved from training data.
- Ignoring duplication can therefore overestimate model quality and distort comparisons.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-evaluation-2107-07112]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-evaluation-2107-07112]].
