---
type: concept
title: Longest Common Subsequence
slug: longest-common-subsequence
date: 2026-04-20
updated: 2026-04-20
aliases: [LCS, 最长公共子序列]
tags: [sequence-algorithm, supervision]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Longest Common Subsequence** (最长公共子序列) — the longest sequence that appears in the same order within two sequences, without requiring contiguous matches.

## Key Points

- [[li-2023-skcoder-2302-06144]] uses the LCS between retrieved code `Y'` and target code `Y` as silver supervision for sketch construction.
- This choice preserves reusable code structure while discarding many task-specific tokens that should be edited later.
- The sketcher is trained to predict LCS-aligned tokens with a token-level classification loss rather than exact generation.
- In the paper's sketch-design comparison, LCS-based sketches outperform anonymization-based and overlap-only sketches by up to `9.13%` EM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-skcoder-2302-06144]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-skcoder-2302-06144]].
