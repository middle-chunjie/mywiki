---
type: concept
title: Kendall's Tau
slug: kendalls-tau
date: 2026-04-20
updated: 2026-04-20
aliases: [Kendall tau, tau correlation, 肯德尔秩相关系数]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Kendall's Tau** (肯德尔秩相关系数) — a rank-correlation statistic that measures how often two scoring systems agree on pairwise preferences.

## Key Points

- [[cui-2022-codeexp-2211-15395]] uses adapted Kendall `τ` to compare automatic metrics against human judgments on code explanations.
- The statistic is computed from concordant, discordant, and tie pairs using `τ = |#Con - #Dis| / |#Con + #Dis + #Tie|`.
- BLEU and METEOR achieve the strongest alignment with overall human scores in the paper's analysis.
- All reported `τ` values remain below `0.5`, which the authors interpret as evidence that current automatic metrics are still limited.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cui-2022-codeexp-2211-15395]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cui-2022-codeexp-2211-15395]].
