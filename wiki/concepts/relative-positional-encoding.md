---
type: concept
title: Relative Positional Encoding
slug: relative-positional-encoding
date: 2026-04-20
updated: 2026-04-20
aliases: [relative position encoding, 相对位置编码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Relative Positional Encoding** (相对位置编码) — a positional encoding scheme that represents pairwise relations between positions and injects them directly into attention computations rather than only assigning each token an absolute index.

## Key Points

- The paper uses relative encoding only for local tree structure, not for all node pairs in the AST.
- A relative vector `` `r_ij` `` is non-zero only when nodes `i` and `j` are adjacent in a parent-child relation; otherwise it is set to zero.
- The formulation follows disentangled attention, adding relative terms `` `(x_i W^Q)(r_ij W_r^K)^T` `` and `` `(r_ji W_r^Q)(x_j W^K)^T` `` to the token attention score.
- Restricting relative encoding to one-hop edges keeps the number of structural relation types tractable while preserving directionality between parent and child.
- Ablation shows that removing the local relative component lowers completion and summarization performance, so the relative term contributes beyond the global absolute bias.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-nd-rethinking]]
- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-nd-rethinking]].
