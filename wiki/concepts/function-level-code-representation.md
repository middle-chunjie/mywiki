---
type: concept
title: Function-level Code Representation
slug: function-level-code-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [函数级代码表示, function representation for code]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Function-level Code Representation** (函数级代码表示) — an embedding of an entire function or code snippet that aims to capture its semantics for downstream tasks such as search, clone detection, and retrieval.

## Key Points

- SCodeR is designed specifically to improve function-level code embeddings rather than token-level prediction or generation quality alone.
- The paper argues that standard pre-trained code models often underperform on this objective because contrastive structure is weak or anisotropy harms semantic separation.
- Positive pairs are built from code comments and AST-derived code/context splits so the representation must encode semantics instead of shallow syntax.
- The learned encoder transfers to code search, clone detection, zero-shot code-to-code search, and notebook markdown ordering.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-softlabeled-2210-09597]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-softlabeled-2210-09597]].
