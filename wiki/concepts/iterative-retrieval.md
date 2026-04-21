---
type: concept
title: Iterative Retrieval
slug: iterative-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [interleaved retrieval]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Iterative retrieval** — a retrieval strategy that issues additional retrieval steps during inference, using intermediate model outputs such as sub-queries to gather more evidence.

## Key Points

- IterDRAG alternates between generation and retrieval rather than relying on a single initial retrieval pass.
- Retrieved evidence from later steps is merged with the original evidence, which improves recall and discounted ranking metrics over one-step DRAG.
- The paper reports especially strong gains for multi-hop datasets, where later sub-queries target missing pieces of evidence.
- Iterative retrieval is one reason the method can scale test-time compute beyond the limit of a single long-context prompt.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-inference]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-inference]].
