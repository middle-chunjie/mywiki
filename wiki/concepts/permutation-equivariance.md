---
type: concept
title: Permutation Equivariance
slug: permutation-equivariance
date: 2026-04-20
updated: 2026-04-20
aliases: [permutation equivalent function, 置换等变性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Permutation Equivariance** (置换等变性) — the property that permuting the inputs produces the same permutation of outputs, meaning the model does not intrinsically encode order.

## Key Points

- The paper re-derives that a simplified Transformer without positional information is permutation equivariant.
- This property explains why self-attention and feed-forward layers alone are insufficient for tasks where token order matters.
- FLOATER is motivated as a way to break permutation equivariance by injecting learned positional structure into every block.
- The argument is used to justify the need for explicit positional encoding even in strong non-recurrent architectures.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2020-encode-2003-09229]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2020-encode-2003-09229]].
