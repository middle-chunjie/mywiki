---
type: concept
title: Synthetic Bug Injection
slug: synthetic-bug-injection
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic corruption, 合成缺陷注入]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Bug Injection** (合成缺陷注入) — a dataset-construction strategy that creates supervised bug examples by deliberately corrupting correct programs according to a controlled rule.

## Key Points

- This paper injects a variable-misuse bug by replacing the true variable `v_u` at a slot with another in-scope variable `v_d != v_u`.
- A paired bug-free copy of the original program is kept for every corrupted example, producing a balanced 50/50 buggy-clean training corpus.
- Slots with fewer than two repair candidates are discarded so the repair task is not trivial.
- The paper explicitly identifies a weakness of this strategy: slot-conditioned training examples do not match inference cases where the queried slot is not the actual bug location.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[vasic-2019-neural-1904-01720]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[vasic-2019-neural-1904-01720]].
