---
type: concept
title: Multi-Conversation Optimization
slug: multi-conversation-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-conv optimization]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Conversation Optimization** — an RL training setup where several context-independent conversations generated for one sample share the same final reward signal and are optimized jointly as parts of a single task trajectory.

## Key Points

- MemAgent treats each memory-update turn and the final answer turn as separate conversations rather than concatenating them into one tool-style transcript.
- The final answer reward is broadcast to all preceding conversations from the same sample, aligning memory writing with downstream answer quality.
- The paper extends DAPO loss computation from a `(group, token)` structure to `(group, conversation, token)`.
- This design is presented as the key algorithmic change needed to train chunked read-write memory workflows with RL.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2026-memagent-2507-02259]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2026-memagent-2507-02259]].
