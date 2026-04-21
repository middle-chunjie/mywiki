---
type: concept
title: Monitor-Guided Decoding
slug: monitor-guided-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [MGD]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Monitor-guided decoding** (监视器引导解码) — a constrained decoding framework in which a stateful monitor queries static analysis at trigger points during generation and masks the LM's next-token distribution to enforce valid continuations.

## Key Points

- The monitor is formalized as `(A_φ, s_0, S, pre, update, maskgen)` and runs concurrently with a frozen LM.
- In this paper, the main trigger is an object dereference such as `obj.`, where static analysis returns the valid members of the inferred type.
- The approach constrains decoding without retraining the model or stuffing repository-wide context directly into the prompt.
- The framework is compositional: the authors explicitly discuss combining multiple monitors for richer semantic properties.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
