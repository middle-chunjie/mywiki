---
type: concept
title: Backoff
slug: backoff
date: 2026-04-20
updated: 2026-04-20
aliases: [back-off, 回退]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Backoff** (回退) — a smoothing strategy for count-based language models that falls back to shorter contexts when a longer context does not provide reliable counts.

## Key Points

- The paper distinguishes its `∞`-gram backoff rule from classical Katz backoff.
- Instead of backing off when the continuation count is zero, it backs off only until the context denominator becomes positive.
- This choice makes the effective order depend only on the observed prefix, so `P_∞(* | context)` remains a valid distribution without extra discounting.
- Backoff is the mechanism that lets the model use the longest observed suffix while still returning probabilities for previously unseen long contexts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-infinigram-2401-17377]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-infinigram-2401-17377]].
