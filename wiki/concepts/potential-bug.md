---
type: concept
title: Potential Bug
slug: potential-bug
date: 2026-04-20
updated: 2026-04-20
aliases: [latent bug, 潜在缺陷]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Potential Bug** (潜在缺陷) — a small edit in a partial code prefix that is not necessarily erroneous in isolation but causes at least one otherwise valid completion to fail.

## Key Points

- [[dinh-2023-large-2306-03438]] defines a potential bug relative to a reference prefix `s` and a completion `c`, not as an unconditional bug in the prefix alone.
- The paper focuses on potential bugs that preserve syntactic validity while altering semantics, making them harder for completion models to detect.
- In buggy-HumanEval, potential bugs are constructed by semantic-opposite operator flips such as `!= -> ==` or `+ -> -`.
- The concept motivates why partial draft code should be treated as noisy intent rather than trusted ground truth during completion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dinh-2023-large-2306-03438]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dinh-2023-large-2306-03438]].
