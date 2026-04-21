---
type: concept
title: Satisfiability Modulo Theories
slug: satisfiability-modulo-theories
date: 2026-04-20
updated: 2026-04-20
aliases: [SMT, satisfiability modulo theories, 模理论可满足性]
tags: [formal-methods, search]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Satisfiability Modulo Theories** (模理论可满足性) — a constraint-solving framework that extends SAT with domain theories such as arithmetic so symbolic systems can search over structured programs more efficiently.

## Key Points

- [[balog-2017-deepcoder-1611-01989]] reviews SMT-based synthesis as a major alternative to pure enumeration for finding programs consistent with input-output examples.
- The paper highlights that theory-aware reasoning can solve arithmetic constraints much faster than naive value enumeration.
- DeepCoder uses [[sketch]] as an SMT-based synthesis back end by treating function choices and arguments as holes constrained by the examples.
- The paper also notes a tradeoff: richer DSL semantics can produce very large constraint systems whose construction may be slower than enumerative search.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[balog-2017-deepcoder-1611-01989]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[balog-2017-deepcoder-1611-01989]].
