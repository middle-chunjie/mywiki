---
type: concept
title: Probabilistic Context-Free Grammar
slug: probabilistic-context-free-grammar
date: 2026-04-20
updated: 2026-04-20
aliases: [PCFG, probabilistic context-free grammar, 概率上下文无关文法]
tags: [grammar, probability]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Probabilistic Context-Free Grammar** (概率上下文无关文法) — a context-free grammar whose production rules carry probabilities, thereby inducing a probability distribution over derivations and generated objects.

## Key Points

- [[fijalkow-2022-scaling]] uses a neural network to assign production probabilities to a CFG, yielding a PCFG over candidate programs.
- The induced distribution `D` is the central object of analysis: both enumeration and sampling algorithms are compared by the expected number of tries needed under `D`.
- Heap Search enumerates programs in non-increasing PCFG probability, which is exactly the loss-optimal order in the paper's framework.
- SQRT Sampling constructs a new PCFG for `sqrt(D)` by taking square roots of rule probabilities and then globally renormalizing the grammar.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
