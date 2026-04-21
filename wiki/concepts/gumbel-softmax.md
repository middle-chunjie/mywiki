---
type: concept
title: Gumbel-Softmax
slug: gumbel-softmax
date: 2026-04-20
updated: 2026-04-20
aliases: [Gumbel-Softmax, Gumbel Softmax, Gumbel-Softmax 重参数化]
tags: [optimization, discrete-learning, transformer]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gumbel-Softmax** (Gumbel-Softmax 重参数化) — a differentiable relaxation for sampling from categorical distributions, used to optimize discrete structural choices with gradient-based methods.

## Key Points

- [[friedman-2023-transformer-2306-01128]] uses Gumbel-Softmax to optimize distributions over discrete gate selections and predicate-matrix rows in Transformer Programs.
- The same relaxation is also applied to the argmax step in hard attention so discrete routing decisions can be trained jointly with the rest of the model.
- The paper anneals the softmax temperature from `3.0` to `0.01` over training, so samples become increasingly close to one-hot selections.
- After training, the model collapses the relaxed distributions to their maximum-likelihood discrete choices before converting the network into Python code.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-transformer-2306-01128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-transformer-2306-01128]].
