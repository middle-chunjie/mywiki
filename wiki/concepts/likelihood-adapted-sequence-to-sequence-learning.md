---
type: concept
title: Likelihood-Adapted Sequence-to-Sequence Learning
slug: likelihood-adapted-sequence-to-sequence-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [likelihood-adapted seq2seq learning, iterative permutation training, 似然自适应序列到序列学习]
tags: [training, generative-retrieval, sequence-to-sequence]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Likelihood-Adapted Sequence-to-Sequence Learning** (似然自适应序列到序列学习) — an iterative training strategy for generative retrieval where the Seq2Seq model's training objective (the permutation of the term-set document identifier) is dynamically updated each iteration to the highest-likelihood permutation under the current model snapshot.

## Key Points

- Because the document identifier is an unordered term-set, there is no fixed ground-truth sequence; training instead requires selecting which permutation to maximize at each update step.
- At iteration $t$, top-K sampling is applied to the difference set $\mathcal{T}(D) \setminus I_{<i}$ conditioned on the current model $\Theta^{t-1}$; the sampled candidate sequences are re-ranked by overall likelihood and the argmax is taken as the new objective $I^t$.
- Initialization matters: ordering terms by their matching-oriented importance scores yields the best starting permutation, outperforming random (`MRR@10 0.757` vs `0.723`) and likelihood-based LM ordering (`0.757` vs `0.715`) on NQ320K.
- Convergence is fast: retrieval accuracy on a held-out validation set stabilizes after only two iterations in practice.
- Non-adaptive training (fixed random permutation throughout) degrades performance (MRR@100 0.745 vs 0.760), confirming that aligning the training target with test-time constrained greedy search is necessary.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-generative-2305-13859]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-generative-2305-13859]].
