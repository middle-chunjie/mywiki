---
type: concept
title: Matching-Oriented Term Selection
slug: matching-oriented-term-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [matching-oriented term extraction, 匹配导向词项选择]
tags: [generative-retrieval, document-identifier, ir]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Matching-Oriented Term Selection** (匹配导向词项选择) — a term-importance estimation method that selects document terms as a document identifier based on their utility for query-document matching, supervised by an InfoNCE contrastive objective over labeled query-document pairs.

## Key Points

- Terms are scored by a BERT-based encoder: each token $t_i^D$ is projected to a scalar importance weight $w_i^D$ via a linear layer and ReLU activation; the model is trained with an InfoNCE loss that rewards terms bridging query-document lexical overlap.
- The objective encourages high weights for terms that co-occur between query $Q$ and its relevant document $D^+$: `min(-log(exp(Σ w_i^Q w_j^D+ / τ) / (exp(...) + Σ_m exp(...))))`.
- The top-$N$ terms by importance form the document identifier `T(D)`; for NQ320K, $N=12$ already guarantees zero identifier collisions across 320k documents.
- Ablation shows matching-oriented selection substantially outperforms random selection (MRR@100 0.760 vs 0.631) and also beats title-only selection (0.760 vs 0.745) on NQ320K, confirming that term-matching signals are more discriminative than title tokens alone.
- The selection module is trained once and frozen; the same importance weights also determine the initial permutation order used to bootstrap the Seq2Seq training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-generative-2305-13859]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-generative-2305-13859]].
