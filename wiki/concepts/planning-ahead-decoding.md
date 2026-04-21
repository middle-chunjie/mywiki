---
type: concept
title: Planning-Ahead Decoding
slug: planning-ahead-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [PAG, planning-ahead constrained beam search, 前瞻解码]
tags: [decoding, generative-retrieval, ir]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Planning-Ahead Decoding** (前瞻解码) — a constrained beam search variant for generative retrieval that augments each prefix's score with a pre-computed document-level relevance estimate, so that prefixes leading to globally relevant documents are less likely to be pruned even when their immediate next-token probability is low.

## Key Points

- Motivation: standard constrained beam search is a greedy local algorithm; since each relevant document has exactly one DocID, pruning any prefix of that DocID is an irrecoverable failure. PAG addresses this by injecting global document-level scores as lookahead signals.
- Modified prefix score: `s'(prefix; q) = max_{d ∈ D sharing prefix} s_simul(q, d) + s(prefix; q)`, where `s_simul` is a simultaneous (set-based) relevance score computed in a single forward pass before beam search begins.
- The simultaneous scores act as priors: only the top-n documents (n=1000) by simultaneous score contribute to the lookahead, keeping the prefix-to-document dictionary compact.
- Final document ranking uses only the sequential score `s(c^d; q)` to avoid over-relying on the set-based signal after decoding completes.
- Empirically much less sensitive to beam size than RIPOR: MRR@10 at beam 10 (0.379) vs beam 100 (0.385) on MSMARCO Dev, a gap of only 1.6% vs RIPOR's much larger sensitivity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2024-planning-2404-14600]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2024-planning-2404-14600]].
