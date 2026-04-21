---
type: concept
title: Simultaneous Decoding
slug: simultaneous-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [simultaneous document scoring, 同步解码]
tags: [decoding, generative-retrieval, ir]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Simultaneous Decoding** (同步解码) — a decoding approach for generative retrieval in which all tokens of a set-based document identifier are scored in a single forward pass (without autoregressive conditioning), enabling efficient document-level relevance estimation across an entire corpus before sequential decoding begins.

## Key Points

- In PAG, simultaneous decoding produces sparse query-token weights `h^q = MaxPool(log(1 + ReLU(E_simul · Q^T)))` in one forward call; the document score is then `s_simul(q, d) = Σ_i h^q[t_i^d]`, summing the query weights of the document's set-based tokens.
- Computationally efficient: for MSMARCO (8.8M passages), the FLOPs for simultaneous decoding (`P_m + |C|·m`) are much less than sequential decoding (`L·k·P_m` with L=8, k=100) on T5-base.
- Does not replace autoregressive decoding but provides priors that guide it; purely simultaneous retrieval achieves MRR@10 = 0.303 (vs PAG's 0.385), so the combination is critical.
- Design inspired by SPLADE sparse retrieval: log-saturation and max-pooling operations convert dense query representations to sparse importance weights over vocabulary.
- For billion-scale corpora, brute-force simultaneous scoring may be slower than sequential decoding; ANN approximation would be needed in that regime.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2024-planning-2404-14600]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2024-planning-2404-14600]].
