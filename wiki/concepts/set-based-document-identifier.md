---
type: concept
title: Set-Based Document Identifier
slug: set-based-document-identifier
date: 2026-04-20
updated: 2026-04-20
aliases: [set-based DocID, bag-of-token DocID, lexical document identifier, 集合式文档标识符]
tags: [generative-retrieval, document-identifier, ir]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Set-Based Document Identifier** (集合式文档标识符) — a document identifier in generative retrieval composed of an unordered set of lexical tokens extracted from the document's content, enabling simultaneous (non-autoregressive) scoring via a bag-of-words-style relevance computation.

## Key Points

- Contrasts with sequential DocIDs (ordered token sequences from residual quantization) — set-based DocIDs have no token ordering, so all tokens can be scored in parallel in one forward pass.
- Construction: a T5-based sparse encoder (pre-trained with MarginMSE + FLOPs regularization) produces sparse document representations; the top-m tokens by importance weight form the set `t^d = {t_1^d, …, t_m^d}` (m=64 in PAG).
- Lexical grounding means each token corresponds to a real subword in the generative model's vocabulary, linking semantic-DocID-based autoregressive decoding to explicit term-matching signals.
- Ablation: set-based DocIDs alone achieve MRR@10 = 0.303 on MSMARCO Dev (vs combined PAG's 0.385), confirming that lexical signals are useful but insufficient without sequential refinement.
- Index memory for set-based DocIDs: 2.77 GB for 8.8M passages at m=64 (vs 0.50 GB for sequential only, 3.27 GB for combined PAG).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2024-planning-2404-14600]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2024-planning-2404-14600]].
