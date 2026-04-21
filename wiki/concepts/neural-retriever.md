---
type: concept
title: Neural Retriever
slug: neural-retriever
date: 2026-04-20
updated: 2026-04-20
aliases: [dense retriever, neural retrieval model, 神经检索器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Neural Retriever** (神经检索器) — a learned retrieval model that embeds queries and candidates into a vector space and ranks candidates by representation similarity rather than hand-written lexical rules alone.

## Key Points

- In LAIL, the retriever is trained after LLM-based labeling so that retrieval aligns with the target model's preference for useful demonstrations.
- The paper uses GraphCodeBERT to encode natural-language requirements into `[CLS]` representations for retrieval.
- Training uses a contrastive loss with one positive example, one hard negative from the low-scoring set, and `63` additional sampled negatives.
- At inference time, the retriever encodes each test requirement once, compares it against cached training-example embeddings with cosine similarity, and returns the top prompt candidates.
- The learned retriever transfers across both LLMs and datasets, showing that it captures reusable structure beyond a single model-instance pairing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-large-2310-09748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-large-2310-09748]].
