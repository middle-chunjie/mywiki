---
type: entity
title: RepLLaMA
slug: repllama
date: 2026-04-20
entity_type: tool
aliases: [RepLLaMA retriever]
tags: []
---

## Description

RepLLaMA is the LLaMA-2-based dense retriever introduced in [[ma-2023-finetuning-2310-08319]]. It encodes queries and passages or documents into `4096`-dimensional vectors for inner-product retrieval.

## Key Contributions

- Uses the final `</s>` token representation as the dense embedding for both queries and documents.
- Achieves state-of-the-art effectiveness on MS MARCO passage and document retrieval among systems compared in the paper.
- Transfers strongly to zero-shot BEIR evaluation without extra contrastive pretraining.

## Related Concepts

- [[dense-retrieval]]
- [[bi-encoder]]
- [[contrastive-learning]]

## Sources

- [[ma-2023-finetuning-2310-08319]]
