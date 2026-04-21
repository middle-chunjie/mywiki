---
type: entity
title: DensePhrases
slug: densephrases
date: 2026-04-20
entity_type: tool
aliases: [Dense Phrases]
tags: []
---

## Description

DensePhrases is the dense phrase retrieval system studied in [[lee-2021-phrase-2109-08133]]. It indexes multiple phrase vectors per passage and is reused in the paper as a unified retriever for phrases, passages, and documents.

## Key Contributions

- Outperforms DPR on top-rank passage retrieval metrics without any architecture change or retraining for passage retrieval.
- Supports document-level adaptation through query-side fine-tuning with `L_doc` for entity linking and grounded dialogue retrieval.
- Combines phrase filtering and OPQ compression to reduce the Wikipedia-scale index from `307GB` to `69GB` and lower with more aggressive filtering.

## Related Concepts

- [[phrase-retrieval]]
- [[dense-passage-retrieval]]
- [[vector-quantization]]

## Sources

- [[lee-2021-phrase-2109-08133]]
