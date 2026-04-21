---
type: entity
title: DRAGON+
slug: dragon-plus
date: 2026-04-20
entity_type: model
aliases: [DRAGON+, DRAGON Plus]
tags: []
---

## Description

DRAGON+ is the dense dual-encoder retriever used to initialize the retrieval component in [[unknown-nd-raditretrievalaugmented-2310-01352]]. In this paper it serves as both the base retriever and the starting point for retriever fine-tuning.

## Key Contributions

- Provides the initial query and document encoders for RA-DIT's retrieval stack.
- Outperforms Contriever and Contriever-MSMARCO in the paper's retriever comparison.
- Remains effective after query-encoder-only fine-tuning with LM-supervised retrieval signals.

## Related Concepts

- [[dense-retrieval]]
- [[bi-encoder]]
- [[lm-supervised-retrieval]]

## Sources

- [[unknown-nd-raditretrievalaugmented-2310-01352]]
