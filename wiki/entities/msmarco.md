---
type: entity
title: MSMARCO
slug: msmarco
date: 2026-04-20
entity_type: dataset
aliases: [MS MARCO, MSMarco]
tags: []
---

## Description

MSMARCO is the large general-domain retrieval dataset used in RouterRetriever as both a single-model baseline and, in one setting, as an additional expert domain. In this paper it represents the strongest general-purpose alternative to routing over domain-specialized experts.

## Key Contributions

- Serves as the training source for the single general-purpose baseline that scores `47.5` average nDCG@10 on the seven-domain BEIR setup.
- Provides an optional expert that raises RouterRetriever from `49.3` to `49.6` average nDCG@10.

## Related Concepts

- [[dense-retrieval]]
- [[information-retrieval]]
- [[zero-shot-generalization]]

## Sources

- [[lee-2024-routerretriever-2409-02685]]
