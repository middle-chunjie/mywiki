---
type: entity
title: ColBERT
slug: colbert
date: 2026-04-20
entity_type: model
aliases: [Contextualized Late Interaction over BERT]
tags: []
---

## Description

ColBERT is a late-interaction retrieval model used in [[gao-2021-coil-2104-07186]] as the main high-effectiveness comparison point. It represents both queries and documents with multiple contextualized token vectors and scores them with all-to-all soft interaction.

## Key Contributions

- Serves as the strongest passage-retrieval baseline closest in spirit to COIL's token-level interaction.
- Illustrates the cost of all-to-all late interaction that COIL aims to avoid with exact token filtering.
- Provides the effectiveness frontier that COIL-full nearly matches while using lower reported latency.

## Related Concepts

- [[dense-retrieval]]
- [[soft-matching]]
- [[contextualized-exact-lexical-match]]

## Sources

- [[gao-2021-coil-2104-07186]]
