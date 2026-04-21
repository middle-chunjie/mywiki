---
type: entity
title: GTR-XXL
slug: gtr-xxl
date: 2026-04-20
entity_type: tool
aliases: [GTR XXL, Google T5 Retriever XXL]
tags: []
---

## Description

GTR-XXL is the large dense retriever used in [[lin-2023-how-2302-07452]] both as a baseline that breaks the supervised/zero-shot tradeoff through scale and as a later-stage teacher in DRAGON's progressive supervision pipeline.

## Key Contributions

- Demonstrates that strong zero-shot dense retrieval was previously achievable with much larger models.
- Supplies a stronger dense supervision signal between Contriever and ColBERTv2 in the final DRAGON curriculum.

## Related Concepts

- [[dense-retrieval]]
- [[zero-shot-retrieval]]
- [[curriculum-learning]]

## Sources

- [[lin-2023-how-2302-07452]]
