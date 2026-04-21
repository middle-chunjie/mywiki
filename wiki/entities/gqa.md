---
type: entity
title: GQA
slug: gqa
date: 2026-04-20
entity_type: dataset
aliases: [Graph Question Answering]
tags: []
---

## Description

GQA is one of the source datasets used in [[ray-nd-cola]] to build both training and evaluation splits for `COLA` single-object and multi-object retrieval. The paper relies on its object and attribute annotations to generate hard compositional queries.

## Key Contributions

- Supplies the main real-image split for `COLA` retrieval experiments.
- Enables seen versus unseen attribute-object evaluation by removing selected query compositions from training.

## Related Concepts

- [[attribute-binding]]
- [[text-to-image-retrieval]]
- [[compositional-generalization]]

## Sources

- [[ray-nd-cola]]
