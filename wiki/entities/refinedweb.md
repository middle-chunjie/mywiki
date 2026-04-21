---
type: entity
title: RefinedWeb
slug: refinedweb
date: 2026-04-20
entity_type: dataset
aliases: [RefinedWeb dataset]
tags: []
---

## Description

RefinedWeb is the large web-scraped pretraining corpus used as NeoBERT's main data source in [[breton-2025-neobert-2502-19587]]. The paper describes it as a `600B`-token dataset that replaces the much older and smaller corpora used for BERT-style encoders.

## Key Contributions

- Supplies the main pretraining data responsible for the largest positive ablation gain on GLUE (`+3.6%` relative).
- Serves as the base corpus from which the longer-sequence RefinedWeb1024+ and RefinedWeb2048+ subsets are derived.

## Related Concepts

- [[pretraining-data-mixture]]
- [[masked-language-modeling]]
- [[long-context-training]]

## Sources

- [[breton-2025-neobert-2502-19587]]
