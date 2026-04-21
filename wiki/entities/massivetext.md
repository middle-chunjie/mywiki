---
type: entity
title: MassiveText
slug: massivetext
date: 2026-04-20
entity_type: dataset
aliases: [Massive Text]
tags: []
---

## Description

MassiveText is the multilingual corpus used for training and large-scale retrieval in [[borgeaud-2022-improving-2112-04426]]. The paper describes it as containing more than `5T` tokens across web, books, news, Wikipedia, and GitHub sources.

## Key Contributions

- Supplies the `600B`-token training retrieval database and the `1.75T`-token evaluation retrieval database for RETRO.
- Makes it possible to study how retrieval quality changes when the datastore is scaled from billions to trillions of tokens.

## Related Concepts

- [[pretraining-data-mixture]]
- [[retrieval-based-language-model]]
- [[data-decontamination]]

## Sources

- [[borgeaud-2022-improving-2112-04426]]
