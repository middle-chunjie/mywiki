---
type: entity
title: DCLM
slug: dclm
date: 2026-04-20
entity_type: dataset
aliases: [DCLM-Baseline]
tags: []
---

## Description

DCLM is the main pre-training corpus used in [[gao-2025-metadata-2501-01956]], where it serves as the strongest of the three evaluated data sources. The paper describes it as a filtered version of a RefinedWeb reproduction and uses it for the headline MeCo results.

## Key Contributions

- Provides the main `160B`-token training data for the paper's `1.6B` language-model experiments.
- Supports the headline comparison in which MeCo matches a `240B`-token standard baseline with only `160B` tokens.
- Supplies the URL metadata distribution used for ablations over top URLs, hashed URLs, and suffixes.

## Related Concepts

- [[language-model-pretraining]]
- [[metadata-conditioning]]
- [[pretraining-data-mixture]]

## Sources

- [[gao-2025-metadata-2501-01956]]
