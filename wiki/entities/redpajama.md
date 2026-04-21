---
type: entity
title: RedPajama
slug: redpajama
date: 2026-04-20
entity_type: dataset
aliases: [RPJ, RedPajama Data]
tags: []
---

## Description

RedPajama is a large open pretraining corpus used in the paper as one of the main reference datasets for `∞`-gram indexing. After decontamination, it contributes `1.4T` tokens and is used both for latency benchmarking and for improving neural LM perplexity.

## Key Contributions

- Supplies the largest single datastore used in the paper's infini-gram experiments.
- Helps drive stronger perplexity gains when combined with the Pile, especially for LLaMA-2 models.
- Serves as the reference corpus for the paper's trillion-token latency benchmarks.

## Related Concepts

- [[data-decontamination]]
- [[language-modeling]]
- [[nonparametric-language-model]]

## Sources

- [[liu-2024-infinigram-2401-17377]]
