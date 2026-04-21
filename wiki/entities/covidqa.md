---
type: entity
title: CovidQA
slug: covidqa
date: 2026-04-20
entity_type: dataset
aliases: [COVID-QA]
tags: []
---

## Description

CovidQA is a low-resource extractive question answering dataset about the COVID-19 domain and one of the three evaluation benchmarks in [[samuel-2023-can-2309-12426]]. It is the dataset on which the paper reports the strongest gains from GPT-4-based synthetic augmentation.

## Key Contributions

- Serves as the clearest positive case for synthetic augmentation in the paper.
- Improves from `25.81` EM / `50.91` F1 on the original trainset to `31.90` EM / `58.66` F1 with one-shot cycle-consistent augmentation.

## Related Concepts

- [[machine-reading-comprehension]]
- [[data-augmentation]]
- [[round-trip-filtration]]

## Sources

- [[samuel-2023-can-2309-12426]]
