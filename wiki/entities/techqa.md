---
type: entity
title: TechQA
slug: techqa
date: 2026-04-20
entity_type: dataset
aliases: [Tech QA]
tags: []
---

## Description

TechQA is a technical-support question answering dataset and one of the three low-resource benchmarks in [[samuel-2023-can-2309-12426]]. In this source it acts as the main negative case, showing that GPT-4-based augmentation does not reliably help in the smallest-data regime.

## Key Contributions

- Highlights the failure mode of synthetic augmentation under extremely limited data and a tiny evaluation set.
- The original trainset reaches `11.11` EM / `39.45` F1, while the strongest augmentation remains below the T5 baseline of `44.44` EM / `59.92` F1.

## Related Concepts

- [[machine-reading-comprehension]]
- [[data-augmentation]]
- [[domain-shift]]

## Sources

- [[samuel-2023-can-2309-12426]]
