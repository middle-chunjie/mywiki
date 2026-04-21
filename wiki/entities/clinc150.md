---
type: entity
title: CLINC150
slug: clinc150
date: 2026-04-20
entity_type: dataset
aliases: [CLINC 150, CLINC, clinc150-dataset]
tags: [intent-recognition, benchmark, dialogue]
---

## Description

CLINC150 is a benchmark dataset for intent classification and out-of-scope prediction, containing 150 intent categories (plus an out-of-scope class) spread across 10 domains with 22,500 training utterances in total. It is widely used as an external labeled source for pre-training intent detection models.

## Key Contributions

- Provides high-quality, domain-diverse intent annotations used for supervised pre-training in MTP-CLNN (8 of 10 domains, excluding "Banking" and "Credit Cards" to avoid target domain leakage).
- Benchmark for evaluating intent classification and OOD detection.

## Related Concepts

- [[intent-recognition]]
- [[new-intent-discovery]]
- [[out-of-distribution-generalization]]

## Sources

- [[zhang-2022-new-2205-12914]]
