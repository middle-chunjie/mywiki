---
type: entity
title: FRet
slug: fret
date: 2026-04-20
entity_type: dataset
aliases: [Few-shot Prompted Retrieval]
tags: []
---

## Description

FRet is the synthetic retrieval training dataset created in [[lee-2024-gecko-2403-20327]]. It contains `6.6M` examples, each with a task description, query, positive passage, and hard negative passage produced through two-step LLM distillation.

## Key Contributions

- Uses few-shot LLM prompting to generate diverse task-query pairs from web passages.
- Relabels positives and negatives with LLM reranking, with positive replacement occurring in about `15%` of examples.
- Provides enough supervision for a zero-shot Gecko model to reach `62.64` average on MTEB.

## Related Concepts

- [[synthetic-data-generation]]
- [[hard-negative-mining]]
- [[knowledge-distillation]]

## Sources

- [[lee-2024-gecko-2403-20327]]
