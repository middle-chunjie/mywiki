---
type: entity
title: GLUE
slug: glue
date: 2026-04-20
entity_type: benchmark
aliases: [General Language Understanding Evaluation]
tags: []
---

## Description

GLUE is the downstream benchmark used to test whether the paper's simplified BERT-style models preserve task performance after pre-training. In [[he-2023-simplifying-2311-01906]], SAS and SAS-P match the Crammed Pre-LN baseline on GLUE within reported statistical variation.

## Key Contributions

- Serves as the downstream evaluation showing that simplified Transformer blocks preserve BERT fine-tuning quality.
- Quantifies the trade-off between throughput gains and end-task performance in the encoder-only setting.

## Related Concepts

- [[masked-language-modeling]]
- [[transformer]]

## Sources

- [[he-2023-simplifying-2311-01906]]
