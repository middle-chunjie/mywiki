---
type: entity
title: nomic-embed-text-v1
slug: nomic-embed-text-v1
date: 2026-04-20
entity_type: tool
aliases: [nomic embed text v1, nomic embed]
tags: []
---

## Description

nomic-embed-text-v1 is the `137M`-parameter English long-context text embedding model introduced in [[nussbaum-2025-nomic-2402-01613]]. It is positioned as a fully reproducible open model that supports inference up to `8192` tokens.

## Key Contributions

- Outperforms Ada-002 and `text-embedding-3-small` on average scores over [[mteb]] and [[loco-benchmark]].
- Demonstrates that a compact encoder can remain competitive on long-context retrieval benchmarks.
- Is released together with weights, code, and curated training data.

## Related Concepts

- [[text-embedding]]
- [[contrastive-finetuning]]
- [[dynamic-ntk-scaling]]

## Sources

- [[nussbaum-2025-nomic-2402-01613]]
