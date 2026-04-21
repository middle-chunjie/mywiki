---
type: entity
title: ShareGPT
slug: sharegpt
date: 2026-04-20
entity_type: dataset
aliases: [ShareGPT-Mix]
tags: []
---

## Description

ShareGPT is the public conversation-sharing source used in [[gudibande-2023-false-2305-15717]] to build the paper's broad-coverage imitation corpus. After deduplication and English filtering, it contributes roughly `50K` dialogue examples to the training mixture.

## Key Contributions

- Supplies the largest component of the paper's broad imitation dataset.
- Provides diverse public user instructions with low nearest-neighbor BLEU similarity of about `8%`.
- Anchors the evidence that broad conversational imitation improves style more than factual capability.

## Related Concepts

- [[model-imitation]]
- [[instruction-tuning]]
- [[distribution-shift]]

## Sources

- [[gudibande-2023-false-2305-15717]]
