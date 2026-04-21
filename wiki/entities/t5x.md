---
type: entity
title: T5X
slug: t5x
date: 2026-04-20
entity_type: tool
aliases: [T5X codebase]
tags: []
---

## Description

T5X is Google's optimized implementation stack for T5/SeqIO-style models and serves as the baseline runtime system against which speculative decoding is compared for T5-XXL.

## Key Contributions

- Provides the baseline walltime numbers that speculative decoding improves over by roughly `2x-3x`.
- Grounds the paper's claim that the method works out of the box on an already optimized serving implementation.

## Related Concepts

- [[sequence-to-sequence]]
- [[speculative-decoding]]
- [[transformer]]

## Sources

- [[leviathan-2023-fast-2211-17192]]
