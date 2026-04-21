---
type: entity
title: Zstandard
slug: zstd
date: 2026-04-20
entity_type: tool
aliases: [zstd, Zstandard]
tags: []
---

## Description

Zstandard is the entropy-coding backend used in [[xiao-2026-embedding-2602-00079]] after spherical transformation, transposition, and byte shuffling.

## Key Contributions

- Provides the final compression stage in both the baseline and the proposed spherical pipeline.
- Enables the reported high-throughput operating point, especially at level `1`.

## Related Concepts

- [[entropy-coding]]
- [[byte-shuffling]]
- [[lossless-compression]]

## Sources

- [[xiao-2026-embedding-2602-00079]]
