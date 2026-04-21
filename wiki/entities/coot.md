---
type: entity
title: COOT
slug: coot
date: 2026-04-20
entity_type: tool
aliases: [COOT model, Cooperative Hierarchical Transformer, coot-video-text]
tags: [model, video-text, hierarchical-transformer, cross-modal]
---

## Description

COOT (Cooperative Hierarchical Transformer) is a two-level hierarchical transformer model for video-text joint embedding learning, introduced by Ging et al. (2020); it consists of a local transformer for clip/sentence-level features and a global transformer for video/paragraph-level features.

## Key Contributions

- Serves as the backbone architecture for CrossCLR; CrossCLR replaces COOT's original losses with the CrossCLR loss, yielding R@1 improvement from 16.7 to 19.5 on Youcook2 Text→Video.
- Establishes the two-level (local + global) hierarchical loss application pattern reused in CrossCLR: `L = L_local + 0.6 * L_global`.

## Related Concepts

- [[hierarchical-transformer]]
- [[text-video-retrieval]]
- [[joint-embedding]]

## Sources

- [[zolfaghari-2021-crossclr-2109-14910]]
