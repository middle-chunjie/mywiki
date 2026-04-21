---
type: entity
title: Hotpot-QA
slug: hotpot-qa
date: 2026-04-20
entity_type: dataset
aliases: [HotpotQA]
tags: []
---

## Description

Hotpot-QA is the large-scale BeIR retrieval dataset used in [[unknown-nd-adaptive-2405-03651]] to test indexing and adaptive retrieval with `5,233,329` items. In this source, it is the main stress test for comparing matrix factorization, adaCUR, and dual-encoder distillation at scale.

## Key Contributions

- Serves as the paper's largest indexing benchmark for approximate cross-encoder retrieval.
- Demonstrates that `MFInd` can index `5` million items in under `3` hours for one reported setting while remaining competitive with stronger but costlier baselines.

## Related Concepts

- [[information-retrieval]]
- [[approximate-nearest-neighbor-search]]
- [[inductive-matrix-factorization]]

## Sources

- [[unknown-nd-adaptive-2405-03651]]
