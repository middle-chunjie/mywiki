---
type: entity
title: SCaNN
slug: scann
date: 2026-04-20
entity_type: tool
aliases: [Scalable Nearest Neighbors, ScaNN]
tags: []
---

## Description

SCaNN is the approximate nearest-neighbor search library used in [[borgeaud-2022-improving-2112-04426]] to query chunk embeddings from RETRO's large retrieval datastore.

## Key Contributions

- Enables approximate `k`-NN retrieval over databases that scale to trillions of tokens.
- Keeps retrieval latency low enough for evaluation and sampling, with the paper reporting around `10 ms` query time for a `2T`-token database.

## Related Concepts

- [[nearest-neighbor-search]]
- [[dense-retrieval]]

## Sources

- [[borgeaud-2022-improving-2112-04426]]
