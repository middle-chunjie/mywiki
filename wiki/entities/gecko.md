---
type: entity
title: Gecko
slug: gecko
date: 2026-04-20
entity_type: tool
aliases: [Gecko-1B, Gecko-1B-256, Gecko-1B-768]
tags: []
---

## Description

Gecko is the compact text embedding model introduced in [[lee-2024-gecko-2403-20327]]. It is a `1.2B` dual encoder trained with LLM-distilled supervision and supports `768`- and `256`-dimensional embeddings through [[matryoshka-representation-learning]].

## Key Contributions

- Combines pre-finetuning with a unified fine-tuning mixture built around [[fret]].
- Reaches `66.31` average on MTEB with `768` dimensions and `64.37` with `256` dimensions.
- Balances retrieval, semantic similarity, and classification better than several larger baselines.

## Related Concepts

- [[knowledge-distillation]]
- [[dense-retrieval]]
- [[matryoshka-representation-learning]]

## Sources

- [[lee-2024-gecko-2403-20327]]
