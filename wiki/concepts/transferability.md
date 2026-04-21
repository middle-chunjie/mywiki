---
type: concept
title: Transferability
slug: transferability
date: 2026-04-20
updated: 2026-04-20
aliases: [transferability, 可迁移性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Transferability** (可迁移性) — the property that a learned component or representation can be moved to a different model or setting while retaining useful behavior.

## Key Points

- LM-Switch transfers across language models by aligning target embeddings to source embeddings with a learned linear map `H`.
- The transferred switch is `H^TWH`, meaning the original control matrix is conjugated into the target model's embedding space.
- The paper reports successful detoxification transfer from GPT-2 Large to models ranging from GPT-2 to GPT-J-6B.
- Transfer is helpful but imperfect; the authors reduce decoding strength to `0.5ε_0` because larger switch magnitudes are less stable after alignment.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
