---
type: concept
title: Intra-block Encoder
slug: intra-block-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [within-block encoder]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Intra-block Encoder** (块内编码器) — a block-local Transformer module that models token interactions inside one block while incorporating an updated block summary carrying cross-block context.

## Key Points

- Each intra-block encoder receives a dispatched block summary from the inter-block encoder together with the block's token embeddings.
- This design injects global context into local token modeling instead of encoding each block independently.
- Longtriever stacks intra-block and inter-block encoders repeatedly, so tokens can indirectly interact across distant blocks.
- The block size hyperparameter controls the trade-off between richer local context and computational efficiency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-longtriever]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-longtriever]].
