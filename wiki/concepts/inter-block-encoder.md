---
type: concept
title: Inter-block Encoder
slug: inter-block-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-block encoder]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Inter-block Encoder** (块间编码器) — a module that exchanges information among block-level summary tokens and a global document token so separated text blocks can share context.

## Key Points

- In Longtriever, the inter-block encoder operates on `[DOC]` plus one `[CLS]` token per block.
- Its self-attention lets each block summary attend to global document state and other block summaries.
- The module is the mechanism that makes Longtriever tightly coupled rather than a purely cascaded hierarchy.
- Ablation shows removing inter-block encoding causes a major performance drop on MS MARCO Dev Doc.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-longtriever]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-longtriever]].
