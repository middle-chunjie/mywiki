---
type: concept
title: Conversational Dense Retrieval
slug: conversational-dense-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [CDR, 对话稠密检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Conversational Dense Retrieval** (对话稠密检索) — dense passage retrieval that encodes an entire multi-turn conversation into a continuous representation for matching against passages.

## Key Points

- ConvAug targets the conversational passage retrieval setting and uses a conversational context encoder over the whole dialogue context.
- The paper positions CDR as an end-to-end alternative to query rewriting, with stronger direct optimization for downstream retrieval.
- Generalization in CDR is limited by sparse observed conversations, motivating augmentation of semantically equivalent and contrastive variants.
- ConvAug is instantiated on top of ANCE and improves both dense and sparse conversational retrievers when used as a training framework.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
