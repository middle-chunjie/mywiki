---
type: concept
title: Inverse Cloze Task
slug: inverse-cloze-task
date: 2026-04-20
updated: 2026-04-20
aliases: [ICT, 逆完形填空任务]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Inverse Cloze Task** (逆完形填空任务) — a retrieval pre-training objective that asks a model to recover the source document of a sentence or span, providing a supervised signal for dense retrievers without manual labels.

## Key Points

- REALM warm-starts both `Embed_input` and `Embed_doc` with ICT before end-to-end latent retrieval training.
- The paper uses ICT to avoid a cold-start regime where irrelevant retrievals teach the encoder to ignore retrieved text.
- ICT provides the retriever with useful embeddings before the harder marginal-likelihood objective is applied.
- REALM reuses the same ICT initialization family as ORQA but improves downstream performance via stronger pre-training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guu-2020-realm-2002-08909]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guu-2020-realm-2002-08909]].
