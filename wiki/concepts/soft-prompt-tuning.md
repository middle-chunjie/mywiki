---
type: concept
title: Soft Prompt Tuning
slug: soft-prompt-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [soft prompt tuning, prompt tuning, 软提示微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Soft Prompt Tuning** (软提示微调) — a parameter-efficient adaptation method that keeps the pretrained language model frozen and learns continuous prompt vectors in its embedding space.

## Key Points

- The paper freezes the LLM and feeds a learned graph token into the decoder, turning graph understanding into a soft-prompt conditioning problem.
- Vanilla prompt tuning is one of the main baselines, implemented with `10` virtual tokens and no graph retrieval.
- G-Retriever extends soft prompt tuning by deriving the prompt from a GNN-encoded retrieved subgraph instead of learning free prompt parameters alone.
- Under a frozen LLM, this graph-conditioned prompt substantially outperforms plain prompt tuning on ExplaGraphs, SceneGraphs, and WebQSP.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-gretriever-2402-07630]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-gretriever-2402-07630]].
