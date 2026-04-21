---
type: concept
title: Few-Shot Indexing
slug: few-shot-indexing
date: 2026-04-20
updated: 2026-04-20
aliases: [少样本索引]
tags: [indexing, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Few-Shot Indexing** (少样本索引) — an indexing strategy that builds retrieval-time identifiers or representations by prompting a pre-trained model with a small set of demonstrations instead of training model parameters on the target corpus.

## Key Points

- In this paper, few-shot indexing generates free-text docids for every document using an LLM and a small demonstration prompt.
- The indexing pipeline first generates pseudo queries for each document, then asks the LLM to turn those pseudo queries into docids.
- The resulting docid bank is created with no fine-tuning, which sharply reduces indexing cost relative to training-based generative retrieval.
- Because the index is externalized as a docid bank, adding or removing documents is conceptually easier than retraining a GR model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[askari-2024-generative-2408-02152]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[askari-2024-generative-2408-02152]].
