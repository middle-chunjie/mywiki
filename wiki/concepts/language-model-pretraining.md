---
type: concept
title: Language Model Pretraining
slug: language-model-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [LM pretraining, 语言模型预训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language Model Pretraining** (语言模型预训练) — the large-scale next-token training stage in which a language model learns generic representations and generation behavior from unlabeled text or code.

## Key Points

- This paper treats sequence construction as a first-class pretraining design choice, not just an implementation detail.
- It shows that data formatting can affect downstream context following, summarization faithfulness, and code correctness even when model architecture is unchanged.
- The reported setup trains LLaMA-style models on `300B` to `500B` tokens with RefinedWeb and the Stack under multiple context lengths.
- The method is positioned as broadly applicable to both pretraining and finetuning, although experiments focus on pretraining.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-fewer-2404-10830]]
- [[gao-2025-metadata-2501-01956]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-fewer-2404-10830]].
