---
type: concept
title: Long-Context Language Model
slug: long-context-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [long context LLM, 长上下文语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Context Language Model** (长上下文语言模型) — a language model designed to preserve useful conditioning and retrieval behavior over prompts containing tens of thousands of tokens.

## Key Points

- Mixtral is trained with a fully dense `32k` context window rather than relying on retrieval-only extensions at inference time.
- On the passkey retrieval task, the paper reports `100%` retrieval accuracy regardless of both context length and passkey location.
- The proof-pile experiment shows monotonic perplexity improvements as larger context windows are provided.
- The paper presents long-context capability as one of Mixtral's core advantages over smaller-context dense baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-mixtral-2401-04088]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-mixtral-2401-04088]].
