---
type: concept
title: Prompt Compression
slug: prompt-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt compaction, prompt reduction, 提示压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Compression** (提示压缩) — the reduction of prompt length while preserving the information needed for a target model to produce behavior close to that of the original prompt.

## Key Points

- [[jiang-2023-llmlingua-2310-05736]] studies prompt compression specifically for black-box LLM APIs where model-side pruning or quantization is unavailable.
- The paper frames compression as minimizing the divergence between generations from the compressed prompt and the original prompt under an explicit token budget `τ = L~/L`.
- LLMLingua combines coarse-grained demonstration filtering, fine-grained token selection, and distribution alignment instead of using one-shot lexical filtering alone.
- The reported results show prompt compression can preserve reasoning and summarization quality even at `5x-20x` compression ratios, depending on the task.
- Prompt compression also reduces downstream generation cost because compressed prompts tend to induce shorter generated outputs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2023-llmlingua-2310-05736]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2023-llmlingua-2310-05736]].
