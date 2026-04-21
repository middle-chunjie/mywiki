---
type: concept
title: Retrieval-Based Few-Shot Prompting
slug: retrieval-based-few-shot-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic retrieval prompting]
tags: [llm, prompting, code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Based Few-Shot Prompting** — a prompting strategy that retrieves task-relevant exemplars from a corpus at inference time and inserts them into the prompt to improve generation on the current input.

## Key Points

- The paper retrieves similar optimization examples from PIE instead of using a fixed few-shot prompt.
- Programs are embedded with CodeBERTScore models trained for `C++`, and nearest neighbors are retrieved with FAISS.
- The preferred configuration uses `K = 4` retrieved examples to build a `slow -> fast` prompt on the fly.
- Dynamic retrieval substantially outperforms static few-shot prompting for program optimization, especially at `Best@8`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shypula-2024-performanceimproving-2302-07867]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shypula-2024-performanceimproving-2302-07867]].
