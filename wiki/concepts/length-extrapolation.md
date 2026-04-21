---
type: concept
title: Length Extrapolation
slug: length-extrapolation
date: 2026-04-20
updated: 2026-04-20
aliases: [context length generalization]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Length extrapolation** — the ability of a sequence model to remain effective when evaluated on input lengths substantially longer than those emphasized during training.

## Key Points

- The paper argues that superposition prompting improves effective length extrapolation by shortening the longest path the transformer perceives.
- On NaturalQuestions-Open, the average maximum path length drops from `2923` tokens in naive RAG to `206` tokens under superposition prompting.
- This helps explain why the method can improve accuracy without modifying model weights, especially when retrieved context exceeds the regime the model was trained on.
- The paper also connects positional assignment to extrapolation behavior, showing equilibrium positioning outperforms left-aligned padding on both ALiBi and RoPE-based models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[merth-2024-superposition-2404-06910]]
- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[merth-2024-superposition-2404-06910]].
