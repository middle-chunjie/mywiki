---
type: concept
title: Style Preference Alignment
slug: style-preference-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [SPS, style preference set, 风格偏好对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Style Preference Alignment** (风格偏好对齐) — a preference-alignment strategy that trains a model to generate responses matching the stylistic quality of human-labeled gold answers, by constructing preference sets from multiple LLMs of varying capability.

## Key Points

- KnowPAT constructs a Style Preference Set (SPS) using gold answer (highest) + answers from ChatGPT, ChatGLM-6B, and Vicuna-7B (decreasing quality order confirmed by LLM ranking benchmarks and human expert verification).
- The assumption is that LLMs of different capability produce answers with different stylistic properties; the model learns to prefer higher-capability styles through contrastive training.
- SPS targets the user-friendliness axis: domain-specific QA must produce appropriately toned, sufficiently detailed, and non-offensive answers.
- Unlike knowledge preference, style preference does not require KG access; the preference signal comes purely from the relative quality of generated text.
- Combined with KPS, SPS contributes to an additive performance gain (ablation: removing SPS costs BLEU-1 −4.99 pts).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-knowledgeable-2311-06503]].
