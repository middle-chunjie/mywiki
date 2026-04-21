---
type: concept
title: Automatic Chain-of-Thought
slug: auto-cot
date: 2026-04-20
updated: 2026-04-20
aliases: [Auto-CoT, automatic chain of thought, 自动思维链]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Automatic Chain-of-Thought** (自动思维链) — an automated prompting approach that uses model-generated reasoning exemplars as chain-of-thought demonstrations instead of manually authored ones.

## Key Points

- The paper positions Auto-CoT as the strongest prior automated CoT baseline for ODMR-style prompting.
- SP-CoT extends the same automation goal but adds dataset generation, validation constraints, and adaptive demonstration retrieval rather than relying on raw auto-generated exemplars alone.
- On ChatGPT over four benchmarks, Auto-CoT reports `22.6` EM / `29.6` F1, while SP-CoT improves this to `28.8` / `36.0`.
- The qualitative analysis also reports better clearness, conciseness, and intermediate-answer recall for SP-CoT than for Auto-CoT.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfprompted-2310-13552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfprompted-2310-13552]].
