---
type: concept
title: Zero-Shot Chain-of-Thought
slug: zero-shot-cot
date: 2026-04-20
updated: 2026-04-20
aliases: [Zero-shot-CoT, zero shot chain of thought, 零样本思维链]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Chain-of-Thought** (零样本思维链) — prompting a language model to produce intermediate reasoning steps without providing worked examples in the prompt.

## Key Points

- The paper treats Zero-shot-CoT as a scalable automated baseline but argues that its generated reasoning quality is unstable for difficult open-domain multi-hop questions.
- On ChatGPT, Zero-shot-CoT averages `20.6` EM / `25.4` F1 across the four ODMR benchmarks, clearly below SP-CoT's `28.8` / `36.0`.
- In the qualitative analysis on MuSiQue, Zero-shot-CoT produces less clear and less direct intermediate reasoning than SP-CoT.
- The comparison motivates reusing retrieved high-quality demonstrations instead of only appending a generic step-by-step trigger phrase.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfprompted-2310-13552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfprompted-2310-13552]].
