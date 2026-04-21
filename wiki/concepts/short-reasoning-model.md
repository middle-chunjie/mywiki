---
type: concept
title: Short Reasoning Model
slug: short-reasoning-model
date: 2026-04-20
updated: 2026-04-20
aliases: [SRM, short CoT model]
tags: [llm, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Short Reasoning Model** (短推理模型) — a reasoning model that retains structured problem-solving behavior while generating relatively short chain-of-thought traces.

## Key Points

- The paper argues that LCPO-trained long-CoT models can be repurposed as effective short-CoT models simply by requesting shorter outputs.
- Under matched generation lengths, L1-Max beats its non-reasoning Qwen baseline and slightly exceeds GPT-4o on the reported average.
- The short-CoT regime is important because it reduces latency and token cost while preserving much of the reasoning benefit.
- Qualitative analysis suggests short and long CoTs use similar high-level strategies but with different frequencies of verification and conclusion steps.
- This result is presented as evidence that RL can distill useful long-form reasoning behaviors into compact traces.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
