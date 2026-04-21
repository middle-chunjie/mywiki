---
type: concept
title: Greedy Decoding
slug: greedy-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [greedy decode, 贪心解码]
tags: [decoding, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Greedy Decoding** (贪心解码) — a decoding strategy that selects the locally highest-probability next token at each step, producing a single deterministic output path.

## Key Points

- This paper uses greedy decoding as the main baseline for chain-of-thought prompting.
- The authors argue greedy decoding suffers from local optimality and low path diversity, which is especially harmful for multi-step reasoning.
- On PaLM-540B, greedy CoT reaches `56.5` on GSM8K, far below self-consistency with majority vote at `74.4`.
- Example analyses show greedy decoding can commit to an early wrong intermediate step and then preserve that error through the final answer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfconsistency-2203-11171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfconsistency-2203-11171]].
