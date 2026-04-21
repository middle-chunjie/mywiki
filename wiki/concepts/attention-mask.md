---
type: concept
title: Attention Mask
slug: attention-mask
date: 2026-04-20
updated: 2026-04-20
aliases: [attention masking, 注意力掩码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Attention Mask** (注意力掩码) — a constraint matrix that determines which token positions are allowed to attend to which other positions during self-attention.

## Key Points

- [[ratner-2023-parallel-2212-10947]] modifies the standard causal mask so each context window remains autoregressive only within itself.
- The PCW mask assigns task tokens a broader receptive field: they can attend to all preceding tokens from all windows plus earlier task tokens.
- This mask is the second core ingredient of PCW, paired with positional-embedding reuse.
- The design preserves training-free compatibility with off-the-shelf decoder-only LLMs while changing the information flow available at inference time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ratner-2023-parallel-2212-10947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ratner-2023-parallel-2212-10947]].
