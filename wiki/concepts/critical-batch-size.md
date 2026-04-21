---
type: concept
title: Critical Batch Size
slug: critical-batch-size
date: 2026-04-20
updated: 2026-04-20
aliases: [critical batch size, 临界批大小]
tags: [optimization, scaling]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Critical Batch Size** (临界批大小) — the loss-dependent batch-size threshold below which increasing batch size preserves most compute efficiency and above which returns diminish sharply.

## Key Points

- The paper models critical batch size as `B_crit(L) ≈ B_* / L^{1 / alpha_B}` with `B_* ≈ 2 x 10^8` tokens and `alpha_B ≈ 0.21`.
- For Transformer language modeling, `B_crit` depends mainly on attained loss rather than directly on model size.
- Training at `B ≈ B_crit` gives a practical time/compute compromise; `B << B_crit` minimizes compute, while `B >> B_crit` minimizes serial steps.
- Kaplan et al. use `B_crit` to convert observed runs at fixed batch size into adjusted quantities such as `S_min` and `C_min`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kaplan-2020-scaling-2001-08361]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kaplan-2020-scaling-2001-08361]].
