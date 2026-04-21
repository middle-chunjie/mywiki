---
type: entity
title: MiniF2F
slug: mini-f2f
date: 2026-04-20
entity_type: benchmark
aliases: [miniF2F, miniF2F-test]
tags: []
---

## Description

MiniF2F is the main theorem-proving benchmark used in [[unknown-nd-leanstar]] to evaluate pass rates for Lean-CoT and Lean-STaR. It provides held-out formalized problems on which the paper reports pass@32 and pass@64.

## Key Contributions

- Serves as the primary benchmark for comparing Lean-STaR against prior theorem-proving systems.
- Shows base-model gains from `29.5%` SFT to `34.8%` Lean-STaR pass@32 and `36.1%` pass@64.
- Shows plus-model gains up to `45.4%` pass@32 and `46.3%` pass@64.

## Related Concepts

- [[formal-theorem-proving]]
- [[sampling-based-proof-search]]
- [[expert-iteration]]

## Sources

- [[unknown-nd-leanstar]]
