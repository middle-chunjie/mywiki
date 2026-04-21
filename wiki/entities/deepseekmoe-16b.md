---
type: entity
title: DeepSeekMoE-16B
slug: deepseekmoe-16b
date: 2026-04-20
entity_type: tool
aliases: [DeepSeekMoE 16B]
tags: []
---

## Description

DeepSeekMoE-16B is the primary Mixture-of-Experts language model analyzed in [[li-2024-your-2410-10814]]. The paper reports it has `28` layers with `64` experts per layer and uses it for most qualitative analyses and detailed ablations.

## Key Contributions

- Serves as the main backbone for RW versus HS comparisons, prompt-robustness analysis, and STS ablations.
- Delivers a no-prompt MTEB average of `43.30` and a PromptEOL average of `53.28` with MoEE (sum).

## Related Concepts

- [[mixture-of-experts]]
- [[routing-weight-embedding]]
- [[prompt-sensitivity]]

## Sources

- [[li-2024-your-2410-10814]]
