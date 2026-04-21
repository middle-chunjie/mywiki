---
type: entity
title: DeepSeekMoE
slug: deepseekmoe
date: 2026-04-20
entity_type: tool
aliases: [DeepSeek MoE]
tags: []
---

## Description

DeepSeekMoE is the sparse FFN architecture used inside DeepSeek-V2. It combines fine-grained routed experts with isolated shared experts so that the model can scale total capacity while keeping activated compute low.

## Key Contributions

- Supplies the MoE block design that replaces almost all dense FFNs in DeepSeek-V2.
- Introduces a structure with `2` shared experts, `160` routed experts, and `6` active routed experts per token in the main model.
- Works together with expert parallelism and routing constraints to make large-scale sparse training economical.

## Related Concepts

- [[mixture-of-experts]]
- [[expert-parallelism]]
- [[transformer]]

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]
