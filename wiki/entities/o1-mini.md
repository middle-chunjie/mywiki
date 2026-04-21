---
type: entity
title: o1-mini
slug: o1-mini
date: 2026-04-20
entity_type: tool
aliases: [OpenAI o1-mini, openai-o1-mini]
tags: [llm, reasoning, openai]
---

## Description

o1-mini is a reasoning-specialized language model from OpenAI, designed for cost-efficient performance on complex reasoning tasks including mathematics and programming. It represents the proprietary upper-bound critic in [[processbench]] evaluation.

## Key Contributions

- Achieves the highest F1 scores on all four [[processbench]] subsets (avg 87.9), significantly outperforming all open-source and other proprietary models.
- Demonstrates a large capability gap between reasoning-specialized models and general-purpose LLMs on process error identification, with especially strong performance on Olympiad-level problems (87.2 OlympiadBench, 82.4 Omni-MATH).

## Related Concepts

- [[process-error-identification]]
- [[mathematical-reasoning]]
- [[critic-model]]
- [[scalable-oversight]]

## Sources

- [[zheng-2024-processbench-2412-06559]]
