---
type: entity
title: OpenELM
slug: openelm
date: 2026-04-20
entity_type: tool
aliases: [OpenELM-3B-Instruct, OpenELM]
tags: []
---

## Description

OpenELM is the RoPE-based LLM family used in [[merth-2024-superposition-2404-06910]] as one evaluation backbone, specifically through the `OpenELM-3B-Instruct` checkpoint.

## Key Contributions

- Provides the paper's RoPE-based testbed for evaluating equilibrium position assignment without extra fine-tuning.
- Under superposition prompting on NaturalQuestions-Open, reaches `0.241` accuracy with `96.8x` theoretical speedup versus naive RAG `0.001`.

## Related Concepts

- [[large-language-model]]
- [[positional-encoding]]
- [[length-extrapolation]]

## Sources

- [[merth-2024-superposition-2404-06910]]
