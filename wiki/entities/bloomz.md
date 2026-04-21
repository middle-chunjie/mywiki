---
type: entity
title: BLOOMZ
slug: bloomz
date: 2026-04-20
entity_type: tool
aliases: [BLOOMZ, bloomz-3b, bloomz-7b1]
tags: []
---

## Description

BLOOMZ is the instruction-tuned LLM family used in [[merth-2024-superposition-2404-06910]] for mid-scale evaluation, with experiments on the `3B` and `7.1B` checkpoints.

## Key Contributions

- Shows that superposition prompting transfers across model sizes, reaching `0.223` accuracy at `98.3x` speedup on `bloomz-3b`.
- On `bloomz-7b1`, superposition reaches `0.253` accuracy and `93.5x` theoretical speedup on NaturalQuestions-Open.

## Related Concepts

- [[large-language-model]]
- [[retrieval-augmented-generation]]
- [[long-context-inference]]

## Sources

- [[merth-2024-superposition-2404-06910]]
