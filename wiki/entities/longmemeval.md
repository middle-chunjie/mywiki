---
type: entity
title: LongMemEval
slug: longmemeval
date: 2026-04-20
entity_type: dataset
aliases: [LongMemEval-S, long-mem-eval]
tags: []
---

## Description

LongMemEval is a benchmark for evaluating long-term interactive memory in chat assistants. The LongMemEval-S split uses ultra-long conversations of roughly 100K tokens each, with diverse question types requiring multi-session memory recall.

## Key Contributions

- Provides a challenging evaluation setting for memory systems that must reason over very long interaction histories (~100K tokens per sample).
- Used in MemSkill as a cross-dataset transfer evaluation: skills learned on LoCoMo are applied directly to LongMemEval without any LongMemEval-specific training.
- Stratified evaluation sample covers approximately 100 examples drawn from about one-fifth of the dataset, ensuring coverage of diverse question types.

## Related Concepts

- [[long-term-memory]]
- [[memory-management]]
- [[self-evolving-agent-memory]]

## Sources

- [[zhang-2026-memskill-2602-02474]]
