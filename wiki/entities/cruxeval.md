---
type: entity
title: CRUXEval
slug: cruxeval
date: 2026-04-20
entity_type: benchmark
aliases: [CRUXEval, Code Reasoning, Understanding and Execution Evaluation]
tags: []
---

## Description

CRUXEval is the benchmark introduced in [[gu-2024-cruxeval-2401-03065]] for testing whether code language models can reason about and execute short Python functions via paired input-prediction and output-prediction tasks.

## Key Contributions

- Provides `800` filtered Python functions of length `3-13` lines with two complementary execution-reasoning tasks.
- Exposes a gap between HumanEval-style code generation gains and actual program-execution reasoning.
- Serves as a testbed for studying prompting, diversity, and fine-tuning effects on code reasoning.

## Related Concepts

- [[benchmark-dataset]]
- [[code-understanding]]
- [[execution-based-evaluation]]
- [[pass-at-k]]

## Sources

- [[gu-2024-cruxeval-2401-03065]]
