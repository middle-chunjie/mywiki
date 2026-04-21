---
type: entity
title: AFL
slug: afl
date: 2026-04-20
entity_type: tool
aliases: [American Fuzzy Lop]
tags: []
---

## Description

AFL is the random-fuzzing baseline compared against ALPHAPROG in [[li-2022-alphaprog]]. The paper uses AFL with a single empty seed to benchmark validity and coverage on the same Brainfuck compiler.

## Key Contributions

- Provides the baseline showing that unguided fuzzing is markedly less effective than ALPHAPROG on structured compiler inputs.
- Reaches only `35%` peak validity and `43,135` tested basic blocks in the reported comparison.

## Related Concepts

- [[compiler-fuzzing]]
- [[basic-block-coverage]]
- [[epsilon-greedy-exploration]]

## Sources

- [[li-2022-alphaprog]]
