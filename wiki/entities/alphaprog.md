---
type: entity
title: ALPHAPROG
slug: alphaprog
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

ALPHAPROG is the reinforcement-learning-based compiler-fuzzing prototype introduced in [[li-2022-alphaprog]]. It generates Brainfuck programs from scratch and scores them with compiler validity, coverage, and control-flow-complexity signals.

## Key Contributions

- Uses a deep Q-network to append one of the eight Brainfuck tokens at each generation step.
- Outperforms AFL on BFC in both valid-program rate and accumulated compiler coverage.
- Helped detect two confirmed bugs in the target compiler.

## Related Concepts

- [[compiler-fuzzing]]
- [[deep-q-learning]]
- [[basic-block-coverage]]
- [[cyclomatic-complexity]]

## Sources

- [[li-2022-alphaprog]]
