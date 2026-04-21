---
type: entity
title: BFC
slug: bfc
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

BFC is the industrial-grade Brainfuck compiler used as the target environment in [[li-2022-alphaprog]]. Its compilation messages and execution behavior provide the reward signals used to train and evaluate ALPHAPROG.

## Key Contributions

- Serves as the target compiler whose validity outcomes and traces define the reinforcement-learning environment.
- Is the system on which ALPHAPROG discovers two confirmed bugs.

## Related Concepts

- [[compiler-fuzzing]]
- [[basic-block-coverage]]
- [[cyclomatic-complexity]]

## Sources

- [[li-2022-alphaprog]]
