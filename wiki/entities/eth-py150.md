---
type: entity
title: ETH-Py150
slug: eth-py150
date: 2026-04-20
entity_type: dataset
aliases: [ETH Py150, Py150]
tags: [benchmark, python]
---

## Description

ETH-Py150 is the public Python code corpus used as the main benchmark in [[vasic-2019-neural-1904-01720]]. The paper derives synthetic bug-free and variable-misuse examples from its function-level code.

## Key Contributions

- Supplies `394K` training, `42K` validation, and `214K` test functions after preprocessing.
- Supports the paper's main comparison between the joint pointer model and enumerative repair.
- Enables controlled synthetic corruption for training and evaluation of variable-misuse localization and repair.

## Related Concepts

- [[variable-misuse-bug]]
- [[synthetic-bug-injection]]
- [[program-repair]]

## Sources

- [[vasic-2019-neural-1904-01720]]
