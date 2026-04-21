---
type: entity
title: EffiBench
slug: effibench
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

EffiBench is a benchmark of `1,000` efficiency-critical LeetCode problems for evaluating the efficiency of automatically generated code. It pairs each problem with an executable [[canonical-solution]] and stress-oriented test cases.

## Key Contributions

- Introduces execution-time and memory-based metrics such as `ET`, `NET`, `MU`, `NMU`, `TMU`, and `NTMU`.
- Provides manually repaired executable reference solutions collected from LeetCode discussions.
- Supports fine-grained efficiency analysis across `11` algorithm subsets and `21` LLMs in [[huang-2024-effibench-2402-02037]].

## Related Concepts

- [[benchmark-dataset]]
- [[code-efficiency]]
- [[canonical-solution]]
- [[test-case-generation]]
- [[execution-based-evaluation]]

## Sources

- [[huang-2024-effibench-2402-02037]]
