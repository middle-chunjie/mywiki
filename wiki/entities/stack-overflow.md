---
type: entity
title: Stack Overflow
slug: stack-overflow
date: 2026-04-20
entity_type: tool
aliases: [StackOverflow]
tags: []
---

## Description

Stack Overflow is the developer question-answering platform used as the raw source corpus for [[li-2024-infibench-2404-07940]]. The paper mines its question and answer posts to build a benchmark that better reflects real coding support scenarios.

## Key Contributions

- Supplies the large raw pool from which InfiBench filters `23.54M` questions and `34.68M` answers.
- Provides highly voted and accepted answers that domain experts use as references when defining evaluation criteria.
- Anchors the benchmark in natural developer workflows rather than synthetic problem design.

## Related Concepts

- [[question-answering]]
- [[benchmark-evaluation]]
- [[data-contamination]]

## Sources

- [[li-2024-infibench-2404-07940]]
