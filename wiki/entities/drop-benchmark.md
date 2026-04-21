---
type: entity
title: DROP
slug: drop-benchmark
date: 2026-04-20
entity_type: tool
aliases:
  - DROP
  - DROP benchmark
  - Discrete Reasoning Over Paragraphs
tags: [benchmark, reading-comprehension, reasoning, nlp]
---

## Description

DROP (Dua et al., 2019) is a reading comprehension benchmark requiring discrete reasoning — including arithmetic, counting, and sorting — over paragraphs. Problems require the model to identify relevant numbers or spans and apply multi-step reasoning.

## Key Contributions

- Focuses on numerical and discrete reasoning over text, making it complementary to GSM8K's math word problems.
- Least-to-most prompting outperforms chain-of-thought on the numerical subset: `82.45%` vs. `74.77%` (non-football) and `73.42%` vs. `59.56%` (football) using `code-davinci-002`.
- The large L2M improvement is attributed to the fact that most DROP numerical problems admit a natural trivial decomposition.

## Related Concepts

- [[mathematical-reasoning]]
- [[least-to-most-prompting]]
- [[chain-of-thought-prompting]]

## Sources

- [[zhou-2023-leasttomost-2205-10625]]
