---
type: entity
title: SCAN
slug: scan-benchmark
date: 2026-04-20
entity_type: tool
aliases:
  - SCAN
  - SCAN benchmark
  - Simplified versions of Commands Action Navigation
tags: [benchmark, compositional-generalization, nlp]
---

## Description

SCAN (Lake & Baroni, 2018) is a compositional generalization benchmark that requires mapping natural language commands (e.g., "jump left and walk twice") to action sequences (e.g., `TURN_LEFT JUMP WALK WALK`). Its most challenging split — the length split — tests on action sequences strictly longer than any in training.

## Key Contributions

- Became the canonical benchmark for evaluating compositional generalization in sequence-to-sequence models.
- Length split: training set contains ~80% of 20,000+ examples with shorter action sequences; test set has longer sequences — neural models without symbolic components typically fail.
- Least-to-most prompting achieves `99.7%` accuracy on the length split with `code-davinci-002` using only 14 exemplars, without training.

## Related Concepts

- [[compositional-generalization]]
- [[easy-to-hard-generalization]]
- [[sequence-to-sequence]]
- [[least-to-most-prompting]]

## Sources

- [[zhou-2023-leasttomost-2205-10625]]
