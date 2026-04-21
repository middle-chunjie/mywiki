---
type: entity
title: MBPP
slug: mbpp
date: 2026-04-20
entity_type: benchmark
aliases:
  - MBPP
  - Mostly Basic Programming Problems
tags: [benchmark, python]
---

## Description

MBPP is the text-to-Python benchmark used in [[chen-2023-teaching-2304-05128]] where each problem has three unit tests but only one is exposed in the prompt.

## Key Contributions

- Provides the partial-test setting that forces the model to reason beyond the visible assertion.
- Demonstrates substantial gains from self-debugging, including Codex `61.4` to `70.8` and GPT-4 `72.8` to `80.6`.

## Related Concepts

- [[text-to-python-generation]]
- [[unit-test-feedback]]
- [[execution-trace-feedback]]

## Sources

- [[chen-2023-teaching-2304-05128]]
