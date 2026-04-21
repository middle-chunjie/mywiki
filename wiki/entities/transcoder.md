---
type: entity
title: TransCoder
slug: transcoder
date: 2026-04-20
entity_type: benchmark
aliases:
  - TransCoder
tags: [benchmark, code-translation]
---

## Description

TransCoder is the code-translation benchmark used in [[chen-2023-teaching-2304-05128]] for `C++`-to-Python translation with executable unit tests.

## Key Contributions

- Provides `560` test problems with `10` unit tests per problem for measuring iterative debugging.
- Shows the largest Codex gain in the paper, from `80.4` to `92.5` with unit-test plus explanation feedback.

## Related Concepts

- [[code-translation]]
- [[unit-test-feedback]]
- [[execution-based-evaluation]]

## Sources

- [[chen-2023-teaching-2304-05128]]
