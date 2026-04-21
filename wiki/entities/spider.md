---
type: entity
title: Spider
slug: spider
date: 2026-04-20
entity_type: benchmark
aliases:
  - Spider
tags: [benchmark, sql]
---

## Description

Spider is the text-to-SQL benchmark used in [[chen-2023-teaching-2304-05128]] to evaluate self-debugging when no unit tests are available.

## Key Contributions

- Supplies the development-set evaluation where explanation-based SELF-DEBUGGING raises Codex from `81.3` to `84.1`.
- Exposes extra-hard SQL cases where the paper reports a `9%` gain.

## Related Concepts

- [[text-to-sql-generation]]
- [[code-explanation]]
- [[self-debugging]]

## Sources

- [[chen-2023-teaching-2304-05128]]
