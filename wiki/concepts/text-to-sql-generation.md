---
type: concept
title: Text-to-SQL Generation
slug: text-to-sql-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [text-to-sql, 文本到SQL生成]
tags: [semantic-parsing, code, databases]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Text-to-SQL Generation** (文本到SQL生成) — the task of mapping natural-language questions plus database context to executable SQL queries.

## Key Points

- [[chen-2023-teaching-2304-05128]] uses [[spider]] dev as the setting without unit tests, making correctness inference harder.
- The SELF-DEBUGGING prompt first infers the required answer shape, then explains the generated SQL using execution results and sample rows.
- Explanation-based feedback improves Codex from `81.3` to `84.1` on Spider.
- The paper finds the biggest gains on extra-hard SQL examples, where missing conditions and wrong joins are common.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-teaching-2304-05128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-teaching-2304-05128]].
