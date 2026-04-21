---
type: concept
title: Line Coverage
slug: line-coverage
date: 2026-04-20
updated: 2026-04-20
aliases: [statement coverage, 行覆盖率]
tags: [software-testing, evaluation]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Line Coverage** (行覆盖率) — a testing metric that measures which executable source lines are run by a test suite.

## Key Points

- This paper uses line coverage rather than path coverage because path coverage is intractable on repository-scale systems.
- Coverage is restricted to executable lines changed by the golden patch, making the metric issue-targeted rather than global.
- The proposed `Delta C` score compares coverage before and after adding generated tests on both pre-patch and post-patch program states.
- Successful generated tests tend to have much higher change coverage than unsuccessful ones, making coverage a useful complementary signal to binary success.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[m-ndler-2024-code-2406-12952]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[m-ndler-2024-code-2406-12952]].
