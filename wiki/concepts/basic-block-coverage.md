---
type: concept
title: Basic Block Coverage
slug: basic-block-coverage
date: 2026-04-20
updated: 2026-04-20
aliases: [基本块覆盖]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Basic Block Coverage** (基本块覆盖) — a testing metric that measures how many distinct straight-line code regions with single entry and exit are executed by a test suite.

## Key Points

- The paper uses accumulated unique compiler basic blocks as its approximation to fuzzing coverage.
- For a generated program `p`, the trace-level quantity `B(T_p)` counts unique basic blocks seen in its execution trace.
- Reward `R_2` normalizes current-trace coverage against the generated suite so far to encourage diverse tests rather than repeated valid ones.
- Coverage is collected from compiler execution traces obtained through dynamic instrumentation with Pin.
- Under the strongest reward, ALPHAPROG exceeds `100,000` tested basic blocks with `30,000` generated programs and substantially outperforms AFL's `43,135`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-alphaprog]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-alphaprog]].
