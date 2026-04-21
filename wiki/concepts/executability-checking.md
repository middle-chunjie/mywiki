---
type: concept
title: Executability Checking
slug: executability-checking
date: 2026-04-20
updated: 2026-04-20
aliases: [executability filtering, 可执行性检查]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Executability Checking** (可执行性检查) — testing whether a generated program can run or be parsed successfully under a specified environment before or during evaluation or selection.

## Key Points

- [[shi-2022-natural-2204-11454]] studies executability as a weaker signal than full execution-result agreement, but still useful for filtering poor candidates.
- On MBPP and Spider, the paper executes candidates under `128GB` and `10s` limits and treats timeout or runtime failure as inexecutable behavior.
- For NL2Bash, the paper approximates executability with bashlex parsing because real command execution is unsafe or impractical.
- Ablations show that restricting baseline rerankers to executable candidates can materially improve results, especially on Spider.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-natural-2204-11454]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-natural-2204-11454]].
