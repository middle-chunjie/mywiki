---
type: concept
title: Issue Reproduction
slug: issue-reproduction
date: 2026-04-20
updated: 2026-04-20
aliases: [bug reproduction, issue reproducing test, 问题复现]
tags: [software-testing, bug-fixing]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Issue Reproduction** (问题复现) — the process of constructing an artifact, often a test case, that reliably triggers a reported bug under the pre-fix program state.

## Key Points

- SWT-Bench defines successful issue reproduction as generating at least one `F -> P` test and no post-patch failures.
- The paper treats GitHub issue text as the natural-language input from which reproduction tests must be synthesized.
- Reproduction is stricter than merely touching relevant code, because `P -> P` tests are valid but unhelpful and `F -> x` tests are only partial evidence.
- The authors show that issue reproduction and code repair overlap only weakly on a per-instance basis.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[m-ndler-2024-code-2406-12952]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[m-ndler-2024-code-2406-12952]].
