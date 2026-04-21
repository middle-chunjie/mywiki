---
type: concept
title: Code Explanation
slug: code-explanation
date: 2026-04-20
updated: 2026-04-20
aliases: [program explanation, 代码解释]
tags: [code, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Explanation** (代码解释) — a natural-language account of what a generated program does, used as an intermediate representation for evaluation or revision.

## Key Points

- [[chen-2023-teaching-2304-05128]] asks the model to explain SQL clauses or Python code line by line before deciding whether to revise.
- On Spider, explanation is paired with execution results such as the returned table shape and sample rows.
- Code explanation improves Codex on Spider from `81.3` to `84.1`, whereas simple feedback alone does not help.
- The paper argues explanation helps the model surface mismatches between the specification and the generated code.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-teaching-2304-05128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-teaching-2304-05128]].
