---
type: concept
title: Prompting with Test Cases
slug: prompting-with-test-cases
date: 2026-04-20
updated: 2026-04-20
aliases: [test-augmented prompting, 带测试用例的提示]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Prompting with Test Cases** (带测试用例的提示) — a prompting strategy that embeds example unit tests or expected behaviors in the prompt so the model can condition on executable task constraints.

## Key Points

- [[wang-2023-executionbased-2212-10481]] builds prompts by concatenating function context with a docstring that may optionally contain test cases.
- The baseline setting is zero-shot with no tests in the prompt, but the paper ablates adding one random test and all annotated tests.
- Injecting one exemplar test case significantly improves execution accuracy for Codex on the English set.
- Adding more than one test case yields little extra benefit, suggesting one test often suffices to communicate the main functionality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-executionbased-2212-10481]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-executionbased-2212-10481]].
