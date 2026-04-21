---
type: concept
title: Tree-of-Thought
slug: tree-of-thought
date: 2026-04-20
updated: 2026-04-20
aliases: [ToT, tree of thought]
tags: [llm, reasoning, search]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tree-of-Thought** (思维树) — a reasoning paradigm that expands and evaluates multiple intermediate reasoning branches in a tree before selecting a solution path.

## Key Points

- The paper treats ToT as a strong structured-prompting baseline that asks the LLM to both generate one-step thought candidates and evaluate them.
- ToT is evaluated with retained branch counts `b = 1` and `b = 3`, plus task-specific step limits such as `9` for 8-Puzzle and `4` for Pocket Cube.
- The method provides flexible search over reasoning branches, but its repeated LLM evaluation makes it much less efficient than XoT.
- In all three core tasks, ToT remains far below XoT in accuracy despite using tens of LLM calls per example.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-everything-2311-04254]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-everything-2311-04254]].
