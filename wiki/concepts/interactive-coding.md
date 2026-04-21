---
type: concept
title: Interactive Coding
slug: interactive-coding
date: 2026-04-20
updated: 2026-04-20
aliases: [interactive code generation, 交互式编程]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Interactive Coding** (交互式编程) — a code-generation setting where an agent iteratively issues executable code, observes runtime feedback, and refines its actions before submitting a final solution.

## Key Points

- [[yang-2023-intercode-2306-14898]] argues that programming is fundamentally a write-execute-test loop rather than a single sequence-transduction step.
- The paper formalizes interactive coding as repeated agent-environment interaction with code as actions and execution output as observations.
- The benchmark instantiates interactive coding for Bash, SQL, and Python, showing that the same abstraction can support multiple languages.
- Interaction is valuable not only for debugging but also for context discovery, modular problem decomposition, and error recovery.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-intercode-2306-14898]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-intercode-2306-14898]].
