---
type: entity
title: SAFIM
slug: safim
date: 2026-04-20
entity_type: tool
aliases: [SAFIM, Syntax-Aware Fill-in-the-Middle]
tags: []
---

## Description

SAFIM is the benchmark introduced in [[gong-2024-evaluation-2403-04814]] for evaluating code LLMs on syntax-aware fill-in-the-middle tasks across multiple programming languages and completion types.

## Key Contributions

- Provides `17,720` examples covering algorithmic block, control-flow, and API function call completion.
- Combines execution-based evaluation with syntax matching and syntax-aware truncation to support fairer model comparisons.
- Reduces contamination risk by sourcing code written after April 2022.

## Related Concepts

- [[benchmark-dataset]]
- [[fill-in-the-middle]]
- [[execution-based-evaluation]]
- [[syntax-aware-truncation]]

## Sources

- [[gong-2024-evaluation-2403-04814]]
