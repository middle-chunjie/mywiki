---
type: concept
title: Syntax-Aware Truncation
slug: syntax-aware-truncation
date: 2026-04-20
updated: 2026-04-20
aliases: [syntax-aware truncation, 语法感知截断]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Syntax-Aware Truncation** (语法感知截断) — a post-processing method that trims model-generated code until the inserted span forms a syntactically valid program fragment in the surrounding abstract syntax tree.

## Key Points

- The paper uses syntax-aware truncation to standardize completion boundaries for both FIM-trained and non-FIM-trained code LLMs.
- For algorithmic block completion, it repeatedly removes trailing lines until the generated span can be parsed as a valid block and the remaining code matches the original AST structure.
- For control-flow and API-call completion, it incrementally extends the generated text until it becomes a valid expression node in context.
- The method sharply improves `Pass@1` and reduces compilation errors for open-ended models such as CodeGen and CodeLLaMa without adding inference-time GPU cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2024-evaluation-2403-04814]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2024-evaluation-2403-04814]].
