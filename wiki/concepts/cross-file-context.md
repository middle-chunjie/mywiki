---
type: concept
title: Cross-file Context
slug: cross-file-context
date: 2026-04-20
updated: 2026-04-20
aliases: [Cross-repository Context, 跨文件上下文]
tags: [code-completion, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-file Context** (跨文件上下文) — code information from other files in the same project that is relevant to a target prediction, such as imported classes, functions, globals, and their signatures.

## Key Points

- The paper treats cross-file context as project-local information that standard in-file code LMs cannot infer reliably from the current file alone.
- CoCoMIC retrieves cross-file context by locating imported entities in a project context graph and collecting their `k`-hop neighbors.
- Retrieved entities are reordered to preserve natural source-code order within each file before being passed to the model.
- Cross-file context improves identifier recall on the benchmark from `75.19%` to `95.55%`.
- Plainly prepending cross-file text helps only marginally because repository context is too long for the backbone model's input window.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2023-cocomic-2212-10007]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2023-cocomic-2212-10007]].
