---
type: concept
title: Repo-Level Code Generation
slug: repo-level-code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [repository-level code generation, 仓库级代码生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Repo-Level Code Generation** (仓库级代码生成) — a code-generation setting in which a model must produce code inside an existing repository while using surrounding files, dependencies, and local conventions as context.

## Key Points

- EvoCodeBench evaluates repo-level generation with three settings: without context, local-file completion, and local-file infilling.
- The benchmark measures not only whether generated code passes tests, but also whether it invokes relevant in-repository dependencies via `Recall@k`.
- Repository context materially improves performance: for `gpt-4`, Pass@1 rises from `7.27` without context to `17.45` or `20.73` with local-file context.
- The paper's error analysis shows missing cross-file context is a major failure source, accounting for `20/50` manually inspected `gpt-4` errors.
- The setting is presented as closer to real software development than snippet-level code generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-evocodebench-2410-22821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-evocodebench-2410-22821]].
