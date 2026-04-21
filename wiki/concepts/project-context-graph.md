---
type: concept
title: Project Context Graph
slug: project-context-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [Repository Context Graph, 项目上下文图]
tags: [graph, retrieval, software-engineering]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Project Context Graph** (项目上下文图) — a multi-relational directed graph over project entities and their dependency or scope relations, used to retrieve repository context relevant to a target coding task.

## Key Points

- CCFINDER builds the graph top-down from a root project node to file nodes and then to classes, functions, and global variables.
- Edges capture both intra-file hierarchy and inter-file import dependencies.
- Retrieval starts from nodes referenced by local import statements and expands to nearby nodes by graph search.
- The graph is designed for module dependency structure rather than data-flow or control-flow analysis.
- The paper uses the graph as the retrieval backbone for cross-file code completion rather than as a standalone program-analysis artifact.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2023-cocomic-2212-10007]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2023-cocomic-2212-10007]].
