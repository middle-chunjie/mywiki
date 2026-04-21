---
type: entity
title: CCFINDER
slug: ccfinder
date: 2026-04-20
entity_type: tool
aliases: [Cross-file Context Finder]
tags: [tool, retrieval]
---

## Description

CCFINDER is the static-analysis retrieval tool introduced in [[ding-2023-cocomic-2212-10007]] to build project context graphs and fetch relevant cross-file entities for code completion.

## Key Contributions

- Builds a multi-relational project context graph over files, classes, functions, and globals.
- Retrieves `k`-hop neighbors of imported entities to supply repository-local context to CoCoMIC.
- Improves identifier recall from `75.19%` to `95.55%` when cross-file context is added.

## Related Concepts

- [[cross-file-context]]
- [[project-context-graph]]
- [[static-analysis]]

## Sources

- [[ding-2023-cocomic-2212-10007]]
