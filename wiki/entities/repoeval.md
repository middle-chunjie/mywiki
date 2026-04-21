---
type: entity
title: RepoEval
slug: repoeval
date: 2026-04-20
entity_type: tool
aliases: [RepoEval]
tags: []
---

## Description

RepoEval is the repository-level code completion benchmark used in [[wu-2024-repoformer-2403-10059]] for line, API, and function completion. It is built from `32` Python repositories and emphasizes cross-file dependency resolution.

## Key Contributions

- Serves as the main benchmark showing that retrieval helps only a minority of completion instances.
- Provides the core evaluation setting where REPOFORMER improves both edit similarity and unit-test pass rate over always-retrieve baselines.
- Supplies the latency testbed used for the paper's online-serving speedup analysis.

## Related Concepts

- [[repository-level-code-completion]]
- [[cross-file-context]]
- [[selective-retrieval]]

## Sources

- [[wu-2024-repoformer-2403-10059]]
