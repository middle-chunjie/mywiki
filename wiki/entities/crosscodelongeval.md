---
type: entity
title: CrossCodeLongEval
slug: crosscodelongeval
date: 2026-04-20
entity_type: tool
aliases: [CrossCodeLongEval]
tags: []
---

## Description

CrossCodeLongEval is the long-form repository-level code completion benchmark introduced in [[wu-2024-repoformer-2403-10059]]. It expands evaluation beyond RepoEval by constructing chunk and function completion tasks from a large pool of Python repositories related to CrossCodeEval.

## Key Contributions

- Adds large-scale chunk and function completion evaluation with `5000` instances for each task.
- Tests REPOFORMER on broader repository coverage than RepoEval, including `944` repositories for chunk completion and `1460` for function completion.
- Shows that selective retrieval remains beneficial for longer generations and larger repositories.

## Related Concepts

- [[repository-level-code-completion]]
- [[selective-retrieval]]
- [[cross-file-context]]

## Sources

- [[wu-2024-repoformer-2403-10059]]
