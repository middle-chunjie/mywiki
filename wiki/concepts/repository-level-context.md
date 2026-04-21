---
type: concept
title: Repository-Level Context
slug: repository-level-context
date: 2026-04-20
updated: 2026-04-20
aliases: [repository-level context, repo-level context, 仓库级上下文]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Repository-Level Context** (仓库级上下文) - code evidence drawn from other files, classes, or structural relations in the same repository to condition a target prediction in the current file.

## Key Points

- The paper argues that useful completion evidence often lies outside the current file, such as imports, parent classes, sibling files, and similarly named files.
- RLPG treats repository structure itself as a retrieval prior, rather than relying only on the left context before the cursor.
- Repository-level context is converted into discrete prompt proposals so it can be inserted into a black-box code model without fine-tuning.
- Even when the resulting prompt is a loosely ordered mash-up of snippets, the paper shows that relevant repository-level context can still improve completion accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shrivastava-2023-repositorylevel-2206-12839]]
- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shrivastava-2023-repositorylevel-2206-12839]].
