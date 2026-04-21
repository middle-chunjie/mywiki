---
type: concept
title: Repository-Level Code Completion
slug: repository-level-code-completion
date: 2026-04-20
updated: 2026-04-20
aliases: [repo-level code completion, 仓库级代码补全]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Repository-Level Code Completion** (仓库级代码补全) — code completion that conditions on relevant context from multiple files in the same repository rather than relying only on the active file.

## Key Points

- The paper frames repository-level completion as the realistic setting for modern development because target APIs and helper classes are often defined outside the current file.
- CROSSCODEEVAL enforces repository-level reasoning by selecting completions whose identifiers become unresolved when intra-project imports are replaced with dummy definitions.
- The benchmark shows that strong code LMs underperform sharply with only in-file context and improve once retrieved cross-file context is prepended.
- The paper treats repository-level completion as jointly a generation problem and a retrieval problem because useful context must be found before it can be used.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-nd-crosscodeeval]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-nd-crosscodeeval]].
