---
type: concept
title: Web Traversal
slug: web-traversal
date: 2026-04-20
updated: 2026-04-20
aliases: [web traversal, 网页遍历]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Web Traversal** (网页遍历) — the task of answering a query by navigating from a root webpage through its subpages to gather sufficient evidence.

## Key Points

- WebWalker formalizes web traversal as taking a root URL `U_root` and query `Q`, then collecting information through site navigation rather than one-shot search retrieval.
- The benchmark emphasizes depth and hop structure, with both `single_source_i` and `multi_source_i` questions requiring traversal over official websites.
- In this paper the action space is constrained to clicking sublinks, so the core challenge is planning and evidence gathering instead of general browser control.
- Performance degrades as page depth and the number of required sources increase, showing that deep traversal remains difficult for current LLM agents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-webwalker-2501-07572]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-webwalker-2501-07572]].
