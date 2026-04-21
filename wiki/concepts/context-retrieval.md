---
type: concept
title: Context Retrieval
slug: context-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieved context, 上下文检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Retrieval** (上下文检索) — the process of selecting external supporting context that is likely to help a model make a correct prediction for the current input.

## Key Points

- The paper retrieves cross-file repository snippets by chunking source code into non-overlapping `10`-line segments and ranking them against the last `10` prompt lines.
- The main baseline uses `BM25` and prepends the top-`5` retrieved chunks, truncated to `512` BPE tokens, before the in-file context.
- The authors also study denser retrievers such as UniXCoder and OpenAI ada embeddings, showing that retrieval quality materially changes completion accuracy.
- An oracle-like retrieval w/ ref. condition quantifies the ceiling of better context retrieval while remaining unrealistic for actual deployment.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-nd-crosscodeeval]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-nd-crosscodeeval]].
