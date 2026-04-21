---
type: concept
title: External Context Utilization
slug: external-context-utilization
date: 2026-04-20
updated: 2026-04-20
aliases: [context utilization, 外部上下文利用]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**External Context Utilization** (外部上下文利用) — the degree to which a model's next-token behavior changes in response to retrieved evidence rather than remaining driven by its internal parameters alone.

## Key Points

- Lumina measures external-context utilization by comparing next-token distributions under relevant documents versus random documents.
- The paper instantiates this comparison with maximum mean discrepancy over token embeddings rather than attention-head heuristics.
- A larger score means the model is more sensitive to semantic changes in the retrieved context when generating the current token.
- The paper validates the measure statistically by showing retrieved-document generations score higher than no-document generations and summarization scores higher than QA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yeh-2026-lumina-2509-21875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yeh-2026-lumina-2509-21875]].
