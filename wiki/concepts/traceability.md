---
type: concept
title: Traceability
slug: traceability
date: 2026-04-20
updated: 2026-04-20
aliases: [evidence traceability, content traceability, 可追溯性]
tags: [citations, reasoning, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Traceability** (可追溯性) — the property that each important fragment of a generated answer can be mapped back to explicit supporting evidence and intermediate reasoning steps.

## Key Points

- SearChain records the retrieved supporting document for each node on the correct reasoning path and uses those node-document pairs when generating final content.
- The system asks the LLM to produce answers with references attached to the corresponding reasoning step, rather than only appending a loose document list.
- The paper frames this as a way to make generated knowledge more credible and auditable for users in knowledge-intensive tasks.
- Against New Bing, SearChain improves Scope of Knowledge Coverage from `1.143` to `2.882` and Accuracy of Marking Position from `0.45` to `0.80`.
- The authors emphasize that this citation-style tracing is achieved without supervised citation annotations or additional LLM training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-searchinthechain-2304-14732]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-searchinthechain-2304-14732]].
