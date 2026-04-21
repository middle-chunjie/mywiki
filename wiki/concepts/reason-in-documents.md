---
type: concept
title: Reason-in-Documents
slug: reason-in-documents
date: 2026-04-20
updated: 2026-04-20
aliases: [document reasoning, 文档内推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reason-in-Documents** (文档内推理) — a separate reasoning stage that analyzes retrieved documents against the current reasoning state and compresses them into concise knowledge for reinjection into the main chain of thought.

## Key Points

- Search-o1 introduces this module because raw retrieved pages are long, redundant, and can disrupt the coherence of the main reasoning trace.
- For each search step, the module first generates an intermediate analysis over the retrieved documents and then produces a refined knowledge step.
- The refined output is inserted back into the main reasoning chain instead of the full retrieved documents.
- The paper attributes Search-o1's gains over plain agentic RAG mainly to this module, especially on multi-hop QA and GPQA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-searcho-2501-05366]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-searcho-2501-05366]].
