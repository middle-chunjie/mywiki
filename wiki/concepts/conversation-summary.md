---
type: concept
title: Conversation Summary
slug: conversation-summary
date: 2026-04-20
updated: 2026-04-20
aliases: [conversation summary, 对话摘要]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Conversation Summary** (对话摘要) — a compact representation of prior dialogue history used to provide global context for downstream memory extraction or retrieval.

## Key Points

- Mem0 uses an asynchronously refreshed summary `S` as one of the inputs to the extraction prompt.
- The summary captures the broader semantic state of the conversation, complementing a local window of recent messages.
- This design separates global context maintenance from the main extraction loop, reducing latency in online processing.
- The paper treats conversation summary as contextual support for extracting new memories, not as the memory store itself.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chhikara-nd-mem]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chhikara-nd-mem]].
