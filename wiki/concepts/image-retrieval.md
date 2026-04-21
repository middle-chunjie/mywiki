---
type: concept
title: Image Retrieval
slug: image-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [image search, 图像检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Image Retrieval** (图像检索) — the task of ranking images in a corpus according to their relevance to a query, which may be another image, text, or an interactive sequence of feedback.

## Key Points

- The paper reframes image retrieval as a multi-turn process in which a caption is incrementally refined through question-answer interaction.
- Retrieval is performed in a shared embedding space by comparing the dialog representation against precomputed image embeddings.
- ChatIR evaluates retrieval on a `50K`-image corpus and reports both Hit@10 and Average Target Rank as interaction proceeds.
- The work shows that dialog can raise retrieval success from roughly `64%` caption-only to `78.3%` after `5` rounds.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[levy-nd-chatting]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[levy-nd-chatting]].
