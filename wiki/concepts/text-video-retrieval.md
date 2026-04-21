---
type: concept
title: Text-Video Retrieval
slug: text-video-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [video-text retrieval, 文本视频检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Text-Video Retrieval** (文本视频检索) — the task of ranking videos for a text query, or texts for a video, by measuring cross-modal semantic similarity.

## Key Points

- The paper studies text-to-video and video-to-text retrieval across five benchmarks: MSR-VTT, MSVD, VATEX, ActivityNet, and DiDeMo.
- It frames the task as matching a text embedding against frame-level video embeddings aggregated into a query-aware video representation.
- The proposed method keeps video and text encoding independent, then computes an `n \times n` similarity matrix for batch retrieval training.
- Standard evaluation uses recall at rank `K` and mean rank, emphasizing top-ranked cross-modal retrieval quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]
- [[zolfaghari-2021-crossclr-2109-14910]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
