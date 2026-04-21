---
type: concept
title: Text-to-Image Retrieval
slug: text-to-image-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [TTI retrieval, text image retrieval, 文本到图像检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Text-to-Image Retrieval** (文本到图像检索) — the retrieval setting in which a text query is used to rank candidate images by semantic relevance.

## Key Points

- ChatIR treats caption-only retrieval as the `0`-round special case `D_0 = (C)` of its broader dialog retrieval formulation.
- The paper compares its conversational setup directly against CLIP and BLIP text-to-image retrieval baselines on COCO and Flickr30K.
- On VisDial round 0, a BLIP caption-only baseline (`63.66%`) and the dialog-trained retriever (`63.61%`) are nearly identical, isolating the benefit of later dialog turns.
- The work argues that one-shot text queries often fail to capture a user's full visual intent, motivating iterative clarification.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[levy-nd-chatting]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[levy-nd-chatting]].
