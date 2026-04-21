---
type: concept
title: Vision-Language Model
slug: vision-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [VLM, vision language model, 视觉语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Vision-Language Model** (视觉语言模型) — a model that jointly processes visual and textual information for tasks such as retrieval, question answering, captioning, or generation.

## Key Points

- ChatIR is explicitly built on foundation vision-language models, using BLIP/BLIP2 components for retrieval and answer generation.
- The image retriever uses BLIP encoders to map dialogs into a visual embedding space shared with image features.
- BLIP2 serves as an off-the-shelf answer provider for scalable comparison of different question generators.
- The paper positions modern VLMs as making chat-based image retrieval practical without task-specific end-to-end dialog training from scratch.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[levy-2023-chatting-2305-20062]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[levy-2023-chatting-2305-20062]].
