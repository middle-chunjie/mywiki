---
type: entity
title: BLIP
slug: blip
date: 2026-04-20
entity_type: tool
aliases: [Bootstrapping Language-Image Pre-training]
tags: []
---

## Description

BLIP is the pretrained vision-language retrieval backbone used in [[levy-nd-chatting]] to instantiate the dialog-conditioned image retriever `F`. The paper fine-tunes BLIP text/image encoders for multi-round retrieval and also compares against BLIP as a text-to-image baseline.

## Key Contributions

- Supplies the pretrained text and image encoders used for the ChatIR retriever.
- Serves as both the base architecture for dialog-conditioned retrieval and a strong caption-only baseline.
- Enables retrieval in a shared image-text embedding space with cosine ranking.

## Related Concepts

- [[text-to-image-retrieval]]
- [[multimodal-retrieval]]
- [[contrastive-learning]]

## Sources

- [[levy-nd-chatting]]
