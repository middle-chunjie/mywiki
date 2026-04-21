---
type: entity
title: VisDial
slug: visdial
date: 2026-04-20
entity_type: dataset
aliases: [Visual Dialog]
tags: []
---

## Description

VisDial is the human-annotated visual dialog dataset repurposed in [[levy-nd-chatting]] as the main supervision and evaluation substrate for chat-based image retrieval. Each dialog is paired with its underlying image, which becomes the retrieval target.

## Key Contributions

- Provides the dialog-image pairs used to train the retriever `F` on partial dialogs of different lengths.
- Supplies human-written questions for comparison against LLM and RL question generators.
- Enables evaluation on a `50K` unseen-image retrieval setting and supports the human-vs-BLIP2 answerer analysis.

## Related Concepts

- [[visual-dialog]]
- [[image-retrieval]]
- [[interactive-retrieval]]

## Sources

- [[levy-nd-chatting]]
