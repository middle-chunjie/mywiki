---
type: entity
title: OpenChat 3.5
slug: openchat-3-5
date: 2026-04-20
entity_type: model
aliases:
  - OpenChat-3.5
tags: []
---

## Description

OpenChat 3.5 is the language model used in [[samarinas-2024-procis]] as the LLM component inside LMGR. It generates candidate Wikipedia concepts and is reused for the grounding stage of the retrieval pipeline.

## Key Contributions

- Generates up to `20` candidate title-description pairs per conversation.
- Acts as both generator and grounding model in the LMGR pipeline.

## Related Concepts

- [[language-model-grounded-retrieval]]
- [[reactive-retrieval]]
- [[proactive-retrieval]]

## Sources

- [[samarinas-2024-procis]]
