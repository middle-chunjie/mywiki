---
type: entity
title: CodeBERT
slug: codebert
date: 2026-04-20
entity_type: tool
aliases: [CodeBERT model]
tags: [model, baseline]
---

## Description

CodeBERT is a pre-trained encoder model for programming and natural languages used as a major baseline in [[ahmad-2021-unified]]. The paper contrasts PLBART against CodeBERT to show the value of pretraining both encoder and decoder.

## Key Contributions

- Serves as a strong encoder-only comparison point across generation and classification tasks.
- Highlights the limitation of requiring a randomly initialized decoder for generative fine-tuning.

## Related Concepts

- [[sequence-to-sequence]]
- [[code-generation]]
- [[code-translation]]

## Sources

- [[ahmad-2021-unified]]
