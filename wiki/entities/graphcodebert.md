---
type: entity
title: GraphCodeBERT
slug: graphcodebert
date: 2026-04-20
entity_type: tool
aliases: [Graph CodeBERT]
tags: [model, baseline]
---

## Description

GraphCodeBERT is a structure-aware pretrained code model used as a baseline in [[ahmad-2021-unified]]. It extends CodeBERT with data-flow information and remains competitive on some understanding-heavy tasks.

## Key Contributions

- Provides a strong baseline that explicitly models data-flow edges between code tokens.
- Serves as a point of comparison showing where PLBART's denoising seq2seq pretraining is competitive and where structured inductive bias still matters.

## Related Concepts

- [[code-translation]]
- [[code-clone-detection]]
- [[vulnerability-detection]]

## Sources

- [[ahmad-2021-unified]]
