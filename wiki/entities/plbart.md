---
type: entity
title: PLBART
slug: plbart
date: 2026-04-20
entity_type: tool
aliases: [Program and Language BART]
tags: [model, pretraining]
---

## Description

PLBART is the pre-trained sequence-to-sequence model introduced in [[ahmad-2021-unified]] for unified program understanding and generation. It adapts the BART recipe to code and developer natural language.

## Key Contributions

- Jointly pretrains over Java, Python, and English developer text with denoising autoencoding.
- Supports summarization, text-to-code generation, code translation, program repair, clone detection, and vulnerability detection with one backbone.
- Improves strongly over prior baselines on translation and CodeBLEU-oriented generation quality.

## Related Concepts

- [[transformer]]
- [[sequence-to-sequence]]
- [[denoising-autoencoding]]
- [[multilingual-pretraining]]

## Sources

- [[ahmad-2021-unified]]
