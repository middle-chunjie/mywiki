---
type: entity
title: EasyOCR
slug: easyocr
date: 2026-04-20
entity_type: tool
aliases: [EasyOCR]
tags: []
---

## Description

EasyOCR is the OCR module used in [[cho-2023-visual-2305-15328]] for VPEval's text-rendering checks. It extracts text strings and their boxes so the evaluator can verify whether a prompt-specified word appears in the image.

## Key Contributions

- Implements the `ocr` primitive inside VPEval.
- Supports text-rendering evaluation with both textual outputs and visual evidence.

## Related Concepts

- [[visual-programming]]
- [[text-to-image-generation]]
- [[explainable-evaluation]]

## Sources

- [[cho-2023-visual-2305-15328]]
