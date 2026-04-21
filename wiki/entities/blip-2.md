---
type: entity
title: BLIP-2
slug: blip-2
date: 2026-04-20
entity_type: tool
aliases: [BLIP-2, BLIP 2]
tags: []
---

## Description

BLIP-2 is the vision-language model used in [[cho-2023-visual-2305-15328]] as the VQA backbone for some VPEval modules and as a captioning/VQA baseline in the human-correlation study. The paper specifically references the Flan-T5 XL variant.

## Key Contributions

- Provides the `vqa` module for prompt elements that are not easily handled by detection or OCR alone.
- Serves as a strong baseline for both VQA-based evaluation and caption-based automatic metrics.
- Improves human correlation further in the `VPEval dagger` variant.

## Related Concepts

- [[visual-programming]]
- [[text-to-image-generation]]
- [[explainable-evaluation]]

## Sources

- [[cho-2023-visual-2305-15328]]
