---
type: entity
title: CodeT5
slug: codet5
date: 2026-04-20
entity_type: model
aliases: [CodeT5 model]
tags: []
---

## Description

CodeT5 is an encoder-decoder code model used as a major backbone in [[cui-2022-codeexp-2211-15395]]. In the paper it is the strongest fine-tuned model family for code explanation generation on most reported metrics.

## Key Contributions

- Achieves the best scores on `6/7` automatic metrics under the raw-then-refined training strategy.
- Reaches near-human overall quality when fine-tuned on CodeExp(refined), with human-evaluation overall score `3.446`.
- Demonstrates that an encoder-decoder architecture is highly competitive for long-form docstring generation.

## Related Concepts

- [[sequence-to-sequence]]
- [[fine-tuning]]
- [[code-explanation-generation]]

## Sources

- [[cui-2022-codeexp-2211-15395]]
