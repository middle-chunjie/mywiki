---
type: entity
title: Dromedary
slug: dromedary
date: 2026-04-20
entity_type: tool
aliases: [Dromedary-65B, Dromedary final, Dromedary non-verbose]
tags: []
---

## Description

Dromedary is the AI assistant produced by applying SELF-ALIGN to [[llama-65b]] in [[sun-2023-principledriven-2305-03047]]. The paper studies both a principle-engraved non-verbose version and a final verbose-cloned version.

## Key Contributions

- Demonstrates that fewer than `300` lines of human supervision can produce a competitive aligned assistant.
- Reaches reported gains over LLaMA and Alpaca on TruthfulQA and HHH Eval while exposing a verbose-quality tradeoff.
- Serves as the paper's concrete testbed for principle-driven alignment from scratch.

## Related Concepts

- [[self-align]]
- [[context-distillation]]
- [[instruction-following]]

## Sources

- [[sun-2023-principledriven-2305-03047]]
