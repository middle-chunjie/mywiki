---
type: entity
title: CodeGeeX2-6B
slug: codegeex2-6b
date: 2026-04-20
entity_type: tool
aliases: [CodeGeeX-6B, CodeGeeX2 6B]
tags: []
---

## Description

CodeGeeX2-6B is the base model fine-tuned and further optimized with RL in [[wu-2024-daco-2403-02528]]. The paper uses it as the main open `6B` model for studying data analysis via code generation.

## Key Contributions

- Serves as the base model for both the SFT system and the Daco-RL system.
- Demonstrates that a smaller open model can learn reasonable data-analysis behavior from GPT-4-generated supervision.
- Provides the experimental platform for testing dense reward shaping over intermediate code steps.

## Related Concepts

- [[code-generation]]
- [[supervised-fine-tuning]]
- [[reinforcement-learning-from-human-feedback]]

## Sources

- [[wu-2024-daco-2403-02528]]
