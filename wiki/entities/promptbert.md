---
type: entity
title: PromptBERT
slug: promptbert
date: 2026-04-20
entity_type: tool
aliases: [PromptBERT]
tags: []
---

## Description

PromptBERT is a sentence embedding baseline discussed in [[jiang-2022-improved-2203-06875]]. The paper contrasts it with PromCSE because PromptBERT uses prompting but still fine-tunes the whole PLM and depends on manually designed discrete prompts.

## Key Contributions

- Serves as a strong prompt-based baseline on standard STS with average Spearman `78.54`.
- Helps isolate the benefit of prompt-only adaptation versus prompting plus full PLM fine-tuning.

## Related Concepts

- [[sentence-embedding]]
- [[contrastive-learning]]
- [[semantic-textual-similarity]]

## Sources

- [[jiang-2022-improved-2203-06875]]
