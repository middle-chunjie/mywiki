---
type: entity
title: CodeSage
slug: codesage
date: 2026-04-20
entity_type: tool
aliases: [CodeSage]
tags: []
---

## Description

CodeSage is the family of encoder-only source-code representation models introduced in [[unknown-nd-code-2402-01935]]. It comprises 130M, 356M, and 1.3B parameter variants trained on large-scale code corpora and text-code pairs.

## Key Contributions

- Establishes a strong off-the-shelf code embedding baseline across semantic search and classification transfer tasks.
- Combines identifier-aware denoising with bimodal contrastive learning instead of relying on standard MLM alone.
- Shows that scaling encoder pretraining for code benefits from hard positive and hard negative design, not just more data.

## Related Concepts

- [[code-representation-learning]]
- [[masked-language-modeling]]
- [[multimodal-contrastive-learning]]

## Sources

- [[unknown-nd-code-2402-01935]]
