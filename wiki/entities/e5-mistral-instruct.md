---
type: entity
title: E5-Mistral-Instruct
slug: e5-mistral-instruct
date: 2026-04-20
entity_type: model
aliases: [E5 Mistral Instruct]
tags: []
---

## Description

E5-Mistral-Instruct is a pre-trained embedding model used in [[unknown-nd-rare-2410-20088]] as a strong retriever checkpoint for continued fine-tuning. The paper reports that this backbone benefits substantially from RARe, especially on out-of-domain and reasoning-heavy retrieval benchmarks.

## Key Contributions

- Serves as the strongest retriever-checkpoint setting for RARe in the paper.
- Improves from `56.96` to `58.28` on BeIR All and from `24.12` to `25.79` on RAR-b after RARe fine-tuning.

## Related Concepts

- [[dense-retrieval]]
- [[contrastive-learning]]
- [[domain-generalization]]

## Sources

- [[unknown-nd-rare-2410-20088]]
