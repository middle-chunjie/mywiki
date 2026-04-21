---
type: concept
title: Contrastive Loss
slug: contrastive-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [对比损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Loss** (对比损失) — a metric-learning objective that pulls positive pairs closer and pushes negative pairs farther apart in embedding space.

## Key Points

- The forum link prediction task is fine-tuned with contrastive loss over linked and unlinked post pairs.
- The paper uses the resulting embedding separation, not just classification accuracy, as the main indicator of representation quality.
- Fine-tuned BERTOverflow dramatically increases the distance gap between linked and unlinked posts compared with its untuned version.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
