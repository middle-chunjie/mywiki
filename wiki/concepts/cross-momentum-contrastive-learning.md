---
type: concept
title: Cross Momentum Contrastive Learning
slug: cross-momentum-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [xMoCo]
tags: [dense-retrieval, contrastive-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross Momentum Contrastive Learning** — a momentum-based contrastive training framework for asymmetric matching tasks that maintains separate fast/slow encoders and queues for both sides, enabling large-scale negative reuse without tying the encoders.

## Key Points

- xMoCo extends MoCo to question-passage retrieval by introducing separate fast and slow encoders for questions and passages.
- The training loss combines question-to-passage and passage-to-question objectives with `λ = 0.5`, so both sides receive learning signal indirectly through the slow encoders.
- Two queues store historical question and passage vectors, letting the model train against many more negatives than ordinary in-batch setups.
- The method is designed for dense retrieval settings where the two inputs are not interchangeable and may benefit from untied parameters.
- In [[yang-2021-xmoco]], xMoCo improves retrieval over DPR across most open-domain QA benchmarks while retaining efficient inference with pre-computed passage vectors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2021-xmoco]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2021-xmoco]].
