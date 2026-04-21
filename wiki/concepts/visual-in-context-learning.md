---
type: concept
title: Visual In-Context Learning
slug: visual-in-context-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [visual ICL, in-context learning for vision, 视觉上下文学习]
tags: [computer-vision, in-context-learning, multimodal]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Visual In-Context Learning** (视觉上下文学习) — an inference paradigm for large vision models in which unseen tasks are solved by conditioning the model on a prompt of image–label example pairs, without any parameter update.

## Key Points

- Formally: `y_q = g_τ(P, x_q)` where `P = {x_c1, y_c1, ..., x_cK, y_cK}` are `K` in-context examples drawn from a source dataset, and `g_τ` is the frozen vision model.
- Performance is highly sensitive to the choice of in-context examples: the gap between best and worst prompt can exceed 70% mIoU on Pascal-5^i foreground segmentation.
- Unlike NLP in-context learning, visual in-context learning has been studied only very recently; [[zhang-nd-what]] provides the first comprehensive investigation.
- Good visual in-context examples tend to be semantically similar and spatially close to the query (similar background, pose, appearance, viewpoint).
- More in-context examples consistently improve performance across random, unsupervised, and supervised selection strategies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-nd-what]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-nd-what]].
