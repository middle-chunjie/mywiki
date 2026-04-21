---
type: concept
title: Joint Embedding
slug: joint-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [联合嵌入, joint embedding space, shared embedding space]
tags: [representation-learning, multimodal, cross-modal]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Joint Embedding** (联合嵌入) — a shared vector space into which items from two or more different modalities are projected such that semantically related items from any modality are geometrically proximate, enabling cross-modal similarity computation.

## Key Points

- A joint embedding requires two (or more) encoder networks `f_x(·)` and `f_y(·)` that map heterogeneous inputs to a common dimensionality where inner products or cosine similarity measure semantic relatedness.
- Learning quality is assessed on downstream tasks such as cross-modal retrieval (Recall@K, MdR) and captioning metrics (BLEU, CIDEr-D, METEOR).
- A key challenge is that cross-modal training (inter-modality loss only) does not guarantee that semantically similar items within a single modality remain proximate in the joint space — CrossCLR addresses this with an explicit intra-modal alignment term.
- Joint embeddings built on frozen pre-trained expert features inherit the semantic structure of those experts, making intra-modal proximity preservation particularly important.
- The choice of negative sampling strategy strongly influences joint embedding geometry: uncritical use of all non-matching pairs as negatives causes semantic collision.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zolfaghari-2021-crossclr-2109-14910]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zolfaghari-2021-crossclr-2109-14910]].
