---
type: concept
title: Attentive Personalization Network
slug: attentive-personalization-network
date: 2026-04-20
updated: 2026-04-20
aliases: [APN]
tags: [personalization, attention, dense-retrieval, recommendation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Attentive Personalization Network** (注意力个性化网络) — a network module introduced in UIA that enriches a current request encoding with personalized context drawn from the user's interaction history via multi-head cross-attention (content-based) and learned user/functionality embeddings (collaborative).

## Key Points

- Computes attention from the current request vector to past request encodings (keys) and clicked item encodings (values): `Attn(Q_j, K_j, V_j) = softmax(Q_j K_j^T / √l) V_j`, using `N_h=12` heads with `l=l_v=64` hidden dim per head.
- Supports cross-task history: user interactions with any information access functionality (search, QBE, CIR) can inform personalization for any current request.
- Collaborative component concatenates user embedding `u ∈ R^{128}` and functionality embedding `f ∈ R^{64}` to the attention output, passed through a ReLU MLP to yield personalized request vector `R_t^{u*}`.
- Ablation shows APN is the most impactful component: removing it drops NDCG@10 by 93%/72%/145% for keyword search/QBE/CIR on Lowe's data.
- Collaborative personalization is beneficial for keyword search and CIR but not consistently for QBE, suggesting content-based personalization suffices when item similarity is the primary signal.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2023-personalized-2304-13654]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2023-personalized-2304-13654]].
