---
type: concept
title: Complementary Item Recommendation
slug: complementary-item-recommendation
date: 2026-04-20
updated: 2026-04-20
aliases: [CIR, complementary recommendation]
tags: [recommendation, information-retrieval, e-commerce]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Complementary Item Recommendation** (互补商品推荐) — a recommendation task where the system suggests items that complement (are functionally paired with) a given anchor item, as opposed to similar or substitute items.

## Key Points

- Complementary items are defined by functional pairing rather than semantic similarity: e.g., recommending "roof ridge vents" to someone viewing "roof shingles," not more shingles.
- Distinguished from substitute/similar-item retrieval even though the input format (anchor item content) is identical; the functionality description `F` is the only differentiating signal in the UIA model.
- Most challenging of the three tasks in UIA: smallest training data in Lowe's (330K vs. 968K QBE and 4M keyword search interactions), highest relative gains from joint training (+45% NDCG@10).
- Requires strong cross-functionality knowledge transfer: user search history indirectly signals product interest patterns useful for identifying complementary pairs.
- Collaborative personalization (user ID embeddings) is beneficial for CIR, suggesting that user identity encodes product-category pairing preferences beyond what content similarity captures.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2023-personalized-2304-13654]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2023-personalized-2304-13654]].
