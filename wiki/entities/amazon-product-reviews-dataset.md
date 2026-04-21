---
type: entity
title: Amazon Product Reviews Dataset
slug: amazon-product-reviews-dataset
date: 2026-04-20
entity_type: tool
aliases: [Amazon reviews dataset, Amazon CDSC benchmark, DVD-Book-Electronics-Kitchen dataset]
tags: [dataset, sentiment-analysis, benchmark, domain-adaptation]
---

## Description

A widely-used benchmark for cross-domain sentiment classification, comprising product reviews from four Amazon domains: Books (B), DVD (D), Electronics (E), and Kitchen (K). Each domain contains 2,000 labeled reviews (1,000 positive, 1,000 negative) and 4,000 unlabeled reviews, enabling 12 cross-domain transfer pairs.

## Key Contributions

- The standard benchmark for evaluating cross-domain sentiment classification methods; used by SCL, SFA, mSDA, DANN, AMN, HATN, IATN, BERT-DAAT, and GAST.
- Enables systematic comparison across all 12 directed domain-pair transfer tasks (e.g., B→D, D→E).
- Unlabeled data from both source and target domains supports semi-supervised and adversarial domain adaptation methods.

## Related Concepts

- [[cross-domain-sentiment-classification]]
- [[sentiment-analysis]]
- [[domain-adaptation]]

## Sources

- [[zhang-2022-graph-2205-08772]]
