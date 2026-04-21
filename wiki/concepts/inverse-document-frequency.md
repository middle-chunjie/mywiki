---
type: concept
title: Inverse Document Frequency
slug: inverse-document-frequency
date: 2026-04-20
updated: 2026-04-20
aliases: [IDF, IDF weighting, 逆文档频率]
tags: [information-retrieval, nlp, text-representation]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Inverse Document Frequency** (逆文档频率) — a weighting scheme that assigns lower weight to tokens appearing frequently across a corpus and higher weight to rare, informative tokens, computed as the negative log of a token's collection frequency.

## Key Points

- In BERTScore and CodeBERTScore, IDF weights are computed from a language-specific test set and used to up-weight rare tokens when aggregating token similarities into precision and recall.
- [[zhou-2023-codebertscore-2302-05527]] follows Zhang et al. (2020) in applying IDF weighting: each token's contribution to the score is proportional to its negative log frequency in the reference corpus.
- IDF weighting prevents common programming tokens (e.g., brackets, semicolons) from dominating the similarity score, though punctuation is also removed via an alphanumeric mask.
- The combination of IDF weighting and the alphanumeric masking in CodeBERTScore means trivial shared tokens have negligible influence on the final metric.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-codebertscore-2302-05527]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-codebertscore-2302-05527]].
