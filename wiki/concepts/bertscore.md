---
type: concept
title: BERTScore
slug: bertscore
date: 2026-04-20
updated: 2026-04-20
aliases: [BERTScore, BERT score, 基于BERT的评估指标]
tags: [evaluation, nlp, text-generation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**BERTScore** (基于BERT的评估指标) — a reference-based text generation evaluation metric that encodes both candidate and reference with a BERT-based model and computes token-level cosine similarities, aggregated into precision, recall, and F1.

## Key Points

- BERTScore encodes candidate and reference sentences independently using a pretrained model, yielding contextual token vectors rather than surface tokens.
- Precision is computed by averaging the maximum cosine similarity of each candidate token to any reference token; recall is the symmetric counterpart over reference tokens.
- The final score is the F1 harmonic mean of precision and recall, and optionally F_β variants that weight recall higher.
- [[zhou-2023-codebertscore-2302-05527]] extends BERTScore to code by (a) prepending natural language context to both code sequences and (b) using language-specific CodeBERT encoders, resulting in CodeBERTScore.
- IDF token weighting (from a reference corpus) further reduces the influence of common tokens on the score.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-codebertscore-2302-05527]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-codebertscore-2302-05527]].
