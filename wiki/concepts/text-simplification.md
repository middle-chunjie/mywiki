---
type: concept
title: Text Simplification
slug: text-simplification
date: 2026-04-20
updated: 2026-04-20
aliases: [lexical simplification, sentence simplification, 文本简化]
tags: [nlp, text-simplification, readability]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Text Simplification** (文本简化) — the NLP task of rewriting text to reduce its linguistic complexity (lexical, syntactic, or both) while preserving its core meaning, typically to improve readability for non-expert or language-learner audiences.

## Key Points

- Lexical simplification substitutes complex words with simpler synonyms; it is closely related to complex word identification, which identifies *which* words need simplification.
- In the domain-adaptation CWI setting, text simplification is used as an auxiliary task: a masked-language-modeling Transformer predicts a simpler candidate for a masked complex word using the BenchLS benchmark.
- A task discriminator with gradient reversal is applied between the simplification task and the main CWI task, encouraging the feature extractor to learn cross-task invariant representations.
- Including text simplification as an auxiliary task yields the best single-word Pearson score on CompLex LCP 2021 (`.7744`) and the best cross-lingual Pearson on CWI 2018 validation (`.8602` EN-News).
- The BenchLS dataset (Paetzold and Specia, 2016) provides simplification candidates for English words, serving as the source of auxiliary training pairs in this work.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
