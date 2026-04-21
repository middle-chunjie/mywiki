---
type: concept
title: Language Identification
slug: language-identification
date: 2026-04-20
updated: 2026-04-20
aliases: [language ID, language detection, 语言识别]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Language Identification** (语言识别) — the task of assigning a language label, often with confidence scores, to a text unit so that downstream processing can target the intended linguistic subset.

## Key Points

- [[penedo-2023-refinedweb-2306-01116]] uses the CCNet `fastText` classifier at the document level to keep only English pages for REFINEDWEB.
- Documents whose top language score falls below `0.65` are removed, because they are often low-confidence or lack meaningful natural-language content.
- The paper notes that the classifier supports `176` languages, even though the released RefinedWeb extract focuses only on English.
- Language identification is responsible for a large early reduction in scale: after the initial preparation stage, only about `48%` of original CommonCrawl documents remain.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[penedo-2023-refinedweb-2306-01116]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[penedo-2023-refinedweb-2306-01116]].
