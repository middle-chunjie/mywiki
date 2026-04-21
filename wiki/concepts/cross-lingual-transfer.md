---
type: concept
title: Cross-Lingual Transfer
slug: cross-lingual-transfer
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-lingual transfer learning, zero-shot cross-lingual, 跨语言迁移]
tags: [nlp, multilingual, transfer-learning, domain-adaptation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cross-Lingual Transfer** (跨语言迁移) — the technique of training a model primarily on labeled data in one or more source languages and applying it to target languages with little or no labeled data, leveraging multilingual representations to bridge the language gap.

## Key Points

- Multilingual pretrained encoders such as XLM-RoBERTa provide language-agnostic token representations that enable zero-shot or few-shot transfer to unseen languages.
- A language discriminator combined with gradient reversal encourages the feature extractor to remove language-identifying information, yielding more language-neutral representations.
- CWI 2018 contains no French training data; cross-lingual adaptation still improves French test MAE over a model trained without explicit language alignment.
- Performance degrades in proportion to linguistic distance and the amount of target-language training signal; German and Spanish transfer better than French in the CWI setting.
- Cross-lingual transfer is conceptually analogous to cross-domain adaptation: the language is treated as the "domain" variable the model must become invariant to.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
