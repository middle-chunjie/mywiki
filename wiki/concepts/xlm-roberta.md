---
type: concept
title: XLM-RoBERTa
slug: xlm-roberta
date: 2026-04-20
updated: 2026-04-20
aliases: [XLM-R, Cross-lingual Language Model RoBERTa]
tags: [nlp, multilingual, pretrained-language-model, transformer]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**XLM-RoBERTa** — a multilingual pretrained Transformer encoder trained on 2.5 TB of filtered CommonCrawl data in 100 languages using the RoBERTa training recipe, providing language-agnostic contextual representations for cross-lingual NLP tasks.

## Key Points

- Proposed by Conneau et al. (2020, ACL); scales multilingual pre-training by using a much larger corpus than mBERT, leading to stronger cross-lingual transfer especially for low-resource languages.
- Shares the RoBERTa architecture (12 or 24 Transformer layers, `d_model = 768` or `1024`); trained with a multilingual SentencePiece tokenizer covering 100 languages.
- Outperforms mBERT on cross-lingual classification, NER, and reading comprehension benchmarks; zero-shot transfer to unseen languages is substantially improved.
- In the CWI cross-lingual setting, XLM-RoBERTa provides the `F_c = 768`-dim pooled sentence features; combined with a character-level BiLSTM and language adaptation, it generalizes to German, Spanish, and French without language-specific fine-tuning data.
- Still a strong baseline encoder for multilingual tasks; language-adversarial training on top of XLM-RoBERTa yields additional gains over vanilla fine-tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
