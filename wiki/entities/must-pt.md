---
type: entity
title: MuST-PT
slug: must-pt
date: 2026-04-20
entity_type: tool
aliases: [Multilingual Snippet Training for Program Translation, MuST-PT model]
tags: [program-translation, model, pretraining]
---

## Description

MuST-PT (Multilingual Snippet Training for Program Translation) is the model proposed in [[zhu-2022-multilingual]], combining a DOBF-initialized encoder-decoder Transformer with three training stages: multilingual snippet denoising auto-encoding, multilingual snippet translation (MuST), and program-level fine-tuning.

## Key Contributions

- Achieves state-of-the-art BLEU on the CodeXGLUE Java-C# and C#-Java translation tasks (87.37 and 85.25 BLEU respectively) as of AAAI 2022.
- Introduces language identifier embeddings `α_{l_i}` added to token inputs, enabling a single encoder-decoder to handle all 42 language pair translations.
- Demonstrates that the MuST training stage alone is a generalizable plug-in that improves Transformer, CodeBERT, and TransCoder baselines.

## Related Concepts

- [[multilingual-snippet-translation]]
- [[program-translation]]
- [[denoising-autoencoding]]
- [[sequence-to-sequence]]

## Sources

- [[zhu-2022-multilingual]]
