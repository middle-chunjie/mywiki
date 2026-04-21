---
type: concept
title: Bidirectional LSTM
slug: bilstm
date: 2026-04-20
updated: 2026-04-20
aliases: [BiLSTM, Bidirectional Long Short-Term Memory, 双向长短期记忆网络]
tags: [rnn, sequence-modeling, nlp, deep-learning]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Bidirectional LSTM** (双向长短期记忆网络) — a recurrent neural architecture that processes sequences in both forward and backward temporal directions using two independent LSTM modules, concatenating their hidden states to produce context-aware representations at each position.

## Key Points

- LSTM gates (input, forget, output) mitigate the vanishing gradient problem present in vanilla RNNs, enabling the model to capture long-range dependencies in sequences.
- The bidirectional extension yields representations informed by both past and future context, which is important for sequence labeling and token-level prediction tasks.
- In the CWI architecture, a character-level BiLSTM encodes the morphological structure of the target word: input characters `[c_1, ..., c_n]` are embedded, passed through the BiLSTM and a dropout layer, yielding feature vector `F_t`.
- Character-level encoding allows the model to generalize to unseen or rare words — particularly important for biomedical terminology with low Transformer vocabulary coverage.
- BiLSTM features are concatenated with Transformer contextual features before the final regression layers, combining local morphological and global sentential signals.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
