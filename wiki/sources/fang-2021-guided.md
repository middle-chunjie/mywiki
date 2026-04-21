---
type: source
subtype: paper
title: Guided Attention Network for Concept Extraction
slug: fang-2021-guided
date: 2026-04-20
language: en
tags: [concept-extraction, sequence-labeling, attention, crf, nlp]
processed: true

raw_file: raw/papers/fang-2021-guided/paper.pdf
raw_md: raw/papers/fang-2021-guided/paper.md
bibtex_file: raw/papers/fang-2021-guided/paper.bib
possibly_outdated: true

authors:
  - Songtao Fang
  - Zhenya Huang
  - Ming He
  - Shiwei Tong
  - Xiaqing Huang
  - Ye Liu
  - Jie Huang
  - Qi Liu
year: 2021
venue: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence
venue_type: conference
arxiv_id:
doi: 10.24963/ijcai.2021/200
url: https://www.ijcai.org/proceedings/2021/200
citation_key: fang2021guided
paper_type: method

read_status: unread

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2021; re-verify against recent literature.

This paper proposes the Guided Attention Concept Extraction Network (GACEN), a sequence labeling model that injects structured document signals into concept extraction. Instead of relying only on token labels, GACEN uses document titles and LDA-derived topic vectors to guide a topic-aware attention module, and uses manually collected clue words plus a neural soft-matching component to drive a position-aware attention module. The two guided representations are interpolated and passed to a CRF decoder. On CSEN, KP-20K, and MTB, GACEN improves F1 over strong baselines such as BERT-CRF and Joint-layer RNN, with the largest gain on CSEN, and it remains especially useful when labeled data is scarce because clue words provide compact supervision beyond span annotations.

## Problem & Motivation

Prior concept extraction systems either rank candidate phrases with handcrafted or statistical features, or apply neural sequence models that still learn only from sparse token-level annotations. The paper argues that humans identify concepts using richer structured cues: document titles expose global topic, topic distributions highlight semantically central words, and local clue phrases such as "is called an" or "we study" indicate where concepts are likely to appear. The authors therefore target a model that can exploit these signals directly, generalize clue-word matching beyond exact string overlap, and improve sample efficiency in low-label settings.

## Method

- **Task formulation**: given sentence `x = {x_1, ..., x_n}`, title `t = {t_1, ..., t_l}`, and clue-word set `c = {c_1, ..., c_k}`, predict BIOES-style labels `y_i ∈ {S-CON, B-CON, I-CON, E-CON, O}` for each token.
- **Topic-based encoder**: compute document topic vector `z_d` and word topic vectors `z_w ∈ R^k` with LDA, concatenate them with token embeddings, and encode title/body separately by Bi-LSTM. Hidden states are `h_i = [\overrightarrow{h_i}; \overleftarrow{h_i}]`.
- **Topic-aware attention**: use title summary/query `q_t` to build topic-aware token features `u_i = α_i h_i`, where `α_i = SoftMax(v_1^T tanh(W_1 h_i + W_2 q_t))`. This biases the model toward title/topic-related words.
- **Soft matching module**: pre-train a neural matcher to locate semantically similar clue words in unseen sentences instead of relying on exact regex matches; the reported matching threshold is `θ = 0.75` and maximum window size is `3`.
- **Position-aware attention**: define relative position `p_i` to the matched clue-word span `(s_1, s_2)`, embed it as `p_i^s`, average clue-word hidden states into query `q`, and compute `m_i = β_i h_i`, where `β_i = SoftMax(v_2^T tanh(W_3 h_i + W_4 q + W_5 p_i^s))`.
- **Fusion and decoding**: aggregate guided features as `h_i' = λ u_i + (1 - λ) m_i` with `λ = 0.5`, concatenate `[h_i; h_i']`, and feed the result into a CRF layer to model label dependencies.
- **Implementation details**: word embeddings use `300` dimensions; Bi-LSTM hidden size is `200`; topic counts are `50` for CSEN, `100` for KP-20K, and `50` for MTB; parameters are initialized from `U(-0.1, 0.1)`.
- **Optimization**: train with Adam, batch size `10`, dropout `0.1`, on `1 × Tesla V100` plus `16` Intel CPUs; the soft-matching module is trained until the loss stops decreasing for about `20` epochs.

## Key Results

- On **CSEN**, GACEN reaches `69.70` precision, `60.21` recall, and `64.60` F1, outperforming BERT-CRF (`55.26` F1) and Joint-layer RNN (`52.71` F1) by large margins.
- On **KP-20K**, GACEN reaches `45.69` F1 versus `44.01` for GACEN-position, `43.77` for GACEN*REs, `43.48` for GACEN-topic, and `41.73` for BERT-CRF.
- On **MTB**, GACEN achieves `66.43` precision, `64.72` recall, and `65.56` F1, improving over BERT-CRF (`60.20` F1) and Joint-layer RNN (`59.86` F1).
- Removing topic-aware attention drops recall sharply: CSEN recall falls from `60.21` to `57.94`, KP-20K from `37.65` to `34.90`, and MTB from `64.72` to `62.63`.
- Replacing neural soft matching with regex matching increases precision but hurts recall; on KP-20K, GACEN*REs gets `59.23` precision but only `34.71` recall, versus GACEN's `58.10` / `37.65`.
- With only `20%` of training data, the paper reports GACEN performing comparably to a BiLSTM-CRF baseline trained with `70%`, indicating better label efficiency.

## Limitations

- The model depends on manually collected clue-word inventories; although the reported annotation cost is under `10` hours, this still adds domain adaptation overhead.
- Topic features rely on an external LDA pipeline with dataset-specific topic counts, which increases preprocessing complexity and may be brittle across domains.
- Evaluation is limited to three concept-extraction datasets and does not test cross-domain transfer or fully end-to-end clue-word discovery.
- The paper compares against 2020-era baselines; it does not address stronger pretrained encoder architectures or more recent span-based extraction methods.

## Concepts Extracted

- [[concept-extraction]]
- [[sequence-labeling]]
- [[long-short-term-memory]]
- [[conditional-random-field]]
- [[latent-dirichlet-allocation]]
- [[soft-matching]]
- [[topic-aware-attention]]
- [[position-aware-attention]]
- [[clue-words]]

## Entities Extracted

- [[songtao-fang]]
- [[zhenya-huang]]
- [[ming-he]]
- [[shiwei-tong]]
- [[xiaqing-huang]]
- [[ye-liu]]
- [[jie-huang]]
- [[qi-liu]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
