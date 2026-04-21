---
type: source
subtype: paper
title: Unsupervised Dense Information Retrieval with Contrastive Learning
slug: izacard-2022-unsupervised-2112-09118
date: 2026-04-20
language: en
tags: [dense-retrieval, contrastive-learning, information-retrieval, multilingual-retrieval, zero-shot-retrieval]
processed: true
raw_file: raw/papers/izacard-2022-unsupervised-2112-09118/paper.pdf
raw_md: raw/papers/izacard-2022-unsupervised-2112-09118/paper.md
bibtex_file: raw/papers/izacard-2022-unsupervised-2112-09118/paper.bib
possibly_outdated: true
authors:
  - Gautier Izacard
  - Mathilde Caron
  - Lucas Hosseini
  - Sebastian Riedel
  - Piotr Bojanowski
  - Armand Joulin
  - Edouard Grave
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2112.09118
doi:
url: http://arxiv.org/abs/2112.09118
citation_key: izacard2022unsupervised
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

The paper introduces Contriever, an unsupervised dense retriever trained with contrastive learning instead of labeled query-document pairs. A shared BERT bi-encoder is optimized on unaligned Wikipedia and CCNet text by constructing positive pairs from random crops of the same document and contrasting them against a large MoCo queue of negatives. The model narrows or surpasses the gap to BM25 in zero-shot retrieval, beating BM25 on 11 of 15 BEIR datasets in Recall@100, and becomes even stronger after MS MARCO fine-tuning. The paper also extends the approach to multilingual retrieval with mContriever, showing strong performance on Mr. TyDi and cross-lingual retrieval on MKQA, including language pairs with different scripts where lexical methods are intrinsically weak.

## Problem & Motivation

Dense retrievers perform well when large supervised retrieval datasets such as MS MARCO are available, but they often fail in zero-shot transfer and are commonly worse than BM25 when no in-domain labels exist. The paper asks whether dense retrieval can be trained fully without supervision and still remain competitive with strong lexical baselines. This matters both for domain transfer, where annotation is expensive, and for multilingual retrieval, where large labeled datasets are sparse outside English and exact lexical matching breaks down across languages and scripts.

## Method

- **Retriever architecture**: uses a shared bi-encoder `f_theta` for queries and documents, with relevance `s(q, d) = f_theta(q)^T f_theta(d)`. Both sides use the same BERT-base-uncased encoder and mean-pool the last-layer token representations.
- **Contrastive objective**: trains with an InfoNCE-style loss over one positive key `k+` and `K` negatives, scaled by temperature `tau`; positives should score higher than negatives under `s(q, d)`.
- **Positive pair construction**: compares the inverse cloze task against independent random cropping. The best variant samples two spans from a `256`-token document, with crop lengths sampled from `5%` to `50%` of document length.
- **Augmentation**: applies token deletion with probability `0.1` in addition to cropping; ablations also test replacement and masking.
- **Negative sampling and momentum encoder**: adopts MoCo with a queue of past negatives and momentum update `theta_k <- m theta_k + (1 - m) theta_q`, which scales beyond pure in-batch negatives.
- **English pre-training hyperparameters**: the main Contriever pre-training setup uses queue size `131072`, momentum `m = 0.9995`, `tau = 0.05`, AdamW with learning rate `5e-5`, batch size `2048`, and `500000` steps on a `50/50` mixture of Wikipedia and CCNet.
- **MS MARCO fine-tuning**: switches to in-batch negatives, uses ASAM with learning rate `1e-5`, batch size `1024`, and `20000` steps; a first pass uses random negatives, then a second pass mines hard negatives, sampled `10%` of the time.
- **Multilingual extension**: mContriever initializes from mBERT, jointly pre-trains on `29` languages with queue size `32768`, momentum `0.999`, `tau = 0.05`, learning rate `5e-5`, and `20000` warmup steps before linear decay.

## Key Results

- On open-domain QA without supervision, Contriever reaches `R@100 = 82.1` on NaturalQuestions versus BM25 `78.3`, and matches BM25 on TriviaQA at `83.2`.
- On BEIR zero-shot retrieval, the unsupervised model beats BM25 on `11/15` datasets for `Recall@100`; the paper reports Contriever as competitive with BM25 except notably on TREC-COVID and Touche-2020.
- In the appendix's unsupervised BEIR table, Contriever achieves average `Recall@100 = 60.1`, ahead of SimCSE `45.4` and REALM `46.9`, though still below BM25 `63.6`.
- After MS MARCO fine-tuning, Contriever reaches BEIR average `Recall@100 = 67.1` (excluding CQA), improving over TAS-B `65.0`, and average `nDCG@10 = 46.6`, above BM25 `43.0`.
- With cross-encoder reranking, `Ours+CE` reaches BEIR average `nDCG@10 = 50.2` and is best on `9` datasets in Table 2.
- In few-shot retrieval, Contriever scores `84.0` on SciFact, `33.6` on NFCorpus, and `36.4` on FiQA without extra supervised pre-training; with MS MARCO it improves to `84.8`, `35.8`, and `38.1`.
- For multilingual retrieval on Mr. TyDi, mContriever + MS MARCO obtains average `MRR@100 = 38.4` and `Recall@100 = 87.0`, outperforming BM25's `33.3` and `74.3`; on cross-lingual MKQA it reaches average `R@100 = 65.6`, above CORA `59.8`.

## Limitations

- The unsupervised model is still weaker than BM25 on ranking-sensitive metrics: on BEIR average `nDCG@10`, unsupervised Contriever gets `36.0` versus BM25 `41.7`.
- It fails badly on specific domains and document types, especially TREC-COVID (`27.4` vs BM25 `65.6` in unsupervised `nDCG@10`) and Touche-2020 (`19.3` vs `36.7`), indicating sensitivity to domain shift and long documents.
- The strongest results depend on substantial compute and data scale, including `500000` pre-training steps, batch sizes up to `2048`, and multi-GPU training.
- Top-ranked retrieval quality often still requires a cross-encoder reranker, so the dense retriever alone is not sufficient for best end-user search quality.
- Multilingual scaling is not monotonic: the appendix explicitly reports a curse of multilinguality, where pre-training on more languages can degrade retrieval quality.

## Concepts Extracted

- [[dense-retrieval]]
- [[bi-encoder]]
- [[contrastive-learning]]
- [[contrastive-loss]]
- [[inverse-cloze-task]]
- [[data-augmentation]]
- [[exponential-moving-average]]
- [[cross-encoder]]
- [[few-shot-learning]]
- [[multilingual-retrieval]]
- [[cross-lingual-retrieval]]

## Entities Extracted

- [[gautier-izacard]]
- [[mathilde-caron]]
- [[lucas-hosseini]]
- [[sebastian-riedel]]
- [[piotr-bojanowski]]
- [[armand-joulin]]
- [[edouard-grave]]
- [[contriever]]
- [[beir]]
- [[bm25]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
