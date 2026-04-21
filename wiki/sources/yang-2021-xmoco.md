---
type: source
subtype: paper
title: "xMoCo: Cross Momentum Contrastive Learning for Open-Domain Question Answering"
slug: yang-2021-xmoco
date: 2026-04-20
language: en
tags: [dense-retrieval, open-domain-qa, contrastive-learning, passage-retrieval, information-retrieval]
processed: true

raw_file: raw/papers/yang-2021-xmoco/paper.pdf
raw_md: raw/papers/yang-2021-xmoco/paper.md
bibtex_file: raw/papers/yang-2021-xmoco/paper.bib
possibly_outdated: true

authors:
  - Nan Yang
  - Furu Wei
  - Binxing Jiao
  - Daxing Jiang
  - Linjun Yang
year: 2021
venue: "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)"
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2021.acl-long.477
url: "https://aclanthology.org/2021.acl-long.477"
citation_key: yang2021xmoco
paper_type: method

read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2021; re-verify against recent literature.

xMoCo adapts momentum contrastive learning to open-domain QA retrieval, where questions and passages are asymmetric and often benefit from different encoders. The key change is to maintain fast and slow encoders for both sides, plus separate question and passage queues, so the model can optimize question-to-passage and passage-to-question matching jointly while still reusing a large pool of negatives. With BERT-base encoders, a `16,384`-item queue, and FAISS retrieval, the method improves over DPR on most Top-20 and Top-100 retrieval benchmarks across Natural Questions, TriviaQA, WebQuestions, and CuratedTREC. Its main contribution is showing that MoCo-style queue training can be made stable and effective for dense passage retrieval without tying the two encoders.

## Problem & Motivation

Dense passage retrieval for open-domain QA relies on a dual-encoder retriever that must rank one relevant passage among millions of candidates. Standard training only sees a small set of negatives per step, creating a mismatch with inference-time retrieval over a very large index. Original MoCo offers a way to maintain a large negative queue efficiently, but its image-style formulation assumes the two sides are interchangeable and only applies gradients directly to one encoder. For question-passage matching, that setup makes passage representations insufficiently trainable and prevents clean use of separate encoders for questions and passages. The paper addresses this mismatch by redesigning momentum contrast for asymmetric text matching.

## Method

- **Dual-encoder retrieval setup**: represent a question and passage with separate encoders and score relevance by inner product, `s(q, p) = E_q(q) · E_p(p)`, so all passage vectors can be pre-computed and searched with [[maximum-inner-product-search]].
- **xMoCo architecture**: use four encoders, `E_q^{fast}`, `E_q^{slow}`, `E_p^{fast}`, and `E_p^{slow}`, plus two queues `Q_q` and `Q_p` that store previously encoded question and passage vectors.
- **Bidirectional contrastive objective**: optimize question-to-passage and passage-to-question losses jointly, `L_qp` and `L_pq`, then combine them as `L = λL_qp + (1 - λ)L_pq` with `λ = 0.5`.
- **Momentum updates**: apply gradients only to the fast encoders, and update the slow encoders by exponential moving average, `E_p^{slow} <- αE_p^{fast} + (1 - α)E_p^{slow}` and `E_q^{slow} <- αE_q^{fast} + (1 - α)E_q^{slow}`; the implementation uses `α = 0.001`.
- **Batch adaptation**: enqueue all slow-encoder vectors from the current minibatch together, which makes the method compatible with the familiar [[in-batch-negative-sampling]] behavior used in dense retrieval.
- **Encoders and representations**: initialize separate question and passage towers from pre-trained [[bert-base-uncased]]; use the sequence start token from the last layer as the dense representation for both sides.
- **Hard negatives**: add an auxiliary loss over a set `P^-` of hard negatives and, in the implemented setting, sample one BM25 hard negative per positive pair from top retrieval results.
- **False-negative handling**: track passage IDs inside the queue and mask items identical to the current positive passage, reducing spurious negatives when the queue is large or datasets are small.
- **Training hyperparameters**: batch size `128`, dropout `0.1`, queue size `16,384`, Adam with learning rate `3e-5`, linear schedule with `5%` warmup; train `100` epochs for TREC/WQ and `40` for larger datasets.
- **Infrastructure**: inference uses [[faiss]] for dense search; BM25 uses Lucene with `b = 0.4` and `k1 = 0.9`; training runs on `16` NVIDIA `32GB` GPUs and takes under `12` hours per model.

## Key Results

- **Single-dataset retrieval**: xMoCo beats [[dpr]] on Top-20 / Top-100 for NQ (`82.3` / `86.0` vs `78.6` / `85.3`), TriviaQA (`80.2` / `85.9` vs `79.0` / `85.1`), WQ (`76.5` / `83.1` vs `72.2` / `81.2`), and TREC (`80.7` / `89.4` vs `80.1` / `88.9`).
- **Multi-dataset retrieval**: xMoCo again improves over DPR on NQ (`82.5` / `86.3` vs `79.4` / `85.7`), TriviaQA (`80.1` / `85.7` vs `78.5` / `84.8`), WQ (`78.2` / `84.8` vs `74.8` / `82.9`), and TREC (`89.4` / `94.1` vs `89.2` / `93.7`).
- **SQuAD remains difficult**: xMoCo underperforms [[bm25]] on SQuAD retrieval, e.g. single-setting Top-20 / Top-100 is `65.1` / `77.5` versus BM25 `68.9` / `80.3`.
- **Queue ablation**: retrieval quality improves as the negative queue grows, then largely saturates around `16k`, supporting the large-negative-pool hypothesis without requiring the very large queues reported elsewhere.
- **Untied encoders matter**: tying question and passage encoders on NQ drops Top-20 / Top-100 from `82.3` / `86.0` to `75.4` / `81.2`, validating the asymmetric design.
- **End-to-end QA gains are modest**: compared with DPR, xMoCo improves answer accuracy only slightly, e.g. NQ `42.4` vs `42.1`, TriviaQA `57.1` vs `56.4`, and multi-setting TREC `48.1` vs `47.3`.

## Limitations

- The method still relies on stale queued representations; momentum updates mitigate but do not eliminate this approximation error.
- False-negative filtering only masks identical passage IDs, not semantically equivalent but differently indexed passages.
- Retrieval gains are stronger for Top-20 than Top-100 and translate into only marginal improvements for end-to-end QA with the paper's simple BERT reader.
- BM25 remains stronger on SQuAD, and linear score fusion with BM25 is not consistently beneficial.
- The study uses BERT-base encoders, one simple BM25-based hard-negative scheme, and no hyperparameter search, so the design space is only partially explored.

## Concepts Extracted

- [[cross-momentum-contrastive-learning]]
- [[dense-passage-retrieval]]
- [[dual-encoder-retrieval]]
- [[open-domain-question-answering]]
- [[contrastive-learning]]
- [[momentum-contrastive-learning]]
- [[negative-sampling]]
- [[hard-negative-sampling]]
- [[in-batch-negative-sampling]]
- [[maximum-inner-product-search]]

## Entities Extracted

- [[nan-yang]]
- [[furu-wei]]
- [[binxing-jiao]]
- [[daxing-jiang]]
- [[linjun-yang]]
- [[microsoft]]
- [[bert-base-uncased]]
- [[dpr]]
- [[bm25]]
- [[faiss]]
- [[natural-questions]]
- [[triviaqa]]
- [[webquestions]]
- [[curatedtrec]]
- [[squad]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
