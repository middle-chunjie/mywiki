---
type: source
subtype: paper
title: Rethinking the Role of Token Retrieval in Multi-Vector Retrieval
slug: lee-nd-rethinking
date: 2026-04-20
language: en
tags: [information-retrieval, dense-retrieval, multi-vector-retrieval, token-retrieval, late-interaction]
processed: true

raw_file: raw/papers/lee-nd-rethinking/paper.pdf
raw_md: raw/papers/lee-nd-rethinking/paper.md
bibtex_file: raw/papers/lee-nd-rethinking/paper.bib
possibly_outdated: true

authors:
  - Jinhyuk Lee
  - Zhuyun Dai
  - Sai Meher Karthik Duddu
  - Tao Lei
  - Iftekhar Naim
  - Ming-Wei Chang
  - Vincent Y. Zhao
year: 2023
venue: NeurIPS 2023
venue_type: conference
arxiv_id: 2304.01982
doi: 10.48550/arXiv.2304.01982
url: https://arxiv.org/abs/2304.01982
citation_key: leendrethinking
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper argues that late-interaction multi-vector retrieval has over-invested in expensive gather-and-rescore pipelines while under-training the first-stage token retriever. It introduces XTR, which changes training so relevant document tokens must be retrieved directly from in-batch token pools, then reuses those retrieved scores at inference instead of loading every token vector from each candidate document. This removes the gathering stage and replaces exhaustive rescoring with a lightweight approximation based on missing-similarity imputation. Across BEIR, LoTTE, open-domain QA passage retrieval, and MIRACL, XTR matches or exceeds strong ColBERT- and GTR-style baselines while making the scoring stage roughly `4,000x` cheaper in estimated FLOPs.

## Problem & Motivation

Multi-vector retrievers such as ColBERT are expressive because they keep token-level query-document interactions, but their scoring function is not directly compatible with efficient maximum inner product search over full documents. In practice this forces a three-stage inference pipeline: token retrieval, gathering all token vectors for candidate documents, and then expensive rescoring. The paper argues that this design creates both a systems bottleneck and a training-inference mismatch: conventional objectives optimize document-level reranking, yet the actual candidate set is determined earlier by token retrieval. If the first-stage token retriever misses salient evidence tokens, later rescoring cannot recover them cheaply. XTR is motivated by the claim that better token retrieval should itself carry most of the retrieval burden.

## Method

- **Baseline formulation**: ColBERT scores a query-document pair as `` `f_ColBERT(Q,D) = (1/n) * sum_i max_j q_i^T d_j` ``, where `n` is the query length and each query token chooses its best-matching document token.
- **In-batch token retrieval objective**: XTR replaces within-document alignment with an alignment mask over all mini-batch tokens, setting `` `Â_ij = 1` `` only when token `j` is among the top-`k_train` retrieved tokens for query token `i` across `mB` batch tokens. The training score becomes `` `f_XTR(Q,D) = (1/Z) * sum_i max_j Â_ij q_i^T d_j` ``, where `` `Z = |{i | exists j, Â_ij > 0}|` `` counts query tokens that retrieved at least one token from `D`.
- **Inference without gathering**: Candidate documents are formed from top-`k'` retrieved tokens per query token. Instead of loading all candidate-document token vectors, XTR scores candidates using only retrieved token similarities plus missing-similarity imputation: `` `f_XTR'(Q,D^) = (1/n) * sum_i max_j [Â_ij q_i^T d_j + (1-Â_ij) m_i]` ``.
- **Missing similarity imputation**: when a candidate document has no retrieved token for a query token, XTR substitutes an imputed score `` `m_i` `` bounded by the last retrieved top-`k'` score for that query token, approximating the sum-of-max score without recomputing pairwise similarities.
- **Complexity reduction**: the paper estimates scoring-stage cost drops from `` `n^2 k' (2 m d + m + 1)` `` for ColBERT-style rescoring to `` `n^2 k' (r + 1)` `` for XTR's lightweight scorer, where `r` is the average number of retrieved tokens per candidate document.
- **Training setup**: XTR is initialized from T5 encoder backbones (`base`, `xxl`), multilingual XTR from `mT5`, tuned with `` `k_train in {32, 64, 128, 256, 320}` ``, trained for `50,000` iterations with learning rate `` `1e-3` ``, and uses in-batch negatives plus fixed RocketQA hard negatives from MS MARCO.
- **Retrieval stack**: inference uses `` `k' = 40,000` `` token retrieval, ScaNN for maximum inner product search during token retrieval, and up to `256` TPU v3 chips depending on model scale.

## Key Results

- **Efficiency**: estimated scoring FLOPs on an MS MARCO-style setting drop from `0.36 x 10^9` for ColBERT rescoring to `0.09 x 10^6` for XTR, roughly a `4,000x` reduction.
- **BEIR**: `XTR_base` reaches average `49.1 nDCG@10`, outperforming `ColBERT` (`45.1`) and `T5-ColBERT_base` (`46.8`); `XTR_xxl` reaches `52.7`, beating `T5-ColBERT_xxl` (`50.8`) and matching or surpassing prior strong zero-shot retrievers.
- **LoTTE**: `XTR_base` achieves pooled `69.0` on LoTTE Search and `60.1` on LoTTE Forum, ahead of `ColBERT` (`67.3`, `58.2`) and far above `BM25` (`48.3`, `47.2`).
- **Open-domain QA passage retrieval**: `XTR_xxl` gets `79.4` top-20 on EntityQuestions, `84.9` on Natural Questions, `83.3` on TriviaQA, and `81.1` on SQuAD, with the paper highlighting a `+4.1` top-20 gain on EntityQuestions over the previous state of the art.
- **Multilingual retrieval**: on MIRACL, `mXTR_base` averages `58.2 nDCG@10` and `mXTR_xl` `65.9`, outperforming multilingual Contriever baselines (`52.7` for the jointly trained variant).
- **Ablation on efficient scoring**: on MS MARCO dev, `XTR_base` with top-`k'` score imputation reaches `MRR@10 = 37.4` and `Recall@1000 = 98.0`, while applying the same efficient scorer to `T5-ColBERT_base` is much worse (`27.7`, `91.8`), showing the benefit comes from the training objective rather than only the inference trick.

## Limitations

- The main training recipe depends on English MS MARCO and fixed RocketQA hard negatives, so transfer to low-resource, non-English, or license-constrained settings is not directly resolved.
- The efficient scorer is still an approximation: if first-stage token retrieval misses relevant evidence, XTR falls back to imputed similarities rather than exact document-token interactions.
- Strong results often rely on large retrieval breadth such as `` `k' = 40,000` `` and large batch sizes, so the practical memory-throughput tradeoff is not completely eliminated.
- The paper emphasizes estimated FLOPs and benchmark accuracy; it does not provide a full production latency, serving-cost, or hardware-utilization study across end-to-end deployments.

## Concepts Extracted

- [[multi-vector-retrieval]]
- [[token-retrieval]]
- [[late-interaction]]
- [[missing-similarity-imputation]]
- [[dense-retrieval]]
- [[information-retrieval]]
- [[maximum-inner-product-search]]
- [[in-batch-negative-sampling]]
- [[passage-retrieval]]
- [[multilingual-retrieval]]
- [[lexical-retrieval]]

## Entities Extracted

- [[jinhyuk-lee-deepmind]]
- [[zhuyun-dai]]
- [[sai-meher-karthik-duddu]]
- [[tao-lei]]
- [[iftekhar-naim]]
- [[ming-wei-chang]]
- [[vincent-y-zhao]]
- [[google-deepmind]]
- [[xtr]]
- [[colbert]]
- [[beir]]
- [[ms-marco]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
