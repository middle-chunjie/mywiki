---
type: source
subtype: paper
title: Distillation for Multilingual Information Retrieval
slug: yang-2024-distillation-2405-00977
date: 2026-04-20
language: en
tags: [mlir, clir, dense-retrieval, knowledge-distillation, multilingual]
processed: true
raw_file: raw/papers/yang-2024-distillation-2405-00977/paper.pdf
raw_md: raw/papers/yang-2024-distillation-2405-00977/paper.md
bibtex_file: raw/papers/yang-2024-distillation-2405-00977/paper.bib
possibly_outdated: false
authors:
  - Eugene Yang
  - Dawn Lawrie
  - James Mayfield
year: 2024
venue: SIGIR 2024
venue_type: conference
arxiv_id: 2405.00977
doi: 10.1145/3626772.3657955
url: http://arxiv.org/abs/2405.00977
citation_key: yang2024distillation
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper extends Translate-Distill from single-target cross-language retrieval to multilingual information retrieval, where one English query must rank documents from several languages on a single scale. Its Multilingual Translate-Distill (MTD) pipeline uses English MS MARCO data, a ColBERTv2 selector, and a MonoT5-with-mT5XXL scorer to generate teacher scores, then translates passages into each target language and trains a ColBERT-X student with a `KL` objective. The main design question is how to mix languages inside each mini-batch; the authors test mix-passages, mix-entries, and round-robin variants. Across CLEF and NeuCLIR collections, MTD consistently beats prior multilingual translate-train baselines, with especially large gains on NeuCLIR, while remaining fairly robust to the exact mixing strategy.

## Problem & Motivation

MLIR is harder than CLIR because the system must rank documents from multiple languages in a single list rather than solve one query-language to one document-language mapping. Surface-form overlap is often unavailable, translated-document pipelines are costly at indexing time, and multilingual encoders trained only monolingually do not reliably produce cross-language relevance scores on a shared scale. Prior Translate-Distill handled only one document language, so the paper asks how to preserve the benefits of translation plus distillation when the student must compare French, German, Spanish, Chinese, Persian, and Russian documents directly.

## Method

- **Training signal**: start from English query-passage pairs and teacher scores, then optimize the student with `L = KL(p_teacher || p_student)` over the sampled passages in each mini-batch.
- **Two-teacher pipeline**: a query-passage selector retrieves top passages for each query, and a stronger scorer reranks them; in experiments the selector is English `ColBERTv2` and the scorer is `MonoT5` with `mT5XXL`.
- **Multilingual Translate-Distill**: for each batch of `n` queries and `m` sampled passage IDs, translate each passage into all target document languages before assigning language realizations to the batch entries.
- **Mix Passages**: each passage in an entry is independently assigned a language, so high- and low-scoring passages are evenly exposed across languages and the model directly learns multilingual ranking.
- **Mix Entries**: all passages associated with one query share the same randomly chosen language, reducing the chance that translation quality becomes an easy language cue.
- **Round Robin Entries**: repeat queries so the same query appears with passages from every language; this equalizes language exposure but reduces the number of distinct queries per batch under fixed GPU memory.
- **Student backbone**: fine-tune `ColBERT-X` from `XLM-RoBERTa large`, keeping the multilingual encoder while using late interaction for retrieval.
- **Optimization**: train on `8 x V100 32GB` GPUs for `200,000` gradient steps with mini-batches of `8` entries per GPU and `6` passages per entry, `AdamW`, learning rate `5 x 10^-6`, and half precision.
- **Indexing and search**: split documents into `180`-token passages with stride `90`, index with `PLAID-X` using `1` residual bit, retrieve passages, then aggregate document scores with `MaxP`.

## Key Results

- Against `ColBERT-X MTT`, MTD improves `nDCG@20` by `5%` to `26%` and `MAP` by `15%` to `47%` across the reported collections.
- On CLEF 2003, `ColBERT-X MTT` scores `0.643 nDCG@20 / 0.451 MAP / 0.827 Recall@1000`; the best MTD variant reaches `0.699 / 0.535 / 0.922`.
- On NeuCLIR 2022 MLIR, `ColBERT-X MTT` scores `0.375 nDCG@20 / 0.236 MAP / 0.612 Recall@1000`; the best MTD variant reaches `0.474 / 0.347 / 0.768`.
- On NeuCLIR 2023 MLIR, `ColBERT-X MTT` scores `0.330 nDCG@20 / 0.281 MAP / 0.760 Recall@1000`; the best MTD variant reaches `0.404 / 0.372 / 0.877`.
- On the CLEF00-03 subset, MTD improves over MTT from `0.613` to `0.674` in `nDCG@10` and from `0.411` to `0.476` in `MAP@100`.
- The three language-mixing strategies are statistically similar on most MAP and Recall comparisons, indicating that MTD is robust as long as multiple languages are present in each mini-batch.
- In the training-language ablation, mix-passages is more robust to train-test language mismatch, and training on both CLEF and NeuCLIR languages can slightly hurt effectiveness relative to training only on the evaluation languages.

## Limitations

- The method still depends on a costly pipeline: English teacher retrieval, high-quality reranking, and machine translation of MS MARCO passages into every target language.
- Query language is fixed to English in the reported experiments, so the paper does not test multilingual or code-switched query settings.
- Round-robin mixing is constrained by GPU memory because repeating queries across languages reduces the number of distinct entries per batch.
- Training on both CLEF and NeuCLIR languages can degrade performance, suggesting either capacity limits in the student model or sensitivity to translation artifacts.
- The paper studies only four MLIR evaluation collections, so robustness beyond CLEF and NeuCLIR remains unverified.

## Concepts Extracted

- [[multilingual-retrieval]]
- [[cross-lingual-retrieval]]
- [[multilingual-dense-retrieval]]
- [[knowledge-distillation]]
- [[dual-encoder]]
- [[late-interaction]]
- [[multilingual-translate-distill]]
- [[language-mixing-strategy]]
- [[kl-divergence]]
- [[neural-machine-translation]]

## Entities Extracted

- [[eugene-yang]]
- [[dawn-lawrie]]
- [[james-mayfield]]
- [[johns-hopkins-university]]
- [[colbert-x]]
- [[mono-t5]]
- [[ms-marco]]
- [[clef-2003]]
- [[trec-neuclir-2022]]
- [[trec-neuclir-2023]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
