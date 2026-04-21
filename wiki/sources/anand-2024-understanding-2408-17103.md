---
type: source
subtype: paper
title: "Understanding the User: An Intent-Based Ranking Dataset"
slug: anand-2024-understanding-2408-17103
date: 2026-04-20
language: en
tags: [ir, ranking, datasets, user-intent, evaluation]
processed: true

raw_file: raw/papers/anand-2024-understanding-2408-17103/paper.pdf
raw_md: raw/papers/anand-2024-understanding-2408-17103/paper.md
bibtex_file: raw/papers/anand-2024-understanding-2408-17103/paper.bib
possibly_outdated: false

authors:
  - Abhijit Anand
  - Jurek Leonhardt
  - Venktesh V.
  - Avishek Anand
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2408.17103
doi:
url: http://arxiv.org/abs/2408.17103
citation_key: anand2024understanding
paper_type: dataset

read_status: unread

domain: ir
---

## Summary

The paper introduces DL-MIA, an intent-annotated ranking dataset built on top of TREC-DL 2021 and 2022 to measure the gap between a user's true information need and the intent inferred from a short query. Instead of evaluating only query-level relevance, the dataset represents each example as `(query, intent, passage, label)` and uses GPT-4, Sentence-BERT clustering, manual review, and crowdsourcing to derive explicit intent descriptions and passage-level intent labels. The final resource contains 24 ambiguous queries, 69 finalized intents, and 2655 annotations. Baseline experiments with BM25, a BERT reranker, and CoLBERTv2 show that making the user intent explicit improves both intent-sensitive ranking and diversity metrics.

## Problem & Motivation

Standard ad-hoc retrieval benchmarks such as MS MARCO and TREC-DL typically expose short keyword queries with relevance labels but do not reveal the precise user intent behind each query. This is a problem when queries are ambiguous or multi-intent, because a ranker can look effective under generic relevance metrics while still misunderstanding what the user actually wants. The paper aims to operationalize this mismatch by constructing an evaluation set with explicit intent descriptions and intent-conditioned passage judgments, so that ranking, diversification, query rewriting, and intent-coverage methods can be evaluated against more faithful representations of user need.

## Method

- **Base data selection**: start from TREC-DL 2021 and 2022 queries and their QRel passages; initially `118` queries are considered.
- **Passage clustering**: encode relevant passages with Sentence-BERT and cluster them when pairwise cosine similarity exceeds `0.8`, reducing redundant evidence before prompting the LLM.
- **Intent generation**: prompt GPT-4 with the query plus clustered passages to produce `5` descriptive intents; decoding uses temperature `0.6` for diversity.
- **Intent clustering and filtering**: cluster generated intents with Sentence-BERT using cosine similarity threshold `0.9`, then manually remove hallucinated, redundant, incomplete, query-copying, or answer-like intents; only queries with `>= 2` intents are retained.
- **Crowdsourced annotation**: annotators label which intents each passage satisfies; when a query has more than `30` relevant passages, it is split into chunks of `30` passages. The study reports `22` annotation sets from `16` annotators, each query annotated at least twice.
- **Manual merging and cleanup**: similar crowd-added intents are merged, and any passage-intent pair with fewer than `2` annotators is dropped to improve reliability.
- **QRel construction**: assign score `0` if no annotator selected an intent for a passage, `1` if at least one annotator selected it, and `2` if all annotators selected it, producing intent-aware QRel files.
- **Benchmarking setup**: evaluate BM25, a BERT cross-attention reranker with max input length `512` and learning rate `1e-5`, and CoLBERTv2; for diversity under intent-as-query evaluation, fuse intent rankings with reciprocal rank fusion using `k = 60`.

## Key Results

- Dataset construction narrows `118` initial TREC-DL queries to `26` intent-bearing candidates and finally to `24` queries after annotation and manual merging.
- The final DL-MIA release contains `69` finalized intents and `2655` intent-labeled tuples of `(query, intent, passage, label)`.
- In the illustrative "What is 311 for" case, passage clustering reduces `53` relevant passages to `18` clusters before intent generation.
- Using original queries but intent-aware QRel labels yields weak performance: BM25 `nDCG@10 = 0.073`, `α-nDCG@10 = 0.144`; BERT reranking reaches `0.060` and `0.114`.
- Treating explicit intents as queries substantially improves results: BM25 reaches `0.116` / `0.250`, BERT reaches `0.169` / `0.375`, and CoLBERTv2 is best at `0.261` / `0.532`.

## Limitations

- The benchmark is small: only `24` final queries, so coverage of web-search ambiguity is limited and per-intent analyses may be noisy.
- Dataset creation depends on GPT-4 prompting plus multiple manual cleanup stages, so reproducibility and cost may be sensitive to model behavior and annotation labor.
- The annotator pool is specialized rather than broad web users, which may bias intent formulations and agreement patterns.
- Experiments cover only a few baseline rankers and two evaluation settings, so conclusions about broader retrieval architectures remain limited.
- The resource is derived only from TREC-DL 2021 and 2022; transfer to other corpora, tasks, or live search traffic is not established.

## Concepts Extracted

- [[information-retrieval]]
- [[user-intent]]
- [[ranking-dataset]]
- [[ad-hoc-retrieval]]
- [[query-ambiguity]]
- [[crowdsourcing]]
- [[relevance-judgment]]
- [[query-rewriting]]
- [[large-language-model]]
- [[sentence-embedding]]
- [[cosine-similarity]]
- [[neural-reranking]]

## Entities Extracted

- [[abhijit-anand]]
- [[jurek-leonhardt]]
- [[venktesh-v]]
- [[avishek-anand]]
- [[l3s-research-center]]
- [[delft-university-of-technology]]
- [[dl-mia]]
- [[ms-marco]]
- [[trec-dl-21]]
- [[trec-dl-22]]
- [[sentence-bert]]
- [[gpt-4]]
- [[bm25]]
- [[colbertv2]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
