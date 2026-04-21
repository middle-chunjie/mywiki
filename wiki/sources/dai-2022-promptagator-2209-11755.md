---
type: source
subtype: paper
title: "Promptagator: Few-shot Dense Retrieval From 8 Examples"
slug: dai-2022-promptagator-2209-11755
date: 2026-04-20
language: en
tags: [dense-retrieval, few-shot-learning, query-generation, reranking, beir]
processed: true

raw_file: raw/papers/dai-2022-promptagator-2209-11755/paper.pdf
raw_md: raw/papers/dai-2022-promptagator-2209-11755/paper.md
bibtex_file: raw/papers/dai-2022-promptagator-2209-11755/paper.bib
possibly_outdated: true

authors:
  - Zhuyun Dai
  - Vincent Y. Zhao
  - Ji Ma
  - Yi Luan
  - Jianmo Ni
  - Jing Lu
  - Anton Bakalov
  - Kelvin Guu
  - Keith B. Hall
  - Ming-Wei Chang
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2209.11755
doi:
url: http://arxiv.org/abs/2209.11755
citation_key: dai2022promptagator
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

PROMPTAGATOR studies few-shot dense retrieval in the setting where each target retrieval task provides only a short instruction and as few as `2-8` labeled query-document examples. The paper uses a `137B` FLAN model to generate synthetic task-specific queries from corpus documents, filters them with round-trip consistency, and then trains a lightweight `110M` dual encoder plus an optional `110M` cross-attention reranker. On 11 BEIR datasets, the few-shot retriever reaches `47.8` average nDCG@10, outperforming MS MARCO-trained ColBERT v2 (`46.2`) and SPLADE v2 (`46.6`) despite avoiding token-level late interaction in the first-stage retriever. The reranker variant reaches `52.8`, showing that a handful of in-task examples can be amplified into effective dense retrieval supervision.

## Problem & Motivation

Prior neural retrieval methods often assume that supervision from one or two large QA-style datasets such as [[ms-marco]] or [[natural-questions]] will transfer broadly, but BEIR reveals strong heterogeneity in query distribution, corpus structure, and search intent. The paper argues that retrieval failure on diverse tasks is not only a corpus-adaptation issue; models also miss task-specific notions of relevance. PROMPTAGATOR targets a more realistic few-shot setting where a practitioner can provide a short task description and a few positive examples, then asks whether a prompted LLM can expand that sparse supervision into enough task-aligned synthetic data to train an efficient dense retriever.

## Method

- **Task formulation**: retrieval is defined as `T = {D, Q, I}` where the corpus `D`, query distribution `Q`, and search intent `I` can differ substantially across tasks; the few-shot BEIR setup provides only `k <= 8` positive query-document pairs per target task.
- **Prompt-based query generation**: PROMPTAGATOR builds a task-specific prompt from labeled examples `{(q_i, d_i)}_{i=1}^k` and a new document `d`, then asks [[flan]] to emit a task-shaped query for that document. The paper uses the `137B` FLAN checkpoint, samples up to `1M` documents per corpus, generates `8` queries per document, and decodes with `temperature = 0.7`.
- **Synthetic data construction**: accepted generations become synthetic positive pairs `(\hat{q}, d)`, intended to match the target task's query distribution and search intent more closely than generic QA-trained question generators.
- **Round-trip filtering**: an initial retriever is trained on the noisy synthetic pairs, then each pair `(q, d)` is kept only if the source document is retrieved in the top `K = 1` results for `q`. This retrieval-specific filtering removes generic, ambiguous, and hallucinated queries without relying on an external QA model.
- **Retriever architecture**: the first-stage model is a shared [[dual-encoder]] initialized from a T5-base v1.1 encoder, mean-pools the top layer, and projects to a `768`-dimensional embedding. The retriever has roughly `110M` parameters and is pretrained on C4 with Contriever-style independent cropping before synthetic-data fine-tuning.
- **Training recipe**: dual-encoder fine-tuning uses cross-entropy with in-batch negatives. Batch size is `128` for small corpora (`<50k` docs) and `6000` otherwise; training runs `1000` steps for small/mid datasets and `5000` for large ones.
- **Reranking extension**: PROMPTAGATOR++ trains a `110M` cross-attention reranker on the same synthetic data, using cross-entropy with `31` sampled negatives from the top `200` passages returned by the retriever; reranker batch size is `64`, with `5000` or `20000` fine-tuning steps depending on corpus size.

## Key Results

- On 11 BEIR datasets, few-shot PROMPTAGATOR reaches `47.8` average nDCG@10, improving over zero-shot PROMPTAGATOR (`45.5`), GPL (`45.5`), ColBERT v2 (`46.2`), and SPLADE v2 (`46.6`).
- The largest task-level gains over zero-shot PROMPTAGATOR appear on argument-heavy retrieval: Touché-2020 improves from `26.6` to `34.5`, and ArguAna from `53.8` to `59.4`.
- PROMPTAGATOR++ raises average nDCG@10 from `47.8` to `52.8`, exceeding zero-shot PROMPTAGATOR++ (`49.9`), monoT5-3B (`51.1`), and UPR (`42.7`).
- Few-shot PROMPTAGATOR outperforms ColBERT v2 by `1.2` average nDCG (`47.8` vs. `46.2`) while serving with a simpler `110M` dual encoder instead of token-level late interaction.
- Round-trip filtering improves performance on `8/11` datasets and adds roughly `2.5` average nDCG@10 in the ablation study.
- The paper reports that eight examples plus LLM-generated supervision can approximately match the effect of `50k` labeled MS MARCO examples for a simple dual encoder.

## Limitations

- Results are sensitive to exemplar quality: Climate-FEVER few-shot prompting underperforms zero-shot when the prompt examples include noisy relevance labels, and performance recovers only after switching to FEVER-style prompts.
- Round-trip filtering can hurt on very small datasets such as NFCorpus and SciFact, likely because filtering plus further fine-tuning overfits when the synthetic pool is already small.
- The method depends on an expensive upstream generator: query synthesis uses a `137B` FLAN model and the paper reports generating `58.46M` queries (`610M` words), so training-time cost is substantial even if serving is cheap.
- The paper does not resolve data-efficiency questions such as how many synthetic pairs are actually needed, nor does it systematically analyze prompt sensitivity beyond a few ablations.
- Main evaluation excludes MS MARCO, Natural Questions, and Quora from the primary benchmark comparison because of overlap concerns, so broad conclusions still depend on BEIR-style transfer tasks.

## Concepts Extracted

- [[information-retrieval]]
- [[dense-retrieval]]
- [[few-shot-learning]]
- [[question-generation]]
- [[prompt-engineering]]
- [[large-language-model]]
- [[synthetic-data]]
- [[dual-encoder]]
- [[round-trip-consistency]]
- [[neural-reranking]]

## Entities Extracted

- [[zhuyun-dai]]
- [[vincent-y-zhao]]
- [[ji-ma]]
- [[yi-luan]]
- [[jianmo-ni]]
- [[jing-lu]]
- [[anton-bakalov]]
- [[kelvin-guu]]
- [[keith-b-hall]]
- [[ming-wei-chang]]
- [[google-research]]
- [[flan]]
- [[beir]]
- [[natural-questions]]
- [[ms-marco]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
