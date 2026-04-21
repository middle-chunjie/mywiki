---
type: source
subtype: paper
title: Generative Retrieval with Few-shot Indexing
slug: askari-2024-generative-2408-02152
date: 2026-04-20
language: en
tags: [generative-retrieval, information-retrieval, few-shot-learning, indexing, llm]
processed: true

raw_file: raw/papers/askari-2024-generative-2408-02152/paper.pdf
raw_md: raw/papers/askari-2024-generative-2408-02152/paper.md
bibtex_file: raw/papers/askari-2024-generative-2408-02152/paper.bib
possibly_outdated: false

authors:
  - Arian Askari
  - Chuan Meng
  - Mohammad Aliannejadi
  - Zhaochun Ren
  - Evangelos Kanoulas
  - Suzan Verberne
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2408.02152
doi:
url: http://arxiv.org/abs/2408.02152
citation_key: askari2024generative
paper_type: method

read_status: unread

domain: ir
---

## Summary

This paper proposes Few-Shot GR, a generative retrieval framework that replaces training-based indexing with prompting-only indexing. For each document `d_i`, the system first uses pseudo queries `\hat{q}_j = QG(d_i)` and then prompts an LLM to generate free-text docids `id_j = LLM(\hat{q}_j)`, building a deduplicated docid bank without fine-tuning. Retrieval uses the same LLM with constrained decoding so the generated identifier must belong to the bank and can be mapped back to a document. The key extension is one-to-many mapping, where each document receives multiple docids to cover diverse query intents. On NQ320K and MS300K, the method matches or surpasses strong training-based generative baselines while cutting indexing cost from hundreds or thousands of GPU-hours to `37` hours on a single A100.

## Problem & Motivation

Prior generative retrieval systems typically treat indexing as model training: they fine-tune a seq2seq model on large query-docid pairs so the model memorizes which identifier corresponds to a relevant document. The authors argue that this design is expensive, brittle under corpus updates, and poorly aligned with the pre-trained text-generation objective of modern large language models. Their goal is to keep the direct docid-generation interface of generative retrieval while removing heavy training and exploiting an LLM's pre-trained knowledge more directly.

## Method

- **Few-shot indexing**: for a corpus `C = {d_1, ..., d_|C|}`, generate `n` pseudo queries per document and then produce `n` free-text docids with `\hat{q}_j = QG(d_i)` and `id_j = LLM(\hat{q}_j)`.
- **One-to-many mapping**: instead of assigning one identifier per document, assign multiple docids per document so different queries can still land on the same relevant item through different lexical realizations.
- **Docid bank**: aggregate all generated docids into a bank `B`, then de-duplicate it so each valid docid maps uniquely to one document at retrieval time.
- **Retrieval objective**: given a query `q`, use the same backbone model to generate `id = LLM(q)` and map the matched valid docid back to its document.
- **Constrained decoding**: apply constrained beam search during generation so the output must be a member of the docid bank `B`, preventing invalid identifiers.
- **Indexing inputs**: follow prior GR work by indexing with pseudo queries rather than raw documents; the pseudo queries are generated with the InPars query generator.
- **Backbone and decoding hyperparameters**: use `llama-3-8B-Instruct`; generate `n = 10` docids per document; constrain docid length to `3-15` tokens.
- **Evaluation setup**: tune on training splits of NQ320K or MS300K; report efficiency on a single `A100 80GB` GPU with batch size `16`.

## Key Results

- **NQ320K**: Few-Shot GR reaches `Recall@1 = 70.1`, `Recall@10 = 87.6`, `MRR@100 = 77.4`, outperforming NOVO on `Recall@1` and `MRR@100` and exceeding GLEN's `69.1 / 86.0 / 75.4`.
- **MS300K**: Few-Shot GR achieves `Recall@1 = 49.6`, `Recall@10 = 81.2`, `MRR@10 = 59.1`, beating GenRET on Recall metrics (`47.9 / 79.8`) and narrowly trailing NOVO on `MRR@10` (`59.2`).
- **One-to-many mapping matters**: with Llama-3 on NQ320K, increasing docids per document from `1` to `10` improves `Recall@10` by `27.2%`, with performance saturating around `10` docids.
- **LLM choice matters**: on NQ320K, `llama-3-8B-Instruct` scores `70.1 / 87.6 / 77.4`, Zephyr-7B-beta scores `69.9 / 87.2 / 77.8`, and T5-base lags at `52.4 / 66.4 / 55.8`.
- **Efficiency**: indexing takes `37` hours for Few-Shot GR versus `240` for DSI-QG and approximately `16,800` for GenRET on a single-A100 equivalent basis; retrieval latency is `98 ms` versus `72 ms` for DSI-QG and GenRET.

## Limitations

- The largest evaluated corpora contain `100K` to `320K` documents; the paper does not show whether prompting-only indexing remains effective at million-document scale.
- Retrieval quality depends on the choice of backbone LLM and on pseudo-query quality, so gains may not transfer uniformly to weaker generators or lower-resource settings.
- Retrieval latency is not universally better than training-based baselines: indexing is far cheaper, but online decoding still takes `98 ms`, slightly slower than the `72 ms` reported for DSI-QG and GenRET.
- The evaluation focuses on two benchmark collections; broader validation on BEIR, conversational retrieval, and dynamic-update scenarios is left to future work.

## Concepts Extracted

- [[generative-retrieval]]
- [[few-shot-indexing]]
- [[large-language-model]]
- [[document-identifier]]
- [[one-to-many-mapping]]
- [[question-generation]]
- [[constrained-decoding]]
- [[information-retrieval]]
- [[sequence-to-sequence]]

## Entities Extracted

- [[arian-askari]]
- [[chuan-meng]]
- [[mohammad-aliannejadi]]
- [[zhaochun-ren]]
- [[evangelos-kanoulas]]
- [[suzan-verberne]]
- [[llama-3-8b-instruct]]
- [[natural-questions]]
- [[ms-marco]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
