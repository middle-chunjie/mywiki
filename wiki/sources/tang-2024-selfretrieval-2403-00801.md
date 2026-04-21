---
type: source
subtype: paper
title: "Self-Retrieval: Building an Information Retrieval System with One Large Language Model"
slug: tang-2024-selfretrieval-2403-00801
date: 2026-04-20
language: en
tags: [information-retrieval, large-language-model, generative-retrieval, reranking, retrieval-augmented-generation]
processed: true

raw_file: raw/papers/tang-2024-selfretrieval-2403-00801/paper.pdf
raw_md: raw/papers/tang-2024-selfretrieval-2403-00801/paper.md
bibtex_file: raw/papers/tang-2024-selfretrieval-2403-00801/paper.bib
possibly_outdated: false

authors:
  - Qiaoyu Tang
  - Jiawei Chen
  - Zhuoqun Li
  - Bowen Yu
  - Yaojie Lu
  - Cheng Fu
  - Haiyang Yu
  - Hongyu Lin
  - Fei Huang
  - Ben He
  - Xianpei Han
  - Le Sun
  - Yongbin Li
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2403.00801
doi:
url: http://arxiv.org/abs/2403.00801
citation_key: tang2024selfretrieval
paper_type: method
read_status: unread
domain: ir
---

## Summary

This paper proposes Self-Retrieval, an end-to-end information retrieval architecture that collapses indexing, retrieval, reranking, and optionally answer generation into a single large language model. Instead of retrieving identifiers or dense vectors, the model internalizes the corpus via self-supervised sentence-to-passage training, generates candidate titles and passages with trie-constrained decoding so outputs exactly match corpus text, and then scores relevance through self-assessment. On KILT-based NQ and TriviaQA passage retrieval, Self-Retrieval reaches `MRR@5 = 69.45` and `66.72` with a 2.8B StableLM backbone, outperforming strong dense and generative baselines; document retrieval on NQ320K also improves over GenRet by `+5.2 R@1`, `+3.8 R@10`, and `+4.8 MRR@100`. The framework further improves downstream RAG exact-match scores over retriever-reader pipelines.

## Problem & Motivation

Current IR systems usually keep indexing, retrieval, reranking, and downstream generation as separate modules, with LLMs plugged into only part of the pipeline. The paper argues that this separation prevents knowledge sharing across components, underuses the language understanding and generation abilities of LLMs, and creates engineering complexity. The core goal is therefore to build a unified IR system where one LLM both memorizes the corpus and performs retrieval-time matching, passage generation, and relevance estimation, while remaining compatible with downstream retrieval-augmented generation.

## Method

- **Indexing as corpus internalization**: for a passage `p = {s_1, ..., s_L}`, the model takes one sentence `s_i` as input and auto-regressively reconstructs the full passage with objective `P(p | s_i, θ)`. This turns indexing into continued pretraining-like self-supervised learning rather than external index construction.
- **Retrieval by title then passage generation**: given query `q`, the model first generates a title `P(t̂ | q; θ)`, then a passage `P(p̂ | q, t̂; θ)`, using the title as global guidance before passage decoding.
- **Trie-constrained decoding**: a prefix tree `T` is built from the corpus so each generation step is restricted to valid next tokens. Retrieval therefore becomes `P(t̂ | q; θ; T)` and `P(p̂ | q, t̂; θ; T)`, ensuring the generated output matches an existing corpus title or passage rather than a hallucinated paraphrase.
- **Self-assessment reranking**: the model predicts responses like "can answer the query" versus rejection responses for `(q, t_i, p_i)`. Title score and passage assessment score are `S_i^T = Softmax(P(t_i | q; θ) / τ)` and `S_i^P = Softmax((1 - P(rejection | q, t_i, p_i; θ)) / δ)`, with final relevance `S = S^T · S^P`.
- **Reranking supervision**: positive instances use the gold passage; negatives are sampled from both the same document and different documents so the model learns fine-grained relevance discrimination, not just corpus memorization.
- **Unified training and downstream integration**: indexing, retrieval, and reranking are all trained as auto-regressive cross-entropy objectives. For RAG, the gold answer is appended after assessment so the same model can continue into answer generation in one turn.
- **Implementation details**: passage retrieval uses StableLM-3B and Llama2-7B; document retrieval uses StableLM-1.6B on NQ320K and StableLM-3B on MS MARCO. Training uses `8 × NVIDIA A100 80GB`, ZeRO stage-2, AdamW, batch size `16` per GPU, `bf16`, `3` epochs, and learning rate `2e-5`.
- **Inference hyperparameters**: the model generates `i = 5` titles and `j = 10` passages per title with beam search, then reranks the resulting `50` candidates; temperature parameters are fixed at `τ = 0.4` and `δ = 0.4`.

## Key Results

- **Passage retrieval on KILT NQ / TriviaQA**: Self-Retrieval (StableLM, `2.8B`) reaches `H@1 = 62.16`, `H@5 = 79.28`, `MRR@5 = 69.45` on NQ and `58.69 / 78.39 / 66.72` on TriviaQA; the Llama 2 (`6.74B`) variant further improves to `63.44 / 79.29 / 70.00` and `59.94 / 81.06 / 68.74`.
- **Improvement over strong dense retrieval**: compared with fine-tuned BGE, Self-Retrieval 3B gains `+5.46 MRR@5` on NQ (`69.45` vs `63.99`) and `+5.07` on TriviaQA (`66.72` vs `61.65`).
- **Beats 2-stage retriever-reranker pipelines**: it surpasses combinations such as GritLM + BGE-Reranker-FT (`66.98 MRR@5` on NQ, `67.21` on TriviaQA) while using one unified model.
- **Document retrieval on NQ320K**: Self-Retrieval obtains `R@1 = 73.3`, `R@10 = 92.6`, `MRR@100 = 80.7`, improving over GenRet by `+5.2`, `+3.8`, and `+4.8` respectively.
- **Document retrieval on MS MARCO**: Self-Retrieval achieves `R@1 = 47.8`, `R@5 = 69.9`, `MRR@10 = 57.2`, roughly matching GenRet on `R@1` (`47.9`) while outperforming older sparse, dense, and generative baselines.
- **Ablations validate all components**: removing indexing drops NQ `MRR@5` from `69.45` to `58.95`; removing title generation drops it to `52.81`; removing self-assessment drops it to `62.77`. Similar degradations appear on TriviaQA (`66.72 → 60.98 / 58.48 / 55.92`).
- **RAG performance also improves**: on NQ, Self-Retrieval 7B reaches EM `53.26` / `52.98` on `10K` / `40K` corpora versus `49.10` / `49.24` for BGE-FT + Llama2-FT; on TriviaQA it reaches `72.14` / `70.40` versus `61.79` / `61.72`.

## Limitations

- The largest reported scale is only `200K` Wikipedia documents and about `3M` passages, so the paper does not establish behavior on web-scale or highly noisy enterprise corpora.
- Retrieval efficiency is lower than sparse or dense retrievers because inference depends on LLM generation plus constrained decoding rather than cheap vector or lexical lookup.
- The approach still depends on corpus titles as useful global anchors; the MS MARCO setting needed Llama2-generated titles to compensate for missing document metadata.
- Dynamic updates are not solved: incremental learning and corpus expansion remain future work, which is a serious systems limitation for practical IR deployments.
- The method inherits the memory bottlenecks of generative retrieval, even though the reported degradation with corpus growth is better than prior generative approaches.

## Concepts Extracted

- [[information-retrieval]]
- [[large-language-model]]
- [[dense-retrieval]]
- [[generative-retrieval]]
- [[constrained-decoding]]
- [[self-supervised-learning]]
- [[reranking]]
- [[retrieval-augmented-generation]]
- [[beam-search]]

## Entities Extracted

- [[qiaoyu-tang]]
- [[jiawei-chen]]
- [[zhuoqun-li]]
- [[bowen-yu]]
- [[yaojie-lu]]
- [[cheng-fu]]
- [[haiyang-yu]]
- [[hongyu-lin]]
- [[fei-huang]]
- [[ben-he]]
- [[xianpei-han]]
- [[le-sun]]
- [[yongbin-li]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
