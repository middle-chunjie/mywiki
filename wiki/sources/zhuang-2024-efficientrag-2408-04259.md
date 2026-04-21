---
type: source
subtype: paper
title: "EfficientRAG: Efficient Retriever for Multi-Hop Question Answering"
slug: zhuang-2024-efficientrag-2408-04259
date: 2026-04-20
language: en
tags: [rag, retrieval, multi-hop-qa, information-retrieval, nlp]
processed: true
raw_file: raw/papers/zhuang-2024-efficientrag-2408-04259/paper.pdf
raw_md: raw/papers/zhuang-2024-efficientrag-2408-04259/paper.md
bibtex_file: raw/papers/zhuang-2024-efficientrag-2408-04259/paper.bib
possibly_outdated: false
authors:
  - Ziyuan Zhuang
  - Zhiyang Zhang
  - Sitao Cheng
  - Fangkai Yang
  - Jia Liu
  - Shujian Huang
  - Qingwei Lin
  - Saravan Rajmohan
  - Dongmei Zhang
  - Qi Zhang
year: 2024
venue: arXiv preprint
venue_type: preprint
arxiv_id: 2408.04259
doi:
url: http://arxiv.org/abs/2408.04259
citation_key: zhuang2024efficientrag
paper_type: method
read_status: unread
domain: ir
---

## Summary

EfficientRAG proposes a lightweight iterative retriever for multi-hop open-domain question answering that avoids repeated LLM calls during query generation. The system consists of two small token-level classifier models — a Labeler/Tagger and a Filter — fine-tuned on DeBERTa-v3-large. The Labeler extracts useful tokens from each retrieved chunk and tags the chunk as `<Continue>` (needs more information) or `<Terminate>` (sufficient or irrelevant). The Filter concatenates the original query with labeled tokens to construct the next-hop query without invoking an LLM. On HotpotQA and 2WikiMQA, EfficientRAG achieves Recall@K of 81.84 and 84.08 respectively while retrieving only ~6 and ~3.7 chunks on average — far fewer than competing iterative methods. End-to-end QA accuracy is competitive with LLM-based iterative baselines at roughly 3x lower latency.

## Problem & Motivation

Standard one-round RAG fails on multi-hop questions where the answer depends on information from multiple documents linked by intermediate entity hops. Existing iterative RAG methods (e.g., Iter-RetGen, SelfAsk, IRCoT) address this by generating new queries at each retrieval round, but all require additional LLM inference calls per iteration — increasing latency, cost, and reliance on carefully tuned few-shot prompts. The authors observe that the relation types in multi-hop chains are limited in variety (compared to entities), and hypothesize that small models can identify these relations. EfficientRAG tests whether a token-classifier approach can replace LLMs for iterative query generation while preserving or improving retrieval efficiency.

## Method

- **Overall framework**: plug-and-play wrapper around any dense retriever. Given a query, retrieves top-k chunks, passes each through the Labeler+Tagger, filters and pools results, then (if needed) constructs a new query via the Filter and iterates until all chunks are tagged `<Terminate>` or max iterations is reached.
- **Labeler+Tagger**: a single DeBERTa-v3-large encoder with two output heads.
  - *Token labeling head*: classifies each token in `concat(query, chunk)` as useful (`1`) or not (`0`); labeled tokens capture the partial answer bridging entity.
  - *Chunk tagging head*: classifies the mean-pooled sequence embedding as `<Continue>` (chunk helps but more info needed) or `<Terminate>` (chunk resolves the query or is irrelevant).
- **Filter**: a separate DeBERTa-v3-large encoder. Input is `concat(query, labeled_tokens_from_Continue_chunks)`. A token classification head extracts the tokens that form the next-hop query by predicting which tokens to retain, replacing the "unknown" part of the query with bridging information.
- **Synthetic training data construction** (using Llama-3-70B-Instruct):
  1. *Decompose*: LLM decomposes multi-hop question into single-hop sub-questions with dependency graph.
  2. *Token labeling*: LLM marks important tokens in each chunk for each sub-question (SpaCy tokenization).
  3. *Next-hop filtering*: LLM generates next-hop query given previous labeled tokens; same token-extraction procedure applied.
  4. *Negative sampling*: hard negatives retrieved as most-similar but irrelevant chunks; tagged `<Terminate>`.
  - Training set sizes: HotpotQA Labeler 357k / Filter 73k; MuSiQue 93k / 25k; 2WikiMQA 70k / 13k.
- **Training**: AdamW, lr `5e-6`, `4×A100` GPUs, ~10 GPU-hours per dataset.
- **Retriever**: Contriever-MSMARCO for both training data synthesis and inference.
- **Generator** (inference): Llama-3-8B-Instruct; baselines also use Llama-3-8B-Instruct for fair comparison.

## Key Results

- **Retrieval (Recall@K)**:
  - HotpotQA: 81.84 recall at K=6.41 avg chunks (vs. Iter-RetGen iter3: 83.05 at K=16.42; SelfASK: 73.42 at K=35.27).
  - 2WikiMQA: 84.08 at K=3.69 (vs. SelfASK: 88.90 at K=33.68; Iter-RetGen iter3: 74.29 at K=17.32).
  - MuSiQue: 49.51 at K=6.09 — below baselines; attributed to smaller chunk count and higher dataset complexity.
- **End-to-end QA accuracy** (Llama-3-8B-Instruct generator, 3 datasets):
  - HotpotQA: EM 50.59 / F1 57.93 / Acc 57.86 (2nd highest Acc).
  - 2WikiMQA: EM 44.18 / F1 51.64 / Acc 53.41 (2nd).
  - MuSiQue: EM 16.44 / F1 21.18 / Acc 20.00.
- **Efficiency** (MuSiQue 200-sample subset):
  - 1.00 LLM calls per query (vs. 7.18 for SelfASK, 3.00 for Iter-RetGen iter3).
  - Latency 3.62 s (vs. 9.68 s Iter-RetGen iter3, 27.47 s SelfASK); ~3x faster than LLM-based iterative methods.
  - GPU utilization 65.55% — similar to iterative baselines.
- **GPT-3.5 generator** (2WikiMQA): EfficientRAG Acc 61.88%, highest among all baselines including Iter-RetGen iter3 (60.60%).
- **Out-of-domain** cross-dataset transfer (HotpotQA ↔ 2WikiMQA): model trained on one dataset transfers well to the other, sometimes exceeding in-domain performance (e.g., 2WikiMQA trained on HotpotQA: Acc 56.59 vs. in-domain 53.41).

## Limitations

- Only Llama-3-8B-Instruct used as the final QA generator; larger LLM generators untested due to resource constraints.
- Evaluated only on open-domain datasets; multi-hop datasets in in-domain (enterprise, vertical) settings not analyzed.
- MuSiQue performance is suboptimal: the dataset's higher average hop count and smaller retrieved chunk count expose weaknesses in the current architecture.
- Transferability across domain-specialized or very large corpora remains unexplored.
- The token-level next-hop query construction (picking tokens verbatim) is structurally rigid and may break when bridging information requires paraphrase or abstraction.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[multihop-question-answering]]
- [[iterative-retrieval]]
- [[iterative-query-generation]]
- [[chunk-labeling]]
- [[token-classification]]
- [[query-decomposition]]
- [[hard-negative-sampling]]
- [[synthetic-data-generation]]
- [[dense-retrieval]]
- [[open-domain-question-answering]]
- [[multi-hop-retrieval]]
- [[chain-of-thought]]

## Entities Extracted

- [[ziyuan-zhuang]]
- [[zhiyang-zhang]]
- [[sitao-cheng]]
- [[fangkai-yang]]
- [[jia-liu]]
- [[shujian-huang]]
- [[qingwei-lin]]
- [[saravan-rajmohan]]
- [[dongmei-zhang]]
- [[qi-zhang]]
- [[nanjing-university]]
- [[microsoft]]
- [[hotpotqa]]
- [[musique]]
- [[2wikimultihopqa]]
- [[contriever]]
- [[deberta]]
- [[llama-3]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
