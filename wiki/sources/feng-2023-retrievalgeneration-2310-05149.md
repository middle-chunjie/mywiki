---
type: source
subtype: paper
title: Retrieval-Generation Synergy Augmented Large Language Models
slug: feng-2023-retrievalgeneration-2310-05149
date: 2026-04-20
language: en
tags: [llm, retrieval, rag, question-answering, multi-hop]
processed: true
raw_file: raw/papers/feng-2023-retrievalgeneration-2310-05149/paper.pdf
raw_md: raw/papers/feng-2023-retrievalgeneration-2310-05149/paper.md
bibtex_file: raw/papers/feng-2023-retrievalgeneration-2310-05149/paper.bib
possibly_outdated: true
authors:
  - Zhangyin Feng
  - Xiaocheng Feng
  - Dezhi Zhao
  - Maojin Yang
  - Bing Qin
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.05149
doi:
url: http://arxiv.org/abs/2310.05149
citation_key: feng2023retrievalgeneration
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

ITRG proposes an iterative retrieval-generation loop for knowledge-intensive question answering that couples generation-augmented retrieval (GAR) with retrieval-augmented generation (RAG). At iteration `t`, the system expands the query with the previously generated document, retrieves top-`k = 5` Wikipedia paragraphs using a dense dual-encoder retriever, and then regenerates a supporting document with either refine or refresh prompting on [[llama-33b]]. This closed loop is meant to combine parametric knowledge inside the model with non-parametric knowledge from an external corpus while progressively discovering better reasoning paths. Across Natural Questions, TriviaQA, 2WikiMultiHopQA, and HotpotQA, the method outperforms vanilla LM, retrieve-then-read, and generate-then-read baselines, with especially strong gains on multi-hop QA.

## Problem & Motivation

The paper studies how large language models can better solve knowledge-intensive questions when their internal knowledge is incomplete or insufficiently connected for multi-step reasoning. Prior approaches usually choose one of two isolated routes: retrieve evidence from an external knowledge base, or ask the model to generate pseudo-documents from its own parameters. The authors argue that this separation wastes a useful feedback loop: generated text can improve retrieval, and retrieved evidence can improve generation. Their goal is therefore to build an iterative framework that exploits both information sources and helps the model uncover the correct reasoning path over several retrieval-generation rounds.

## Method

- **Task setup**: given a question `q` and a document corpus `D = {d_i}_{i=1}^{|D|}`, ITRG alternates GAR and RAG for up to `T = 5` iterations, retrieving top-`k = 5` paragraphs from a December 2018 Wikipedia dump at each step.
- **Generation-augmented retrieval (GAR)**: at iteration `t > 1`, the query is expanded as `q_t = [q; y_{t-1}]`, where `y_{t-1}` is the document generated in the previous round; at `t = 1`, the query is just `q`. This is a simple query-expansion mechanism intended to reduce semantic gaps between the question and relevant passages.
- **Dense retriever**: retrieval uses a dual-encoder architecture that mean-pools token representations to form document and query embeddings `E(d)` and `E(q)`, then ranks passages by cosine similarity `s(d, q) = cos(E(d), E(q))`.
- **Retrieval-augmented generation, refine variant**: the model updates only with newly retrieved evidence `R_update = R_t - R_{t-1}` and generates `y_t = M(prompt(y_{t-1}, q, R_update))`. If `R_update` is empty, it keeps `y_t = y_{t-1}`.
- **Retrieval-augmented generation, refresh variant**: to avoid propagating errors in earlier generations, refresh discards the prior pseudo-document and regenerates directly from retrieved evidence via `y_t = M(prompt(q, R_t))`.
- **LLM backend and decoding**: the implementation uses [[llama-33b]] with greedy decoding, generating up to `200` tokens for the intermediate document and `15` tokens for the final answer.
- **Evaluation protocol**: experiments use `0`-shot, `1`-shot, and `5`-shot prompting on four QA datasets, each subsampled to `500` examples. Answers are scored by exact match after lowercasing and removing articles, punctuation, and duplicate whitespace.

## Key Results

- On Natural Questions, ITRG (refresh) reaches `37.6 / 38.4 / 38.0` EM in `0`/`1`/`5`-shot, beating LLaMA-33B vanilla LM at `27.0 / 29.4 / 32.4`; the `5`-shot gain is `+5.6` EM.
- On TriviaQA, ITRG (refine) is best with `79.0 / 79.4 / 80.6` EM, outperforming vanilla LM (`74.8 / 70.8 / 75.8`) and generate-then-read (`73.6 / 77.2 / 77.6`).
- On 2WikiMultiHopQA, ITRG (refresh) achieves `32.2 / 36.2 / 38.6` EM and exceeds vanilla LM by `+7.8`, `+8.6`, and `+6.8` points across the three shot settings.
- On HotpotQA, ITRG (refresh) reaches `31.0 / 32.6 / 33.4` EM versus `22.6 / 25.0 / 27.0` for vanilla LM, giving gains of `+8.4`, `+7.6`, and `+6.4`.
- Iteration helps both answer quality and document quality: for ITRG (refresh) in `5`-shot, Natural Questions EM improves from `34.0` at iteration `1` to `38.0` at iteration `5`, while 2WikiMultiHopQA rises from `34.8` to `38.6`; answer recall also improves on NQ from `44.0` to `48.0`.

## Limitations

- The evaluation uses only `500` randomly sampled examples per dataset because of cost, so result stability on the full benchmarks is not established.
- The retrieval setup is fixed to a pre-trained dense retriever over the December 2018 Wikipedia dump; the paper does not study sensitivity to retriever quality, corpus freshness, or alternative indexing strategies.
- Iteration increases inference cost because each question may trigger up to `T = 5` retrieval-generation rounds, but the paper does not report latency or cost breakdowns.
- Only one backend LLM family ([[llama-33b]]) and greedy decoding are tested, so it is unclear how the framework transfers to stronger or smaller models, or to alternative decoding strategies.

## Concepts Extracted

- [[large-language-model]]
- [[retrieval-augmented-generation]]
- [[generation-augmented-retrieval]]
- [[dense-retrieval]]
- [[dual-encoder]]
- [[query-expansion]]
- [[open-domain-question-answering]]
- [[multi-hop-reasoning]]
- [[in-context-learning]]
- [[few-shot-learning]]
- [[parametric-knowledge]]
- [[non-parametric-knowledge]]

## Entities Extracted

- [[zhangyin-feng]]
- [[xiaocheng-feng]]
- [[dezhi-zhao]]
- [[maojin-yang]]
- [[bing-qin]]
- [[harbin-institute-of-technology]]
- [[llama-33b]]
- [[natural-questions]]
- [[triviaqa]]
- [[2wiki-multihopqa]]
- [[hotpotqa]]
- [[gpt-3-5]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
