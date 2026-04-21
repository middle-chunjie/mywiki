---
type: source
subtype: paper
title: "Retrieval-Augmented Generation for Large Language Models: A Survey"
slug: gao-2024-retrievalaugmented-2312-10997
date: 2026-04-20
language: en
tags: [rag, llm, retrieval, survey, evaluation]
processed: true

raw_file: raw/papers/gao-2024-retrievalaugmented-2312-10997/paper.pdf
raw_md: raw/papers/gao-2024-retrievalaugmented-2312-10997/paper.md
bibtex_file: raw/papers/gao-2024-retrievalaugmented-2312-10997/paper.bib
possibly_outdated: false

authors:
  - Yunfan Gao
  - Yun Xiong
  - Xinyu Gao
  - Kangxiang Jia
  - Jinliu Pan
  - Yuxi Bi
  - Yi Dai
  - Jiawei Sun
  - Meng Wang
  - Haofen Wang
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2312.10997
doi:
url: http://arxiv.org/abs/2312.10997
citation_key: gao2024retrievalaugmented
paper_type: survey

read_status: unread

domain: llm
---

## Summary

This survey synthesizes retrieval-augmented generation (RAG) for large language models from more than 100 studies and organizes the area into three paradigms: Naive RAG, Advanced RAG, and Modular RAG. The paper decomposes the design space into retrieval, generation, and augmentation, then reviews concrete choices such as retrieval source and granularity, indexing and query optimization, embedding models, reranking, context compression, iterative or adaptive retrieval, and joint use with fine-tuning. Beyond system design, it also summarizes downstream tasks, evaluation targets, and benchmark/tooling ecosystems for RAG, covering both retrieval quality and answer quality. The paper is useful as a taxonomy and engineering map rather than as a single-method contribution or a unified empirical benchmark.

## Problem & Motivation

Large language models are strong general-purpose generators but remain limited by hallucination, stale parametric knowledge, and weak traceability when asked knowledge-intensive or time-sensitive questions. RAG addresses this by coupling generation with external knowledge retrieval, but the literature had grown quickly without a coherent synthesis of paradigms, modules, and evaluation practice. The paper aims to systematize that landscape: what kinds of retrieval sources and units are used, how retrieval and generation are optimized, when augmentation happens, how RAG compares with fine-tuning, and which tasks and metrics are actually used to assess progress.

## Method

- **Survey scope and taxonomy**: reviews `100+` RAG papers and organizes them into `3` paradigms: Naive RAG (`indexing -> retrieval -> generation`), Advanced RAG (pre-retrieval and post-retrieval optimization), and Modular RAG (module substitution, routing, memory, search, task adapters, and flexible flows).
- **Retrieval design space**: compares retrieval sources across unstructured text, semi-structured data such as PDF, structured data such as knowledge graphs, and even LLM-generated content; also contrasts retrieval granularity from `Token`, `Phrase`, `Sentence`, and `Proposition` up to `Chunk` and `Document`.
- **Indexing and pre-retrieval optimization**: summarizes chunking choices such as fixed-size chunks of `100`, `256`, or `512` tokens, recursive splitting, sliding windows, metadata attachment, time-aware retrieval, reverse HyDE, and hierarchical / KG-backed index structures.
- **Query and retriever optimization**: covers multi-query expansion, sub-query decomposition, Chain-of-Verification, query rewriting, HyDE, step-back prompting, metadata or semantic routing, plus sparse-dense hybrid retrieval and retriever tuning with LM supervision; examples include `KL`-based alignment in REPLUG / RA-DIT style training.
- **Generation-stage processing**: describes post-retrieval reranking, context selection, and prompt compression to mitigate "lost in the middle"; representative methods include LLMLingua-style token compression, filter-reranker pipelines, LLM critique, and task- or structure-aware LLM fine-tuning.
- **Augmentation and control flow**: distinguishes iterative retrieval, recursive retrieval, and adaptive retrieval, where models decide whether and when to retrieve; highlighted mechanisms include confidence-triggered retrieval and Self-RAG reflection tokens such as `retrieve` and `critic`.
- **Evaluation framework**: systematizes RAG evaluation along `3` quality scores (context relevance, answer faithfulness, answer relevance) and `4` required abilities (noise robustness, negative rejection, information integration, counterfactual robustness), then maps them to benchmarks and tools.

## Key Results

- The survey covers `100+` RAG studies and compresses them into `3` paradigms and `3` core stages, giving a workable taxonomy for the field.
- Its evaluation review spans `26` downstream tasks and nearly `50` datasets, from single-hop and multi-hop QA to IE, dialogue, reasoning, code search, and fact verification.
- For embedding selection, the paper highlights benchmark scale rather than a winner: MTEB covers `8` tasks and `58` datasets, while C-MTEB covers `6` tasks and `35` datasets.
- The discussion section notes that long-context LLMs can already handle contexts beyond `200,000` tokens, but argues that RAG still matters for latency, selective evidence access, and citation traceability.
- A cited robustness result reports that adding irrelevant documents can increase accuracy by over `30%` in some settings, underscoring that RAG quality depends on interaction between retrieval noise and generation rather than on simple "more relevant is always better" assumptions.
- The paper identifies evaluation tools and benchmarks with explicit target splits, including RGB / RECALL / CRUD and tool-based evaluators such as RAGAS and ARES.

## Limitations

- This is a survey, not a unified benchmark paper: it does not run a controlled empirical comparison under a single experimental protocol.
- A non-trivial portion of the surveyed ecosystem includes arXiv preprints, tool docs, and blog-style engineering writeups, so evidence quality is heterogeneous.
- Some sections are breadth-first rather than deep; for example, multimodal RAG, production systems, and semi-structured retrieval are discussed but not analyzed with the same granularity as text-centric RAG pipelines.
- The fast-moving nature of RAG means the paper is immediately incomplete with respect to post-2024 developments in agentic retrieval, long-context integration, and production evaluation practice.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[information-retrieval]]
- [[vector-database]]
- [[query-rewriting]]
- [[query-expansion]]
- [[hybrid-retrieval]]
- [[context-compression]]
- [[adaptive-retrieval]]
- [[knowledge-graph]]

## Entities Extracted

- [[yunfan-gao]]
- [[yun-xiong]]
- [[xinyu-gao]]
- [[kangxiang-jia]]
- [[jinliu-pan]]
- [[yuxi-bi]]
- [[yi-dai]]
- [[jiawei-sun]]
- [[meng-wang]]
- [[haofen-wang]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
