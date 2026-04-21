---
type: source
subtype: paper
title: Context-Augmented Code Generation Using Programming Knowledge Graphs
slug: unknown-nd-contextaugmented-2601-20810
date: 2026-04-20
language: en
tags: [code-generation, retrieval-augmented-generation, knowledge-graph, llm, software-engineering]
processed: true

raw_file: raw/papers/unknown-nd-contextaugmented-2601-20810/paper.pdf
raw_md: raw/papers/unknown-nd-contextaugmented-2601-20810/paper.md
bibtex_file: raw/papers/unknown-nd-contextaugmented-2601-20810/paper.bib
possibly_outdated: false

authors:
  - Shahd Seddik
  - Fahd Seddik
  - Iman Saberi
  - Fatemeh Fard
  - Minh Hieu Huynh
  - Patanamon Thongtanunam
year: 2026
venue: arXiv preprint
venue_type: preprint
arxiv_id: 2601.20810
doi:
url: https://openreview.net/pdf?id=EHfn5fbFHw
citation_key: unknownndcontextaugmented
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

The paper proposes Programming Knowledge Graphs (PKGs) as a structured retrieval layer for code generation with large language models. Instead of retrieving whole rows or documents, it indexes code and tutorial corpora as fine-grained graph nodes: functions and nested code blocks for code-centric retrieval, plus JSON path-value leaves for text-centric retrieval. Retrieval is paired with subtree pruning to control context size and a simple reranker that selects among candidates produced by NoRAG and RAG variants. On open-source models, block-level PKG retrieval improves average pass@1 over NoRAG from `49.0%` to `55.0%` on HumanEval and from `45.4%` to `48.2%` on MBPP, while reranking lifts the averages further to `59.8%` and `60.6%`, respectively. The paper's main claim is that retrieval structure and candidate selection matter more than naive retrieval volume.

## Problem & Motivation

The paper studies why retrieval-augmented code generation often fails even when relevant external knowledge exists. Flat retrieval over raw question-answer pairs or tutorial rows can introduce noisy, weakly aligned, or overlong context, which in turn distracts the generator and increases hallucination risk. The authors argue that programming knowledge is heterogeneous across code, documentation, and examples, so the retrieval unit itself must preserve syntax, hierarchy, and granularity. Their goal is therefore to design a structure-aware representation that retrieves smaller and more useful pieces of evidence, prunes irrelevant branches before prompting, and uses reranking to recover from retrieval-induced regressions.

## Method

- **PKG schema**: represent the corpus as a typed directed graph `G = (V, E, τ, φ)`, where each node stores a textual payload `φ(v)` and node type `τ(v)`; retrieval is cosine-similarity search over embedded node payloads.
- **Code-centric PKG**: preprocess PythonAlpaca from `143,000` Q/A pairs into `115,000` Python functions. For each function `F`, create one NAME node, one IMPL node, and one BLOCK node for every extracted AST block, with containment edges `NAME -> IMPL -> BLOCK` and nested `BLOCK -> BLOCK` parent edges.
- **Text-centric PKG**: convert each tutorial document into JSON, then create PATHVALUE nodes only for leaf paths with primitive values. Each payload is serialized as a path-value pair such as `path = ...; value = ...`, preserving hierarchical structure through parent-child JSON edges.
- **Embedding and storage**: compute node embeddings `z_v = E(φ(v))` and store the graph in Neo4J `5.20.0` with a vector index for approximate nearest-neighbor search; dense retrieval is compared against Voyage-Code-2, while sparse retrieval uses BM25.
- **Retrieval modes**: evaluate three retrieval granularities, `IMPL` (Func-PKG), `BLOCK` (Block-PKG), and `PATHVALUE` (JSON-PKG), retrieving the top node by `v*(q) = argmax Sim(q, v)`.
- **Tree pruning**: for a retrieved code node, score candidate pruned subtrees `G_{v*}^{-u}` by query similarity and keep `G_pruned = argmax_u Sim(q, G_{v*}^{-u})`, which removes one direct child branch to reduce noise before prompt construction.
- **Prompt augmentation**: serialize the retained subtree or path-value payload into context `C(q)` and append it to the original programming query under a deterministic prompt template.
- **Reranking**: generate one candidate per condition, filter by Python parse validity and sandbox executability, then select `c* = argmax_{c in C_R} Sim(q, c)` over surviving candidates.
- **Evaluation setup**: test CodeLlama-7B, CodeLlama-13B, Llama3.1-8B, StarCoder2-7B, and DeepSeek-Coder-7B on a single A100 GPU using greedy decoding with `t = 0` and `max_new_tokens = 512`; benchmarks are HumanEval and MBPP.

## Key Results

- **PKG scale**: the code-centric PKG contains `425,058` nodes and `434,518` relations; the text-centric PKG contains `288,583` path-value nodes and `287,936` relations.
- **HumanEval, open models**: average pass@1 is `49.0%` for NoRAG, `34.8%` for BM25, `52.8%` for Func-PKG, `55.0%` for Block-PKG, and `59.8%` after reranking; the oracle reranker reaches `69.8%`.
- **MBPP, open models**: average pass@1 is `45.4%` for NoRAG, `35.2%` for BM25, `41.8%` for Func-PKG, `48.2%` for Block-PKG, `48.8%` for JSON-PKG, and `60.6%` after reranking.
- **Model-specific gains**: on HumanEval, Llama3.1-8B improves from `55%` (NoRAG) to `61%` (Block-PKG) and `66%` (reranked); on MBPP, StarCoder2-7B improves from `46%` to `51%` and then `62%`.
- **Closed-source models**: retrieval gives only marginal gains for stronger proprietary models; e.g. GPT-4o on MBPP improves from `81.4%` to `83.4%` with reranking, while Claude Sonnet 4 is mostly flat or slightly worse than NoRAG.
- **Context efficiency**: Block-PKG adds only `84-87` tokens on average, compared with `182-188` for Func-PKG, `218-226` for BM25, and `339-349` for dense row retrieval.

## Limitations

- The evaluation is limited to Python code and English tutorial corpora, so the conclusions may not transfer directly to other programming languages, documentation styles, or repository-specific settings.
- Retrieval is still high variance: DeepSeek-Coder-7B is harmed by Block-PKG on both benchmarks, and PKG underperforms NoRAG on some topic clusters such as string manipulation and data structures.
- The reranker is intentionally simple and similarity-based, which leaves a substantial oracle gap of roughly `10` pass@1 points on HumanEval and can choose mathematically plausible but incorrect code.
- The text-centric pipeline depends on LLM-produced JSON structure, which can introduce schema noise, and the graph construction/storage pipeline adds preprocessing and maintenance overhead.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[code-generation]]
- [[knowledge-graph]]
- [[abstract-syntax-tree]]
- [[dense-retrieval]]
- [[bm25]]
- [[reranking]]
- [[graph-pruning]]

## Entities Extracted

- [[shahd-seddik]]
- [[fahd-seddik]]
- [[iman-saberi]]
- [[fatemeh-fard]]
- [[minh-hieu-huynh]]
- [[patanamon-thongtanunam]]
- [[university-of-british-columbia]]
- [[university-of-melbourne]]
- [[humaneval]]
- [[mbpp]]
- [[pythonalpaca]]
- [[neo4j]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
