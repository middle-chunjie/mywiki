---
type: source
subtype: paper
title: "Tool Graph Retriever: Exploring Dependency Graph-based Tool Retrieval for Large Language Models"
slug: unknown-nd-tool-2508-05152
date: 2026-04-20
language: en
tags: [tool-retrieval, agents, llm, graph-learning, dependency-modeling]
processed: true

raw_file: raw/papers/unknown-nd-tool-2508-05152/paper.pdf
raw_md: raw/papers/unknown-nd-tool-2508-05152/paper.md
bibtex_file: raw/papers/unknown-nd-tool-2508-05152/paper.bib
possibly_outdated: false

authors:
  - Linfeng Gao
  - Yaoxiang Wang
  - Minlong Peng
  - Jialong Tang
  - Yuzhe Shang
  - Mingming Sun
  - Jinsong Su
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2508.05152
doi:
url: https://openreview.net/pdf?id=fyIE0G23ob
citation_key: unknownndtool
paper_type: method

read_status: unread

domain: agents
---

## Summary

The paper proposes Tool Graph Retriever (TGR), a retrieval method for tool-augmented LLM agents that explicitly models dependencies among tools instead of ranking each tool independently by semantic similarity alone. TGR first constructs `TDI300K` to train a three-way dependency discriminator over tool pairs, then builds a directed tool dependency graph and applies parameter-free graph convolution `D^{-1/2}(A + I)D^{-1/2}X` to propagate prerequisite information into tool embeddings. Retrieval is finally performed by scoring query embeddings against the updated tool embeddings. On API-Bank and ToolBench-I1, TGR consistently improves Recall, NDCG, and Pass Rate over BM25, TF-IDF, Paraphrase-MiniLM-L3-v2, and ToolBench-IR, showing that dependency structure helps recover prerequisite tools that are semantically weakly matched to the user query.

## Problem & Motivation

The paper targets a failure mode in tool retrieval for LLM agents: dense or lexical retrievers often retrieve only tools whose descriptions are semantically close to the query, while missing prerequisite tools needed earlier in the execution chain. This is especially harmful when an apparently relevant tool depends on authentication, validation, or intermediate-information tools whose descriptions are not query-aligned. The authors therefore treat tool dependency as additional retrieval signal and ask whether dependency-aware representations can improve both ranking quality and downstream execution completeness.

## Method

- **Dependency definition**: tool `t_a` depends on `t_b` if `t_a` requires output from `t_b` as input, or if `t_b` must be called first for verification or authorization.
- **TDI300K construction**: the dependency discriminator is trained on a three-class dataset of pairs `<t_a, t_b>` with labels `{t_a -> t_b, no dependency, t_a <- t_b}`. The pretraining split is balanced with `92,000` instances per class; the finetuning split is imbalanced with `1,029 / 33,365 / 1,056` instances.
- **Pretraining pipeline**: starting from CodeSearchNet functions, three `gpt-3.5-turbo` agents extract tool documents, generate dependent tool documents, and validate whether the generated dependency satisfies the paper's criteria.
- **Discriminator objective**: pretraining uses 3-class cross-entropy `L(y, y_hat) = -sum_{k=1}^{3} y_k log(y_hat_k)` on concatenated tool documents separated by `[SEP]`, with `[CLS]` used for classification.
- **Imbalanced finetuning loss**: to reduce dominance from the no-dependency class, the paper averages loss per category, `L(y, y_hat) = -sum_{k=1}^{3} (sum_{i=1}^{N_k} y_{i,k} log(y_hat_{i,k}) / N_k)`.
- **Encoder choice**: the discriminator is built on `BERT-base-uncased`; `20%` of the finetuning set is held out as validation, and the best validation checkpoint is kept for testing.
- **Graph construction**: predicted dependencies define a directed graph `G = (V, E)` where nodes are tools and edges represent prerequisite relations.
- **Graph-based tool encoding**: TGR updates tool embeddings with parameter-free graph convolution `G(X, A) = D^{-1/2}(A + I)D^{-1/2}X`, where `X` is the initial tool embedding matrix and `A`/`D` are adjacency and degree matrices.
- **Embedding initialization**: for ToolBench, tool embeddings come from structured tool documents following ToolBench-IR conventions; for API-Bank, only tool descriptions are encoded to reduce domain mismatch between queries and long tool documents.
- **Online retrieval**: the user query is embedded into the same space as the updated tool vectors; cosine-style similarity ranking is then used to return top-`k` tools.
- **Evaluation setup**: retrieval is tested on `311` API-Bank samples and ToolBench-I1 category-level subsets; metrics are Recall, NDCG, and `pass@k = (1 / |Q|) sum_q I(gt(q) subseteq retrieved_k(q))`.

## Key Results

- **Dependency discriminator**: validation `Precision = 0.775`, `Recall = 0.814`, `F1 = 0.792`; test `Precision = 0.893`, `Recall = 0.760`, `F1 = 0.817`.
- **API-Bank, ToolBench-IR base retriever**: TGR improves `Recall@10` from `0.790` to `0.878`, `NDCG@10` from `0.670` to `0.712`, and `Pass Rate@10` from `0.624` to `0.788`.
- **API-Bank, Paraphrase-MiniLM-L3-v2 base retriever**: TGR improves `Recall@10` from `0.763` to `0.834`, `NDCG@10` from `0.609` to `0.659`, and `Pass Rate@10` from `0.592` to `0.698`.
- **ToolBench-I1, ToolBench-IR base retriever**: TGR improves `Recall@5` from `0.709` to `0.742`, `NDCG@10` from `0.807` to `0.829`, and `Pass Rate@10` from `0.690` to `0.730`.
- **Graph quality matters**: on API-Bank, manually annotated graphs outperform discriminator-built graphs; for ToolBench-IR, `Pass Rate@10` rises from `0.788` with `+TGR-d` to `0.817` with `+TGR-m`.
- **Graph density trend**: connected-node ratios are `0.420` on API-Bank and `0.824` on ToolBench, and denser graphs yield larger recall gains in the paper's category-wise analysis.

## Limitations

- The dependency discriminator is still imperfect, and manually annotated graphs outperform predicted graphs, indicating that dependency-identification quality is the current bottleneck.
- Graph construction requires pairwise dependency checks and has `O(N^2)` time complexity, which becomes problematic for large tool inventories.
- Experiments on ToolBench are limited to `I1`; cross-category settings `I2/I3` are left to future work because graph construction is too expensive.
- The dependency formulation only covers prerequisite and verification relations between tool pairs; richer multi-tool workflows or latent task structures are not modeled.
- The pretraining dataset is synthesized with `gpt-3.5-turbo`, so generator bias may affect what kinds of dependency patterns the discriminator learns.

## Concepts Extracted

- [[tool-retrieval]]
- [[tool-dependency]]
- [[dependency-graph]]
- [[graph-convolution]]
- [[dense-retrieval]]
- [[tool-embedding]]
- [[tool-augmented-agent]]
- [[large-language-model]]

## Entities Extracted

- [[linfeng-gao]]
- [[yaoxiang-wang]]
- [[minlong-peng]]
- [[jialong-tang]]
- [[yuzhe-shang]]
- [[mingming-sun]]
- [[jinsong-su]]
- [[xiamen-university]]
- [[baidu]]
- [[alibaba]]
- [[tdi300k]]
- [[api-bank]]
- [[toolbench]]
- [[bert-base-uncased]]
- [[toolbench-ir]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
