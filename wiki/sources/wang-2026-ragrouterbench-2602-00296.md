---
type: source
subtype: paper
title: "RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing"
slug: wang-2026-ragrouterbench-2602-00296
date: 2026-04-20
language: en
tags: [rag, retrieval, routing, benchmark, evaluation]
processed: true

raw_file: raw/papers/wang-2026-ragrouterbench-2602-00296/paper.pdf
raw_md: raw/papers/wang-2026-ragrouterbench-2602-00296/paper.md
bibtex_file: raw/papers/wang-2026-ragrouterbench-2602-00296/paper.bib
possibly_outdated: false

authors:
  - Ziqi Wang
  - Xi Zhu
  - Shuhang Lin
  - Haochen Xue
  - Minghao Guo
  - Yongfeng Zhang
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2602.00296
doi: 10.48550/arXiv.2602.00296
url: http://arxiv.org/abs/2602.00296
citation_key: wang2026ragrouterbench
paper_type: benchmark

read_status: unread

domain: ir
---

## Summary

RAGRouter-Bench introduces a benchmark for adaptive retrieval-augmented generation that treats paradigm selection as a routing problem conditioned on both the query and the corpus. The paper standardizes five paradigms, namely LLM-only, NaiveRAG, GraphRAG, HybridRAG, and IterativeRAG, under a unified retriever abstraction and evaluates them across four corpora, three query types, and dual effectiveness-efficiency axes. Beyond reporting answer quality, the benchmark explicitly characterizes corpora with structural graph metrics and semantic-space diagnostics such as intrinsic dimension and hubness. The central empirical claim is that no single RAG paradigm is uniformly best: performance depends on query-corpus compatibility, while more sophisticated retrieval mechanisms often fail to dominate once token cost is accounted for.

## Problem & Motivation

Prior adaptive RAG work is largely query-centric: it assumes routing should depend on question complexity while treating the corpus as passive background. This paper argues that the corpus is an active constraint on retrieval because graph sparsity, semantic crowding, and document scale can change which paradigm is appropriate for the same query type. The authors therefore target a missing benchmarking gap: a controlled setting where multiple RAG paradigms are instantiated comparably, queries are diversified across factual, reasoning, and summary needs, corpora are diagnosed quantitatively, and routing can be studied as a decision over effectiveness-efficiency trade-offs rather than raw answer quality alone.

## Method

- The routing objective is formalized as `` `\pi^* = \arg\max_{\pi \in \Pi} \mathcal{U}(\pi; q, \mathcal{C})` ``, where utility can reflect answer quality, token cost, or a combined trade-off for a query-corpus pair.
- The benchmark covers four corpora totaling `21,460` documents: MuSiQue, QuALITY, UltraDomain-Legal, and GraphRAGBench-Medical, with query augmentation producing `7,727` queries across factual, reasoning, and summary categories.
- Corpus preprocessing uses sliding-window chunking with `` `L = 512` `` tokens, overlap `` `O = 100` ``, `cl100k_base` tokenization, DeepSeek-V3 triplet extraction at temperature `` `0.0` ``, concurrency `15`, timeout `60s`, retries `3`, and `text-embedding-3-small` embeddings of dimension `` `d = 1536` `` indexed with FAISS.
- The benchmark standardizes five paradigms. NaiveRAG retrieves top-`100` chunks by cosine similarity. GraphRAG extracts up to `20` seed entities, runs Personalized PageRank with `` `\alpha = 0.85` ``, keeps up to `100` nodes and `500` triplets, and maps triplets back to source text. HybridRAG fuses NaiveRAG and GraphRAG rankings with Reciprocal Rank Fusion using smoothing constant `` `k = 60` ``. IterativeRAG runs a retrieve-generate-evaluate loop for up to `` `T = 3` `` iterations.
- Shared generation settings use DeepSeek-V3 or LLaMA-3.1-8B with temperature `0.3`, max output `1000`, retrieval token budget `8000`, cosine similarity, and minimum relevance threshold `0.4`. LLM-only uses temperature `0.7`.
- Query construction follows a generate-and-validate pipeline: single-hop factual questions are sampled from chunks, reasoning questions are created from graph random walks over `2`-hop and `3`-hop chains, and summary questions are built from entity-centered document clusters. Validation applies grounding, shortcut-detection, and closed-book leakage checks; human verification on `200` sampled queries reaches `94%` agreement with automated judgments and Cohen's `` `\kappa = 0.85` ``.
- Evaluation combines answer quality and efficiency: Semantic F1, Coverage, Faithfulness, and [[llm-as-a-judge]] correctness, with token accounting decomposed into retrieval and generation costs. GPT-4o-mini is used as the judge model in the appendix implementation details.

## Key Results

- The benchmark reports `7,727` queries in the main text, distributed as `4,086` reasoning (`52.9%`), `2,320` factual (`30.0%`), and `1,321` summary (`17.1%`), over `21,460` source documents.
- DeepSeek-V3 results show no universal winner. HybridRAG has the best overall average on MuSiQue (`38.6%`), Legal (`36.1%`), and Medical (`64.7%`), while NaiveRAG is best on QuALITY (`48.8%`).
- Query-corpus interactions are strong: GraphRAG reaches `90.2%` on MuSiQue factual questions but only `70.7%` on QuALITY factual questions; HybridRAG reaches `32.8%` on MuSiQue reasoning and `72.2%` on Legal factual questions; NaiveRAG reaches `83.7%` on QuALITY factual and `69.2%` factual with LLaMA-3.1-8B.
- Corpus diagnostics align with performance reversals: QuALITY is described as graph-sparse and narrative-heavy, while Legal shows high hubness and low dispersion that hurt pure vector retrieval; Medical benefits most from complementary retrieval, with DeepSeek HybridRAG at `64.7%` vs. NaiveRAG `61.1%` and GraphRAG `53.5%`.
- Efficiency varies by orders of magnitude. Average per-query context/token cost in the main table ranges from about `7k` tokens for Medical IterativeRAG to `230k` for Legal HybridRAG. In the appendix, total token consumption ranges from `13.27M` for Medical IterativeRAG to `293.10M` for Legal HybridRAG.
- Model choice matters substantially: on MuSiQue, HybridRAG with DeepSeek-V3 achieves `38.6%` LLM-as-a-Judge accuracy versus `20.3%` with LLaMA-3.1-8B, a gap of `18.3` points.

## Limitations

- The study benchmarks paradigm-level abstractions rather than exhaustively tuned implementations, so conclusions about GraphRAG or HybridRAG may not transfer directly to optimized production systems.
- Many benchmark queries are synthetically generated and then filtered; the authors acknowledge that this may not capture the noise profile of real user traffic.
- Several evaluation signals rely on model-based judgments, including LLM-generated validation and LLM-as-a-Judge scoring, which introduces evaluator dependence even with human spot checks.
- The paper enforces a unified `8k` context budget and shared retriever abstractions for fairness, but that simplification may suppress strengths of paradigms that benefit from custom prompts, rerankers, or domain-specific graph construction.
- The manuscript contains an internal count mismatch: Section 4.2 states `7,727` total queries, while Appendix A.2 describes `5,310` validated final queries after adding the Medical set, without fully reconciling the discrepancy.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[adaptive-rag-routing]]
- [[query-corpus-compatibility]]
- [[graph-rag]]
- [[hybrid-rag]]
- [[iterative-rag]]
- [[llm-as-a-judge]]
- [[reciprocal-rank-fusion]]
- [[personalized-pagerank]]
- [[hubness]]
- [[intrinsic-dimension]]

## Entities Extracted

- [[ziqi-wang-rutgers]]
- [[xi-zhu]]
- [[shuhang-lin]]
- [[haochen-xue]]
- [[minghao-guo]]
- [[yongfeng-zhang]]
- [[ragrouter-bench]]
- [[deepseek-v3]]
- [[llama-3-1-8b]]
- [[text-embedding-3-small]]
- [[gpt-4o]]
- [[musique]]
- [[quality]]
- [[ultradomain-legal]]
- [[graphragbench-medical]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
