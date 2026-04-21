---
type: source
subtype: paper
title: "Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation"
slug: zhong-2024-mixofgranularity-2406-00456
date: 2026-04-20
language: en
tags: [rag, retrieval, chunking, medical-qa, mixture-of-experts]
processed: true
raw_file: raw/papers/zhong-2024-mixofgranularity-2406-00456/paper.pdf
raw_md: raw/papers/zhong-2024-mixofgranularity-2406-00456/paper.md
bibtex_file: raw/papers/zhong-2024-mixofgranularity-2406-00456/paper.bib
possibly_outdated: false
authors:
  - Zijie Zhong
  - Hanwen Liu
  - Xiaoya Cui
  - Xiaofan Zhang
  - Zengchang Qin
year: 2024
venue: arXiv preprint
venue_type: preprint
arxiv_id: 2406.00456
doi:
url: http://arxiv.org/abs/2406.00456
citation_key: zhong2024mixofgranularity
paper_type: method
read_status: unread
domain: ir
---

## Summary

This paper introduces Mix-of-Granularity (MoG), a method that dynamically selects the optimal chunk size for retrieval in RAG systems using a lightweight router. Rather than committing to one fixed granularity, MoG pre-indexes a corpus at `n_gra = 5` levels (each level doubling the preceding chunk size) and trains an MLP router—fed RoBERTa query embeddings—to predict granularity-level weights. To bypass non-differentiable top-k selection during training, the authors propose soft labels derived from offline BM25 + semantic similarity, supervising the router with binary cross-entropy. MoG is extended to MoGG by reorganizing documents as a graph (BM25-similarity edges) so that hopping ranges replace fixed window sizes, enabling retrieval of thematically related but textually distant passages. Evaluated on the MIRAGE medical QA benchmark, MoG and MoGG consistently outperform the single-granularity MedRAG baseline across five backbone LLMs.

## Problem & Motivation

Different knowledge sources (medical textbooks vs. knowledge graphs vs. Wikipedia) have heterogeneous information densities, making a single chunk size suboptimal. Even within one corpus, fine-grained queries benefit from small precise snippets while broad queries need larger contextual chunks. Current practice requires tedious manual grid-search over chunk sizes, does not adapt per query, and fails entirely when relevant information is scattered across distant passages. MoG addresses all three failure modes with a trainable router and, in the MoGG extension, a graph-based context reorganization.

## Method

**MoG core:**
- Corpus is indexed at `n_gra` granularity levels; level-j chunk = two adjacent level-(j-1) chunks (non-overlapping, `level 1` = finest).
- For each query `q`, BM25 retrieves `k_r = 3` top candidates per level, forming a pool of `n_gra × k_r` candidates.
- `q` is encoded with RoBERTa; MLP router outputs weight vector `w ∈ R^{n_gra}`.
- Selection: `chunk_r = top-k argmax(t_rs(c) · w)` over finest chunks; then resolve to parent chunk at `g_r = argmax_g w_g`.
- **Soft-label training**: For each query, BM25 retrieves the best snippet `S_best` per level; semantic similarity to the ground-truth label `l` is computed (TF-IDF / RoBERTa / hitrate); assign soft labels `sl = [0.8, 0.2, 0, ...]` for best and runner-up, train with BCE: `l_bce = Σ_i [-sl_i log(w_i) - (1-sl_i) log(1-w_i)]`.
- Router: ~12 GB GPU memory for training, ~6 GB for inference; ~4h per 1,000-epoch training session on GeForce 3090/4090; Adam optimizer, lr = `0.001`.

**MoGG extension:**
- Documents split into 1–2 sentence nodes; BM25-based edges added if similarity ≥ threshold `T_graph` (each node queries `k_graph = 3` nearest neighbors).
- Granularity level reinterpreted as hopping range (hop-1, hop-2, …); duplicate nodes counted once.
- Otherwise identical to MoG retrieval and training pipeline.

**Storage:** 5 granularity levels ≈ 2.7× original corpus size (one corpus + 5 embedding sets).

## Key Results

- **MoG vs MedRAG (MedCorp, Table 1):** Average accuracy improvement across 5 LLMs: +0.4–6.2 pp; largest gains on weaker LLMs: GLM3 +1.2pp avg, Qwen1.5 +6.2pp avg; GPT-3.5: +0.15pp avg. MoG's best avg: 0.6648 (Llama3) vs MedRAG 0.5900.
- **MoGG vs MoG (Textbooks, Table 2):** Despite training on only 0.2% of MedCorp's snippets, MoGG further improves avg accuracy over MedRAG by more than MoG trained on all of MedCorp; e.g., GPT-3.5: MoGG 0.6168 vs MoGG MedRAG baseline 0.5250; Llama3: MoGG 0.6262 vs MoGG MedRAG 0.5127.
- **Soft labels:** RoBERTa-based soft labels yield best overall performance (avg 0.4856 on PubMedQA, 0.4608 overall); hitrate score is best for MedMCQA but RoBERTa most robust.
- **Inference time:** Router adds ~60% overhead to inference; increasing granularity levels from 1→5 adds only marginal additional time (LLM API latency dominates).
- **Error analysis:** ~95% of degraded samples (vs CoT) are caused by noise injected via retrieved context, not by MoG's routing decisions.

## Limitations

1. Granularity levels are manually specified; automatic discovery (e.g., adaptive chunk boundary detection) is left as future work.
2. The router uses only query semantic embeddings; query type, expected response length, or domain signals are not incorporated.
3. No length normalization in similarity scores, known to matter for IR; acknowledged but not addressed.
4. Security risk: a compromised router could redirect retrieval to malicious sources, injecting harmful context into the LLM.
5. Evaluation confined to medical QA; generalization to other knowledge-intensive domains is not directly demonstrated.

## Concepts Extracted

- [[mix-of-granularity]]
- [[retrieval-augmented-generation]]
- [[multi-granularity-retrieval]]
- [[retrieval-granularity]]
- [[dynamic-chunking]]
- [[dual-encoder-architecture]]
- [[mixture-of-experts]]
- [[soft-label]]
- [[bm25]]
- [[graph-structured-rag]]
- [[medical-question-answering]]
- [[chain-of-thought]]

## Entities Extracted

- [[zijie-zhong]]
- [[hanwen-liu]]
- [[xiaoya-cui]]
- [[xiaofan-zhang]]
- [[zengchang-qin]]
- [[shanghai-artificial-intelligence-laboratory]]
- [[beihang-university]]
- [[mirage-benchmark]]
- [[medrag]]
- [[medcorp]]
- [[faiss]]
- [[neo4j]]
- [[bioasq]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
