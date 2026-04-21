---
type: source
subtype: paper
title: Towards Completeness-Oriented Tool Retrieval for Large Language Models
slug: qu-2024-completenessoriented-2405-16089
date: 2026-04-20
language: en
tags: [tool-retrieval, llm, dense-retrieval, graph-learning, tool-learning]
processed: true

raw_file: raw/papers/qu-2024-completenessoriented-2405-16089/paper.pdf
raw_md: raw/papers/qu-2024-completenessoriented-2405-16089/paper.md
bibtex_file: raw/papers/qu-2024-completenessoriented-2405-16089/paper.bib
possibly_outdated: false

authors:
  - Changle Qu
  - Sunhao Dai
  - Xiaochi Wei
  - Hengyi Cai
  - Shuaiqiang Wang
  - Dawei Yin
  - Jun Xu
  - Ji-Rong Wen
year: 2024
venue: CIKM 2024
venue_type: conference
arxiv_id: 2405.16089
doi: 10.1145/3627673.3679847
url: http://arxiv.org/abs/2405.16089
citation_key: qu2024completenessoriented
paper_type: method

read_status: unread

domain: llm
---

## Summary

This paper argues that tool retrieval for large language models should optimize completeness rather than only semantic relevance. It proposes COLT, a two-stage framework that first fine-tunes a PLM bi-encoder for query-tool semantic matching and then refines query, scene, and tool representations with dual-view graph collaborative learning over query-scene, query-tool, and scene-tool bipartite graphs. The training objective combines cross-view contrastive loss with a list-wise multi-label loss so the retriever covers the full ground-truth tool set instead of over-ranking near-duplicate tools. The paper also introduces the COMP@K metric and the ToolLens benchmark. Across ToolLens and ToolBench, COLT consistently improves Recall, NDCG, and especially completeness metrics, and leads to better downstream GPT-4-evaluated tool-use quality.

## Problem & Motivation

Existing tool retrievers for LLM systems mostly rank tools by semantic similarity between a query and a tool description. That objective is insufficient for multifaceted user requests that require several complementary tools at once, because semantically similar tools can be redundant while another essential tool is missed. The paper therefore reframes tool retrieval as a completeness-oriented problem: the retriever should return a diverse but jointly sufficient tool set for downstream tool-augmented generation. It also argues that standard retrieval metrics such as Recall and NDCG under-measure this requirement, motivating both a new metric and a new benchmark.

## Method

- **Task setup**: given query `q` and tool set `T = {(t_i, d_i)}_{i=1}^N`, retrieve the top-`K` tools that maximize both semantic relevance and completeness of the ground-truth tool set.
- **Stage 1, semantic learning**: encode queries and tools with a PLM bi-encoder and score them by cosine similarity, `y_SL(q, t) = sim(e_q, e_t)`, where `e_q` and `e_t` are mean-pooled final-layer embeddings.
- **Semantic objective**: optimize InfoNCE over one positive tool and `k` negatives, `-log exp(sim(q, t+)) / (exp(sim(q, t+)) + sum_j exp(sim(q, t_j-)))`, so the initial embedding space aligns queries with relevant tools.
- **Scene abstraction**: group the full ground-truth tool set for each query into a latent "scene" so the model can reason about collaborative tool usage rather than isolated pairwise matches.
- **Graphs**: build three bipartite graphs, query-scene (`Q-S`), query-tool (`Q-T`), and scene-tool (`S-T`), to encode both direct relevance and higher-order tool co-usage structure.
- **Scene-centric view**: run LightGCN-style normalized propagation on `Q-S`, `e_q^{S(i)} = sum_{s in N_q^S} 1 / (sqrt(|N_q^S|) sqrt(|N_s^Q|)) e_s^{S(i-1)}`, with initial scene embedding `e_s^{S(0)} = 1 / |N_s^T| sum_{t in N_s^T} e_t`.
- **Tool-centric view**: run analogous propagation on `Q-T`, producing `e_q^T` and `e_t^T`; derive `e_s^T = 1 / |N_s^T| sum_{t in N_s^T} e_t^T` from the learned tool embeddings.
- **Final score**: combine the two views as `y(q, t) = sim(e_q^S, e_t^T) + sim(e_q^T, e_t^T)`, so ranking uses both scene-mediated and direct query-tool evidence.
- **Collaborative objective**: align the two views with cross-view contrastive losses on queries and scenes, `L_Q^C` and `L_S^C`, each using temperature `tau`.
- **Completeness-oriented ranking loss**: for a list of length `L`, define target selection probabilities over the full labeled tool list and minimize a list-wise multi-label loss `L_list` so the model learns to cover all required tools instead of over-optimizing one tool at a time.
- **Total loss**: train collaborative learning with `L = L_list + lambda (L_Q^C + L_S^C)`.
- **Implementation details**: dense retrieval backbones are trained for `5` epochs with `lr = 2e-5`, `weight_decay = 0.01`, and AdamW in BEIR; collaborative learning uses batch size `2048`, tunes `lr` over `{1e-3, 5e-3, 1e-4, 5e-4, 1e-5}`, `weight_decay` over `{1e-5, 1e-6, 1e-7}`, and propagation depth `I in {1, 2, 3}`.

## Key Results

- On ToolLens, COLT lifts Contriever from `R@3 = 83.58` / `C@3 = 59.46` after fine-tuning to `R@3 = 93.64`, `N@3 = 94.53`, and `C@3 = 84.55`, with `C@5 = 94.08`.
- On ToolBench I2, Contriever + COLT reaches `R@3 = 75.72`, `N@3 = 78.57`, `C@3 = 51.97`, and `C@5 = 70.10`, outperforming both BM25 and fine-tuned dense retrievers.
- On ToolBench I3, Contriever + COLT obtains `R@3 = 76.63`, `N@3 = 81.21`, `C@3 = 52.00`, and `C@5 = 68.47`, again the best overall setting in the table.
- The biggest relative gains are on completeness metrics: for example, ANCE on ToolLens improves from `C@3 = 54.23` after fine-tuning to `80.50` with COLT, and from `C@5 = 85.83` to `94.40`.
- In downstream pairwise evaluation on `100` ToolLens test queries, COLT achieves the best GPT-4-based Elo scores across Coherence (`1116`), Relevance (`1110`), Comprehensiveness (`1121`), and Overall (`1096`).
- Ablations show every component matters: on ToolLens with Contriever backbone, removing semantic learning drops `R@|N| / C@|N|` from `92.76 / 82.95` to `65.21 / 30.90`, while removing collaborative learning drops them to `80.60 / 54.44`.
- The paper reports that `BERT-mini (11M)` with COLT can outperform a plain `BERT-large (340M)`, indicating that collaborative retrieval structure can matter more than raw encoder scale.
- The introduced ToolLens dataset contains `18,770` queries and `464` tools, with each query requiring `1-3` verified tools; ToolBench I2 has `82,507` queries and `11,473` tools, and ToolBench I3 has `23,734` queries and `1,419` tools.

## Limitations

- The method is evaluated on curated benchmarks where each query needs at most `1-4` tools, so scalability to much larger tool sets and more compositional workflows is not fully established.
- COLT improves retrieval quality, but the paper does not measure end-to-end latency, memory overhead, or online serving complexity of the two-stage graph-enhanced retriever.
- ToolLens construction relies heavily on GPT-4 plus repeated human verification, which may introduce benchmark-specific biases and a costly data creation process.
- The downstream validation uses GPT-4 pairwise judgments on only `100` sampled ToolLens queries rather than full human task-completion studies.
- The scene abstraction is effective empirically, but scenes are induced from ground-truth tool groups rather than discovered from noisy real interaction logs.

## Concepts Extracted

- [[tool-retrieval]]
- [[dense-retrieval]]
- [[dual-encoder-retrieval]]
- [[graph-neural-network]]
- [[lightgcn]]
- [[heterogeneous-graph]]
- [[multi-view-learning]]
- [[graph-contrastive-learning]]
- [[infonce-loss]]
- [[learning-to-rank]]

## Entities Extracted

- [[changle-qu]]
- [[sunhao-dai]]
- [[xiaochi-wei]]
- [[hengyi-cai]]
- [[shuaiqiang-wang]]
- [[dawei-yin]]
- [[jun-xu]]
- [[ji-rong-wen]]
- [[renmin-university-of-china]]
- [[baidu]]
- [[chinese-academy-of-sciences]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
