---
type: source
subtype: paper
title: A Reproducibility Study of PLAID
slug: macavaney-2024-reproducibility
date: 2026-04-20
language: en
tags: [information-retrieval, late-interaction, neural-retrieval, efficiency, reproducibility]
processed: true

raw_file: raw/papers/macavaney-2024-reproducibility/paper.pdf
raw_md: raw/papers/macavaney-2024-reproducibility/paper.md
bibtex_file: raw/papers/macavaney-2024-reproducibility/paper.bib
possibly_outdated: false

authors:
  - Sean MacAvaney
  - Nicola Tonellotto
year: 2024
venue: SIGIR 2024
venue_type: conference
arxiv_id:
doi: 10.1145/3626772.3657856
url: https://dl.acm.org/doi/10.1145/3626772.3657856
citation_key: macavaney2024reproducibility
paper_type: benchmark

read_status: unread

domain: ir
---

## Summary

This paper revisits PLAID, the centroid-based retrieval engine for ColBERTv2, and asks whether its advertised efficiency-effectiveness trade-offs hold under careful reproduction and stronger baselines. The authors reproduce PLAID's recommended operating points on MS MARCO Dev and TREC DL 2019, add exhaustive-search comparisons via `RBO`, sweep the three PLAID parameters `nprobe`, `t_cs`, and `ndocs`, and compare against BM25 reranking plus LADR. They show that the published settings reproduce well but are not uniquely optimal: `ndocs` drives most of the trade-off surface, lexical reranking is substantially faster at low latency, and LADR can dominate PLAID on densely judged evaluation. A cluster analysis further suggests PLAID's candidate generation often behaves close to lexical matching.

## Problem & Motivation

Late-interaction retrieval models such as ColBERTv2 are effective but expensive because exact top-`k` ranking requires scoring large numbers of token-level interactions. PLAID was proposed as an efficient retrieval engine that uses token centroids and progressive pruning to approximate exhaustive ColBERTv2 search, but the original paper left open how sensitive its three new parameters are, how well it transfers to a more densely judged benchmark, and whether a simple lexical-first reranking pipeline is already competitive. This study targets those gaps so that deployment decisions can be made on actual efficiency-effectiveness frontiers rather than on a small set of hand-picked operating points.

## Method

- **Reproduced engine**: start from the released ColBERTv2 checkpoint and PLAID codebase, and evaluate on a single CPU thread with all embeddings resident in memory.
- **PLAID retrieval pipeline**: for each query token, retrieve the nearest token centroids using `nprobe`; prune centroids whose max query similarity is below `t_cs`; keep the top `ndocs` documents under centroid-level approximate scoring; then fully decompress and exactly score the top `ndocs / 4` documents with ColBERTv2.
- **Reference operational points**: reproduce the paper's settings `(a) = (nprobe 1, t_cs 0.50, ndocs 256)`, `(b) = (2, 0.45, 1024)`, and `(c) = (4, 0.40, 4096)` while setting final retrieval depth to `k = 1000`.
- **Index settings**: use `nbits = 2` and `nclusters = 2^18`, matching the recommended ColBERTv2 / PLAID configuration.
- **Datasets**: evaluate on MS MARCO v1 passage Dev with `6,980` queries and sparse labels, plus TREC DL 2019 with `43` queries and about `215` assessments per query.
- **Metrics**: report `RR@10`, `R@1k`, and `RBO` with persistence `p = 0.99` on MS MARCO; `nDCG@10`, `nDCG@1k`, and `R@1k` on TREC DL 2019; and single-thread latency in `ms/query`.
- **Parameter study**: sweep `nprobe in {1, 2, 4, 8}`, `t_cs in {0.3, 0.4, 0.45, 0.5, 0.6}`, and `ndocs in {256, 1024, 4096, 8192}` to expose interdependence rather than evaluating isolated settings.
- **Baselines**: compare against exhaustive ColBERTv2 search, BM25-first `[[reranking]]` with candidate sizes `n in {200, 500, 1000, 2000, 5000, 10000}`, and `[[ladr]]` with initial BM25 candidates `n in {100, 500, 1000}`, neighbor fanout `k in {64, 128}`, and adaptive expansion `c in {10, 20, 50}`.
- **Cluster analysis**: log token-to-cluster assignments from the ColBERTv2 MS MARCO index and analyze majority-token and majority-cluster proportions to quantify how lexical PLAID's centroid retrieval actually is.

## Key Results

- **Core reproduction**: PLAID `(a)`, `(b)`, and `(c)` achieve MS MARCO `RR@10 = 0.394 / 0.397 / 0.397`, `R@1k = 0.833 / 0.933 / 0.975`, `RBO = 0.612 / 0.890 / 0.983`, and latency `80.5 / 103.4 / 163.9 ms/q`.
- **Dense-label evaluation**: on TREC DL 2019, the same three settings yield `nDCG@10 = 0.739 / 0.745 / 0.745`, `nDCG@1k = 0.553 / 0.707 / 0.760`, and `R@1k = 0.555 / 0.786 / 0.871`, close to exhaustive ColBERTv2 (`0.745 / 0.769 / 0.894`).
- **Reproducibility vs original paper**: absolute latency is much lower than the original report (`185.5 / 222.3 / 352.3 ms/q`), but relative trends are preserved; for example, reproduced setting `(b)` is still about `37%` faster than `(c)`.
- **Parameter sensitivity**: `ndocs = 256` hurts effectiveness while saving only about `10 ms/q` relative to `1024`; `ndocs = 8192` adds noticeable latency without further gains, so the authors recommend `ndocs` roughly in `[1024, 4096]`.
- **Other hyperparameters**: increasing `nprobe` raises latency monotonically and values as low as `1` can damage effectiveness; `t_cs` usually has a smaller effect and works best around `0.4-0.5`.
- **Low-latency baselines**: BM25-first reranking reaches as low as `9 ms/q` at `n = 200`, versus `73 ms/q` for the fastest PLAID setting discussed in the paper, while still obtaining `RR@10 = 0.373` on MS MARCO Dev.
- **LADR frontier**: on TREC DL 2019, LADR's Pareto frontier fully eclipses PLAID in both recall and nDCG, but on MS MARCO Dev its approximation to exhaustive ColBERTv2 remains weaker, with `RBO` saturating around `0.96`.
- **Cluster analysis**: `39%` of PLAID clusters have majority-token proportion above `0.95` and the median is `0.86`; conversely, `33%` of tokens have majority-cluster proportion above `0.95` with median `0.62`, indicating heavy but not exclusive lexical alignment.

## Limitations

- The paper is a reproducibility and analysis study of an existing engine, not a new retrieval algorithm, so it mostly informs tuning and baseline selection rather than architectural innovation.
- All latency measurements are single-thread CPU numbers with embeddings preloaded in memory; the conclusions may shift under multi-threaded or GPU serving.
- MS MARCO Dev has sparse judgments, which makes effectiveness and especially exhaustive-search approximation harder to interpret than on densely judged collections.
- The cluster analysis is descriptive rather than causal: it shows strong lexical dominance in centroid assignments, but does not prove which exact retrieval errors arise from that behavior.
- Lexical-first baselines such as BM25 reranking and LADR introduce extra system components, including a lexical index and, for LADR, a document-neighbor graph.

## Concepts Extracted

- [[late-interaction]]
- [[lexical-retrieval]]
- [[approximate-nearest-neighbor-search]]
- [[reranking]]
- [[rank-biased-overlap]]
- [[pseudo-relevance-feedback]]
- [[dynamic-pruning]]
- [[token-clustering]]

## Entities Extracted

- [[sean-macavaney]]
- [[nicola-tonellotto]]
- [[plaid]]
- [[colbertv2]]
- [[bm25]]
- [[ladr]]
- [[ms-marco]]
- [[trec-dl-2019]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
