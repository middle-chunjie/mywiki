---
type: source
subtype: paper
title: Accelerating Retrieval-Augmented Language Model Serving with Speculation
slug: zhang-2024-accelerating-2401-14021
date: 2026-04-20
language: en
tags: [rag, inference-serving, speculative-decoding, retrieval, nlp]
processed: true
raw_file: raw/papers/zhang-2024-accelerating-2401-14021/paper.pdf
raw_md: raw/papers/zhang-2024-accelerating-2401-14021/paper.md
bibtex_file: raw/papers/zhang-2024-accelerating-2401-14021/paper.bib
possibly_outdated: true
authors:
  - Zhihao Zhang
  - Alan Zhu
  - Lijie Yang
  - Yihua Xu
  - Lanting Li
  - Phitchaya Mangpo Phothilimthana
  - Zhihao Jia
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2401.14021
doi:
url: http://arxiv.org/abs/2401.14021
citation_key: zhang2024accelerating
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2024; re-verify against recent literature in LLM/IR serving.

RaLMSpec is a speculation-inspired serving framework that reduces the latency of iterative retrieval-augmented language models (RaLM) while provably preserving model output. The key insight is that retrieved documents exhibit temporal and spatial locality — the same or adjacent corpus entries are retrieved repeatedly during generation — enabling a request-level local cache as a fast "speculative retriever". Consecutive speculative retrievals from the cache are verified in a single batched knowledge-base query; mismatches trigger rollback and correction. Three complementary techniques — cache prefetching, an Optimal Speculation Stride Scheduler (OS3), and asynchronous verification — further exploit available slack. Evaluated on GPT-2-medium, OPT-1.3B, and LLaMA-2-7B across four QA datasets, RaLMSpec achieves up to `2.39×` speedup with an exact dense retriever; for KNN-LM serving it reaches `7.59×`.

## Problem & Motivation

Iterative RaLM methods (e.g., FLARE, kNN-LM, In-Context RALM) deliver higher generation quality than one-shot RAG by repeatedly querying the knowledge base as generation proceeds. However, their quality advantage comes at the cost of high retrieval overhead: each autoregressive step may trigger a separate retrieval, making sequential latency prohibitive for deployment. The authors ask whether iterative RaLM serving can be accelerated without changing model outputs or retrieval semantics.

## Method

- **Speculative retrieval**: maintain a per-request local cache `Q` initialized with the first retrieval from the corpus `C`. For the next `s` (speculation stride) decode steps, retrieve from `Q` rather than `C` using the same scoring metric — possible because distance ranking is locally computable for both dense (L2/IP) and sparse (BM25) retrievers.
- **Batched verification**: after `s` speculation steps, issue a single batched query `{q_1, …, q_s}` to the real corpus. Find the first mismatch index `m = argmin_i (d̂_i ≠ d_i)`. If `m ≤ s`, roll back token generation to position `m`, regenerate with the correct document `d_m`, and continue. Update the cache with top-`k` results from the verification batch (cache prefetching when `k > 1`).
- **Optimal Speculation Stride Scheduler (OS3)**: formalize the speedup objective as expected verified-documents per unit time. For synchronous verification: `(1 − γ^s) / [(1 − γ)(sa + b)]`, where `a` = speculation step latency, `b` = verification latency, `γ` = speculation accuracy. For asynchronous verification, the denominator uses `γ^s·((s−1)a + max(a,b)) + (1−γ^s)·(sa+b)`. OS3 estimates `a`, `b`, `γ` with a sliding window of size `w = 5` (capped at `γ_max = 0.6`) and solves the optimal `s` online.
- **Asynchronous verification**: when `b < a`, the verification step can be overlapped with the next speculation step — if verification passes, no extra latency; if it fails, the fallback cost equals the baseline (no net overhead).
- **KNN-LM adaptation**: populate the cache with the next `n = 10` spatially adjacent corpus entries (spatial locality). Relax the verification criterion to matching the predicted next token rather than the full document set (critical for large `k`, e.g., `k = 1024`).
- **Hyperparameters**: max prompt 512 tokens, max generation 128 tokens, retrieved doc chunk ≤ 256 tokens, default `s = 3` without OS3, prefetch size 20.

## Key Results

- **Naive iterative RaLM (GPT-2/OPT/LLaMA-2-7B, exact dense retriever EDR)**: RaLMSpec+PSA achieves `2.39×` (GPT-2), `2.33×` (OPT), `1.75×` (LLaMA-2) speedup vs. sequential baseline.
- **Approximate dense retriever (ADR)**: `1.05×` (GPT-2), `1.39×` (OPT), `1.04×` (LLaMA-2) — lower gain because retrieval latency no longer dominates.
- **Sparse retriever (BM25)**: `1.53×` (GPT-2), `1.77×` (OPT), `1.31×` (LLaMA-2).
- **LLaMA-2-13B (EDR)**: up to `1.85×` speedup on Web Questions; ADR gains are marginal.
- **KNN-LM (EDR)**: up to `7.59×`; with approximate dense retriever up to `2.45×`.
- **Ablation**: OS3 contributes the largest individual gain; prefetching adds moderate improvement; asynchronous verification adds smaller but complementary gains. Prefetch size 20 outperforms 256 for ADR/SR (extra retrieval overhead dominates).
- Hardware: Oracle Cloud VM.GPU.A10 (1× NVIDIA A10, 15 CPUs).

## Limitations

- Speedup is bottlenecked by the fraction of retrieval latency in end-to-end latency: when generation dominates (ADR, faster retrievers, larger models), gains shrink.
- Python GIL prevents true asynchronous verification; results reported for this component use simulated (ideal) latency.
- KNN-LM evaluation omits asynchronous verification (left for future work).
- Speculation accuracy estimation assumes temporal stationarity of `γ`; adversarial or highly dynamic queries may violate this.
- Does not address multi-document cross-attention architectures (e.g., RETRO) where retrieval-model coupling is tighter.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[retrieval-augmented-language-model]]
- [[iterative-retrieval-augmented-generation]]
- [[speculative-retrieval]]
- [[batched-verification]]
- [[speculation-stride-scheduling]]
- [[speculative-decoding]]
- [[knn-language-model]]
- [[dense-retrieval]]
- [[dense-passage-retrieval]]
- [[bm25]]
- [[sparse-retrieval]]
- [[approximate-nearest-neighbor-search]]
- [[open-domain-question-answering]]
- [[large-language-model]]

## Entities Extracted

- [[zhihao-zhang]]
- [[alan-zhu]]
- [[lijie-yang]]
- [[yihua-xu]]
- [[lanting-li]]
- [[phitchaya-mangpo-phothilimthana]]
- [[zhihao-jia]]
- [[carnegie-mellon-university]]
- [[google-deepmind]]
- [[faiss]]
- [[dpr]]
- [[pyserini]]
- [[llama-2]]
- [[gpt-2]]
- [[natural-questions]]
- [[triviaqa]]
- [[wikitext-103]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
