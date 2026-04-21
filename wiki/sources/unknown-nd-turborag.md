---
type: source
subtype: paper
title: "TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text"
slug: unknown-nd-turborag
date: 2026-04-20
language: en
tags: [rag, retrieval, kv-cache, llm-inference, latency]
processed: true

raw_file: raw/papers/unknown-nd-turborag/paper.pdf
raw_md: raw/papers/unknown-nd-turborag/paper.md
bibtex_file: raw/papers/unknown-nd-turborag/paper.bib
possibly_outdated: false

authors:
  - Songshuo Lu
  - Hua Wang
  - Yutian Rong
  - Zhi Chen
  - Yaohua Tang
year: 2025
venue: EMNLP 2025
venue_type: conference
arxiv_id:
doi:
url: https://openreview.net/forum?id=x7NbaU8RSU
citation_key: unknownndturborag
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

TurboRAG redesigns retrieval-augmented generation by moving document-side KV-cache computation offline and reusing those cached states during online inference. The core idea is to encode each retrieved chunk once, store its key-value cache, then assemble a request-time context by concatenating cached chunks rather than recomputing prefill over a long augmented prompt. To make this work without severe accuracy loss, the paper introduces independent attention across retrieved chunks, analyzes RoPE under cache concatenation, and advocates reordered position IDs, followed by supervised fine-tuning on the modified masking/position regime. On LongBench multi-document QA, TurboRAG-reordered reports comparable answer quality to naive RAG while cutting TTFT by `8.6x` on average and up to `9.4x`, with `98.46%` lower online TFLOPs in the reported batch-scaling study.

## Problem & Motivation

Standard RAG repeatedly concatenates retrieved chunks with the user query and recomputes document-side KV caches online, which makes prefill expensive and inflates time-to-first-token for long contexts. The paper argues that this cost is especially wasteful because many retrieved chunks recur across requests and cross-document attention is empirically sparse. The authors therefore target a system-level reformulation: preserve RAG's grounding benefits while eliminating redundant online computation and keeping answer quality close to the naive concatenation baseline.

## Method

- **Problem setup**: given query `q`, retrieve top-`k` chunks `[c_1, ..., c_k]` of equalized length `l`; naive RAG computes prefill on the concatenated sequence `[c_1, ..., c_k, q]` and stores the resulting document-side KV cache online.
- **Attention formalization**: per decoding step, keys and values follow `K_i = X_i^c W_K`, `V_i = X_i^c W_V`, and attention scores use `Q_t K_i^T / sqrt(d)`, so repeated long-context prefills dominate compute.
- **Independent attention**: TurboRAG masks cross-document attention among retrieved chunks, so each chunk attends only to itself during cached document encoding, while the query/answer tokens still attend over the assembled document cache at inference time.
- **Position-ID rearrangement**: instead of composite positions like `[0, ..., l, 0, ..., l]`, the reordered variant rewrites chunk positions to `[0, ..., l, l+1, ..., 2l, ...]` so RoPE preserves the intended relative offsets after cache concatenation.
- **RoPE compatibility insight**: because RoPE depends on relative position differences, retaining stored `K` and `V` while recomputing query-side positions with reordered indices makes offline chunk caches usable in the online assembled context.
- **Training adaptation**: the base LM is `Qwen2-7B`; SFT uses `50%` document-QA data and `50%` general tasks, `32` `NVIDIA A100 80GB` GPUs, batch size `256`, learning rate `1e-5`, and AdamW so the model learns the modified mask/position regime.
- **Inference pipeline**: offline, build document embeddings, index them in a vector database, and precompute per-chunk KV caches; online, encode the query, retrieve relevant chunks, fetch cached KV states, concatenate them, run query prefill, then decode normally.

## Key Results

- RGB benchmark, Chinese average accuracy: naive RAG `95.3`, TurboRAG-reordered `95.7`, TurboRAG-composite `94.8`; without fine-tuning the reordered/composite variants drop to `92.3` / `91.8`.
- RGB benchmark, English average accuracy: naive RAG `98.2`, TurboRAG-reordered `96.8`, TurboRAG-composite `96.7`; without fine-tuning they fall to `92.9` / `90.2`.
- LongBench multi-document QA average score: naive `32.99`, TurboRAG-composite `31.76`, TurboRAG-reordered `36.29`.
- LongBench TTFT: average `1165 ms` for naive RAG vs `134 ms` for TurboRAG-reordered, an `8.6x` speedup; the peak reported speedup is `9.4x` on MuSiQue (`1610 ms` to `171 ms`).
- General-capability regression on OpenCompass is small: MMLU `69.57 -> 70.73`, TriviaQA `56.90 -> 56.47`, GSM-8K `79.12 -> 79.45`, MATH `39.54 -> 40.58`.
- Batch-scaling study at recall length `8192` and query length `128`: TTFT improves from `711 ms` to `175 ms` at batch size `1` and from `5812 ms` to `1429 ms` at batch size `8`; reported online compute drops by about `98.46%`.

## Limitations

- The method depends on storing and serving large offline per-chunk KV caches; the paper emphasizes latency savings but gives limited analysis of cache storage overhead, refresh cost, and document-update handling.
- Accuracy degrades sharply without task-specific fine-tuning, with nearly `20%` drops under noisier RGB settings, so the approach is not plug-and-play for arbitrary pretrained LLMs.
- Independent attention assumes cross-document interactions are mostly unnecessary; tasks that genuinely require document-to-document reasoning could suffer more than the reported benchmarks reveal.
- Experiments are centered on a `Qwen2-7B` backbone and a small set of QA/regression benchmarks, so architecture-level generality beyond the tested setup remains partly unvalidated.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[key-value-cache]]
- [[independent-attention]]
- [[position-id-reordering]]
- [[rotary-positional-embedding]]
- [[attention-mask]]
- [[vector-database]]
- [[time-to-first-token]]
- [[large-language-model]]

## Entities Extracted

- [[songshuo-lu]]
- [[hua-wang]]
- [[yutian-rong]]
- [[zhi-chen]]
- [[yaohua-tang]]
- [[moore-threads-ai]]
- [[qwen2-7b]]
- [[gpt-4o]]
- [[longbench]]
- [[nvidia-a100]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
