---
type: source
subtype: paper
title: Generative Representational Instruction Tuning
slug: muennighoff-2024-generative-2402-09906
date: 2026-04-20
language: en
tags: [llm, embeddings, retrieval, instruction-tuning, rag]
processed: true

raw_file: raw/papers/muennighoff-2024-generative-2402-09906/paper.pdf
raw_md: raw/papers/muennighoff-2024-generative-2402-09906/paper.md
bibtex_file: raw/papers/muennighoff-2024-generative-2402-09906/paper.bib
possibly_outdated: false

authors:
  - Niklas Muennighoff
  - Hongjin Su
  - Liang Wang
  - Nan Yang
  - Furu Wei
  - Tao Yu
  - Amanpreet Singh
  - Douwe Kiela
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.09906
doi:
url: http://arxiv.org/abs/2402.09906
citation_key: muennighoff2024generative
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper introduces GRIT, a unified fine-tuning recipe that makes one large language model perform both generation and text embedding well instead of specializing to only one side. GritLM switches behavior through instruction and format tokens, uses bidirectional attention plus mean pooling for representational tasks, and keeps causal decoding for generative tasks. The resulting GritLM 7B reaches a `66.8` average on MTEB while preserving a `55.5` average on generative benchmarks, and GritLM 8x7B pushes the generative average to `65.7` while remaining competitive on embedding. Beyond benchmark wins, the unification enables single-model reranking and retrieval-augmented generation caching, with reported long-document RAG latency reductions above `60%`.

## Problem & Motivation

Instruction-tuned LLMs are strong generators but typically poor embedders when their hidden states are used directly, while embedding-specialized models lose generative ability after contrastive fine-tuning. This split is operationally awkward for systems such as retrieval-augmented generation, where one often needs both a retriever and a generator. The paper asks whether embedding and generation are really separate model families or just different output interfaces for the same underlying language understanding. GRIT is proposed as a simple joint training recipe that preserves both capabilities, reduces duplicated inference passes in RAG, and simplifies serving compared with maintaining separate embedding and chat endpoints.

## Method

- **Unified training objective**: optimize embedding and generation jointly with `L_GRIT = lambda_Rep * L_Rep + lambda_Gen * L_Gen`, where `L_Rep` is a contrastive loss over query-document pairs and `L_Gen` is next-token language-modeling loss over assistant responses only.
- **Representational mode**: for embedding tasks, GritLM uses bidirectional attention over the input sample and applies mean pooling to the final hidden states, excluding instruction and format tokens from the pool while still letting them influence the representation through self-attention.
- **Generative mode**: for text generation, the same backbone uses causal attention and a language-modeling head; the paper studies token-level versus sample-level loss aggregation and keeps a mixed setting `Mix (32 -> 8)` with initial loss ratio `L_Rep / L_Gen = 4.1`.
- **Representation loss**: the embedding objective is InfoNCE-style with in-batch negatives, `L_Rep = -(1/M) * sum_i log exp(tau * sim(q_i, d_i)) / sum_j exp(tau * sim(q_i, d_j))`, using cosine similarity after pooling.
- **Backbones and data**: final models start from [[mistral-7b]] and [[mixtral-8x7b]], using adapted E5/E5S embedding data plus Tulu 2 generative data; the embedding recipe prefers E5-style data, same-dataset in-batch negatives, bidirectional attention, and mean pooling.
- **Training configuration**: GritLM 7B uses embedding batch size `2048`, generative batch size `256`, and `1253` steps; GritLM 8x7B reduces embedding batch size to `256` because of compute limits. Standard lengths are `256` tokens for embedding queries and `2048` for embedding documents and generative samples.
- **Optimization details**: training uses Adam with learning rate `2e-5`, `3%` linear warmup, linear decay to `0`, `beta1 = 0.9`, `beta2 = 0.999`, no weight decay, BF16 mixed precision, FP32 pooling/similarity computation, FlashAttention 2, FSDP, and gradient checkpointing.
- **System implications**: the unified model can act as both bi-encoder retrieval model and generative reranker, and it supports RAG caching variants such as query caching and [[document-caching]] because embedding and generation share the same parameters.

## Key Results

- [[gritlm]] 7B reaches `66.8` MTEB average, edging out E5 Mistral 7B at `66.6`, BGE Large at `64.2`, and OpenAI v3 embeddings at `64.6`.
- On generative evaluation, GritLM 7B averages `55.5`, slightly above its generative-only counterpart at `55.2` and well above Llama 2 70B at `46.4`; GritLM 8x7B reaches `65.7`, ahead of Mixtral 8x7B Instruct at `60.3` and Tulu 2 70B at `65.1`.
- The unified model matches single-objective variants closely: embedding-only 7B scores `66.8` on MTEB and generative-only 7B scores `55.2` on the generative average, supporting the paper's no-performance-loss claim.
- Using the same model for retrieval and reranking improves the average retrieval-stage score from `57.4` to `57.9` when reranking the top `10` documents, with gains on `15/16` datasets.
- In Natural Questions RAG, document caching improves match from `30.47` to `33.38` and cuts latency for a `4000`-token document from `14.18s` to `5.25s` on CPU and from `0.39s` to `0.27s` on GPU.
- Ablations show several stable choices: embedding batch size `4096` beats `256` by `+1.0` average embedding points (`64.2` vs `63.2` in the relevant setup), same-dataset in-batch negatives lift retrieval from `54.9` to `56.2`, and BF16 mixed precision remains close to FP32 (`66.5` vs `66.3` embedding average).

## Limitations

- The unified recipe still increases fine-tuning cost relative to generation-only instruction tuning because each step optimizes two objectives instead of one.
- GritLM 7B is expensive as an embedding model compared with lighter specialists: it emits `4096`-dimensional vectors, requiring about `4x` the storage of `1024`-dimensional models such as BGE Large.
- Some benefits depend on large training batches; GritLM 8x7B loses embedding quality partly because its embedding batch size had to drop from `2048` to `256`.
- RAG caching is not universally lossless: query caching drops Natural Questions match from `30.50` to `25.46`, and combined query-document caching variants fall close to the no-RAG baseline because of attention and formatting mismatch.
- Document caching trades latency for storage, requiring about `30TB` of key-value states for the paper's `2,681,468`-document index.
- Although the backbones use sliding-window attention and can accept arbitrary-length inputs, the paper explicitly notes that embedding quality beyond `512` evaluation tokens is still under-benchmarked.

## Concepts Extracted

- [[generative-representational-instruction-tuning]]
- [[representational-instruction-tuning]]
- [[instruction-tuning]]
- [[large-language-model]]
- [[text-embedding]]
- [[retrieval-augmented-generation]]
- [[bidirectional-attention]]
- [[mean-pooling]]
- [[contrastive-learning]]
- [[in-batch-negatives]]
- [[sliding-window-attention]]
- [[document-caching]]

## Entities Extracted

- [[niklas-muennighoff]]
- [[hongjin-su]]
- [[liang-wang-microsoft]]
- [[nan-yang]]
- [[furu-wei]]
- [[tao-yu]]
- [[amanpreet-singh-contextual-ai]]
- [[douwe-kiela]]
- [[contextual-ai]]
- [[university-of-hong-kong]]
- [[microsoft]]
- [[mistral-7b]]
- [[mixtral-8x7b]]
- [[mteb]]
- [[gritlm]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
