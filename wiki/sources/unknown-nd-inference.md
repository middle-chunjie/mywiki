---
type: source
subtype: paper
title: Inference Scaling for Long-Context Retrieval Augmented Generation
slug: unknown-nd-inference
date: 2026-04-20
language: en
tags: [rag, long-context, inference-scaling, retrieval, question-answering]
processed: true

raw_file: raw/papers/unknown-nd-inference/paper.pdf
raw_md: raw/papers/unknown-nd-inference/paper.md
bibtex_file: raw/papers/unknown-nd-inference/paper.bib
possibly_outdated: false

authors:
  - Zhenrui Yue
  - Honglei Zhuang
  - Aijun Bai
  - Kai Hui
  - Rolf Jagerman
  - Hansi Zeng
  - Zhen Qin
  - Dong Wang
  - Xuanhui Wang
  - Michael Bendersky
year: 2025
venue: ICLR 2025
venue_type: conference
arxiv_id:
doi:
url: https://openreview.net/pdf?id=FSjIrOm1vz
citation_key: unknownndinference
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

This ICLR 2025 paper studies how to scale retrieval-augmented generation at inference time when using long-context LLMs. Rather than only increasing the number of retrieved documents, it introduces demonstration-based RAG (DRAG) and iterative demonstration-based RAG (IterDRAG), which combine retrieved evidence, in-context examples, and iterative sub-query generation to use test-time compute more effectively. The paper defines effective context length as the total input tokens consumed across all inference calls, then shows that optimally configured long-context RAG improves almost linearly with the order of magnitude of that budget. It also proposes a computation allocation model that predicts strong parameter settings for different compute constraints, turning inference scaling for RAG from ad hoc prompt tuning into a more systematic optimization problem.

## Problem & Motivation

Long-context LLMs can accept much more retrieved evidence, but simply stuffing more documents into the prompt often plateaus or even hurts performance because the model struggles to identify the useful evidence in noisy ultra-long contexts. The paper asks two linked questions: how much knowledge-intensive QA performance can improve when test-time compute is scaled optimally, and whether the best allocation of that compute across retrieval, demonstrations, and iterative reasoning can be predicted instead of found by exhaustive search.

## Method

- **Inference budget**: define effective context length as the total number of input tokens across all inference calls before the final answer; for one-shot methods this is prompt length, and for iterative methods it is the sum over iterations.
- **Inference parameters**: optimize over `theta = (k, m, n)`, where `k` is the number of retrieved documents, `m` is the number of in-context demonstrations, and `n` is the number of retrieval-generation iterations.
- **Optimal-performance objective**: for a budget `L_max`, estimate `P*(L_max) = max_theta (1 / |X|) sum_i P(y_i, f(x_i; theta))` subject to `l(x_i; theta) <= L_max` for all examples.
- **DRAG**: prepend multiple demonstrations that each contain retrieved documents, question, and answer; retrieve top-`k` documents for both demonstrations and test queries; reverse document order so higher-ranked passages are placed closer to the query.
- **IterDRAG**: extend DRAG with iterative query decomposition and interleaved retrieval; each step generates either a sub-query, an intermediate answer, or the final answer, and the system allows up to `5` iterations before forcing the final answer.
- **Constrained generation**: use Self-Ask style constrained decoding so each iterative response begins with `Follow up:`, `Intermediate answer:`, or `So the final answer is:`.
- **Retrieval setup**: index Wikipedia passages from KILT with Gecko-1B embeddings; truncate each retrieved document to at most `1024` whitespace tokens.
- **Modeling compute allocation**: fit `sigma^-1(P(theta)) ~= (a + b odot i)^T log(theta) + c`, where `i = (i_doc, i_shot, 0)` captures task-specific informativeness of documents and shots; estimate parameters with ordinary least squares and a small `epsilon = 0.01` shift to avoid `log(0)`.
- **Scaling regime**: DRAG scales through larger `k` and `m`, while IterDRAG additionally scales through repeated retrieval-generation loops, letting total compute extend beyond a single context window.

## Key Results

- Scaling inference compute with optimized DRAG / IterDRAG yields up to `58.9%` gains over standard RAG on benchmark QA datasets, according to the abstract.
- The proposed computation allocation model achieves `R^2 = 0.903` and `MSE = 0.085`, and its full formulation outperforms ablations that drop informativeness terms or use simpler output mappings.
- In domain generalization at `1M` effective-context tokens, predicted configurations reach `96.6%` of oracle performance while substantially outperforming an `8`-shot baseline allocation.
- Iterative retrieval improves retrieval metrics over DRAG by an average of `30.5%`; on `2WikiMultiHopQA`, the gain reaches `57.1%`.
- With `k = 5` documents and `m = 4` shots, IterDRAG beats CoT on accuracy: HotpotQA `52.8` vs `45.6`, MuSiQue `25.9` vs `10.8`, and 2WikiMultiHopQA `72.3` vs `36.7`.

## Limitations

- The compute analysis excludes retrieval cost and output tokens, so the reported scaling laws target LLM input compute rather than full end-to-end serving cost.
- Gains diminish beyond roughly `1M` effective-context tokens, and extrapolation to `5M` is less reliable than shorter-range prediction.
- Retrieval quality remains a major bottleneck; recall can keep increasing while NDCG / MRR plateau, so adding more documents can introduce distraction instead of useful evidence.
- The empirical study is centered on Gemini 1.5 Flash plus benchmark QA datasets, so transfer to other model families, retrievers, and non-QA tasks is not fully established.
- IterDRAG depends on successful query decomposition and faithful intermediate reasoning; failure cases still include missing retrieval, incorrect reasoning, hallucination, and evaluation mismatches.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[long-context-modeling]]
- [[in-context-learning]]
- [[inference-time-scaling]]
- [[test-time-compute]]
- [[effective-context-length]]
- [[query-decomposition]]
- [[iterative-retrieval]]
- [[self-ask]]
- [[constrained-decoding]]
- [[multihop-question-answering]]
- [[compositionality-gap]]
- [[computation-allocation-model]]

## Entities Extracted

- [[zhenrui-yue]]
- [[honglei-zhuang]]
- [[aijun-bai]]
- [[kai-hui]]
- [[rolf-jagerman]]
- [[hansi-zeng]]
- [[zhen-qin]]
- [[dong-wang]]
- [[xuanhui-wang]]
- [[michael-bendersky]]
- [[google-deepmind]]
- [[university-of-illinois-urbana-champaign]]
- [[university-of-massachusetts-amherst]]
- [[gemini-1-5-flash]]
- [[gecko-1b]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
