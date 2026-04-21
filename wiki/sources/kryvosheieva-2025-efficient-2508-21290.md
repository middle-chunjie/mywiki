---
type: source
subtype: paper
title: Efficient Code Embeddings from Code Generation Models
slug: kryvosheieva-2025-efficient-2508-21290
date: 2026-04-20
language: en
tags: [code-embedding, code-retrieval, text-embedding, contrastive-learning, llm]
processed: true

raw_file: raw/papers/kryvosheieva-2025-efficient-2508-21290/paper.pdf
raw_md: raw/papers/kryvosheieva-2025-efficient-2508-21290/paper.md
bibtex_file: raw/papers/kryvosheieva-2025-efficient-2508-21290/paper.bib
possibly_outdated: false

authors:
  - Daria Kryvosheieva
  - Saba Sturua
  - Michael Günther
  - Scott Martens
  - Han Xiao
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2508.21290
doi: 10.48550/arXiv.2508.21290
url: http://arxiv.org/abs/2508.21290
citation_key: kryvosheieva2025efficient
paper_type: method

read_status: unread

domain: ir
---

## Summary

The paper introduces `jina-code-embeddings`, a pair of compact code embedding models built by adapting code-generation LLMs rather than encoder-style code retrievers. The `0.5B` and `1.5B` models start from Qwen2.5-Coder backbones, use last-token pooling over the final hidden states, add task-specific instruction prefixes for five retrieval settings, and optimize a contrastive objective with Matryoshka representation learning to support embedding truncation. Training mixes aligned code-text pairs from MTEB/CoIR-style resources, adapted public datasets, forum QA, and GPT-4o-generated synthetic data. On the reported evaluation suite, the models beat similarly sized general-purpose embedders and several much larger baselines on average, while staying competitive with specialized commercial systems on code search, TechQA, and cross-language retrieval.

## Problem & Motivation

The paper targets a data and architecture mismatch in code retrieval. Existing code embedders often rely on BERT-like encoders and relatively narrow aligned supervision such as docstrings or comments, which limits semantic coverage for real development workflows. Meanwhile, code generation models are already pretrained on large amounts of text and code, but that knowledge is rarely repurposed for retrieval. The authors therefore ask whether compact autoregressive code LLMs can be converted into strong embedding models for natural-language-to-code retrieval, technical QA, code-to-code matching, comment retrieval, and completion retrieval without requiring very large general-purpose embedding backbones.

## Method

- **Backbone and embedding readout**: initialize from Qwen2.5-Coder variants and convert the final hidden layer into embeddings with last-token pooling; the released models have `494M` and `1.54B` parameters.
- **Task decomposition**: define five retrieval regimes, `NL2Code`, `TechQA`, `Code2Code`, `Code2NL`, and `Code2Completion`, and prepend English task instructions separately to queries and documents.
- **Similarity training**: for a batch `B = {(q_i, d_i)}_{i=1}^n`, compute normalized query/document embeddings, form a cosine-similarity matrix `S_dense(B)`, and optimize an InfoNCE objective of the form `` `L_NCE = -Σ_i log exp(S_{i,i}/τ) / Σ_k exp(S_{i,k}/τ)` ``.
- **Hyperparameters**: keep temperature `` `τ = 0.05` ``, sequence length `` `512` ``, and batch size `` `n = 512` `` for the `0.5B` model and `` `n = 256` `` for the `1.5B` model.
- **Truncatable embeddings**: apply [[matryoshka-representation-learning]] during training so the same representation can be shortened at inference for cheaper retrieval.
- **Training data**: combine MTEB code tasks, CoSQA+, adapted public resources such as CodeSearchNet/CommitPackFT/SWE-Bench-family datasets, forum QA, and GPT-4o synthetic data validated by manual spot checks.
- **Optimization budget**: train for `` `1500` `` steps on `` `4 × 80GB A100` `` GPUs; reported wall-clock time is about `` `8.3 h` `` for `0.5B` and `` `12 h` `` for `1.5B`.
- **Pooling ablation**: compare last-token pooling against mean pooling and latent attention pooling under the same data and training schedule.

## Key Results

- On the full evaluation suite, `JCE-0.5B` / `JCE-1.5B` reach `78.41%` / `79.04%` overall average, versus `74.11%` for jina-embeddings-v4 and `73.49%` for Qwen3-Embedding-0.6B; only voyage-code-3 is slightly higher at `79.23%`.
- On `MTEB Code AVG`, the two models score `78.72%` and `78.94%`, beating Qwen3-Embedding-0.6B (`74.69%`), jina-embeddings-v4 (`74.87%`), and gemini-embedding-001 (`76.48%`), while remaining below voyage-code-3 (`79.84%`).
- The `1.5B` model is especially strong on `CodeSearchNetRetrieval` (`91.38%`), `CodeEditSearchRetrieval` (`84.43%`), `CommitPackFT` (`65.06%`), `AppsRetrieval` (`86.63%`), and `HumanEval` (`98.41%`).
- The compact models also post strong structured-query results, including `WikiSQL = 98.31%` (`0.5B`) and `98.02%` (`1.5B`), and outperform larger general embedders on `MLQuestions` with `81.05%` / `77.46%`.
- In the pooling ablation, last-token pooling yields the best averages: `78.41%` overall and `78.72%` on MTEB Code, versus `77.20%` / `77.18%` for mean pooling and `78.27%` / `78.41%` for latent attention.

## Limitations

- The gains are not universal: voyage-code-3 and gemini-embedding-001 remain stronger on several tasks such as `MBPP`, `AppsRetrieval`, `StackOverflowQA`, `DS-1000`, and `CodeChefXLang`.
- The paper depends partly on GPT-4o-generated synthetic data, and validation is described only as manual inspection of samples rather than systematic error analysis.
- Training uses sequence length `` `512` ``, so long-context code retrieval behavior is not directly stress-tested despite evaluation at longer inference lengths for some baselines.
- Some benchmark scores for large API models are taken from public MTEB sources instead of fully reproduced in-house, which weakens apples-to-apples comparability.
- The paper emphasizes retrieval quality, but gives little analysis of indexing cost, latency, memory footprint after truncation, or downstream RAG impact in real IDE pipelines.

## Concepts Extracted

- [[code-embedding]]
- [[last-token-pooling]]
- [[text-embedding]]
- [[code-search]]
- [[query-code-matching]]
- [[contrastive-learning]]
- [[infonce-loss]]
- [[instruction-tuning]]
- [[decoder-only-language-model]]
- [[matryoshka-representation-learning]]
- [[cross-lingual-retrieval]]
- [[cosine-similarity]]

## Entities Extracted

- [[daria-kryvosheieva]]
- [[saba-sturua]]
- [[michael-gunther]]
- [[scott-martens]]
- [[han-xiao]]
- [[massachusetts-institute-of-technology]]
- [[jina-ai]]
- [[qwen2-5-coder]]
- [[mteb]]
- [[codesearchnet]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
