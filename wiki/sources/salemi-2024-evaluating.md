---
type: source
subtype: paper
title: Evaluating Retrieval Quality in Retrieval-Augmented Generation
slug: salemi-2024-evaluating
date: 2026-04-20
language: en
tags: [rag, retrieval-evaluation, ir, llm-evaluation]
processed: true

raw_file: raw/papers/salemi-2024-evaluating/paper.pdf
raw_md: raw/papers/salemi-2024-evaluating/paper.md
bibtex_file: raw/papers/salemi-2024-evaluating/paper.bib
possibly_outdated: false

authors:
  - Alireza Salemi
  - Hamed Zamani
year: 2024
venue: SIGIR 2024
venue_type: conference
arxiv_id: 2404.13781
doi: 10.1145/3626772.3657957
url: https://dl.acm.org/doi/10.1145/3626772.3657957
citation_key: salemi2024evaluating
paper_type: method

read_status: unread

domain: ir
---

## Summary

This paper studies how to evaluate the retrieval component of a [[retrieval-augmented-generation]] system without repeatedly running expensive full end-to-end inference. The authors propose eRAG, which feeds each retrieved document to the downstream [[large-language-model]] individually, scores the resulting output against task ground truth, and treats that per-document downstream score as the document's relevance label. The method is evaluated on KILT tasks spanning [[question-answering]], [[fact-verification]], and dialogue generation, with BM25 and Contriever retrievers and T5-FiD generators. Across datasets, eRAG correlates much more strongly with downstream RAG quality than provenance labels or generic [[llm-as-a-judge]] annotations, while also reducing runtime and GPU memory substantially relative to end-to-end evaluation.

## Problem & Motivation

Standard end-to-end RAG evaluation only yields a list-level score, making it hard to understand which retrieved document actually helped generation. It is also expensive because the generator must consume the whole retrieved list each time the retriever output changes, which is especially problematic for interleaving-style comparison and retriever optimization. The paper further shows that conventional query-document relevance labels, including KILT provenance annotations and generic LLM-based judgments, correlate weakly with downstream RAG quality. The core motivation is therefore to obtain retrieval metrics that are both document-level and faithful to the actual downstream behavior of the generator that consumes retrieved evidence.

## Method

- **Core setup**: given retriever `R`, query `q`, retrieved list `R_k`, downstream model `M`, and task metric `E_M`, the standard RAG output is `y_hat = M(q, R_k)`.
- **eRAG document annotation**: each retrieved document `d in R_k` is fed to the same downstream model individually, and its relevance label is defined as `` `G_q[d] = E_M(M(q, {d}), y)` `` where `y` is the gold downstream target.
- **Aggregation into retrieval scores**: the per-document labels `G_q` are aggregated with standard [[retrieval-evaluation]] metrics, including Precision, Recall, MAP, MRR, NDCG, and Hit Ratio; for non-binary labels, Precision uses the average score and Hit Ratio uses the maximum score.
- **Complexity argument**: end-to-end transformer evaluation over `k` retrieved documents of average length `d` and output length `l` costs `` `O(l k^2 d^2)` ``, while eRAG costs `` `O(l k d^2)` `` because documents are scored independently.
- **Datasets and tasks**: experiments use KILT validation sets for [[natural-questions]], [[triviaqa]], [[hotpotqa]], [[fever]], and [[wizard-of-wikipedia]], covering open-domain QA, fact verification, and long-form dialogue generation.
- **Annotator and downstream models**: eRAG uses [[mistral-7b]] to generate relevance annotations, while the main downstream RAG model is T5-small with [[fusion-in-decoder]] over `k = 50` retrieved documents; a T5-base FiD variant and an in-prompt-augmentation baseline are used in ablations.
- **Training configuration**: the main FiD model is trained with AdamW, weight decay `` `1e-2` ``, learning rate `` `5e-5` ``, `10` epochs, linear warmup over the first `5%` of steps, and effective batch size `64` on A100 GPUs.
- **Retrieval backends**: [[bm25]] implemented in [[pyserini]] provides the sparse baseline, while [[contriever]] with [[faiss]] flat indexing provides the dense baseline; KILT Wikipedia is segmented into passages of at most `100` words.

## Key Results

- eRAG improves Kendall's `tau` correlation with downstream RAG performance by an absolute `0.168` to `0.494` over baselines across the evaluated datasets, matching the headline claim in the abstract.
- With BM25, the best eRAG Kendall `tau` reaches `0.529` on NQ, `0.486` on TriviaQA, `0.629` on HotpotQA, `0.592` on FEVER, and `0.504` on WoW; the corresponding KILT provenance best scores are much lower at `0.216`, `0.187`, `0.139`, `0.050`, and `0.019`.
- With Contriever, eRAG remains strongest, with best Kendall `tau` values of `0.522` on NQ, `0.482` on TriviaQA, `0.639` on HotpotQA, `0.481` on FEVER, and `0.540` on WoW.
- Generic LLM annotation is often weak or even negative: for example, the Mistral-7B judge baseline yields best Kendall `tau` of only `0.189` on BM25 TriviaQA, `0.021` on BM25 FEVER, and `-0.005` on BM25 WoW.
- eRAG is on average `2.468x` faster than end-to-end evaluation; runtime drops from `918` to `351` seconds on NQ and from `3395` to `1044` seconds on FEVER.
- GPU memory falls sharply: on NQ, end-to-end evaluation needs `75.0 GB`, versus `4.9 GB` for query-level eRAG and `1.5 GB` for document-level eRAG; the paper summarizes this as `7-15x` lower memory in query-level mode and `30-48x` lower memory in document-level mode.

## Limitations

- The study is limited to five KILT validation sets, because test-set labels are unavailable; it does not establish that the same correlations hold on broader RAG tasks or production settings.
- eRAG is cheaper than end-to-end evaluation but still requires running the downstream model once per retrieved document, so cost still scales linearly with `k`.
- The method defines relevance through the behavior of a particular downstream model and metric, so its labels may not transfer cleanly across different generators, prompts, or task metrics.
- Dialogue generation is only partially covered: for WoW, the paper reports Precision and Hit Ratio but not ranking metrics that require integer relevance labels.
- The empirical analysis focuses on BM25/Contriever retrieval and mostly T5-FiD style generators, so the conclusions for other retrievers or modern decoder-only RAG stacks remain inferential.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[retrieval-evaluation]]
- [[document-level-relevance]]
- [[llm-as-a-judge]]
- [[large-language-model]]
- [[fusion-in-decoder]]
- [[sparse-retrieval]]
- [[dense-retrieval]]
- [[question-answering]]
- [[fact-verification]]

## Entities Extracted

- [[alireza-salemi]]
- [[hamed-zamani]]
- [[university-of-massachusetts-amherst]]
- [[mistral-7b]]
- [[bm25]]
- [[pyserini]]
- [[contriever]]
- [[faiss]]
- [[natural-questions]]
- [[triviaqa]]
- [[hotpotqa]]
- [[fever]]
- [[wizard-of-wikipedia]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
