---
type: source
subtype: paper
title: "CodeRAG-Bench: Can Retrieval Augment Code Generation?"
slug: wang-2024-coderagbench-2406-14497
date: 2026-04-20
language: en
tags: [retrieval, rag, code-generation, benchmark, software-engineering]
processed: true

raw_file: raw/papers/wang-2024-coderagbench-2406-14497/paper.pdf
raw_md: raw/papers/wang-2024-coderagbench-2406-14497/paper.md
bibtex_file: raw/papers/wang-2024-coderagbench-2406-14497/paper.bib
possibly_outdated: false

authors:
  - Zora Zhiruo Wang
  - Akari Asai
  - Xinyan Velocity Yu
  - Frank F. Xu
  - Yiqing Xie
  - Graham Neubig
  - Daniel Fried
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2406.14497
doi:
url: http://arxiv.org/abs/2406.14497
citation_key: wang2024coderagbench
paper_type: benchmark

read_status: unread
read_date:
rating:

domain: retrieval
---

## Summary

CodeRAG-Bench is a benchmark for retrieval-augmented code generation that unifies basic programming, open-domain coding, repository-level editing, and code retrieval under one evaluation stack. The benchmark aggregates roughly `9k` coding tasks, `25M` retrieval documents, and five retrieval sources: programming solutions, tutorials, library documentation, StackOverflow posts, and GitHub files. It also annotates canonical documents so retrieval quality and end-to-end generation can be analyzed separately. Across 10 retrievers and 10 generators, the paper shows that strong retrieved context can substantially improve execution-based code generation, especially for repository-level tasks and harder open-domain problems, while current retrievers still miss useful evidence and generators often fail to fully exploit retrieved context.

## Problem & Motivation

Existing code-generation evaluations mostly test NL-to-code generation without external evidence, even though realistic coding often depends on documentation, repositories, tutorials, and forum posts. Prior retrieval-augmented generation work is largely text-centric, while code tasks vary across basic algorithmic problems, library-heavy open-domain tasks, and repository-level editing. The paper aims to provide a unified benchmark that separates retrieval quality from generation quality, supports reproducible execution-based evaluation, and measures when retrieval truly helps code generation rather than being assumed beneficial by default.

## Method

- **Benchmark composition**: unify six code-generation tasks plus one code-retrieval task across four categories: HumanEval, MBPP, LiveCodeBench, DS-1000, ODEX, RepoEval, SWE-bench-Lite, and CodeSearchNet-Py; the benchmark covers roughly `9k` coding tasks and `25M` retrieval documents.
- **Retrieval corpora**: build five document sources with different granularity and provenance: programming solutions (`1.1k` docs, avg length `194.6`), tutorials (`79.4k`, `1502.5`), library docs (`34k`, `953.4`), StackOverflow posts (`23.5M`, `689.2`), and GitHub files (`1.7M`, `5135.4`).
- **Canonical annotation**: attach gold supporting documents per task type, including solution documents for HumanEval/MBPP, verified documentation entries for DS-1000 and ODEX (averaging `1.4` and `1.2` entries), `20`-line missing-function snippets for RepoEval, and gold edited files for SWE-bench.
- **Retrieval evaluation**: test `10` retrievers spanning sparse, dense, code-specific, and proprietary APIs; the primary metric is `NDCG@10`, with Precision and Recall also reported.
- **Generation evaluation**: test `10` code or general LMs; code correctness is measured with `pass@k`, and RACG is evaluated both with no retrieval and with gold or retrieved context.
- **Retrieval setup**: BM25 is implemented in `pyserini` with `k1 = 1.2` and `b = 0.75`; dense retrieval uses `sentence-transformers`; for generation, the system prepends the top-`5` retrieved documents and omits few-shot examples.
- **Decoding setup**: generation uses `t = 0.2`, `top_p = 0.95`, and one sampled response; SWE-bench-Lite additionally uses `n = 21` sampling plus majority-vote reranking following Agentless.
- **Open retrieval analysis**: beyond canonical corpora, the paper studies source-specific open retrieval and chunking strategies, comparing full-text retrieval, first-`N`-token truncation, reranked chunks, and pre-retrieval chunking for `N = 200 ... 1500`.

## Key Results

- **Best open retriever**: SFR-Mistral achieves the strongest average retrieval quality with `67.0` average `NDCG@10`, outperforming BM25's `57.7`; on SWE-bench-Lite it reaches `62.7`.
- **Canonical-context gains**: with gold documents, GPT-4o on SWE-bench-Lite improves from `2.3` to `30.7` pass@1, and on ODEX-hard from `20.7` to `27.6`.
- **Repository-level benefit**: on RepoEval, GPT-4o rises from `32.4` without retrieval to `62.2` with local retrieved context in the open-retrieval setting; StarCoder2 rises from `26.5` to `51.2` with OpenAI retrieval over local files.
- **Task dependence of retrieval**: BM25 remains very strong on repository-local code retrieval (`93.2` on RepoEval), while dense models dominate harder open-domain settings, e.g. Voyage-code gets `33.1` on DS-1000 and SFR-Mistral gets `37.1` on ODEX.
- **Open-source evidence is unevenly useful**: on HumanEval, StackOverflow and tutorial retrieval help more than documentation or GitHub; on ODEX, programming-solution retrieval yields `3.8-4.3` point gains and can outperform canonical documentation.
- **Chunking matters**: on HumanEval open retrieval, pre-retrieval chunking improves tutorial/doc/SO retrieval from `27.4/29.3/30.5` to `31.1/32.9/33.5`, while reranking chunks degrades to `9.1/9.1/14.0`.

## Limitations

- The benchmark is restricted to Python, so claims about RACG do not automatically transfer to other programming languages.
- Coverage is broad but still omits important scenarios such as code debugging and other software-engineering workflows beyond the included tasks.
- Most experiments use relatively vanilla retrieval, reranking, and generation pipelines; stronger backbones or more advanced RACG methods may change the reported conclusions.
- Retrieval quality remains weak on open-domain and repository-level tasks, and generation models still show limited context capacity and context-utilization ability even when useful documents are retrieved.
- Some gains in basic programming may come from near-solution evidence in tutorials or StackOverflow rather than deeper reasoning over genuinely novel external knowledge.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[code-generation]]
- [[benchmark-dataset]]
- [[code-search]]
- [[code-embedding]]
- [[execution-based-evaluation]]
- [[functional-correctness]]
- [[document-reranking]]
- [[chunking]]
- [[context-utilization]]
- [[data-contamination]]
- [[open-domain-code-generation]]

## Entities Extracted

- [[zhiruo-wang]]
- [[akari-asai]]
- [[xinyan-yu]]
- [[frank-f-xu]]
- [[yiqing-xie]]
- [[graham-neubig]]
- [[daniel-fried]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
