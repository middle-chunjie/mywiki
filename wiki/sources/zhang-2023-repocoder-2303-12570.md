---
type: source
subtype: paper
title: "RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation"
slug: zhang-2023-repocoder-2303-12570
date: 2026-04-20
language: en
tags: [code-completion, retrieval-augmented-generation, repository-level, code-generation, benchmark]
processed: true
raw_file: raw/papers/zhang-2023-repocoder-2303-12570/paper.pdf
raw_md: raw/papers/zhang-2023-repocoder-2303-12570/paper.md
bibtex_file: raw/papers/zhang-2023-repocoder-2303-12570/paper.bib
possibly_outdated: true
authors:
  - Fengji Zhang
  - Bei Chen
  - Yue Zhang
  - Jacky Keung
  - Jin Liu
  - Daoguang Zan
  - Yi Mao
  - Jian-Guang Lou
  - Weizhu Chen
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: "2303.12570"
doi:
url: http://arxiv.org/abs/2303.12570
citation_key: zhang2023repocoder
paper_type: method
read_status: unread
domain: nlp
---

## Summary

> ⚠️ **Possibly outdated** — published 2023 in the volatile LLM/code-completion domain; newer work (e.g., REPOFORMER, CrossCodeEval, DeepSeek-Coder) may supersede specific numbers.

RepoCoder proposes an iterative retrieval-generation framework for repository-level code completion. Given unfinished in-file code, a similarity-based retriever locates relevant snippets from the repository; a frozen code language model generates a completion; that completion is then fed back as an augmented retrieval query, and the cycle repeats. The paper also introduces RepoEval, the first benchmark covering line, API invocation, and function-body completion with unit-test-based evaluation. Experiments with GPT-3.5-Turbo and CodeGen (350M–6B) show RepoCoder improves over in-file baselines by more than 10% EM across all settings and consistently beats single-pass RAG.

## Problem & Motivation

Repository-level code completion requires leveraging context scattered across many files — shared utilities, naming conventions, custom API signatures — that the current file alone cannot provide. Prior work either uses brittle heuristics for prompt construction or fine-tunes on labeled repository data (expensive, cannot generalize). A naive RAG approach suffers from a *retrieval-target gap*: querying with unfinished code `X` is a poor proxy for the intended completion `Y`, so the most useful snippets (e.g., exact API call signatures) are not retrieved in the first pass. RepoCoder directly addresses this gap.

## Method

**Overall pipeline**: Given in-file unfinished code `X` and repository corpus `C_repo = {c_1, c_2, …}`:
- **Iteration 1**: retrieve `C_ret^1 = R(C_repo, X)` using last `S_w` lines of `X` as query; generate `Ŷ^1 = M(C_ret^1, X)`.
- **Iteration i (i > 1)**: construct query from last `(S_w - S_s)` lines of `X` concatenated with first `S_s` lines of `Ŷ^(i-1)`; retrieve `C_ret^i`; generate `Ŷ^i = M(C_ret^i, X)`.
- Parameters `M` and `R` are frozen throughout; no fine-tuning required.

**Retrieval**: sliding window with size `S_w` and step `S_s` chunks repository files into code snippets. Default retriever: sparse bag-of-words with **Jaccard similarity** `|S_q ∩ S_c| / |S_q ∪ S_c|`. Dense retriever (UniXcoder cosine similarity) also tested — performance comparable to sparse.

**Hyperparameters (RepoEval)**:
- GPT-3.5-Turbo prompt budget: `4096` tokens; CodeGen: `2048` tokens.
- Retrieved snippets fill half the prompt budget; max `K = 10` snippets.
- Line/API completion: `S_w = 20`, `S_s = 10`, max output `100` tokens.
- Function body: `S_w = 50`, `S_s = 10`, max output `500` tokens.
- Retrieved snippets ordered ascending by similarity score; each snippet prefixed with its file path.

**RepoEval benchmark**: 14 Python GitHub repositories (open-source, created after 2022-01-01, 100+ stars, >80% Python, with unit tests). Three tasks: line completion (1600 samples, 8 repos), API invocation completion (1600 samples, 8 repos), function body completion (373 samples, 6 repos, evaluated by unit-test Pass Rate).

## Key Results

- **Line completion, GPT-3.5-Turbo**: RepoCoder Iter-3 reaches `EM = 57.00%`, `ES = 75.30%` vs. In-File `EM = 40.56%`, `ES = 65.06%` (>16 pp EM improvement).
- **API invocation, GPT-3.5-Turbo**: RepoCoder Iter-4 reaches `EM = 49.56%` vs. In-File `EM = 34.06%` (>15 pp improvement).
- **Function body, GPT-3.5-Turbo**: Pass Rate `42.63%` (Iter-2) vs. In-File `23.32%` (>19 pp).
- Two iterations of RepoCoder consistently outperform single-pass RAG across all model sizes.
- Dense retriever (UniXcoder) performs comparably to the sparse Jaccard retriever on both tasks.
- Recall of ground-truth API invocation examples increases from Iter-1 to Iter-2, confirming the generation-augmented query mechanism works.
- Oracle upper bound for GPT-3.5-Turbo: `EM ≈ 57.75%` (line), showing RepoCoder Iter-3 nearly closes the gap.
- CodeGen-350M + RepoCoder matches GPT-3.5-Turbo In-File on line completion.

## Limitations

- **Low code-duplication repositories**: RepoCoder depends on finding similar snippets. Repositories with minimal code duplication yield smaller gains; "rl" and "vizier" repos show limited improvement.
- **Unstable multi-iteration behavior**: iterations beyond 2–3 do not monotonically improve performance; some cases regress between iterations due to misleading retrieved snippets (different API parameter sets across files).
- **Optimal stopping is unsolved**: no automatic criterion to determine when to stop iterating without performance degradation.
- **Latency**: each iteration adds a full retrieval and generation step, making real-time deployment costly; the paper notes quantization and caching as future mitigations.
- **Prompt engineering unexplored**: only one prompt template tested; better templates likely improve results.
- **Limited generation model diversity**: only GPT-3.5-Turbo and CodeGen tested; no experiments with GPT-4, StarCoder, or WizardCoder.

## Concepts Extracted

- [[repository-level-code-completion]]
- [[retrieval-augmented-generation]]
- [[iterative-retrieval]]
- [[code-completion]]
- [[code-language-model]]
- [[sparse-retrieval]]
- [[dense-retrieval]]
- [[sliding-window]]
- [[jaccard-similarity]]
- [[cross-file-context]]
- [[execution-based-evaluation]]

## Entities Extracted

- [[fengji-zhang-cityu]]
- [[bei-chen]]
- [[yue-zhang]]
- [[jacky-keung]]
- [[jin-liu]]
- [[daoguang-zan]]
- [[yi-mao]]
- [[jian-guang-lou]]
- [[weizhu-chen]]
- [[repocoder]]
- [[repoeval]]
- [[codegen]]
- [[unixcoder]]
- [[microsoft]]
- [[city-university-of-hong-kong]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
