---
type: source
subtype: paper
title: "SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation"
slug: unknown-nd-seakr
date: 2026-04-20
language: en
tags: [rag, adaptive-rag, uncertainty, question-answering, llm]
processed: true

raw_file: raw/papers/unknown-nd-seakr/paper.pdf
raw_md: raw/papers/unknown-nd-seakr/paper.md
bibtex_file: raw/papers/unknown-nd-seakr/paper.bib
possibly_outdated: false

authors:
  - Zijun Yao
  - Weijian Qi
  - Liangming Pan
  - Shulin Cao
  - Linmei Hu
  - Weichuan Liu
  - Lei Hou
  - Juanzi Li
year: 2025
venue: ACL 2025 (Long Papers)
venue_type: conference
arxiv_id:
doi:
url: https://openreview.net/pdf?id=NhIaRz9Qf5
citation_key: unknownndseakr
paper_type: method

read_status: unread

domain: llm
---

## Summary

SeaKR proposes an adaptive retrieval-augmented generation framework that uses a language model's internal-state uncertainty to decide when retrieval is necessary and how retrieved evidence should be integrated. Instead of relying on output probabilities or explicit self-reports, the method estimates uncertainty from the regularized Gram determinant of `EOS` hidden states sampled across multiple generations, then uses that score to trigger retrieval, re-rank candidate snippets, and choose between rationale-based and evidence-based final reasoning. Implemented with `LLaMA-2-Chat 7B`, `BM25`, `Elasticsearch`, and `vLLM`, SeaKR consistently improves over non-adaptive and adaptive baselines on both multi-hop and simple QA benchmarks, with particularly strong gains on 2WikiMultiHopQA and HotpotQA.

## Problem & Motivation

Standard retrieval-augmented generation retrieves evidence for every query, which is inefficient and can inject misleading or conflicting passages when the model already has sufficient parametric knowledge. Existing adaptive RAG methods mostly decide whether to retrieve from output tokens or explicit model self-reports, but those signals are vulnerable to hallucination and self-bias. SeaKR targets this gap by using internal-state self-awareness as a more faithful signal of knowledge insufficiency, and by extending adaptivity beyond retrieval triggering to evidence selection and final reasoning strategy choice.

## Method

- **Backbone setup**: uses `LLaMA-2-Chat 7B` as the generator and a `BM25` search engine implemented with `Elasticsearch`; the external corpus is an English Wikipedia dump.
- **Iterative reasoning loop**: for question `q`, SeaKR maintains rationale buffer `R = {r_i}` and knowledge buffer `K = {k_i}` while generating one rationale step at a time with chain-of-thought prompting.
- **Retrieval trigger**: builds context `c_r` from in-context examples, `q`, and prior rationales; retrieval is invoked only when self-aware uncertainty exceeds a threshold, i.e. `U(c_r) > δ`.
- **Query generation**: performs pseudo-generation `r_s = LLM(c_r)` and removes low-probability uncertain tokens from `r_s` to form the search query.
- **Self-aware re-ranking**: retrieves top `N = 3` snippets, evaluates each candidate snippet by the uncertainty of the resulting context, and keeps the snippet that minimizes `U(·)` rather than the snippet ranked highest by the search engine.
- **Self-aware reasoning**: after iterative retrieval halts, compares two answering strategies, one conditioned on generated rationales and one conditioned on concatenated retrieved evidence, and selects the answer with lower uncertainty.
- **Uncertainty estimator**: samples `k = 20` generations for the same context, extracts the `EOS` hidden state from the middle layer `l = L / 2`, computes a regularized Gram matrix over those states, and uses its determinant as the uncertainty score.
- **Prompting details**: uses `10` in-context examples; retrieval and re-ranking pseudo-generations stop at `.` and final answer generation stops at newline `\n`.

## Key Results

- On complex QA, SeaKR reaches `30.2 / 36.0` EM/F1 on `[[2wikimultihopqa]]`, `27.9 / 39.7` on `[[hotpotqa]]`, and `19.5 / 23.5` on `[[iirc]]`.
- Relative to the best reported baseline, SeaKR improves F1 by `+6.0` on 2WikiMultiHopQA, `+5.5` on HotpotQA, and `+0.6` on IIRC.
- On simple QA, SeaKR achieves `25.6 / 35.5` EM/F1 on NaturalQuestions, `54.4 / 63.1` on `[[triviaqa]]`, and `27.1 / 36.5` on `[[squad]]`.
- In ablations on sampled subsets, removing self-aware re-ranking lowers F1 to `35.0 / 36.6 / 35.0` on 2Wiki, HotpotQA, and NQ, while removing self-aware retrieval yields `35.7 / 37.6 / 35.8`, indicating that adaptive knowledge integration contributes at least as much as retrieval triggering.
- Swapping the backbone from `LLaMA-2-Chat 7B` to `LLaMA-3 8B Instruct` raises F1 to `48.1 / 47.7 / 43.0` on 2Wiki, HotpotQA, and NQ in the scaling study.

## Limitations

- SeaKR requires access to internal hidden states, so it is not directly applicable to closed commercial APIs that expose only output tokens.
- The evaluation focuses on short-form question answering and does not test long-form generation, creative writing, or broader task families.
- Computing the uncertainty score requires `20` pseudo-generations per decision point, making the method computationally expensive even with `vLLM` acceleration.
- The scaling study stops at models around `8B` parameters, so the paper does not establish behavior on much larger LLMs.
- The authors note that future advances in information retrieval could reduce the relative value of SeaKR's self-aware re-ranking stage.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[self-awareness]]
- [[hallucination]]
- [[large-language-model]]
- [[information-retrieval]]
- [[bm25]]
- [[multihop-question-answering]]
- [[open-domain-question-answering]]
- [[in-context-learning]]

## Entities Extracted

- [[zijun-yao]]
- [[weijian-qi]]
- [[liangming-pan]]
- [[shulin-cao]]
- [[linmei-hu]]
- [[weichuan-liu]]
- [[lei-hou]]
- [[juanzi-li]]
- [[tsinghua-university]]
- [[beijing-institute-of-technology]]
- [[llama-2-chat]]
- [[vllm]]
- [[elasticsearch]]
- [[wikipedia]]
- [[hotpotqa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
