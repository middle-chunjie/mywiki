---
type: source
subtype: paper
title: "Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP"
slug: khattab-2023-demonstratesearchpredict-2212-14024
date: 2026-04-20
language: en
tags: [retrieval, in-context-learning, question-answering, multi-hop, conversational-qa]
processed: true
raw_file: raw/papers/khattab-2023-demonstratesearchpredict-2212-14024/paper.pdf
raw_md: raw/papers/khattab-2023-demonstratesearchpredict-2212-14024/paper.md
bibtex_file: raw/papers/khattab-2023-demonstratesearchpredict-2212-14024/paper.bib
possibly_outdated: true
authors:
  - Omar Khattab
  - Keshav Santhanam
  - Xiang Lisa Li
  - David Hall
  - Percy Liang
  - Christopher Potts
  - Matei Zaharia
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2212.14024
doi:
url: http://arxiv.org/abs/2212.14024
citation_key: khattab2023demonstratesearchpredict
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper proposes DEMONSTRATE-SEARCH-PREDICT (DSP), a framework for composing frozen language models and retrieval models as explicit natural-language programs for knowledge-intensive NLP. Instead of a rigid retrieve-then-read pipeline, DSP structures inference into three stages: bootstrapping demonstrations from end-task labels, iteratively searching with LM-generated queries, and making grounded predictions over retrieved evidence. Implemented as a Python library over GPT-3.5 and ColBERTv2, DSP supports open-domain QA, multi-hop QA, and conversational QA without fine-tuning either component. Across Open-SQuAD, HotPotQA, and QReCC, the authors report consistent gains over vanilla in-context learning, retrieve-then-read baselines, and self-ask, arguing that deliberate control flow and pipeline-aware demonstrations unlock a broader design space for retrieval-augmented in-context learning.

## Problem & Motivation

Simple retrieval-augmented prompting usually treats the retriever as a one-shot preprocessor: retrieve passages, append them to the prompt, and let the LM answer. The paper argues that this is too rigid for knowledge-intensive tasks where queries must be decomposed, conversational context must be resolved, or evidence must be accumulated across hops. The motivation is to treat both the language model and retrieval model as reusable frozen modules that exchange natural-language intermediate states, so developers can build task-aware pipelines with lower annotation and deployment cost. DSP is designed to make these pipelines explicit, modular, and bootstrappable from only end-task supervision rather than manually labeled intermediate queries or rewrites.

## Method

- **Overall abstraction**: DSP programs pass `Example` objects through three stages, `x -> demonstrate(x) -> search(x) -> predict(x)`, where each transformation reads and writes textual fields such as questions, demonstrations, hop summaries, queries, passages, rationales, and answers.
- **Foundation modules**: the LM is a frozen generator/scorer and the RM is a frozen top-`k` retriever over a large passage index. In experiments, the LM is GPT-3.5 (`text-davinci-002`) and the RM is CoLBERTv2.
- **DEMONSTRATE stage**: `annotate(train, fn)` runs a zero-shot or weakly supervised attempt function over training examples, caches intermediate fields when the final answer is correct, and converts successful traces into demonstrations. In the multi-hop case, the paper uses logic equivalent to `return d if d_pred == d_answer else None`.
- **Demonstration selection**: DSP exposes `sample(train, k)`, `knn(train, cast)`, and `crossval(train, n, k)` so prompts can use random subsets, RM-nearest demonstrations, or cross-validated prompt sets. The paper explicitly discusses settings like `|train| = 100`, `k = 5` and larger-scale `|train| = 1000` or `100000`.
- **SEARCH stage**: a simple baseline issues `retrieve(query=x.question, k=2)`, but the main DSP design generates intermediate queries with the LM. In `multihop_search_v2`, the system loops for up to `max_hops = 3`, generates `(summary, query)` pairs, stops early if `query == 'N/A'`, retrieves `k = 5` passages per hop, and carries the accumulated summary into later hops.
- **Fused retrieval**: for recall and robustness, DSP can generate multiple queries and merge result lists with a CombSUM-style fusion rule. The paper illustrates `n = 10` generated queries per hop, selecting the summary with the highest average log-probability while fusing retrieved passages.
- **PREDICT stage**: a `Template` maps fields in an `Example` to a prompt and parses completions back into structured outputs. The default QA template asks for a rationale and answer, enabling chain-of-thought plus answer extraction.
- **Candidate aggregation**: DSP supports majority voting over sampled completions, i.e. self-consistency, and also pipeline-of-thought branching where multiple full program traces are sampled with `n = 5` and `t = 0.7` before taking the majority answer.
- **Task-specific settings**: open-domain QA retrieves `k = 7` passages, generates `n = 20` reasoning chains, and annotates `k = 3` successful demonstrations out of `16` sampled training examples. The HotPotQA multi-hop program fixes the number of hops to `2`, uses fused retrieval with `n = 10` queries per hop, and distributes a total of `k = 5` passages across hops. Conversational QA generates `n = 10` rewritten queries, conditions on `5` retrieved passages, and uses `4` sampled conversations for demonstrations.
- **Evaluation protocol**: unless noted otherwise, each program receives up to `16` training examples, validation/test sets are subsampled to `1000` questions or `400` conversations, and results are averaged over `5` seeds.

## Key Results

- Open-SQuAD: DSP reaches `36.6` EM and `49.0` F1, beating vanilla LM (`16.2` EM, `25.6` F1) and retrieve-then-read (`33.8` EM, `46.1` F1).
- HotPotQA: DSP achieves `51.4` EM and `62.9` F1 versus vanilla LM at `28.3` EM / `36.4` F1 and retrieve-then-read at `36.9` EM / `46.1` F1.
- QReCC: DSP obtains `35.0` F1 and `25.3` nF1, improving over vanilla LM (`29.8` F1, `18.4` nF1) and retrieve-then-read (`31.6` F1, `22.2` nF1).
- Relative gains reported by the authors span `37-120%` over vanilla LM, `8-39%` over retrieve-then-read, and `80-290%` over self-ask, depending on task and metric.
- The self-ask baseline performs especially poorly in this setup: `9.3` EM / `17.2` F1 on Open-SQuAD and `25.2` EM / `33.2` F1 on HotPotQA, far below DSP.

## Limitations

- The paper is an early arXiv report and evaluates only a small set of development tasks with one main LM/RM combination, so generalization across models, corpora, and domains is not fully established.
- DSP still relies on manual program design and task-specific prompt engineering; the framework is modular, but the actual pipelines are hand-authored rather than automatically discovered.
- Many reported gains come from development splits and subsampled evaluation budgets (`1000` questions or `400` conversations), while several held-out tasks such as Open-NaturalQuestions and FEVER are deferred to future versions.
- The approach remains bounded by LM context window and API cost, especially when using `16` demonstrations, multiple retrieved passages, `n = 20` self-consistency samples, or multi-query fusion.
- Evidence quality depends heavily on the retriever and corpus alignment; the paper simulates time alignment with different Wikipedia dumps, but does not solve freshness, domain shift, or retrieval failure in a principled way.

## Concepts Extracted

- [[in-context-learning]]
- [[retrieval-augmented-language-model]]
- [[demonstration-selection]]
- [[few-shot-learning]]
- [[few-shot-prompting]]
- [[dense-retrieval]]
- [[multi-hop-reasoning]]
- [[conversational-search]]
- [[query-rewriting]]
- [[self-consistency]]
- [[chain-of-thought]]
- [[retrieval-augmented-generation]]

## Entities Extracted

- [[omar-khattab]]
- [[keshav-santhanam]]
- [[xiang-lisa-li]]
- [[david-hall]]
- [[percy-liang]]
- [[christopher-potts]]
- [[matei-zaharia]]
- [[stanford-university]]
- [[colbertv2]]
- [[gpt-3-5]]
- [[hotpotqa]]
- [[qrecc]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
