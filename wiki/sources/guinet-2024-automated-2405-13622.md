---
type: source
subtype: paper
title: Automated Evaluation of Retrieval-Augmented Language Models with Task-Specific Exam Generation
slug: guinet-2024-automated-2405-13622
date: 2026-04-20
language: en
tags: [rag, evaluation, irt, retrieval, llm]
processed: true

raw_file: raw/papers/guinet-2024-automated-2405-13622/paper.pdf
raw_md: raw/papers/guinet-2024-automated-2405-13622/paper.md
bibtex_file: raw/papers/guinet-2024-automated-2405-13622/paper.bib
possibly_outdated: false

authors:
  - Gauthier Guinet
  - Behrooz Omidvar-Tehrani
  - Anoop Deoras
  - Laurent Callot
year: 2024
venue: ICML 2024
venue_type: conference
arxiv_id: 2405.13622
doi: 10.48550/arXiv.2405.13622
url: http://arxiv.org/abs/2405.13622
citation_key: guinet2024automated
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper proposes an automated evaluation pipeline for task-specific retrieval-augmented generation by turning each knowledge corpus into a synthetic multiple-choice exam and scoring RAG systems on the resulting questions. Instead of relying on human annotation or generic benchmarks, the framework generates candidate questions from task documents, filters low-quality or leaky items, and then estimates both model ability and question quality with a hierarchical Item Response Theory model. The benchmark spans four corpora, including ArXiv abstracts, StackExchange QA, AWS DevOps guides, and SEC filings. Across these settings, the method shows that retriever choice often matters more than LLM scale, while the IRT layer provides a more interpretable and robust decomposition of performance into LLM, retrieval, and in-context learning effects.

## Problem & Motivation

RAG systems are hard to evaluate when the target workload is narrow, domain-specific, or grounded in proprietary corpora. Public LLM benchmarks do not reliably predict performance on such tasks, while human evaluation is expensive, slow, and difficult to standardize across many pipeline variants. The authors therefore target an automated evaluation procedure that is predictive enough to rank candidate RAG systems on a specific task and prescriptive enough to explain which subsystem, such as the base LLM, retrieval strategy, or prompting setup, is actually limiting task performance.

## Method

- **Task setup**: each task `t in T` is an open-domain QA problem backed by a task-specific corpus. The benchmark uses four corpora: `1249` AWS DevOps webpages, `13000` ArXiv abstracts, `977` StackExchange questions, and `493` SEC filing sections.
- **Synthetic exam generation**: for each source document, an LLM proposes multiple-choice question candidates with exactly one correct option. Appendix A states the generator is `LlamaV2-70B`, after comparing against `LlamaV2-13B`, `ClaudeInstant`, and `ClaudeV2`.
- **Parsing and self-containment filters**: malformed generations are removed with regex parsing; questions that explicitly mention the originating "paper", "article", or other documentation cues are discarded so the exam remains answerable without source leakage.
- **Candidate shuffling**: answer options are randomly permuted to keep trivial answer-position priors near chance, with the fixed-answer baseline staying around `25%`.
- **Discriminator-quality filtering**: the paper removes questions whose distractors are too close to either the evidence document or the correct answer, using Jaccard and embedding similarity constraints such as `J(k, c) + t1 < max_i J(k, d_i)` and `max_i J_n(c, d_i) >= t3`, plus analogous embedding checks `S(...)`.
- **Pointwise grading**: each RAG pipeline is treated as a student and chooses the option with maximum length-penalized log-likelihood; the raw score is the fraction of correctly answered questions.
- **IRT question model**: each question `q_i` is parameterized by guessing `g_i`, discrimination `d_i`, and difficulty `b_i`, with correctness probability `P(X = 1 | theta, g_i, d_i, b_i) = g_i + (1 - g_i) / (1 + exp(-d_i(theta - b_i)))`.
- **Hierarchical IRT decomposition**: model ability is decomposed as `theta_m = theta_llm(m) + theta_ret(m) + theta_icl(m)`, so the framework can estimate separate latent contributions from the base LLM, retriever, and in-context demonstration regime.
- **Parameter estimation**: the model maximizes `L = sum_{m, i} r_{i,m} log p_i(theta) + (1 - r_{i,m}) log(1 - p_i(theta))` with L-BFGS-B. Initialization is `theta = 0`, discrimination `a_i = 1`, difficulty `b_i = 0`, guessing `c_i = 0.25`, with bounds `0.1 <= a_i <= 1.5`, `0.01 <= b_i <= 1`, `0.2 <= c_i <= 0.4`, and `-3 <= theta <= 3`.
- **Exam informativeness**: question utility is measured by the item information function `I(theta | g_i, d_i, b_i) = d_i^2 * ((p_i(theta) - g_i)^2 / (1 - g_i)^2) * ((1 - p_i(theta)) / p_i(theta))`, and aggregated as `Ibar_R(theta) = (1 / |R|) sum_i I(theta | g_i, d_i, b_i)`.
- **Iterative exam improvement**: Algorithm 1 alternates IRT fitting with exam pruning, dropping the least discriminative `r`-quantile of questions with `r ≈ 10%` to produce increasingly informative exams.

## Key Results

- **Generated exam sizes**: after filtering, the surviving exams contain `275` DevOps questions, `381` ArXiv questions, `148` StackExchange questions, and `515` SEC questions; ArXiv and StackExchange also incur large parse failures (`119` and `143`, respectively), showing why filtering matters.
- **Best raw accuracies**: the strongest scores in Table 2 are `72.6` on DevOps with `LlamaV2-70B + MultiQA` or `Oracle`, `75.7` on StackExchange with `LlamaV2-70B + MultiQA`, `77.7` on ArXiv with `LlamaV2-70B + DPRV2`, and `77.1` on SEC with `LlamaV2-70B + Oracle`.
- **Retriever choice can dominate model scaling**: on SEC, hierarchical IRT assigns retrieval abilities from `-1.39` for `SIAM` to `0.22` for `DPRV2`, while LLM abilities range only from `-0.48` for `Mistral-7B` to `0.18` for `LlamaV2-70B`, supporting the claim that retrieval selection can matter more than larger models.
- **Dense vs sparse vs hybrid is task-dependent**: BM25 beats dense retrieval on some corpora, e.g. ArXiv ability `0.60` vs `0.62` for MultiQA but stronger raw `76.9` with `LlamaV2-70B`, while hybrid `DPRV2` is the most robust overall on ArXiv (`0.72`) and competitive across tasks.
- **ICL usually helps**: hierarchical IRT gives `ICL@0` negative ability on all tasks (`-0.54`, `-0.77`, `-0.11`, `-0.83`), whereas `ICL@1` and `ICL@2` are consistently positive, with the largest gains on ArXiv (`0.90` and `1.06`).
- **IRT improves fit over a naive baseline**: RMSE drops from `0.49` to `0.44` on DevOps, `0.47` to `0.42` on ArXiv, `0.48` to `0.43` on StackExchange, and `0.49` to `0.42` on SEC relative to a mean-prediction baseline.

## Limitations

- The evaluation target is mainly factual retrieval and answer selection; synthetic multiple-choice exams do not fully capture open-ended generation quality, nuanced faithfulness, or conversational utility.
- Meta-evaluation remains unresolved: the paper argues its framework is interpretable and actionable, but it does not establish a definitive external gold standard for judging whether the exam itself is the best possible evaluator.
- The pipeline depends on LLM-generated questions plus heuristic parsing and filtering; Appendix A reports substantial failure counts before filtering, especially on StackExchange and ArXiv, so exam quality is still generation-dependent.
- The benchmark covers only four corpora and a relatively small family of 2023-era LLMs and retrievers, leaving multilingual settings, more recent models, and richer RAG design axes such as query reformulation or post-generation verification to future work.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[item-response-theory]]
- [[synthetic-exam-generation]]
- [[question-quality-filtering]]
- [[exam-informativeness]]
- [[dense-retrieval]]
- [[hybrid-retrieval]]
- [[in-context-learning]]
- [[cross-encoder]]
- [[bloom-taxonomy]]

## Entities Extracted

- [[gauthier-guinet]]
- [[behrooz-omidvar-tehrani]]
- [[anoop-deoras]]
- [[laurent-callot]]
- [[amazon-science]]
- [[auto-rag-eval]]
- [[bm25]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
