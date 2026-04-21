---
type: source
subtype: paper
title: Self-Knowledge Guided Retrieval Augmentation for Large Language Models
slug: wang-2023-selfknowledge-2310-05002
date: 2026-04-20
language: en
tags: [llm, retrieval, self-knowledge, question-answering, evaluation]
processed: true

raw_file: raw/papers/wang-2023-selfknowledge-2310-05002/paper.pdf
raw_md: raw/papers/wang-2023-selfknowledge-2310-05002/paper.md
bibtex_file: raw/papers/wang-2023-selfknowledge-2310-05002/paper.bib
possibly_outdated: true

authors:
  - Yile Wang
  - Peng Li
  - Maosong Sun
  - Yang Liu
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.05002
doi:
url: http://arxiv.org/abs/2310.05002
citation_key: wang2023selfknowledge
paper_type: method

read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper studies when retrieval helps large language models and proposes Self-Knowledge guided Retrieval augmentation (SKR), which tries to infer whether a model already knows how to answer a question before invoking external evidence. It first labels training questions by comparing answer quality with and without retrieval, then predicts whether a new question is "LLM-known" or "LLM-unknown" using prompting, in-context learning, a BERT-base classifier, or SimCSE-based `k`-nearest-neighbor search. Retrieval is applied only for predicted unknown questions. Across five QA benchmarks, the `SKR_knn` variant gives the strongest average results among the proposed variants, reaching `68.15` with InstructGPT and `70.62` with ChatGPT, showing that adaptive retrieval can outperform always-retrieve pipelines when retrieved context is sometimes distracting.

## Problem & Motivation

The paper starts from the observation that retrieval augmentation is not uniformly beneficial for large language models: relevant passages can still distract the model and degrade answers when the model's internal parametric knowledge is already sufficient. The central problem is therefore not only how to retrieve evidence, but how to decide whether retrieval should be used at all for a given question. The authors frame this as eliciting model self-knowledge: identifying what the model knows well enough to answer directly versus what requires external non-parametric knowledge.

## Method

- **Self-knowledge collection**: for each training question `q_i`, the model answers with few-shot ICL as `â(M, q_i) = M(q_1 ∘ a_1, ..., q_d ∘ a_d, q_i)` and again with retrieval augmentation as `â^R(M, q_i) = M(q_1 ∘ p_1 ∘ a_1, ..., q_d ∘ p_d ∘ a_d, q_i ∘ p_i)`.
- **Retriever setup**: retrieved evidence `p_i = {p_i1, ..., p_ik} = R(q_i, C)` comes from a dense passage retriever over Wikipedia passage chunks; in the final implementation they feed top-`3` passages to the LLM.
- **Label construction**: each training question is placed into `D+` if `E[â(M, q_i)] >= E[â^R(M, q_i)]` and into `D-` otherwise; questions where both variants are wrong are discarded. `D+` means the question is effectively LLM-known, `D-` means retrieval is beneficial.
- **Direct prompting**: ask the target model `Do you need additional information to answer this question?` and map `No` to LLM-known versus `Yes` to LLM-unknown.
- **In-context self-knowledge elicitation**: provide demonstrations sampled from both `D+` and `D-`, with templated answers `No, I don't need...` or `Yes, I need...`; the experiments use `4` CoT demonstrations in few-shot settings.
- **Classifier-based elicitation**: train a two-way BERT-base classifier over `D+ ∪ D-` with `ŷ_i = softmax(W h_cls(q_i) + b)` and cross-entropy supervision, where `h_cls(q_i)` is the sentence representation of question `q_i`.
- **kNN-based elicitation**: encode questions with a fixed sentence encoder such as SimCSE, compute cosine similarity `sim(q_t, q_i) = e(q_t) · e(q_i) / (||e(q_t)|| ||e(q_i)||)`, retrieve top-`k` neighbors, and predict positive when `l / (k - l) >= m / n`, where `l` is the number of positive neighbors, `k - l` negative neighbors, and `m`, `n` are the sizes of `D+`, `D-`; `k` is tuned in the range `3~10`.
- **Adaptive retrieval policy**: if the question is predicted LLM-known, answer directly from the few-shot prompt; if predicted LLM-unknown, prepend retrieved passages and then answer. Evaluated LLMs are InstructGPT (`text-davinci-003`) and ChatGPT (`gpt-3.5-turbo-0301`).

## Key Results

- On InstructGPT, `SKR_knn` achieves the best average score among SKR variants at `68.15`, versus `67.17` for `SKR_icl`, `66.80` for `SKR_cls`, and `65.77` for `SKR_prompt`.
- On ChatGPT, `SKR_knn` reaches `70.62`, again the strongest SKR variant and slightly above `SKR_cls` at `70.33` and `SKR_icl` at `69.32`.
- Dataset-level InstructGPT gains for `SKR_knn` include `48.00/58.47` on TemporalQA, `79.83` on TabularQA, `71.62` on StrategyQA, and `74.34` on TruthfulQA.
- Dataset-level ChatGPT gains for `SKR_knn` include `61.14/66.13` on TemporalQA, `76.75` on TabularQA, and `82.30` on TruthfulQA.
- Prompt-only self-knowledge is relatively weak: the paper reports about `70%~73%` accuracy on questions the model believed it could answer directly, implying roughly `30%` "unknown unknowns".
- The paper reports beneficial-guidance rates from `55%` on StrategyQA to `78%` on TruthfulQA for `SKR_knn`, showing that nearest-neighbor self-knowledge transfers across datasets better than direct prompting.

## Limitations

- The paper only probes self-knowledge through retrieval augmentation, so it does not separate memorization, understanding, and reasoning failures into finer categories.
- Evaluation is limited to five general QA benchmarks; the authors explicitly note that broader domains and more difficult boundary cases remain unexplored.
- The retrieval module itself is fairly simple: DPR over Wikipedia with top-`3` passages, without stronger evidence selection or retrieval-reasoning integration.
- Direct prompting and ICL variants are unstable, while the classifier depends on training data quality and cross-dataset transfer; even the best kNN strategy is only moderately above a `50%` uninformed choice baseline on harder datasets.
- Because the work is in fast-moving LLM/RAG territory and published in 2023, later tool-use and adaptive-RAG methods may supersede some design choices.

## Concepts Extracted

- [[self-knowledge]]
- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[dense-passage-retrieval]]
- [[in-context-learning]]
- [[chain-of-thought-prompting]]
- [[nearest-neighbor-search]]

## Entities Extracted

- [[yile-wang]]
- [[peng-li]]
- [[maosong-sun]]
- [[yang-liu]]
- [[tsinghua-university]]
- [[shanghai-artificial-intelligence-laboratory]]
- [[instructgpt]]
- [[chatgpt]]
- [[wikipedia]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
