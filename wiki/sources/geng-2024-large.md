---
type: source
subtype: paper
title: "Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning"
slug: geng-2024-large
date: 2026-04-20
language: en
tags: [llm, code-summarization, in-context-learning, few-shot-learning, software-engineering]
processed: true
raw_file: raw/papers/geng-2024-large/paper.pdf
raw_md: raw/papers/geng-2024-large/paper.md
bibtex_file: raw/papers/geng-2024-large/paper.bib
possibly_outdated: false
authors:
  - Mingyang Geng
  - Shangwen Wang
  - Dezun Dong
  - Haotian Wang
  - Ge Li
  - Zhi Jin
  - Xiaoguang Mao
  - Xiangke Liao
year: 2024
venue: ICSE 2024
venue_type: conference
arxiv_id:
doi: 10.1145/3597503.3608134
url: https://dl.acm.org/doi/10.1145/3597503.3608134
citation_key: geng2024large
paper_type: method
read_status: unread
domain: llm
---

## Summary

This paper studies multi-intent code comment generation through few-shot in-context learning instead of supervised training. The authors argue that code-domain large language models already absorb mappings between code and heterogeneous real-world comments during pretraining, so the main challenge is to elicit the right intent with prompting. They instantiate this idea with Codex prompts that specify one of five intent categories, retrieve demonstrations by lexical or semantic similarity, and optionally rerank multiple sampled outputs against comments from similar code. On Funcom and TLC, vanilla 10-shot Codex surpasses the prior supervised baseline DOME, while semantic demonstration selection plus token-based reranking pushes average BLEU to `72.4` and `68.8`, establishing a much stronger multi-intent comment generation baseline.

## Problem & Motivation

Existing code comment generation systems mostly assume a one-to-one mapping from code to a single comment, but real developer comments often serve different intents such as explaining functionality, rationale, usage, implementation details, or properties. This mismatch reduces practical usefulness because maintainers frequently need different perspectives on the same method. The paper asks whether code-oriented large language models can generate intent-specific comments without task-specific fine-tuning, and whether prompt construction plus output post-processing can unlock that capability better than a supervised model trained on limited labeled data.

## Method

- **Task formulation**: multi-intent comment generation is cast as prompting a model with `P = {x_test + CD + NL}`, where `x_test` is the target method, `CD = {(x_i, y_i)}_{i=1}^n` is a set of demonstrations, and `NL` is an intent-specific natural-language instruction.
- **Intent-aware prompt template**: each prompt states the programming language and one target intent among `what`, `why`, `how-to-use`, `how-it-is-done`, and `property`; demonstrations are separated by `###`, and the prompt must satisfy `size(P) <= 8000` tokens because Codex's context window is limited.
- **Few-shot settings**: the study evaluates zero-shot, one-shot, `5`-shot, and `10`-shot prompting; `10` examples is the maximum because of context length constraints.
- **Base LLM**: experiments use OpenAI Codex `code-davinci-002` with temperature `0.5`; no fine-tuning is performed.
- **Token-based demonstration retrieval**: Java keywords are removed, identifiers are split by camel case or underscore, and subtokens are lowercased before computing Jaccard similarity `s_token = |tokens_target ∩ tokens_candidate| / |tokens_target ∪ tokens_candidate|`.
- **Semantic demonstration retrieval**: code snippets are embedded with the sentence-transformer model `st-codesearch-distilroberta-base`, and nearest demonstrations are selected by cosine similarity in embedding space.
- **Reranking strategy**: the model is sampled multiple times, then candidate comments are reranked against the comment attached to the most similar retrieved code snippet using either token overlap or semantic similarity.
- **Datasets**: evaluation uses the Java Funcom and TLC datasets with intent labels inherited from prior work; the `others` intent is excluded as ambiguous.
- **Scale and protocol**: Funcom contributes `1,175,696` train and `68,757` test examples across five intents, while TLC contributes `52,258` train and `4,236` test examples; each RQ1/RQ2 setting is repeated `100` times to estimate average performance, and RQ3 reranks over `100` sampled runs.

## Key Results

- Vanilla prompting improves steadily with more demonstrations: average Funcom BLEU rises from `21.2` (`0`-shot) to `33.4` (`10`-shot), and TLC BLEU rises from `18.8` to `27.2`.
- `10`-shot Codex beats DOME on average across both datasets: Funcom `33.4 / 76.1 / 24.1` vs. DOME `31.8 / 42.5 / 20.5` for BLEU / ROUGE-L / METEOR, and TLC `27.2 / 66.7 / 19.2` vs. `22.2 / 36.7 / 16.5`.
- Retrieval quality matters more than raw shot count in some settings: `1`-shot with token-based selection reaches average BLEU `39.2` on Funcom and `36.1` on TLC, already exceeding vanilla `10`-shot.
- Under `10`-shot prompting, semantic demonstration selection boosts average BLEU from `33.4` to `65.9` on Funcom and from `27.2` to `62.8` on TLC.
- The best configuration, `10`-shot + semantic selection + token reranking, reaches average BLEU / ROUGE-L / METEOR of `72.4 / 91.8 / 71.6` on Funcom and `68.8 / 86.3 / 68.6` on TLC.
- Human evaluation on `100` sampled methods shows the best variant also leads in perceived quality, scoring naturalness `4.3`, adequacy `4.1`, and usefulness `3.8` on a `5`-point Likert scale.

## Limitations

- The study only evaluates Java, so transfer to other programming languages is unverified.
- Codex may have seen overlapping open-source code during pretraining, so data leakage cannot be ruled out completely.
- Even with `100` repetitions, the reported numbers remain subject to sampling variance and retrieval randomness.
- The approach assumes access to a corpus from which similar demonstrations and reference comments can be retrieved.
- Performance remains weaker on the `how-it-is-done` intent than on easier intent categories such as `what` or `property`.

## Concepts Extracted

- [[large-language-model]]
- [[in-context-learning]]
- [[few-shot-learning]]
- [[prompt-engineering]]
- [[code-summarization]]
- [[multi-intent-comment-generation]]
- [[demonstration-selection]]
- [[reranking]]
- [[jaccard-similarity]]
- [[sentence-transformer]]
- [[cosine-similarity]]

## Entities Extracted

- [[mingyang-geng]]
- [[shangwen-wang]]
- [[dezun-dong]]
- [[haotian-wang]]
- [[ge-li]]
- [[zhi-jin]]
- [[xiaoguang-mao]]
- [[xiangke-liao]]
- [[national-university-of-defense-technology]]
- [[peking-university]]
- [[openai-codex]]
- [[dome]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
