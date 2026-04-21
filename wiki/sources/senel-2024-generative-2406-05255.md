---
type: source
subtype: paper
title: "Generative Explore-Exploit: Training-free Optimization of Generative Recommender Systems using LLM Optimizers"
slug: senel-2024-generative-2406-05255
date: 2026-04-20
language: en
tags: [recommender-systems, generative-recommendation, click-through-rate, llm, question-generation]
processed: true

raw_file: raw/papers/senel-2024-generative-2406-05255/paper.pdf
raw_md: raw/papers/senel-2024-generative-2406-05255/paper.md
bibtex_file: raw/papers/senel-2024-generative-2406-05255/paper.bib
possibly_outdated: false

authors:
  - Lütfi Kerem Senel
  - Besnik Fetahu
  - Davis Yoshida
  - Zhiyu Chen
  - Giuseppe Castellucci
  - Nikhita Vedula
  - Jason Choi
  - Shervin Malmasi
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2406.05255
doi:
url: http://arxiv.org/abs/2406.05255
citation_key: senel2024generative
paper_type: method

read_status: unread
read_date:
rating:

domain: ir
---

## Summary

The paper proposes a training-free framework for optimizing generative recommenders with an LLM-based optimizer that consumes historical question/CTR pairs and alternates exploration with exploitation. Instead of fine-tuning, the system iteratively updates a question pool for each topic by dropping low-CTR items and generating replacements. A two-stage user simulator first scores question-persona relevance with GPT-4 on a `1-10` scale and then converts those scores into clicks through a softmax choice model. Across e-commerce categories and Wikipedia articles, the Explore-Exploit variant consistently beats full-ctr, partial-ctr, random-ctr, and no-drop baselines, improving both relevance scores and click-through rate. The paper positions CTR-guided prompt optimization as a practical way to adapt open-set recommendation without model retraining.

## Problem & Motivation

The paper studies open-set recommendation settings where the system must generate candidate items rather than rank a fixed catalog. In such settings, user preferences are latent and only indirectly observable through engagement signals such as click-through rate. Fine-tuning a large language model for every topic or feedback update is too expensive, especially when the system must cover many contexts. The authors therefore ask whether an LLM can be optimized purely through prompting: using past generated questions plus observed CTRs to iteratively steer future generations toward the hidden preferences of a user population.

## Method

- **Task setup**: for a topic `t`, generate an item pool `IP` of `N = 5` questions that maximizes CTR over a user population `U = {u_1, ..., u_n}` with hidden preferences.
- **Initialization**: construct `IP_0` by prompting the LLM to generate `5` short questions for the topic with no user feedback available in the first round.
- **Iterative refinement loop**: at each iteration `i`, drop the `n = 1` worst question from `IP_i`, generate `1` replacement, and evaluate the updated pool; the main experiments run for `I = 15` iterations.
- **`full-ctr` optimizer**: provide the LLM with all previously generated questions and their observed CTR values, then prompt it to write a novel question likely to achieve high CTR without duplicating earlier questions.
- **Explore-Exploit optimizer**: split generation into an explore phase and an exploit phase; exploration sees the current `IP_i` without CTR values, while exploitation conditions on the best-performing question/topic pattern from `IP_i` to generate a high-CTR variant.
- **Relevance scoring**: for question `q_i`, persona `p_j`, and topic `t`, GPT-4 computes `r_{i,j} = QS(q_i, p_j, t)` on a discrete scale `r in {1, ..., 10}`; the simulator uses temperature `1` during scoring to induce within-persona variation.
- **Click simulation**: simulate `S = 5000` interactions per iteration by sampling a persona and `K = 3` questions, then compute click probability as `P(click | p_j, q_i) = exp(r_{i,j} / T) / (exp(RS / T) + Σ_k exp(r_{k,j} / T))` with `T = 1.5` and rejection score `RS = 11`.
- **Domains and personas**: evaluate on `50` Amazon product categories and `50` Wikipedia articles; e-commerce personas encode shopping preferences such as quality or ethical considerations, while general-knowledge personas focus on discussion, history, events, people, or locations.
- **LLM configuration**: all generation and simulation experiments use `gpt-4-1106-preview`; the method relies on in-context learning rather than parameter updates or reward-model training.

## Key Results

- **Relevance scoring is usable but noisy**: on the e-commerce evaluation, human annotators agree on the better question in `70.2%` of pairs (`132/188`), and GPT-4 aligns with that agreed preference in `77.3%` of cases (`102/132`).
- **Explore-Exploit improves recommendation quality most strongly**: from `IP_0` to `IP_15`, the paper reports more than `+2` relevance-score points and more than `+11` CTR points for single-persona e-commerce populations, plus more than `+7` CTR points for the `3`-persona setting.
- **Concrete CTR gains are large in harder personas**: for the Quality persona, last-iteration CTR reaches `23.4%` with Explore-Exploit versus `18.2%` for full-ctr, `15.1%` for partial-ctr, and `6.0%` for no-drop; for Ethical Considerations, Explore-Exploit reaches `20.3%` versus `15.7%` for full-ctr.
- **Generalization beyond shopping**: on the general-knowledge domain, Explore-Exploit is reported to deliver the strongest improvement trend, with persona-level CTR gains ranging from about `+3` to `+25` points between the initial and final iterations.
- **Human preference follows the simulator trend**: in pairwise comparison of `IP_0` versus `IP_15` for the best Explore-Exploit system, annotators prefer the final pool in `88%` of cases (`22/25`).
- **Effect size is statistically significant**: the e-commerce comparison between Explore-Exploit and competing approaches is reported as significant with `p < .001` under a Z-test for proportions.

## Limitations

- The optimization loop is validated only with an offline simulator rather than real user traffic, so deployment behavior under sparse, delayed, or strategic feedback remains untested.
- Personalization is cohort-level rather than user-level; the method models personas instead of persistent individual histories.
- The study focuses on question generation, leaving other generative recommendation tasks such as summarization or headline generation as future work.
- The experiments depend on GPT-4-quality instruction following and numerical reasoning; the authors report that smaller open models were not strong enough in preliminary tests.
- Baseline coverage is incomplete: the paper does not adapt classical bandits or hybrid retrieval-generation optimizers to the dynamic item-pool setting.

## Concepts Extracted

- [[generative-recommender-system]]
- [[training-free-optimization]]
- [[explore-exploit]]
- [[click-through-rate]]
- [[large-language-model]]
- [[in-context-learning]]
- [[question-generation]]
- [[user-simulation]]
- [[relevance-scoring]]
- [[persona]]

## Entities Extracted

- [[lutfi-kerem-senel]]
- [[besnik-fetahu]]
- [[davis-yoshida]]
- [[zhiyu-chen]]
- [[giuseppe-castellucci]]
- [[nikhita-vedula]]
- [[jason-choi]]
- [[shervin-malmasi]]
- [[gpt-4]]
- [[openai]]
- [[wikipedia]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
