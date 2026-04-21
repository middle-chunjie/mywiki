---
type: source
subtype: paper
title: "Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning"
slug: wang-nd-large
date: 2026-04-20
language: en
tags: [llm, in-context-learning, bayesian-inference, demonstration-selection, prompt-tuning]
processed: true

raw_file: raw/papers/wang-nd-large/paper.pdf
raw_md: raw/papers/wang-nd-large/paper.md
bibtex_file: raw/papers/wang-nd-large/paper.bib
possibly_outdated: true

authors:
  - Xinyi Wang
  - Wanrong Zhu
  - Michael Saxon
  - Mark Steyvers
  - William Yang Wang
year: 2023
venue: NeurIPS 2023
venue_type: conference
arxiv_id: 2301.11916
doi:
url: https://arxiv.org/abs/2301.11916
citation_key: wangndlarge
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper reframes in-context learning as approximate Bayesian inference over a latent task concept that encodes both label semantics and prompt format. On this view, strong demonstrations are examples that most strongly induce the task-specific latent variable. The authors turn this into a two-stage algorithm: learn task-specific concept tokens by prompt tuning a small LM, then score candidate demonstrations by how likely they are to predict those tokens, and transfer the selected examples to larger models unchanged. Across eight text-classification datasets and eight GPT-family models, the method improves average 4-shot accuracy over uniform and similarity baselines, and it also yields gains on GSM8K, making the paper a concrete bridge from latent-variable theory to practical prompt construction.

## Problem & Motivation

In-context learning works surprisingly well, but its performance is highly sensitive to which demonstrations are chosen, their format, and their order. Prior explanations either depend on synthetic settings, strong assumptions such as Hidden Markov data generation, or heuristics like nearest-neighbor similarity that do not explain why some examples transfer across models better than others. This paper aims to close that gap by giving a more realistic Bayesian account of prompting in real LLMs and then deriving a practical demonstration-selection rule from that account. The target is a cheap selection procedure that can be run with a small LM yet still improve larger LLMs at inference time.

## Method

- **Latent-variable formulation**: model continuation as `P_M(w_{t+1:T} | w_{1:t}) = \int_\Theta P_M(w_{t+1:T} | \theta) P_M(\theta | w_{1:t}) d\theta`, where `\theta` is a latent task concept carrying task and formatting information.
- **Task structure**: analyze both causal directions `X -> Y <- \theta` and `Y -> X <- \theta`; for the direct direction, the Bayes-optimal predictor is `argmax_y P_M^d(Y = y | \theta^d, X)`.
- **Latent concept learning**: add `c` new concept tokens `\hat{\theta}^d` for each task and freeze the original LM parameters; only the new embeddings are optimized.
- **Training objective**: minimize `\mathcal{L}(\hat{\theta}^d) = \mathbb{E}[\ell(X, Y; \hat{\theta}^d)]`, with `\ell = -\log P_M^d(Y | \hat{\theta}^d, X)` for `X -> Y <- \theta` and `\ell = -\log P_M^d(X | \hat{\theta}^d, Y)` for `Y -> X <- \theta`.
- **Posterior approximation**: estimate the task posterior by normalizing over a diverse task set `\mathcal{S}` as `\hat{P}_{M'}^d(\hat{\theta}^d | w_{1:t}) = P_{M'}^d(\hat{\theta}^d | w_{1:t}) / \sum_{t \in \mathcal{S}} P_{M'}^t(\hat{\theta}^t | w_{1:t})`.
- **Selection rule**: score each candidate example by `\hat{P}_{M'}^d(\hat{\theta}^d | X_i^d, Y_i^d)` and choose the top `k` demonstrations, using an independence approximation to avoid `O(|\mathcal{D}^d|^k)` combinatorial search.
- **Default setup**: use `k = 4` demonstrations, `c = 10` concept tokens per dataset, a candidate pool of `100` training examples, and at most `1000` sampled test examples per run.
- **Optimization details**: concept tokens are learned with [[gpt2-large]] using learning rate `1e-4`, batch size `16`, and `10k` training steps; training takes roughly `20-40` hours on a single A100, V100, or A6000 GPU.
- **Transfer protocol**: demonstrations are selected once with a small LM and then reused directly on larger LLMs, motivated by the assumption that related models share sufficiently similar pretraining distributions.

## Key Results

- Across eight GPT-family models and eight text-classification datasets, average 4-shot accuracy increases from `57.9` with uniform selection and `61.4` with similarity-based selection to `65.0` with the proposed method, a reported `12.5%` relative gain over uniform.
- For [[gpt2-large]], the average rises from `57.4` (uniform) and `59.7` (similar) to `64.8` (ours).
- For GPT3-curie (`6.7B`), the average rises from `62.3` (uniform) and `66.8` (similar) to `69.2` (ours), supporting transfer from small-model-selected demonstrations to larger GPT models.
- On individual GPT2-large datasets, accuracy improves from `77.1 -> 86.2` on SST2, `51.3 -> 60.4` on FPB, `62.7 -> 69.1` on COLA, and `54.4 -> 56.5` on DBpedia when moving from uniform to the proposed selector.
- On [[gsm8k]], [[chatgpt]] reaches `81.2` with demonstrations selected using [[llama-2]] `7B`, compared with `76.5` for uniform and `78.1` for similarity selection.
- On [[gsm8k]], [[llama-2]] `13B` improves from `17.0` (uniform) and `18.3` (similar) to `21.6` with the proposed demonstrations.
- The `k` ablation shows gains for `k = 2, 4, 8, 16`; the largest average improvement is at `k = 4`, where GPT3-ada rises from `56.8` to `63.4`.
- The `c` ablation peaks at `c = 10` with average `64.2`, compared with `56.6` for `c = 5`, `62.3` for `c = 15`, and `60.92` for `c = 20`.

## Limitations

- The theory assumes the LM approximates the true language distribution well enough for the Bayesian derivation to be meaningful; this is only approximately true in practice.
- Demonstration selection relies on an independence approximation across examples, which keeps search tractable but may miss interaction effects between demonstrations.
- The learned concept-token classifier is model-specific: the latent prefixes themselves do not transfer across models, only the selected demonstrations do.
- The core evaluation is dominated by text-classification tasks, with [[gsm8k]] serving as an auxiliary case study rather than the main theoretical target.
- The method shows high variance on some datasets such as EmoC, and gains on some non-GPT models are smaller than on GPT-family models.
- The paper leaves the construction of the diverse task subset `\mathcal{S}` mostly heuristic, so performance can depend on upstream task coverage.

## Concepts Extracted

- [[large-language-model]]
- [[in-context-learning]]
- [[few-shot-learning]]
- [[latent-variable-model]]
- [[bayesian-inference]]
- [[empirical-bayes]]
- [[demonstration-selection]]
- [[prompt-tuning]]
- [[text-classification]]
- [[chain-of-thought-prompting]]
- [[topic-model]]
- [[bayes-optimal-classifier]]

## Entities Extracted

- [[xinyi-wang]]
- [[wanrong-zhu]]
- [[michael-saxon]]
- [[mark-steyvers]]
- [[william-yang-wang]]
- [[university-of-california-santa-barbara]]
- [[university-of-california-irvine]]
- [[gpt2-large]]
- [[gpt-3]]
- [[llama-2]]
- [[chatgpt]]
- [[gsm8k]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
