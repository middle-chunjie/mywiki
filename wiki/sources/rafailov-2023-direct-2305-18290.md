---
type: source
subtype: paper
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
slug: rafailov-2023-direct-2305-18290
date: 2026-04-20
language: en
tags: [alignment, llm, preferences, rlhf, optimization]
processed: true
raw_file: raw/papers/rafailov-2023-direct-2305-18290/paper.pdf
raw_md: raw/papers/rafailov-2023-direct-2305-18290/paper.md
bibtex_file: raw/papers/rafailov-2023-direct-2305-18290/paper.bib
possibly_outdated: true
authors:
  - Rafael Rafailov
  - Archit Sharma
  - Eric Mitchell
  - Stefano Ermon
  - Christopher D. Manning
  - Chelsea Finn
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2305.18290
doi:
url: http://arxiv.org/abs/2305.18290
citation_key: rafailov2023direct
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces Direct Preference Optimization (DPO), a preference-learning algorithm that replaces the standard RLHF pipeline of reward-model fitting plus PPO fine-tuning with a single binary classification objective on preferred versus dispreferred completions. The key observation is that the KL-constrained reward-maximization objective used in RLHF admits a closed-form optimal policy, allowing rewards to be reparameterized by policy-to-reference log-probability ratios. This yields a simple logistic loss that directly optimizes the policy while preserving the same underlying objective as constrained RLHF. Across controlled sentiment generation, TL;DR summarization, and Anthropic-HH dialogue, DPO matches or exceeds PPO-based baselines, improves implementation simplicity, and reduces the need for reward-model training, online sampling, and extensive hyperparameter tuning.

## Problem & Motivation

RLHF had become the dominant recipe for aligning large language models, but its two-stage pipeline is awkward in practice: one first fits a reward model from pairwise preferences and then optimizes a policy with PPO under a KL constraint to a reference model. That setup increases engineering complexity, training instability, and compute cost because optimization requires additional models, online sampling, and careful hyperparameter tuning. The paper asks whether the same preference-alignment objective can be optimized directly from offline preference pairs, without explicit reward modeling or reinforcement learning. DPO is proposed as a simpler answer that keeps the KL-regularized alignment objective but collapses the pipeline into ordinary supervised optimization.

## Method

- Start from the standard RLHF objective `max_pi E_{x,y~pi}[r(x,y)] - beta * D_KL(pi(y|x) || pi_ref(y|x))`, where `beta` controls how far the aligned policy may drift from the reference policy.
- Use the closed-form optimizer of that objective: `pi_r(y|x) = (1 / Z(x)) * pi_ref(y|x) * exp(r(x,y) / beta)`, with partition function `Z(x) = sum_y pi_ref(y|x) exp(r(x,y) / beta)`.
- Reparameterize rewards via the policy ratio: `r(x,y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log Z(x)`. Under the Bradley-Terry preference model, the `log Z(x)` term cancels when comparing two responses.
- Optimize the policy directly with `L_DPO = -E_{(x,y_w,y_l)} [log sigma(beta * log(pi_theta(y_w|x) / pi_ref(y_w|x)) - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x)))]`.
- Interpret the implicit reward as `r_hat_theta(x,y) = beta * log(pi_theta(y|x) / pi_ref(y|x))`; the gradient increases the log-probability of preferred responses and decreases that of rejected responses, with dynamic weighting based on current mis-ordering.
- Default training settings are `beta = 0.1`, batch size `64`, optimizer `RMSprop`, learning rate `1e-6`, and linear warmup over `150` steps; for TL;DR summarization the paper uses `beta = 0.5`.
- In controlled sentiment experiments, the setup uses `gpt2-large` as base model, `25,000` IMDb prefixes of length `2-8` tokens, `4` sampled completions per prefix, `6` preference pairs per prefix, SFT for `1` epoch, and reward-model training for `3` epochs as the PPO baseline pipeline.

## Key Results

- In controlled sentiment generation, a `22`-run sweep shows DPO achieving the best reward-versus-KL frontier, strictly outperforming PPO and even PPO-GT, which has access to the ground-truth reward.
- On Reddit TL;DR summarization, DPO reaches about `61%` GPT-4 win rate at temperature `0.0`, compared with PPO's best result of about `57%`, while also being more robust to sampling temperature.
- In human evaluation on TL;DR, DPO samples are preferred over PPO-0 samples `58%` of the time; GPT-4(C) reports `54%`, and human-human agreement is `65%`.
- Under distribution shift to CNN/DailyMail, DPO beats PPO on GPT-4 win rate against ground-truth summaries: `0.36` vs `0.26` at temperature `0.0`, and `0.31` vs `0.23` at temperature `0.25`.
- On Anthropic-HH single-turn dialogue, the paper reports DPO as the only computationally efficient method that improves over the preferred completions in the dataset, while matching or exceeding a much more expensive Best-of-`128` baseline.

## Limitations

- The empirical study only scales to models up to roughly `6B` parameters, so the paper does not establish behavior at today's largest alignment scales.
- Out-of-distribution generalization is only lightly tested, mainly with TL;DR-to-CNN/DailyMail transfer; broader robustness claims remain tentative.
- Several headline comparisons, such as the reward-KL frontier, are presented visually rather than with full tabulated numbers, which limits exact quantitative auditing.
- The evaluation pipeline leans heavily on GPT-4 as a proxy judge, even though the paper does include a smaller human study for validation.
- DPO still depends on high-quality offline preference data and a carefully chosen reference policy; it is not a substitute for data curation.

## Concepts Extracted

- [[direct-preference-optimization]]
- [[reinforcement-learning-from-human-feedback]]
- [[reward-model]]
- [[proximal-policy-optimization]]
- [[kl-divergence]]
- [[instruction-tuning]]
- [[sentiment-control]]
- [[large-language-model]]

## Entities Extracted

- [[rafael-rafailov]]
- [[archit-sharma]]
- [[eric-mitchell]]
- [[stefano-ermon]]
- [[christopher-d-manning]]
- [[chelsea-finn]]
- [[stanford-university]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
