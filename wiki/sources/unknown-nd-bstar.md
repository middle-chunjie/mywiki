---
type: source
subtype: paper
title: "B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners"
slug: unknown-nd-bstar
date: 2026-04-20
language: en
tags: [self-improvement, reasoning, reward-model, online-learning, math]
processed: true

raw_file: raw/papers/unknown-nd-bstar/paper.pdf
raw_md: raw/papers/unknown-nd-bstar/paper.md
bibtex_file: raw/papers/unknown-nd-bstar/paper.bib
possibly_outdated: false

authors:
  - Weihao Zeng
  - Yuzhen Huang
  - Lulu Zhao
  - Yijun Wang
  - Zifei Shan
  - Junxian He
year: 2025
venue: ICLR 2025
venue_type: conference
arxiv_id:
doi:
url: https://openreview.net/pdf?id=P6dwZJpJ4m
citation_key: unknownndbstar
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

B-STaR studies why iterative self-improvement for reasoning models plateaus after only a few rounds and argues that the bottleneck is not a single scalar “data quality” issue but a moving balance between exploration and exploitation. The paper formalizes exploration as the policy’s ability to generate diverse correct candidates and exploitation as the reward’s ability to rank and filter those candidates effectively. It introduces a balance score that combines the number and purity of selected correct responses, then uses this score to adapt sampling temperature and reward threshold at each iteration. Across math, coding, and commonsense reasoning, B-STaR improves both greedy accuracy and multi-sample exploration metrics over STaR/ResT-EM, iterative RFT, and online RFT baselines.

## Problem & Motivation

Iterative self-improvement methods such as STaR, RFT, and online RFT can improve reasoning models without large volumes of human-labeled traces, but their gains saturate quickly. The paper argues that prior work treats sampling and reward configurations as static even though the policy and reward interaction changes over training. When the model stops producing diverse high-quality candidates, exploration collapses; when the reward no longer separates good and bad candidates well, exploitation weakens. B-STaR is motivated by the need to monitor these dynamics explicitly and to adjust the training loop so self-generated data remain both plentiful and clean enough to support continued improvement.

## Method

- **Self-improvement loop**: starting from policy `P_0`, each iteration samples candidates, scores them with a reward, filters selected responses, and updates the policy with SFT-style rejection fine-tuning.
- **Exploration metrics**: the paper tracks `Pass@K`, `Pass@K-S`, and Distinct Equations. `Pass@K-S` measures whether at least `S` unique correct responses appear among `K` samples, making exploration less noisy than plain `Pass@K`.
- **Exploitation metrics**: it tracks Best-of-`K` and `Reward@K-S`, where `Reward@K-S` asks whether the top `S` reward-ranked candidates are all correct.
- **Reward design**: for math, the sparse answer reward is `r = 1(\hat{a} = a*)`; with a verifier it becomes `r = 1(\hat{a} = a*) + r_prm(x, \hat{y})`, where the PRM score is the minimum step score over the solution.
- **Balance score**: for each query, B-STaR defines `bs_i = min(n_i' / n*, 1) * (n_i' / n_i)`, where `n_i'` is the number of unique correct selected responses, `n_i` is the number of selected responses, and `n*` is the target count of correct responses per query.
- **Dynamic control variables**: at each iteration B-STaR selects temperature `t_i` and reward threshold `tau_i` that maximize average balance score on a small subset of training queries, then applies them to the full iteration.
- **Math case study setup**: the main analysis uses `Mistral-7B`, SFT for `3` epochs on MATH, then `9` online-RFT iterations of `500` steps each, with batch size `128` and `32` sampled candidates per query in the dynamics study.
- **Main experiment setup**: for mathematical reasoning the larger main run sets total samples per iteration `N = 67,500`, queries per iteration `M = 11,500`, sample size `64`, temperature search `0.5` to `1.2`, and threshold search `-1.0` to `1.0`.
- **PRM training details**: the process reward model is also built on `Mistral-7B`, trained with learning rate `2e-6` for `2` epochs on about `270K` automatically generated process annotations, with scores normalized to `[-1, 1]`.
- **Other domains**: for coding on APPS the base model is `Llama-3-8B`, unit tests provide binary reward, and only temperature is adjusted; for ARC-Challenge the method starts from `Mistral-7B-Instruct` without an SFT stage.

## Key Results

- On `Mistral-7B`, B-STaR reaches `53.8` Pass@1 on GSM8K versus `46.8` for online RFT with reward model and `46.6` for iterative RFT with reward model.
- On MATH, B-STaR reaches `27.8` Pass@1, `67.2` Pass@32, and `42.2` Pass@32-4, outperforming online RFT with reward model (`23.2`, `62.6`, `39.2`).
- On APPS, B-STaR improves Pass@1 from `17.3` under online RFT to `19.6`, and raises Pass@32 from `45.8` to `49.3`.
- On ARC-Challenge, B-STaR reaches `73.0` Pass@1 versus `71.2` for online RFT and `70.7` for Rest-EM.
- Dynamic tuning matters: on GSM8K/MATH, temperature-only adjustment gives `53.1` / `25.0`, threshold-only gives `49.1` / `24.6`, and joint adjustment gives the best `53.8` / `27.8`.
- In the reported math trajectory, B-STaR increases balance score from `0.470` at step `500` to `0.679` at step `4500`, while the selected temperature shifts from `0.5` to mostly `1.1` and the threshold relaxes from `0` to `-0.1`.
- The method also transfers to a stronger `Llama-3.1-8B` setup, reaching `61.6` on GSM8K, `29.2` on MATH, `18.1` on APPS, and `86.3` on ARC-Challenge.

## Limitations

- The formulation depends on tasks with reliable external verification, such as exact-answer matching or unit tests; open-ended reasoning settings are not directly handled.
- The reward model is fixed during training, so exploitation can still degrade if policy outputs drift beyond the verifier’s competence.
- B-STaR only adjusts a small set of scalar configurations (`t` and `tau`) rather than learning richer decoding or reward-update policies.
- The strongest analysis is concentrated on mathematical reasoning; coding and commonsense experiments are less deeply instrumented.
- On hard math tasks, the authors explicitly note that a `7B` PRM may not be sufficiently discriminative, limiting gains from verifier-based exploitation.

## Concepts Extracted

- [[self-improvement]]
- [[self-training]]
- [[exploration-exploitation-tradeoff]]
- [[process-reward-model]]
- [[reward-model]]
- [[rejection-sampling-fine-tuning]]
- [[on-policy-learning]]
- [[pass-at-k]]
- [[balance-score]]
- [[mathematical-reasoning]]
- [[commonsense-reasoning]]

## Entities Extracted

- [[weihao-zeng]]
- [[yuzhen-huang]]
- [[lulu-zhao]]
- [[yijun-wang]]
- [[zifei-shan]]
- [[junxian-he-hkust]]
- [[hkust]]
- [[baai]]
- [[tencent]]
- [[mistral-7b]]
- [[llama-3-8b]]
- [[math]]
- [[gsm8k]]
- [[apps]]
- [[arc-challenge]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
