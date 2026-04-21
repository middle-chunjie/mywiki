---
type: source
subtype: paper
title: Large Language Models to Enhance Bayesian Optimization
slug: unknown-nd-large-2402-03921
date: 2026-04-20
language: en
tags: [llm, bayesian-optimization, hyperparameter-optimization, automl, in-context-learning]
processed: true

raw_file: raw/papers/unknown-nd-large-2402-03921/paper.pdf
raw_md: raw/papers/unknown-nd-large-2402-03921/paper.md
bibtex_file: raw/papers/unknown-nd-large-2402-03921/paper.bib
possibly_outdated: false

authors:
  - Tennison Liu
  - Nicolas Astorga
  - Nabeel Seedat
  - Mihaela van der Schaar
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.03921
doi:
url: https://arxiv.org/abs/2402.03921
citation_key: unknownndlarge3921
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper proposes LLAMBO, a modular framework that injects large language models into multiple stages of Bayesian optimization for hyperparameter tuning. Instead of only using an LLM as a heuristic generator, the method rephrases the optimization state in natural language so the model can warm-start search, act as a surrogate model, and sample promising future configurations conditioned on target performance. The paper argues that LLM priors, few-shot generalization, and contextual understanding are especially useful when observations are scarce, which is exactly the low-data regime where Bayesian optimization is most brittle. Across public, private, and synthetic HPO benchmarks, the authors report that LLAMBO improves early-stage search efficiency and can outperform standard BO baselines despite worse computational cost and less principled uncertainty calibration than Gaussian processes.

## Problem & Motivation

Classical Bayesian optimization is attractive for expensive black-box objectives, but its sample efficiency depends heavily on good surrogate models and good candidate proposal mechanisms under very sparse observations. The paper targets hyperparameter optimization as a realistic low-dimensional black-box setting where an LLM may already encode useful priors about model behavior, hyperparameter interactions, and task metadata. The core motivation is that in-context learning may let an LLM inject prior knowledge and contextual reasoning into BO without collecting transfer datasets or finetuning a task-specific optimizer.

## Method

- **Problem formulation**: optimize an expensive black-box objective `f: H -> S` by finding `h* = arg min_h f(h)` while only observing a small history `D_n = {(h_i, s_i)}_{i=1}^n`.
- **Natural-language interface to BO**: serialize the search space, task metadata, and optimization history into prompts so the LLM can condition on `<MODEL CARD>`, `<DATA CARD>`, and past observations `D_n`.
- **Warmstarting**: use zero-shot prompting to propose the initial `n = 5` configurations under three prompt settings: no context, partial context, and full context with dataset statistics.
- **Discriminative surrogate model**: estimate `p(s | h; D_n)` by asking the LLM to predict the score of a candidate configuration from few-shot examples; repeat prediction `K = 10` times and use empirical mean and standard deviation as the surrogate output.
- **Order-robust uncertainty estimation**: improve the Monte Carlo surrogate by shuffling the order of in-context examples before repeated predictions, motivated by prompt order sensitivity in left-to-right LLM inference.
- **Generative surrogate reformulation**: reinterpret TPE-style scoring through `p(s <= tau | h)` so the LLM can score whether a configuration belongs to the good region rather than directly modeling `l(h)` and `g(h)`.
- **Candidate sampling**: sample candidates from `p(h | s'; D_n)` using a target value `s' = s_min - alpha * (s_max - s_min)`, where `alpha` controls exploration versus exploitation.
- **Acquisition and end-to-end BO**: sample `M = 20` candidate points, score them with expected improvement `a(h) = E[max(p(s | h) - f(h_best), 0)]`, and evaluate the highest-scoring point next.
- **Implementation details**: experiments use `gpt-3.5-turbo-0301` with `temperature = 0.7`, `top_p = 0.95`, and compare against GP, TPE, SMAC3, Optuna, SKOpt, and related BO baselines.

## Key Results

- **Warmstart correlations**: average hyperparameter correlation among sampled initial points rises from `0.2546` for random initialization to `0.3006` with no-context prompting, `0.3905` with partial context, and `0.4301` with full-context prompting.
- **Candidate-sampling ablation**: the paper reports the best sampled-point regret around `alpha = 0.01`, while the deployed end-to-end system uses `alpha = -0.1` as a better exploration-exploitation compromise.
- **Prompt-design reliability**: candidate acceptance rate is `91.60% ± 0.45%` for full LLAMBO, `88.8% ± 0.39%` for the no-context ablation, and only `69.26% ± 0.79%` when task-specific instructions are removed.
- **End-to-end evaluation scale**: component studies cover `74` tasks from Bayesmark and HPOBench, while end-to-end Bayesmark-style evaluation runs `50` HPO tasks with `5` seeds and `25` trials per task.
- **Average-rank outcomes**: on public Bayesmark tasks, LLAMBO reaches average rank `2.02` on DecisionTree and `3.64` on RandomForest; on private/synthetic tasks it reaches `2.08` on RandomForest and `3.35` on DecisionTree, outperforming the reported baselines in those settings.

## Limitations

- The method trades classical BO efficiency for expensive LLM inference, so wall-clock cost and API latency are substantially higher than GP-, TPE-, or RF-based optimizers.
- The study is concentrated on relatively low-dimensional hyperparameter optimization; the paper does not demonstrate robustness on genuinely high-dimensional BO problems such as architecture search or control.
- Uncertainty quality is weaker than Gaussian processes: the discriminative LLAMBO surrogate improves prediction accuracy but still trails GP in calibration-oriented metrics such as log predictive density and empirical coverage.
- Performance depends on prompt engineering choices, prompt ordering, and the underlying LLM, so robustness across stronger or weaker models is not fully established.
- The manuscript has a minor reporting inconsistency between the main text and appendix on the count of private/synthetic datasets used in end-to-end evaluation, which slightly weakens reproducibility clarity.

## Concepts Extracted

- [[bayesian-optimization]]
- [[hyperparameter-optimization]]
- [[large-language-model]]
- [[in-context-learning]]
- [[zero-shot-prompting]]
- [[warmstarting]]
- [[surrogate-model]]
- [[candidate-sampling]]
- [[expected-improvement]]
- [[tree-structured-parzen-estimator]]
- [[gaussian-process]]
- [[confidence-calibration]]

## Entities Extracted

- [[tennison-liu]]
- [[nicolas-astorga]]
- [[nabeel-seedat]]
- [[mihaela-van-der-schaar]]
- [[university-of-cambridge]]
- [[llambo]]
- [[gpt-3-5-turbo]]
- [[bayesmark]]
- [[hpobench]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
