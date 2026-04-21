---
type: source
subtype: paper
title: Generating Adversarial Computer Programs using Optimized Obfuscations
slug: srikant-2021-generating-2103-11882
date: 2026-04-20
language: en
tags: [adversarial-attack, program-obfuscation, program-summarization, robustness, code-intelligence]
processed: true
raw_file: raw/papers/srikant-2021-generating-2103-11882/paper.pdf
raw_md: raw/papers/srikant-2021-generating-2103-11882/paper.md
bibtex_file: raw/papers/srikant-2021-generating-2103-11882/paper.bib
possibly_outdated: true
authors:
  - Shashank Srikant
  - Sijia Liu
  - Tamara Mitrovska
  - Shiyu Chang
  - Quanfu Fan
  - Gaoyuan Zhang
  - Una-May O'Reilly
year: 2021
venue: arXiv
venue_type: preprint
arxiv_id: 2103.11882
doi:
url: http://arxiv.org/abs/2103.11882
citation_key: srikant2021generating
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2021; re-verify against recent literature.

This paper studies adversarial robustness for machine-learning models of source code by treating semantics-preserving program obfuscations as adversarial perturbations. It formulates adversarial program generation as an optimization over both where to perturb a program and what token-level perturbation to apply, using a binary site-selection vector `z` and per-site perturbation variables `u`. The authors instantiate the framework against SEQ2SEQ program summarizers on Python and Java, compare projected-gradient-based joint optimization (JO) with alternating optimization (AO), and add randomized smoothing (RS) to make the discrete loss landscape easier to optimize. AO+RS substantially improves attack success over the prior baseline and also yields stronger adversarial training, making the paper notable for explicitly separating site choice from perturbation choice in code attacks.

## Problem & Motivation

The paper targets learned models that summarize or classify programs, where an attacker wants to change the model's prediction without changing program functionality. Unlike image attacks, code perturbations must remain both hard for humans to notice and semantically valid for the compiler or interpreter. Program obfuscations satisfy that constraint, but prior work largely chose perturbation sites randomly and optimized only the replacement token. The authors argue that adversarial robustness of code models cannot be assessed rigorously unless both the perturbation location and the perturbation content are optimized jointly under a formal attack budget.

## Method

- **Transformation space**: the attack uses 6 semantics-preserving obfuscations, combining 4 replace transforms (renaming local variables, function parameters, object fields, and replacing boolean literals with equivalent expressions) with 2 insert transforms (print statements and dead code).
- **Program representation**: a program `P` is tokenized into `n` tokens over vocabulary `Ω`; perturbation sites are encoded by `z ∈ {0,1}^n` with budget `1^T z <= k`, where `k` is the attacker's perturbation strength.
- **Site perturbation variables**: each token position has a one-hot perturbation choice `u_i ∈ {0,1}^{|Ω|}` with `1^T u_i = 1`; the perturbed program is `P' = (1 - z) · P + z · u`.
- **Attack objective**: adversarial generation is posed as `min_{z,u} l_attack((1-z) · P + z · u; P, θ)`, where `l_attack` is untargeted cross-entropy on the downstream model `θ`.
- **Continuous relaxation and JO**: the discrete constraints are relaxed to `z ∈ [0,1]^n` and `u_i ∈ [0,1]^{|Ω|}`, then optimized with projected gradient descent via updates in Eqs. `4-5`; discrete attacks are recovered by randomized sampling.
- **Projection subproblems**: the projection over `z` and `u_i` is decomposed into separate convex problems, including `min_z ||z - z^(t)||_2^2` subject to `1^T z <= k`; the required scalar roots are solved by bisection.
- **Alternating optimization**: AO alternates between optimizing `z` with `u` fixed and optimizing `u` with `z` fixed, following Eq. `7`; it costs roughly `2x` a JO iteration but empirically converges faster.
- **Randomized smoothing**: the smoothed objective is `l_smooth(z,u) = E_{ξ,τ}[l_attack(z + μξ, u + μτ)]` with `μ = 0.01`; the Monte Carlo approximation uses `m = 10` samples, and the paper finds smoothing `u` is the most useful part.
- **Experimental setup**: attacks are evaluated on program summarization datasets of roughly `150K` Python functions and `700K` Java functions, using a SEQ2SEQ model trained with cross-entropy; the main comparisons use `k = 1` and `k = 5`, AO for `3` iterations, and JO for `10`.

## Key Results

- **Python, `k = 1`**: baseline ASR/F1 is `19.87 / 78.18`; AO reaches `23.16 / 74.78`, JO `23.32 / 74.56`, AO+RS `30.25 / 69.52`, and JO+RS `23.95 / 74.24`.
- **Python, `k = 5`**: baseline ASR/F1 is `37.50 / 59.54`; AO improves to `43.53 / 53.75`, JO to `41.95 / 56.06`, AO+RS to `51.68 / 47.92`, and JO+RS to `48.70 / 51.55`.
- **Java**: on the Java benchmark, baseline ASR rises from `22.93` to `29.08` under AO+RS at `k = 1`, and from `33.16` to `40.53` at `k = 5`; the corresponding F1 values drop from `70.75` to `63.90` and from `59.45` to `51.91`.
- **Optimization behavior**: AO reaches strong performance in about `3` iterations, while JO needs about `10`; under AO+RS, ASR is about `50` at `k = 5` and about `60` at `k = 20`, so a small perturbation budget attains roughly `80%` of the best observed attack rate.
- **Adversarial training**: using AO+RS as the inner maximizer lowers ASR under an AO+RS attack from `30.25` with no AT to `13.75`, versus `19.11` when trained with the baseline attacker.
- **Evaluation pool**: the paper attacks `2800` correctly classified Python programs and `2300` correctly classified Java programs, instead of mixing in already-misclassified examples.

## Limitations

- The evaluation is narrow: a single downstream architecture (SEQ2SEQ) and a single task (program summarization), with code2seq and other code models deferred to future work.
- The perturbation space is limited to 6 handcrafted obfuscations; the paper omits try-catch insertion and loop unrolling because they add optimization variables with little measured benefit.
- The optimization is approximate because it relies on continuous relaxation and randomized sampling rather than solving the original discrete combinatorial problem exactly.
- Some candidate transformations are imperfect semantic proxies for the task; the appendix notes that omitting print statements barely changes ASR and suggests they may affect summarization semantics.
- As a 2021 preprint in a fast-moving NLP/code-model area, its conclusions should be re-checked against transformer-based and large-scale code models developed later.

## Concepts Extracted

- [[adversarial-attack]]
- [[program-obfuscation]]
- [[program-summarization]]
- [[sequence-to-sequence]]
- [[site-selection]]
- [[site-perturbation]]
- [[projected-gradient-descent]]
- [[joint-optimization]]
- [[alternating-optimization]]
- [[randomized-smoothing]]
- [[adversarial-training]]

## Entities Extracted

- [[shashank-srikant]]
- [[sijia-liu]]
- [[tamara-mitrovska]]
- [[shiyu-chang]]
- [[quanfu-fan]]
- [[gaoyuan-zhang]]
- [[una-may-oreilly]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
