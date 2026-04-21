---
type: source
subtype: paper
title: Unsupervised Evaluation of Code LLMs with Round-Trip Correctness
slug: allamanis-2024-unsupervised-2402-08699
date: 2026-04-20
language: en
tags: [code-llm, evaluation, round-trip, program-synthesis, code-editing]
processed: true

raw_file: raw/papers/allamanis-2024-unsupervised-2402-08699/paper.pdf
raw_md: raw/papers/allamanis-2024-unsupervised-2402-08699/paper.md
bibtex_file: raw/papers/allamanis-2024-unsupervised-2402-08699/paper.bib
possibly_outdated: false

authors:
  - Miltiadis Allamanis
  - Sheena Panthaplackel
  - Pengcheng Yin
year: 2024
venue: ICML 2024
venue_type: conference
arxiv_id: 2402.08699
doi:
url: http://arxiv.org/abs/2402.08699
citation_key: allamanis2024unsupervised
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

This paper proposes round-trip correctness (RTC), an unsupervised evaluation framework for code large language models that measures whether a model can map code to natural language and back while preserving semantics. Instead of relying on human-written labels, the authors instantiate RTC for code synthesis and code editing: a forward model produces a description or edit description, then a backward model reconstructs code and is judged by unit tests or exact match. Across HumanEval and ARCADE, RTC correlates strongly with standard pass@1 metrics, while a broader evaluation over 58 open-source Python projects exposes large cross-domain performance variation that narrow benchmarks miss. The paper positions RTC as a practical complement to curated benchmarks, especially when annotation is expensive or noisy.

## Problem & Motivation

Evaluation of code-capable LLMs is bottlenecked by small, manually curated benchmarks such as HumanEval, MBPP, ARCADE, and DS-1000. Those benchmarks are costly to expand and cover only narrow coding domains, which makes them insufficient for users working across diverse real-world repositories, libraries, and workflows. The authors aim to replace some of that annotation cost with an unsupervised procedure that can still reflect coding ability, scale to broader domains, and extend beyond plain code synthesis into tasks such as code editing where reliable gold labels are especially hard to obtain.

## Method

- **General RTC objective**: given a forward model `M: X -> Y` and backward model `M^{-1}: Y -> X`, evaluate whether `x` survives the round trip `x -> y -> x_hat` with `RTC_sim(x) = E_{y ~ M(x)} E_{x_hat ~ M^{-1}(y)} [sim(x_hat, x)]`.
- **Monte Carlo approximation**: estimate RTC with `N_f` forward samples and `N_b` backward samples via `(1 / (N_f N_b)) * sum_y sum_x_hat sim(x_hat, x)`.
- **Forward lift**: isolate how informative the forward prediction is with `L_M^{sim}(x) = RTC_sim(x) - E_{x_hat ~ M^{-1}(epsilon)} [sim(x_hat, x)]`, where `epsilon` is an uninformative utterance.
- **SYNTHESISRTC**: take a coherent code region plus surrounding file context, ask the forward model for a concise natural-language description, replace the region with a TODO containing that description, and ask the backward model to regenerate code; use unit-test success as `sim(·)`.
- **EDITINGRTC**: represent input as old code plus new code, ask the forward model to describe the edit, then ask the backward model to reconstruct the new code from the old code and the sampled edit description; use exact match as `sim(·)`.
- **Default experimental settings**: unless otherwise stated, use `N_f = 3`, `N_b = 1`, forward temperature `0.8`, backward temperature `0.1`, identical `3-shot` prompting, and cap forward outputs at `128` characters.
- **HumanEval / ARCADE calibration**: for HumanEval, remove the reference docstring and ask the model to describe the ground-truth function body before regenerating it; for ARCADE, describe notebook-turn code solutions and regenerate them from the notebook context.
- **Cross-domain sampling pipeline**: start from `77` permissively licensed Python projects with passing test suites, sample CST-aligned statement ranges of `32-384` characters, exclude test files, discard deletions with no test effect (about `40%` of sampled ranges), keep up to `100` ranges per project, require at least `80` valid samples per project, and limit visible context to `<= 1024` characters.
- **Editing benchmark setup**: evaluate EDITINGRTC on `1,000` CodeReviewer examples; for this section, sample `3` forward descriptions at temperature `1.0` and `1` backward edit at temperature `0.0`.

## Key Results

- On narrow-domain calibration benchmarks, `RTC_pass` correlates strongly with `pass@1`: Pearson `r = 0.96` on HumanEval and `r = 0.96` on ARCADE; Spearman `rho = 0.90` and `rho = 0.81`, respectively.
- HumanEval results preserve model ordering: `DSC33B-IT` reaches `40.2% RTC_pass`, `Gemini v1 Pro` `34.8%`, and `StarCoder2 15B` `31.7%`; on ARCADE, the top values are `15.1%`, `12.1%`, and `11.1%`.
- RTC is reasonably stable under the reported settings: repeating the HumanEval experiments `10` times yields standard deviation `sigma = 1.11%`.
- The broader synthesis dataset contains `5,961` code regions from `58` open-source Python projects; cross-project RTC varies widely, and Gemini Pro vs Gemini Nano 2 project-level performance is only moderately correlated (`r = 0.75`, `rho = 0.76`).
- Cross-domain forward lift is substantial: average `L_M` is `7.0%` for Gemini Nano 2 and `21.5%` for Gemini Pro, both higher than on HumanEval.
- For EDITINGRTC on CodeReviewer, Gemini Pro outperforms Gemini Nano 2 on unsupervised exact-match RTC (`12.9%` vs `5.2%`) and lift (`12.4%` vs `4.8%`), while supervised edit-description BLEU remains very low (`1.0` vs `0.5`).

## Limitations

- RTC depends on the quality of the similarity function `sim(·)`; weak proxies can produce misleading scores.
- Forward and backward abilities are coupled, so poor descriptions from `M` can hide the true capability of `M^{-1}`.
- The paper notes an adversarial failure mode: a forward model could simply copy the input and artificially inflate RTC.
- Unit tests and exact match are only proxies for semantic equivalence, not guarantees of full behavioral equivalence.
- The evaluation instantiates RTC only for synthesis and editing, so larger-context repository tasks are left open.

## Concepts Extracted

- [[round-trip-correctness]]
- [[unsupervised-evaluation]]
- [[code-synthesis]]
- [[code-editing]]
- [[forward-lift]]
- [[property-based-testing]]
- [[semantic-equivalence]]
- [[execution-based-evaluation]]
- [[pass-at-k]]
- [[cross-domain-evaluation]]

## Entities Extracted

- [[miltiadis-allamanis]]
- [[sheena-panthaplackel]]
- [[pengcheng-yin]]
- [[google-deepmind]]
- [[humaneval]]
- [[arcade]]
- [[codereviewer]]
- [[gemini-pro]]
- [[gemini-nano-2]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
