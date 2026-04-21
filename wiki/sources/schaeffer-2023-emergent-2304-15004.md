---
type: source
subtype: paper
title: Are Emergent Abilities of Large Language Models a Mirage?
slug: schaeffer-2023-emergent-2304-15004
date: 2026-04-20
language: en
tags: [llm, evaluation, emergence, metrics, scaling-laws]
processed: true

raw_file: raw/papers/schaeffer-2023-emergent-2304-15004/paper.pdf
raw_md: raw/papers/schaeffer-2023-emergent-2304-15004/paper.md
bibtex_file: raw/papers/schaeffer-2023-emergent-2304-15004/paper.bib
possibly_outdated: true

authors:
  - Rylan Schaeffer
  - Brando Miranda
  - Sanmi Koyejo
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2304.15004
doi:
url: http://arxiv.org/abs/2304.15004
citation_key: schaeffer2023emergent
paper_type: theory

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper argues that many reported emergent abilities in large language models are measurement artifacts rather than abrupt capability phase changes. Its core claim is that when model outputs improve smoothly with scale, nonlinear or discontinuous metrics such as exact-match accuracy and multiple-choice grade can still create sharp-looking transitions, especially under limited test-set resolution. The authors formalize this with a simple scaling-law-based model, then test the hypothesis in three ways: metric swaps on GPT-3/InstructGPT arithmetic tasks, a meta-analysis of BIG-Bench emergence claims, and induced "emergence" on vision models by redefining evaluation metrics. Across all three analyses, the paper concludes that alleged emergence often disappears under better statistics or smoother metrics, making many prior claims less about model behavior than about evaluation design.

## Problem & Motivation

Prior work claimed that larger language models suddenly acquire abilities absent in smaller models, with sharp and unpredictable transitions at specific scales. If true, such emergence would matter for capability forecasting, alignment, and safety because dangerous behaviors could appear without warning. This paper questions that interpretation. The authors observe that many canonical emergence plots use metrics like exact string match or multiple-choice grade, which can amplify small token-level improvements into thresholded task-level jumps. They therefore ask whether the apparent phenomenon reflects genuine discontinuities in model behavior, or instead the interaction between smooth scaling and metric choice plus finite-sample measurement noise.

## Method

- **Alternative hypothesis**: assume model-family performance changes smoothly with scale and that claimed emergence is produced by evaluation, not by abrupt changes in outputs.
- **Scaling-law setup**: posit per-token cross-entropy follows a power law in model size, `L_CE(N) = (N / c)^alpha`, where `N` is parameter count, `c > 0`, and `alpha < 0`.
- **Per-token correctness**: map loss to the probability of predicting the correct token as `p(correct token) = exp(-L_CE(N)) = exp(-(N / c)^alpha)`.
- **Nonlinear metric effect**: for sequence-level exact accuracy on length-`L` outputs, approximate `Accuracy(N) ~= p(correct token)^L = exp(-(N / c)^alpha)^L`, which creates geometric decay with target length and visually sharp transitions on linear-log plots.
- **Linear metric effect**: for token edit distance, derive `Token Edit Distance(N) ~= L * (1 - exp(-(N / c)^alpha))`, which should vary much more smoothly with model scale.
- **Discontinuous metric effect**: analyze metrics such as Multiple Choice Grade and compare them against continuous alternatives like Brier Score to test whether thresholding alone can induce emergence-like curves.
- **Prediction 1**: changing from nonlinear/discontinuous metrics to smoother metrics while holding model outputs fixed should remove apparent emergence.
- **Prediction 2**: increasing test-set size should reveal that small models already have non-zero, above-chance performance under thresholded metrics, reducing the illusion of sudden onset.
- **Prediction 3**: target length should predictably modulate emergence plots, with approximately geometric degradation under accuracy and approximately quasilinear degradation under token edit distance.
- **Empirical tests**: evaluate public GPT-3/InstructGPT models on `2-shot` `2-digit x 2-digit` multiplication and `2-shot` `4-digit + 4-digit` addition; meta-analyze BIG-Bench task-metric-model-family triplets; then induce similar effects in CIFAR100 autoencoders, Omniglot autoregressive transformers, and MNIST convolutional networks by redefining metrics.
- **Resolution analysis**: formalize the measurement floor from finite test data, arguing that for small accuracies the effective resolution is bounded by roughly `1 / test_set_size`, so sparse data can make smaller models falsely appear incapable.

## Key Results

- On GPT-3/InstructGPT arithmetic tasks, the same model outputs look emergent under accuracy but smooth and predictable under token edit distance.
- Increasing evaluation data shows smaller GPT-family models have above-chance accuracy rather than literal zero performance, weakening the "sudden appearance" interpretation.
- The public GPT family analyzed spans `350M`, `1.3B`, `6.7B`, and `175B` parameter models.
- In BIG-Bench preferred metrics, possible emergence appears in at most `5 / 39` metrics rather than being widespread across task-model pairs.
- In hand-annotated BIG-Bench emergence data, only `4 / 39` preferred metrics show emergence, and more than `92%` of claimed emergent abilities fall under just two metrics: Multiple Choice Grade and Exact String Match.
- For LaMDA task-model pairs that look emergent under Multiple Choice Grade, the effect disappears when the evaluation metric is switched to Brier Score.
- The authors induce new "emergent" behaviors in vision: CIFAR100 autoencoders under a thresholded reconstruction metric, Omniglot autoregressive transformers under subset accuracy, and MNIST convolutional networks under all-`K`-correct accuracy.

## Limitations

- The paper does **not** prove that genuine emergent abilities are impossible; it argues that many previously reported cases may be mirages caused by metrics and statistics.
- Several derivations rely on simplifying assumptions, especially near-independent token errors, which the authors acknowledge are not literally true.
- Much of the meta-analysis is constrained by public reporting because many model families and raw outputs are unavailable for independent re-evaluation.
- The work focuses on evaluation geometry and sample resolution rather than mechanistic causes inside training dynamics, so it cannot rule out real discontinuities in other settings.
- Some demonstrations of induced emergence are qualitative rather than exhaustive, showing possibility rather than a complete taxonomy of artifact-producing metrics.

## Concepts Extracted

- [[large-language-model]]
- [[model-scaling]]
- [[scaling-law]]
- [[power-law-scaling]]
- [[benchmark-evaluation]]
- [[automatic-evaluation-metric]]
- [[autoregressive-model]]
- [[emergent-ability]]
- [[token-edit-distance]]
- [[brier-score]]
- [[multiple-choice-grade]]
- [[exact-string-match]]

## Entities Extracted

- [[rylan-schaeffer]]
- [[brando-miranda]]
- [[sanmi-koyejo]]
- [[stanford-university]]
- [[gpt-3]]
- [[instructgpt]]
- [[openai]]
- [[big-bench]]
- [[lamda]]
- [[google-research]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
