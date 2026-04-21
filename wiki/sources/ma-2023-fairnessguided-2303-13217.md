---
type: source
subtype: paper
title: Fairness-guided Few-shot Prompting for Large Language Models
slug: ma-2023-fairnessguided-2303-13217
date: 2026-04-20
language: en
tags: [llm, prompting, in-context-learning, fairness, calibration]
processed: true

raw_file: raw/papers/ma-2023-fairnessguided-2303-13217/paper.pdf
raw_md: raw/papers/ma-2023-fairnessguided-2303-13217/paper.md
bibtex_file: raw/papers/ma-2023-fairnessguided-2303-13217/paper.bib
possibly_outdated: true

authors:
  - Huan Ma
  - Changqing Zhang
  - Yatao Bian
  - Lemao Liu
  - Zhirui Zhang
  - Peilin Zhao
  - Shu Zhang
  - Huazhu Fu
  - Qinghua Hu
  - Bingzhe Wu
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2303.13217
doi:
url: http://arxiv.org/abs/2303.13217
citation_key: ma2023fairnessguided
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper studies instability in few-shot in-context learning and argues that a prompt's quality is strongly tied to its predictive bias. It defines a fairness score by feeding a content-free input into the model and measuring the entropy of the label distribution: prompts whose label probabilities are closer to uniform are treated as less biased and empirically perform better. Based on this signal, the authors propose T-fair-Prompting, which ranks individual demonstrations by fairness, and G-fair-Prompting, which greedily builds a prompt to maximize fairness improvement at each step. Across BLOOM and LLaMA models on SST-2, AGNews, TREC, CoLA, and RTE, the greedy variant usually outperforms random, diversity-based, and similarity-based prompting strategies.

## Problem & Motivation

Few-shot prompting for large language models is highly unstable: changing which demonstrations are chosen, how many are used, or in what order they appear can produce large accuracy swings. Prior prompt-search methods usually optimize only one dimension, often rely on heuristics such as diversity or embedding similarity, and may require extra resources like a development set or embedding access. This paper asks whether prompt quality can instead be estimated from the model's own biased predictions on semantically empty inputs, so that prompt search remains feasible even for black-box LLM APIs.

## Method

- **Prompted classification setup**: given a labeled pool `S = {(x_i, y_i)}_{i=1}^N`, a template `Γ(x_i, y_i)` maps each example into text, and a prompt is formed as `ρ = Γ(x_1, y_1) ⊕ ... ⊕ Γ(x_n, y_n)`. Prediction uses normalized class probabilities `p̂(y | ρ ⊕ τ)` and `ŷ = argmax_y p̂(y | ρ ⊕ τ)`.
- **Fairness / predictive-bias metric**: append a content-free input `η` such as `[N/A]` to the prompt and measure entropy `fair(ρ) = -Σ_{y∈Y} p(y | ρ ⊕ η) log p(y | ρ ⊕ η)`. A higher entropy means the prompt induces a more uniform label distribution and is considered less biased.
- **T-fair-Prompting**: evaluate each individual demonstration with one-shot prompting, sort demonstrations by fairness, and select the top-`k` items to form the final prompt. This reduces search from exhaustive `Θ(Σ_{k=1}^N C_N^k k!)` to `Θ(N)`, but depends strongly on the choice of `k`.
- **G-fair-Prompting**: greedily insert the remaining demonstration that maximizes `fair(Γ(x_i, y_i) ⊕ ρ)` subject to improvement over the current `fair(ρ)`, and stop when no addition improves fairness. The worst-case complexity is `O(N^2)`.
- **Local-to-global search view**: T-fair uses only individual-sample fairness, whereas G-fair starts from local bias but progressively optimizes the fairness of the whole prompt, jointly handling selection and ordering.
- **Experimental setup**: main enumeration studies use `4` demonstrations per training set; baselines that need larger pools select `4` examples from `16`. Models include BLOOM `176B`, LLaMA `33B`, and LLaMA `65B`; datasets include SST-2, AGNews, TREC, CoLA, and RTE, with RTE excluded for LLaMA because its maximum input length is `512`.

## Key Results

- Fairness correlates strongly with downstream accuracy across enumerated prompts; the paper reports that prompts found by greedy fairness search usually land in the top `20%` of all candidates, and on BLOOM `176B` often approach the fairest prompt found by enumeration.
- On BLOOM `176B`, G-fair-Prompting improves AGNews from `73.9` to `79.6` accuracy and TREC from `47.9` to `66.8`, substantially beating random prompting and the compared diversity/similarity baselines.
- On LLaMA `33B`, G-fair-Prompting reaches `80.2` on TREC versus `68.1` for random prompting; on LLaMA `65B`, it reaches `72.0` on CoLA versus `66.2` for random prompting.
- T-fair-Prompting is useful but unstable with respect to the number of shots: `Top-2` often beats `Top-4` by more than `5` points, showing that simply keeping more "fair" examples does not guarantee a better prompt.
- Correlation between performance before and after calibration is mostly positive, with Pearson `r` up to `0.8012` on CoLA with LLaMA `65B`, supporting the claim that better uncalibrated prompts often remain better after calibration.
- Exhaustive prompt search is expensive even in small settings: the paper notes that enumerating all candidates for RTE on BLOOM with `4` demonstrations costs more than `120` A100 GPU hours, motivating approximate search.

## Limitations

- The proposed fairness score is an indirect surrogate based on content-free entropy; it is empirically correlated with accuracy, but not theoretically guaranteed to rank prompts correctly across all tasks.
- The study focuses mainly on classification-style ICL with fixed prompt templates and discrete label spaces, so it does not show whether the method transfers to generation-heavy tasks.
- Exhaustive analysis is limited to very small demonstration sets, mainly `4` examples, because full enumeration becomes intractable as `N` grows.
- G-fair-Prompting can still fail when the model is not strongly sensitive to prompt bias, which the paper observes on some CoLA settings.
- Despite the name, the paper studies predictive bias over labels rather than social fairness; any extension to social-bias mitigation is left as future work.

## Concepts Extracted

- [[large-language-model]]
- [[in-context-learning]]
- [[few-shot-prompting]]
- [[demonstration-selection]]
- [[prompt-order-sensitivity]]
- [[predictive-bias]]
- [[content-free-input]]
- [[prompt-optimization]]
- [[greedy-search]]
- [[calibration]]

## Entities Extracted

- [[huan-ma]]
- [[changqing-zhang]]
- [[yatao-bian]]
- [[lemao-liu]]
- [[zhirui-zhang]]
- [[peilin-zhao]]
- [[shu-zhang]]
- [[huazhu-fu]]
- [[qinghua-hu]]
- [[bingzhe-wu]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
