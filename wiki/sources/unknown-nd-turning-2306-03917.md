---
type: source
subtype: paper
title: Turning large language models into cognitive models
slug: unknown-nd-turning-2306-03917
date: 2026-04-20
language: en
tags: [llm, cognitive-modeling, decision-making, finetuning, behavioral-science]
processed: true

raw_file: raw/papers/unknown-nd-turning-2306-03917/paper.pdf
raw_md: raw/papers/unknown-nd-turning-2306-03917/paper.md
bibtex_file: raw/papers/unknown-nd-turning-2306-03917/paper.bib
possibly_outdated: true

authors:
  - Marcel Binz
  - Eric Schulz
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2306.03917
doi:
url: https://arxiv.org/abs/2306.03917
citation_key: unknownndturning
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper asks whether a pretrained large language model can be adapted into a cognitive model of human choice. The authors use frozen final-layer embeddings from `LLaMA-65B` and fit a regularized linear readout, called CENTaUR, on behavioral data from risky-choice and exploration tasks. This simple adaptation beats both the raw LLaMA baseline and domain-specific cognitive baselines on choices13k (`NLL = 48002.3` vs. `49448.1` for BEAST) and the horizon task (`25968.6` vs. `29042.5` for the hybrid model). The learned representations also support participant-level random effects (`23929.5` NLL) and transfer to an unseen experiential-symbolic task (`4521.1` NLL), suggesting that pretrained LLM embeddings can serve as a reusable substrate for modeling human behavior.

## Problem & Motivation

Previous work showed that large language models can sometimes behave in human-like ways inside psychological experiments, but they often remain systematically unhuman in the strategies they use. The paper targets this gap directly: instead of treating an off-the-shelf LLM as a cognitive theory, it asks whether domain-specific behavioral finetuning can align pretrained representations with human decision patterns. The broader motivation is to move from narrow, task-specific cognitive models toward a more general architecture that can absorb multiple experimental paradigms and potentially generalize across them.

## Method

- **Backbone representation**: use frozen embeddings from the final hidden layer of `LLaMA-65B`, prompted with the same information available to a human participant on each trial.
- **Readout model**: fit a regularized logistic regression / linear head on top of the embedding, with choice probability `p(y = 1 | h) = σ(w^T h + b)`. The resulting model family is named [[centaur]].
- **Tasks and data**: train on decisions from description via [[choices-13k]] (`>13,000` choice problems; `14,711` participants; `>1,000,000` choices) and decisions from experience via [[horizon-task]] (`60` participants; `67,200` choices).
- **Prompt construction**: each trial is converted into text. Description trials explicitly encode outcome probabilities and payoffs; horizon-task prompts list past machine outcomes and ask for the next choice.
- **Training protocol**: evaluate with `100`-fold cross-validation; per fold use `90%` train, `9%` validation, and `1%` test splits; standardize all input features before fitting.
- **Regularization search**: nested validation selects the `ℓ_2` coefficient `α ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1}`.
- **Optimization**: implement fitting in PyTorch with default `LBFGS`; for individual-difference analysis, add random effects for each participant and each embedding dimension.
- **Hold-out transfer**: train a joint model on choices13k + horizon-task, then evaluate on the [[experiential-symbolic-task]] with `8` folds; validation tunes both `α` and inverse temperature `τ^-1 ∈ {0.05, 0.10, ..., 1.0}`.
- **Baselines**: compare against random guessing, raw [[llama]] log-probability readout, [[beast]] on choices13k, and a hybrid exploration model on the horizon task.

## Key Results

- On [[choices-13k]], raw [[llama]] is near chance (`NLL = 96248.5`), [[beast]] reaches `49448.1`, and [[centaur]] improves to `48002.3`.
- On [[horizon-task]], raw [[llama]] obtains `NLL = 46211.4`, the hybrid baseline gets `29042.5`, and [[centaur]] improves further to `25968.6`.
- Simulated regret is much closer to humans after adaptation: on choices13k, human `1.24 ± 0.01`, [[centaur]] `1.35 ± 0.01`, raw [[llama]] `1.85 ± 0.01`; on horizon, human `2.33 ± 0.05`, [[centaur]] `2.38 ± 0.01`, raw [[llama]] `7.21 ± 0.02`.
- Participant-level modeling also benefits from the embeddings: [[centaur]] is the best model for `52/60` horizon-task participants.
- Adding participant random effects reduces [[centaur]] horizon-task NLL from `25968.6` to `23929.5`, beating a hybrid random-effects baseline at `24166.0`.
- Transfer to an unseen [[experiential-symbolic-task]] remains strong: [[centaur]] gets `NLL = 4521.1`, versus `5977.7` for random guessing and `6307.9` for raw [[llama]].

## Limitations

- The adaptation is shallow: only a linear readout is trained on frozen embeddings, so the paper does not test whether end-to-end model adaptation would improve or destabilize the cognitive fit.
- The empirical scope is narrow, covering only a few decision-making paradigms with relatively small participant diversity outside choices13k.
- Prompting abstracts each trial into text and explicitly ignores some across-trial learning effects, especially when tasks are prompted independently.
- The strongest generalization claim is based on one hold-out task, so the evidence for truly domain-general cognitive modeling remains preliminary.
- Because the paper is a `2023` preprint in a fast-moving [[large-language-model]] area, both the model choice (`LLaMA-65B`) and the conclusions should be rechecked against newer representation models and behavioral benchmarks.

## Concepts Extracted

- [[large-language-model]]
- [[cognitive-model]]
- [[fine-tuning]]
- [[in-context-learning]]
- [[linear-probing]]
- [[logistic-regression]]
- [[representation-learning]]
- [[multi-task-learning]]
- [[cross-task-generalization]]
- [[exploration-exploitation-tradeoff]]
- [[decision-making]]

## Entities Extracted

- [[marcel-binz]]
- [[eric-schulz]]
- [[llama]]
- [[centaur]]
- [[choices-13k]]
- [[horizon-task]]
- [[beast]]
- [[experiential-symbolic-task]]
- [[max-planck-institute-for-biological-cybernetics]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
