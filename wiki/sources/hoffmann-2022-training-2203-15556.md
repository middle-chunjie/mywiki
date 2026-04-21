---
type: source
subtype: paper
title: Training Compute-Optimal Large Language Models
slug: hoffmann-2022-training-2203-15556
date: 2026-04-20
language: en
tags: [llm, scaling, pretraining, compute-optimality, chinchilla]
processed: true

raw_file: raw/papers/hoffmann-2022-training-2203-15556/paper.pdf
raw_md: raw/papers/hoffmann-2022-training-2203-15556/paper.md
bibtex_file: raw/papers/hoffmann-2022-training-2203-15556/paper.bib
possibly_outdated: true

authors:
  - Jordan Hoffmann
  - Sebastian Borgeaud
  - Arthur Mensch
  - Elena Buchatskaya
  - Trevor Cai
  - Eliza Rutherford
  - Diego de Las Casas
  - Lisa Anne Hendricks
  - Johannes Welbl
  - Aidan Clark
  - Tom Hennigan
  - Eric Noland
  - Katie Millican
  - George van den Driessche
  - Bogdan Damoc
  - Aurelia Guy
  - Simon Osindero
  - Karen Simonyan
  - Erich Elsen
  - Jack W. Rae
  - Oriol Vinyals
  - Laurent Sifre
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2203.15556
doi:
url: http://arxiv.org/abs/2203.15556
citation_key: hoffmann2022training
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

This paper revisits scaling laws for dense autoregressive transformers under a fixed training compute budget and argues that frontier LLMs were being over-sized and under-trained. By fitting loss surfaces from more than `400` runs spanning roughly `70M` to `16B` parameters and `5B` to `500B` training tokens, the authors estimate that compute-optimal scaling should increase parameter count and training data at nearly the same rate, rather than mostly increasing parameters. They validate the prediction with Chinchilla, a `70B` model trained on `1.4T` tokens using approximately the same compute as Gopher `280B`, and show broad downstream gains on MMLU, BIG-bench, reading comprehension, and closed-book QA while also reducing inference cost.

## Problem & Motivation

Recent large language models had largely followed the Kaplan et al. scaling prescription in practice: grow parameter count aggressively while keeping training data around `300B` tokens. The authors argue this leaves large dense transformers under-trained for their compute budget and makes both pretraining and downstream deployment unnecessarily expensive. Their goal is to estimate, for a fixed FLOP budget `C`, the optimal pair `(N, D)` of model parameters and training tokens that minimizes final language-model loss.

## Method

- The target quantity is the compute-constrained optimum `N_opt(C), D_opt(C) = argmin_{N,D: FLOPs(N,D)=C} L(N,D)`, where `L(N,D)` is final pretraining loss after training a transformer with `N` parameters on `D` tokens.
- The empirical basis is a scaling sweep of `400+` autoregressive transformer runs covering roughly `70M` to `16B` parameters and `5B` to `500B` tokens, with multiple training horizons per model size.
- **Approach 1: minimum over training curves.** For each fixed `N`, train several runs with different cosine-schedule lengths, smooth the loss curves, then extract the loss-minimizing `(N,D)` along iso-FLOP slices. This yields `N_opt ∝ C^0.50` and `D_opt ∝ C^0.50`.
- **Approach 2: IsoFLOP profiles.** Hold FLOPs fixed, vary model size, choose token budgets so `FLOPs(N,D)=C`, then fit the loss valley across model sizes. This yields `N_opt ∝ C^0.49` and `D_opt ∝ C^0.51`.
- **Approach 3: parametric loss modeling.** Fit `` `L̂(N,D) = E + A / N^α + B / D^β` `` using Huber loss and `LBFGS`; the fitted exponents are approximately `` `α = 0.34` `` and `` `β = 0.28` ``, implying `N_opt ∝ C^0.46` and `D_opt ∝ C^0.54`.
- The analysis assumes an autoregressive [[transformer]] family and a FLOP accounting close to the standard `` `C ≈ 6ND` `` approximation, with appendix calculations showing only small deviations.
- Matching the learning-rate schedule to the intended token horizon is treated as critical: the cosine cycle decays by `` `10x` `` over approximately the full training run, and overshooting the target step count by more than about `25%` measurably hurts performance.
- The validation model, [[chinchilla]], uses the Gopher architecture family with `80` layers, `64` heads, key/value size `128`, `` `d_model = 8192` ``, feed-forward width `` `4 * d_model` ``, maximum learning rate `` `1e-4` ``, and batch size `1.5M -> 3M` tokens.
- Chinchilla is trained on [[massivetext]] for `1.4T` tokens, uses `AdamW` instead of Adam, a modified [[sentencepiece]] tokenizer without NFKC normalization, `bfloat16` forward/backward passes, and a `float32` optimizer-state copy of the weights.

## Key Results

- The three scaling analyses all predict near-equal parameter/data scaling: `(a, b) = (0.50, 0.50)`, `(0.49, 0.51)`, and `(0.46, 0.54)`, versus Kaplan et al. `(0.73, 0.27)`.
- For a `67B` model on the compute-optimal frontier, Approach 1 predicts `5.76e23` FLOPs and about `1.5T` training tokens; for `175B`, the same approach predicts about `3.7T` tokens; for `280B`, about `5.9T` tokens.
- At roughly the Gopher compute budget, the paper predicts an optimal model around `40B` to `70B` parameters instead of `280B`, and validates that claim with Chinchilla `70B`.
- [[chinchilla]] reaches `67.6%` 5-shot accuracy on [[mmlu]], improving over [[gopher]] by `7.6` points (`60.0% -> 67.6%`).
- On [[big-bench]], Chinchilla improves average accuracy from `54.4%` to `65.1%` and outperforms Gopher on `58/62` tasks.
- On reading comprehension, Chinchilla reaches `77.4%` on LAMBADA versus Gopher `74.5%`, `86.8%` on RACE-m versus `75.1%`, and `82.3%` on RACE-h versus `71.6%`.
- On closed-book QA, Chinchilla improves Natural Questions from `24.5%` to `31.5%` in 5-shot and from `28.2%` to `35.5%` in 64-shot; on TriviaQA (unfiltered, test), 5-shot rises from `63.6%` to `73.2%`.
- Despite being `4x` smaller than Gopher in parameter count, Chinchilla uses the same pretraining compute budget and materially lowers inference and fine-tuning cost.

## Limitations

- The large-scale validation is narrow: the main head-to-head evidence is essentially Gopher versus Chinchilla, with limited intermediate-scale confirmations near the predicted frontier.
- The scaling analysis assumes a power-law frontier in compute, parameters, and data; the paper itself observes curvature at high compute, suggesting the extrapolated optimal model sizes may still be too large.
- Most scaling runs are in the single-epoch or sub-epoch regime, so the conclusions may not transfer cleanly to multi-epoch training.
- Some language-modeling gains may be inflated by train/test leakage because Chinchilla sees `4x` more data than Gopher; the authors therefore place more weight on downstream benchmarks than on raw LM metrics.
- Better compute-optimal training does not remove harms: the paper still reports bias, toxicity, and privacy risks from web-scale data, with only limited safety analysis.

## Concepts Extracted

- [[large-language-model]]
- [[autoregressive-language-model]]
- [[transformer]]
- [[model-scaling]]
- [[scaling-law]]
- [[compute-optimal-scaling]]
- [[training-tokens]]
- [[language-model-pretraining]]
- [[few-shot-learning]]
- [[in-context-learning]]

## Entities Extracted

- [[jordan-hoffmann]]
- [[sebastian-borgeaud]]
- [[arthur-mensch]]
- [[elena-buchatskaya]]
- [[trevor-cai]]
- [[eliza-rutherford]]
- [[diego-de-las-casas]]
- [[lisa-anne-hendricks]]
- [[johannes-welbl]]
- [[aidan-clark]]
- [[tom-hennigan]]
- [[eric-noland]]
- [[katie-millican]]
- [[george-van-den-driessche]]
- [[bogdan-damoc]]
- [[aurelia-guy]]
- [[simon-osindero]]
- [[karen-simonyan]]
- [[erich-elsen]]
- [[jack-w-rae]]
- [[oriol-vinyals]]
- [[laurent-sifre]]
- [[deepmind]]
- [[gopher]]
- [[chinchilla]]
- [[gpt-3]]
- [[jurassic-1]]
- [[massivetext]]
- [[mmlu]]
- [[big-bench]]
- [[the-pile]]
- [[sentencepiece]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
