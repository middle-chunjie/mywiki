---
type: source
subtype: paper
title: Understanding the Behaviour of Contrastive Loss
slug: wang-2021-understanding-2012-09740
date: 2026-04-20
language: en
tags: [contrastive-learning, self-supervised-learning, representation-learning, computer-vision, metric-learning]
processed: true

raw_file: raw/papers/wang-2021-understanding-2012-09740/paper.pdf
raw_md: raw/papers/wang-2021-understanding-2012-09740/paper.md
bibtex_file: raw/papers/wang-2021-understanding-2012-09740/paper.bib
possibly_outdated: false

authors:
  - Feng Wang
  - Huaping Liu
year: 2021
venue: arXiv
venue_type: preprint
arxiv_id: 2012.09740
doi:
url: http://arxiv.org/abs/2012.09740
citation_key: wang2021understanding
paper_type: theory

read_status: unread

domain: representation-learning
---

## Summary

This paper analyzes why softmax-based contrastive loss works in unsupervised representation learning rather than treating it as a black-box objective. Wang and Liu show that the loss is inherently hardness-aware: gradients on negative samples scale with their similarity to the anchor, and temperature `tau` controls how sharply the objective focuses on the hardest negatives. They connect this mechanism to embedding uniformity and argue that contrastive learning faces a uniformity-tolerance dilemma: making embeddings more uniformly separated can also push semantically similar samples apart. Across CIFAR-10, CIFAR-100, SVHN, and ImageNet-100, they show that intermediate temperatures usually yield the best downstream accuracy, while explicit hard negative mining can recover strong performance even with simpler losses.

## Problem & Motivation

The paper addresses a gap between the empirical success of unsupervised contrastive learning and the limited understanding of why the softmax contrastive loss works so well. Prior work had established the importance of uniformity, but not how the temperature parameter changes optimization dynamics or downstream representation quality. The authors aim to explain the loss through gradient analysis, determine how temperature modulates penalties on hard negatives, and clarify why overly aggressive instance discrimination can damage semantic structure by repelling samples that are different instances but semantically close.

## Method

- **Base objective**: analyze the standard softmax contrastive loss `` `L(x_i) = -log(exp(s_ii / tau) / (sum_{k != i} exp(s_ik / tau) + exp(s_ii / tau)))` `` where similarities are `` `s_ij = f(x_i)^T g(x_j)` `` on a hypersphere.
- **Gradient view**: derive `` `dL/ds_ii = -(1/tau) sum_{k != i} P_ik` `` and `` `dL/ds_ij = (1/tau) P_ij` ``, showing that negative penalties are proportional to `` `exp(s_ij / tau)` `` and therefore emphasize harder negatives.
- **Penalty distribution**: define the relative negative penalty `` `r_i(s_ij) = exp(s_ij / tau) / sum_{k != i} exp(s_ik / tau)` ``; decreasing `tau` sharpens this Boltzmann distribution onto the nearest negatives, while `tau -> +infty` approaches a simple linear contrastive objective.
- **Extreme-case analysis**: show `` `tau -> 0+` `` approximates a triplet-like objective focused on the single hardest negative, while `` `tau -> +infty` `` reduces to a simple contrastive loss without hardness-aware weighting.
- **Explicit hard negative sampling**: define a hard contrastive loss that only keeps negatives above an upper `alpha` quantile, `` `L_hard(x_i) = -log(exp(s_ii / tau) / (sum_{s_ik >= s_alpha^(i)} exp(s_ik / tau) + exp(s_ii / tau)))` ``.
- **Uniformity and tolerance metrics**: measure uniformity with `` `L_uniformity(f; t) = log E[e^{-t ||f(x)-f(y)||_2^2}]` `` and tolerance with `` `T = E[(f(x)^T f(y)) I_{l(x)=l(y)}]` ``.
- **Experimental setup**: pretrain on CIFAR-10, CIFAR-100, SVHN, and ImageNet-100 using `ResNet-18` for the first three datasets and `ResNet-50` for ImageNet-100, a memory bank, SGD with momentum `0.9`, batch size `128`, initial learning rate `0.03`, and `200` epochs.
- **Hard-loss hyperparameters**: use `alpha = 0.0819` for CIFAR-10 and CIFAR-100, `0.0315` for SVHN, and `0.034` for ImageNet-100, each with `4095` negative samples retained.

## Key Results

- Ordinary contrastive loss shows a reverse-U temperature curve: best top-1 linear accuracy is `83.27%` on CIFAR-10, `56.44%` on CIFAR-100, `95.47%` on SVHN, and `75.10%` on ImageNet-100, all at `tau = 0.3`.
- Very small or large temperatures are worse for ordinary contrastive loss: CIFAR-10 drops from `83.27%` at `tau = 0.3` to `79.75%` at `tau = 0.07` and `82.21%` at `tau = 1.0`.
- The simple loss without hardness-aware weighting is much weaker: CIFAR-10 `74.83%`, CIFAR-100 `39.31%`, SVHN `70.83%`, and ImageNet-100 `48.09%`.
- Explicit hard negative sampling recovers or exceeds ordinary contrastive performance: hard contrastive reaches `84.19%` on CIFAR-10, `57.54%` on CIFAR-100, `95.26%` on SVHN, and `74.70%` on ImageNet-100 at larger temperatures.
- Hard simple loss is also competitive once hard negatives are selected explicitly, peaking at `84.84%` on CIFAR-10 and `74.31%` on ImageNet-100, supporting the claim that hardness awareness is the critical ingredient.
- Uniformity rises as temperature decreases, while tolerance to same-class samples rises as temperature increases, quantitatively demonstrating the paper's uniformity-tolerance dilemma.

## Limitations

- The paper is mostly analytic and diagnostic; it does not directly solve the semantic-destruction problem beyond showing that explicit hard negative sampling helps.
- Empirical validation is restricted to vision benchmarks and linear evaluation, so the conclusions may not transfer unchanged to language or multimodal contrastive settings.
- The tolerance metric uses supervised class labels during analysis, even though the training setup is unsupervised; this is useful diagnostically but not part of the training objective.
- The theory is based on similarity-gradient behavior and asymptotic temperature limits rather than a complete generalization theory for contrastive representation learning.

## Concepts Extracted

- [[contrastive-loss]]
- [[contrastive-learning]]
- [[unsupervised-contrastive-learning]]
- [[representation-learning]]
- [[hard-negative-sampling]]
- [[embedding-uniformity]]
- [[instance-discrimination]]
- [[hardness-aware-loss]]
- [[uniformity-tolerance-dilemma]]

## Entities Extracted

- [[feng-wang-tsinghua]]
- [[huaping-liu]]
- [[tsinghua-university]]
- [[beijing-national-research-center-for-information-science-and-technology]]
- [[cifar-10]]
- [[cifar-100]]
- [[svhn]]
- [[imagenet-100]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
