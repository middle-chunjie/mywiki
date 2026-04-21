---
type: source
subtype: paper
title: "Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards?"
slug: rodriguez-2021-evaluation
date: 2026-04-20
language: en
tags: [leaderboards, evaluation, item-response-theory, question-answering, benchmarks]
processed: true

raw_file: raw/papers/rodriguez-2021-evaluation/paper.pdf
raw_md: raw/papers/rodriguez-2021-evaluation/paper.md
bibtex_file: raw/papers/rodriguez-2021-evaluation/paper.bib
possibly_outdated: true

authors:
  - Pedro Rodriguez
  - Joe Barrow
  - Alexander Miserlis Hoyle
  - John P. Lalor
  - Robin Jia
  - Jordan Boyd-Graber
year: 2021
venue: "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)"
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2021.acl-long.346
url: https://aclanthology.org/2021.acl-long.346
citation_key: rodriguez2021evaluation
paper_type: method

read_status: unread
read_date:
rating:

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2021; re-verify against recent literature. This paper argues that leaderboard examples are not equally informative and proposes Difficulty and Ability Discriminating (DAD) leaderboards, a Bayesian [[item-response-theory]] framework that jointly models model skill and example properties. On SQuAD 2.0, the authors fit item difficulty, discriminability, and feasibility parameters with mean-field [[variational-inference]] over `161` development subjects, `115` test subjects, and `11,873` items. Compared with mean-accuracy ranking, IRT-based ability estimates are more stable across subsamples and transfer better from development to test settings. The model also surfaces likely [[annotation-error]], partitions datasets by difficulty, and supports more sample-efficient annotation through [[fisher-information]]-based item selection.

## Problem & Motivation

Standard NLP [[leaderboard]]s reduce evaluation to a single aggregate score, which hides that some examples are trivial, some are genuinely discriminative, and some are flawed or effectively unsolvable. The paper argues that this obscures real scientific progress, encourages SOTA chasing, and makes it hard to know whether a benchmark is still informative. The goal is not to discard leaderboards, but to redesign them so they rank systems more reliably, expose where gains come from, identify bad examples, and guide future annotation when evaluation budgets are limited.

## Method

- The paper defines a DAD leaderboard in which each submitted system is a subject with latent skill `\theta_j`, and each evaluation example is an item with latent [[difficulty]] `\beta_i`, [[discriminability]] `\gamma_i`, and [[feasibility]] `\lambda_i`.
- The response model is `p(r_{ij} = 1 | \theta_j, \beta_i, \gamma_i, \lambda_i) = \lambda_i / (1 + e^{-\gamma_i(\theta_j - \beta_i)})`, where higher skill relative to difficulty raises correctness probability, higher discriminability sharpens separation, and lower feasibility caps solvability.
- Three one-dimensional variants are compared: IRT-base fixes `\gamma_i = 1` and `\lambda_i = 1`; IRT-disc learns `\gamma_i` but fixes `\lambda_i = 1`; IRT-feas learns all three item parameters. A multidimensional extension, IRT-vec, rewrites the exponent as `-\gamma_i(\sum_k \theta_{j,k} - \beta_{i,k})` and uses `10` ability dimensions.
- Priors are `\theta_j, \beta_i, \gamma_i \sim \mathcal{N}(\mu_z, \tau_z^{-1})`, `\lambda_i \sim U[0,1]`, with hyperpriors `\mu_z \sim \mathcal{N}(0, 10^6)` and `\tau_z \sim \Gamma(1, 1)`.
- Inference uses mean-field [[variational-inference]] with factorized `q(\mu) q(\tau) \prod_{i,j} q(\theta_j) q(\beta_i) q(\gamma_i)`, Gaussian variational families for `\theta`, `\beta`, `\gamma`, Gaussian families for `\mu_z`, and Gamma families for `\tau_z`. The ELBO is optimized with ADAM.
- Experiments use the SQuAD 2.0 leaderboard with `161` development subjects, `115` test subjects, and `11,873` items (`1.9M` subject-item pairs). Response prediction holds out `10%` of responses and reports ROC AUC, macro F1, and accuracy on an imbalanced label distribution with `80.4%` correct responses.
- A logistic-regression baseline is implemented in [[vowpal-wabbit]] using subject IDs, item IDs, question/context/title words, answer statistics, topic features, and optionally IRT-derived features, mainly to test whether simpler feature-based models can match the Bayesian approach.
- For [[benchmark-reliability]] analysis, the authors repeatedly subsample evaluation data, refit IRT-feas, and compare Kendall rank correlation of IRT ability versus classical mean accuracy across mutually exclusive dev partitions and between dev and test rankings.
- For cold-start annotation, item selection is driven by [[fisher-information]] `I_i(\theta_j) = \gamma_i^2 p_{ij}(1 - p_{ij})`, aggregated as `Info(i) = \sum_j I_i(\theta_j)`. The procedure seeds annotation with the `25` most discriminative items and then adaptively selects new items.
- Implementation details reported in the appendix: IRT-base, IRT-disc, and IRT-feas train for `1000` epochs with learning rate `0.1` and no early stopping; IRT-vec trains for `2500` epochs. The linear baseline runs Hyperopt for `20` iterations over learning rate, `L2`, and hash bits.

## Key Results

- On held-out response prediction over `1.9M` subject-item pairs, all four IRT variants outperform the best linear model on ROC AUC; the paper highlights IRT-vec as the strongest predictor and notes that the strongest linear ablation mainly uses IRT-derived features.
- Cross-model agreement is high: Kendall correlation of learned ability is `0.947` between IRT-feas and IRT-disc, `0.907` between IRT-disc and IRT-base, and `0.895` between IRT-feas and IRT-base.
- Generalization from development to test is slightly better under IRT ability than exact-match ranking: in Table 3, `Ability_dev` vs `Ability_test` reaches Kendall `\tau = 0.950`, compared with `EM_dev` vs `Ability_test` at `0.931`.
- For active evaluation construction, three IRT-based sampling strategies outperform random sampling at low annotation budgets, while pure difficulty-based selection performs worse; the gain is reported across ten trials with `95%` confidence bands.
- Manual inspection of `60` SQuAD items shows that negative discriminability is strongly associated with flawed or wrong items; the three-way annotation task among three authors reaches Krippendorff's `\alpha = 0.344`.
- The feasibility parameter exposes likely unsolvable items: in SQuAD 2.0, the bottom `5%` of items have `\lambda < 0.434`, the bottom `7.5%` have `\lambda < 0.698`, and the bottom `10%` have `\lambda < 0.931`.
- Reported runtime is practical for leaderboard analysis: IRT-feas averages `113 ± 2.31` seconds on CPU, while IRT-vec averages `110 ± 0.5` seconds on GPU.

## Limitations

- The method requires item-level predictions for every system on every evaluation example, which are often available only to benchmark organizers rather than public leaderboard users.
- The empirical case study is centered on one benchmark, SQuAD 2.0, so external validity to other NLP tasks and newer benchmark ecosystems is not established.
- The authors note that mutually exclusive subsampling is not fully independent, which weakens the cleanliness of the reliability analysis.
- Although textual features help the linear baseline, the Bayesian IRT models do not directly incorporate item content, limiting interpretability about which linguistic properties drive difficulty.
- Multidimensional IRT improves predictive accuracy but remains hard to interpret, and the appendix reports limited success extracting meaningful clusters from the `10`-dimensional model.
- The paper itself notes that IRT-feas can overfit relative to IRT-disc in some dev-to-test correlation analyses, so the most expressive model is not uniformly best.

## Concepts Extracted

- [[leaderboard]]
- [[item-response-theory]]
- [[difficulty]]
- [[discriminability]]
- [[feasibility]]
- [[variational-inference]]
- [[fisher-information]]
- [[active-learning]]
- [[annotation-error]]
- [[benchmark-reliability]]
- [[question-answering]]
- [[overfitting]]

## Entities Extracted

- [[pedro-rodriguez]]
- [[joe-barrow]]
- [[alexander-hoyle]]
- [[john-p-lalor]]
- [[robin-jia]]
- [[jordan-boyd-graber]]
- [[university-of-maryland]]
- [[university-of-notre-dame]]
- [[university-of-southern-california]]
- [[squad-2-0]]
- [[pyro]]
- [[pytorch]]
- [[vowpal-wabbit]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
