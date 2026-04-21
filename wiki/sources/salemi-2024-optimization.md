---
type: source
subtype: paper
title: Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation
slug: salemi-2024-optimization
date: 2026-04-20
language: en
tags: [personalization, retrieval-augmentation, dense-retrieval, llm, ir]
processed: true
raw_file: raw/papers/salemi-2024-optimization/paper.pdf
raw_md: raw/papers/salemi-2024-optimization/paper.md
bibtex_file: raw/papers/salemi-2024-optimization/paper.bib
possibly_outdated: false
authors:
  - Alireza Salemi
  - Surya Kallumadi
  - Hamed Zamani
year: 2024
venue: SIGIR 2024
venue_type: conference
arxiv_id: 2404.05970
doi: 10.1145/3626772.3657783
url: https://dl.acm.org/doi/10.1145/3626772.3657783
citation_key: salemi2024optimization
paper_type: method
read_status: unread
domain: llm
---

## Summary

The paper studies how to personalize a frozen large language model by optimizing the retriever inside a retrieval-augmented generation pipeline rather than fine-tuning model weights. It proposes two retriever-training methods, ROPG-RL and ROPG-KD, that use downstream generation quality as supervision, plus two retriever-selection models, RSPG-Pre and RSPG-Post, that choose among multiple retrieval strategies on a per-input basis. Experiments on the seven-task LaMP benchmark with FlanT5-XXL show statistically significant gains on six datasets, with an average `5.5%` improvement over prior state of the art and `15.3%` average improvement over a non-personalized LLM. The work is notable because it frames LLM personalization as a retrieval optimization and selection problem under fixed-model and privacy-conscious constraints.

## Problem & Motivation

The paper addresses personalization for LLM-based generation when directly tuning per-user model parameters is too expensive, too slow, or risky from a privacy perspective. In retrieval-based personalization, the key bottleneck is selecting which personal documents from a user profile should be shown to the LLM, but standard learning-to-rank training is not directly applicable because there are no query-document relevance labels for the downstream personalized generation task. The authors therefore optimize retrieval using the LLM's own downstream performance as supervision and further argue that different inputs benefit from different retrieval strategies, motivating query-level retriever selection.

## Method

- **Pipeline formulation**: for input `x` from user `u`, retrieve profile items with `L = R(phi_q(x); P_u)`, build a personalized prompt `phi_p(x, L)`, and generate with a frozen LLM `M`; only retrieval components are updated.
- **ROPG-RL policy**: parameterize the retriever as `pi_theta(d|x) = exp(R_theta(phi_q(x), d)) / sum_{d' in P_u} exp(R_theta(phi_q(x), d'))`, approximated over top-`l` candidates with `l = 16`.
- **Single-document action space**: sample one profile document per trajectory to keep reward computation tractable; the paper reports little benefit from sampling multiple documents without replacement.
- **RL reward**: use `Reward(d; x, y) = Eval(y, M(phi_p(x, [d]))) - Eval(y, M(phi_p(x, [d_b])))`, where `d_b` is the baseline document selected by the initial retriever.
- **RL objective**: maximize expected reward with vanilla policy gradient, i.e. `E_{d ~ pi_theta}[Reward(d; x, y) log pi_theta(d|x)]`.
- **ROPG-KD target**: build a target distribution `p_t(d|x) ∝ exp(Eval(y, M(phi_p(x, [d]))))` over candidate profile items and minimize KL divergence between `p_t` and `pi_theta`.
- **Retriever backbone**: instantiate both optimization methods on Contriever, a shared-encoder dense bi-encoder with dot-product scoring; inference uses exact `kNN` over independently scored profile documents.
- **Retriever pool for selection**: compare and select among six options: no retrieval, recency, `BM25`, zero-shot `Contriever`, `ROPG-RL`, and `ROPG-KD`.
- **RSPG-Pre / RSPG-Post**: train retriever-selection models with a KL objective over retriever choices; `RSPG-Pre` scores each personalized prompt before generation, while `RSPG-Post` scores prompt plus generated output.
- **Selection encoder and training**: use `Longformer` because selector inputs can be long; training uses Adam with learning rate `1e-5`, `5%` warmup, gradient clipping `1`, retriever training for `10` epochs, selector training for `20` epochs, effective batch size `64`, LLM max length `512`, selector max length `1024`, beam size `4`, and top-`4` retrieved profile items at prompt construction time.

## Key Results

- Across all LaMP datasets, the best method yields `5.5%` average improvement over the previous state of the art and `15.3%` average improvement over the non-personalized baseline, with per-dataset gains over non-personalized ranging from `1.0%` to `33.8%`.
- `RSPG-Post` is the best overall system on `6/7` datasets: LaMP-1 accuracy `0.672`, LaMP-2 accuracy/F1 `0.430/0.339`, LaMP-3 MAE/RMSE `0.264/0.568`, LaMP-4 ROUGE-1/L `0.203/0.186`, LaMP-6 ROUGE-1/L `0.433/0.418`, and LaMP-7 ROUGE-1/L `0.461/0.409`.
- The main exception is LaMP-5, where retrieval selection does not beat the strongest base retriever; `Contriever` reaches ROUGE-L `0.433`, while `RSPG-Pre` ties the best ROUGE-1 at `0.483`.
- Fine-tuning Contriever with `ROPG-RL` improves over zero-shot Contriever on LaMP-1, LaMP-3, LaMP-4, and LaMP-7, while `ROPG-KD` additionally improves LaMP-6 and is stronger on tasks with binary feedback such as LaMP-1.
- Retriever selection substantially outperforms unsupervised query-performance-prediction baselines: `RSPG-Post` reaches selection success rates of `0.874` on LaMP-1, `0.962` on LaMP-2, `0.920` on LaMP-3, and `0.898` on LaMP-7.
- There is still headroom: the best method reaches only `68.3%` and `75.4%` of oracle upper-bound performance on LaMP-3 and LaMP-4, respectively.

## Limitations

- The prompt generator `phi_p` is template-based and fixed; prompt construction itself is not optimized.
- Evaluation is limited to the seven LaMP tasks, all centered on short text classification or short text generation rather than long-form personalized writing.
- The LLM is fixed to `FlanT5-XXL (11B)`, so the paper does not test joint retriever-generator optimization or transfer to newer instruction-tuned models.
- Training approximates the user profile with top-`l = 16` candidates and uses single-document trajectories for RL, which may miss higher-order interactions among retrieved profile items.
- The paper notes RL instability/overfitting concerns and uses exact `kNN`, leaving scalability and robustness under larger user profiles insufficiently explored.

## Concepts Extracted

- [[large-language-model]]
- [[retrieval-augmented-generation]]
- [[personalization]]
- [[dense-retrieval]]
- [[bi-encoder]]
- [[reinforcement-learning]]
- [[policy-gradient]]
- [[knowledge-distillation]]
- [[kl-divergence]]
- [[learning-to-rank]]
- [[retriever-selection]]
- [[query-performance-prediction]]

## Entities Extracted

- [[alireza-salemi]]
- [[surya-kallumadi]]
- [[hamed-zamani]]
- [[university-of-massachusetts-amherst]]
- [[lowes-companies]]
- [[lamp]]
- [[contriever]]
- [[flan-t5-xxl]]
- [[longformer]]
- [[pytorch]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
