---
type: source
subtype: paper
title: "Stochastic RAG: End-to-End Retrieval-Augmented Generation through Expected Utility Maximization"
slug: zamani-2024-stochastic
date: 2026-04-20
language: en
tags: [rag, retrieval, end-to-end-optimization, information-retrieval, nlp]
processed: true
raw_file: raw/papers/zamani-2024-stochastic/paper.pdf
raw_md: raw/papers/zamani-2024-stochastic/paper.md
bibtex_file: raw/papers/zamani-2024-stochastic/paper.bib
possibly_outdated: true
authors:
  - Hamed Zamani
  - Michael Bendersky
year: 2024
venue: "SIGIR 2024"
venue_type: conference
arxiv_id:
doi: 10.1145/3626772.3657923
url: https://dl.acm.org/doi/10.1145/3626772.3657923
citation_key: zamani2024stochastic
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2024 in a volatile domain (retrieval, RAG); re-verify against recent literature.

This paper proposes Stochastic RAG, a framework for end-to-end optimization of retrieval-augmented generation by relaxing two simplifying assumptions common in prior work: (1) top-k marginalization that treats retrieval as deterministic, and (2) document independence. The retrieval process is recast as stochastic sampling without replacement from the document score distribution, enabling gradient-based optimization via straight-through Gumbel-top-k. Any differentiable utility function (exact match, BLEU, ROUGE) can be used as the training signal. Applied to FiD-Light, Stochastic RAG advances state-of-the-art results on six of seven KILT benchmark datasets, covering open-domain QA, fact verification, slot-filling, and dialogue tasks.

## Problem & Motivation

Most RAG systems optimize retrieval and generation components separately, or use simplifying assumptions for joint optimization: (1) top-k approximation of marginalization over all document sets, and (2) document independence — feeding each retrieved document to the LM independently rather than jointly. These approximations decouple retrieval quality from the downstream generation objective. The challenge is that selecting the top-k documents and ranking are non-differentiable operations, so standard gradient descent cannot propagate signals from the generation loss back to the retrieval model.

## Method

**RAG Expected Utility (Eq. 1):**
`RAG_EU = (1/n) Σ_{(x,y)∈T} Σ_{ŷ∈Y} U(y, ŷ) · p(ŷ | x; G_θ, R_φ)`

where `U` is any utility function bounded in `[0,1]` and `U(y,y)=1`.

**Stochastic reformulation (Eq. 3):**
`p(ŷ | x; G_θ, R_φ) = E_{d ~ p(d|x;R_φ)} [p(ŷ | x, d; G_θ)]`

This expectation is approximated by Monte Carlo sampling from the retrieval distribution.

**Sampling without replacement (Eq. 5–6):**
Document-level probabilities via softmax: `p(d_i | x; R_φ) = exp(s^φ_{xd_i}) / Σ_d exp(s^φ_{xd})`

Sampling without replacement uses the Plackett-Luce chain rule: `p(d | x; R_φ) = Π_{i=1}^{|d|} p(d_i | x; R_φ) / (1 - Σ_{j<i} p(d_j | x; R_φ))`

**Straight-through Gumbel-top-k (Eq. 7):**
Perturb each document score with Gumbel noise: `G_d ~ -log(-log(U))` where `U ~ Uniform(0,1)`.
Forward pass: argmax (top-k selection); backward pass: softmax gradients flow through.
This makes the sampling step differentiable.

**Output space Y estimation:**
Every `N=10,000` training steps, run beam search on training inputs to collect 100 candidate outputs; randomly sample `m=10`; always include the gold output `y`. Pre-compute utility values for these candidates for the next `N` steps.

**Model backbone:** FiD-Light (T5-Base 220M or T5-XL 3B) on top of a dense retrieval model pre-trained on MS MARCO. Input token limit `384`, output `64`. Batch size `128`, up to `k=40` retrieved passages (T5-Base) or `k=8` (T5-XL). Learning rate `1e-3` (T5-Base) / `5e-4` (T5-XL), Adafactor optimizer, `50,000` training steps. T5X framework on TPUs.

## Key Results

KILT leaderboard blind test (Table 1), KILT-score metrics:
- NQ (KILT-EM): FiD-Light XL 51.1 → **Stochastic RAG XL 53.0** (+1.9)
- HotpotQA (KILT-EM): 29.2 → **31.1** (+1.9)
- TriviaQA (KILT-EM): 63.7 → **64.7** (+1.0)
- FEVER (KILT-AC): 84.5 → **84.8** (+0.3)
- T-REx (KILT-AC): 76.3 → **78.3** (+2.0)
- zsRE (KILT-AC): 84.0 → **87.0** (+3.0; also beats GripRank by large margin)
- WoW (KILT-F1): 13.1 → 14.2 (+1.1; GripRank 14.7 remains best)

State-of-the-art on 6/7 datasets. Base (220M) also improves on all 7 datasets. Model is robust to number of Monte Carlo samples (Figure 1), with slight improvement from more samples on TriviaQA.

## Limitations

- Evaluated exclusively on short-text generation tasks (output ≤64 tokens); applicability to long-form generation is untested.
- Output space `Y` is estimated by a fixed-size beam-search sample, which may miss high-utility outputs and introduces a refresh latency every `N=10,000` steps.
- The framework assumes a bounded, task-specific utility function; utility functions for open-ended generation remain undefined.
- Stochastic sampling is applied only at training time; inference still uses deterministic top-k retrieval.
- Evaluated only on Wikipedia-based KILT; generalization to web-scale or domain-specific corpora is not demonstrated.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[retrieval-enhanced-machine-learning]]
- [[sampling-without-replacement]]
- [[gumbel-top-k]]
- [[gumbel-softmax]]
- [[expected-utility-maximization]]
- [[end-to-end-rag-optimization]]
- [[fusion-in-decoder]]
- [[dense-retrieval]]
- [[plackett-luce-ranking]]
- [[kilt-score]]
- [[beam-search]]
- [[knowledge-distillation]]

## Entities Extracted

- [[hamed-zamani]]
- [[michael-bendersky]]
- [[university-of-massachusetts-amherst]]
- [[google]]
- [[fid-light]]
- [[kilt-benchmark]]
- [[natural-questions]]
- [[triviaqa]]
- [[hotpotqa]]
- [[fever]]
- [[t-rex-dataset]]
- [[zsre-dataset]]
- [[wizard-of-wikipedia]]
- [[ms-marco]]
- [[t5]]
- [[t5x]]
- [[alireza-salemi]]
- [[sebastian-hofstatter]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
