---
type: source
subtype: paper
title: Diversify Question Generation with Retrieval-Augmented Style Transfer
slug: gou-2023-diversify
date: 2026-04-20
language: en
tags: [question-generation, retrieval, style-transfer, reinforcement-learning, nlp]
processed: true

raw_file: raw/papers/gou-2023-diversify/paper.pdf
raw_md: raw/papers/gou-2023-diversify/paper.md
bibtex_file: raw/papers/gou-2023-diversify/paper.bib
possibly_outdated: true

authors:
  - Qi Gou
  - Zehua Xia
  - Bowen Yu
  - Haiyang Yu
  - Fei Huang
  - Yongbin Li
  - Cam-Tu Nguyen
year: 2023
venue: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.104
url: https://aclanthology.org/2023.emnlp-main.104
citation_key: gou2023diversify
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper proposes RAST, a retrieval-augmented style transfer framework for diverse question generation from a passage-answer pair. Instead of relying only on latent variables or stochastic decoding, the method constructs question style templates by masking context-sensitive spans, retrieves alternative templates from an external corpus with DPR-style dense retrieval, and uses a T5-based style transfer model to produce final questions. Training jointly optimizes the retriever and generator with reinforcement learning, where a generative QA model supplies a consistency reward and Jaccard overlap with the retrieved template supplies a diversity reward. Across SQuAD and NewsQA, RAST substantially improves diversity metrics while keeping consistency close to or above strong baselines such as CVAE, Composition Sampling, and Nucleus-T5.

## Problem & Motivation

Question generation has an inherent one-to-many structure: the same passage and answer can support multiple valid questions with different wording and focus. Prior diverse QG approaches mostly depend on internal latent variables, content selection, or sampling heuristics, which are harder to control and do not explicitly leverage external expression patterns. The paper argues that external question templates offer a more interpretable way to induce stylistic diversity, but existing template-based approaches either require manual question-type annotations or paired rewriting data. RAST is motivated by closing that gap: it aims to retrieve style templates automatically and use them to diversify surface form without sacrificing answer consistency.

## Method

- **Task formulation**: for context `x = {c, a}` and external template corpus `Z`, the model factors diverse QG as `p(y | x, Z) = E[p(z_0 | x) p(z | z_0) p(y | z, x)]`, where `z_0` is an initial template and `z` is a retrieved alternative style.
- **Template construction**: question templates are built by keeping stopwords and interrogatives while masking named entities, noun phrases, and context-sensitive tokens with `[MASK]`; near-duplicates are removed with pairwise Jaccard similarity.
- **Retriever**: RAST initializes a DPR-style dense retriever with two BERT encoders, using `q(z) = BERT_1(z)`, `q(z_0) = BERT_2(z_0)`, and retrieval scores `p_phi(z | z_0) ∝ exp(q(z)^T q(z_0))`; maximum inner product search is used for efficient lookup.
- **Generator**: a T5-base style transfer model generates the final question auto-regressively with `p_theta(y | x, z) = ∏_t p_theta(y_t | x, z, y_<t)`. Vanilla QG is also implemented with T5-base to supply a proxy question at inference.
- **Supervised warm start**: before RL, the generator is trained with cross-entropy `L_theta^CE = -Σ_i y_i log p(ŷ_i | x, z̃_0)` on corrupted templates `z̃_0`, where corruption includes random entity replacement, noun insertion, `[MASK]` deletion, and template substitution to improve robustness to noisy retrieval.
- **RL objective**: joint training minimizes `L^RL(θ, φ) = -E[r(y^s, z^s)]` with REINFORCE; the style transfer gradient uses a self-critical baseline `b` from the greedy output, and the full update adds a KL regularizer `β KL(p_theta || p_theta^*)`.
- **Rewards**: consistency is computed with a generative QA model as `L_qa = -(1/T_a) Σ_i log p(a_i | c, y^s, a_<i)` and `r_cons = exp(-L_qa)`; diversity is `r_div = |z^s ∩ y^s| / |z^s ∪ y^s|`; total reward is `r = r_cons + λ r_div`.
- **Diversity-driven sampling**: during training, retrieved templates are clustered by Jaccard similarity into `k` groups and one style is sampled per cluster to avoid collapse onto near-duplicate templates; during inference, the top retrieved style is used.
- **Implementation details**: inference generates `N = 5` questions for evaluation; supervised training uses `top-p = 0.9`, `top-k = 30`, `5` epochs, and batch sizes `64/32/6` for SQuAD split 1 / SQuAD split 2 / NewsQA; RL uses `7` epochs, warmup ratio `0.2`, retrieval counts `100` (train) and `500` (eval), `β = 0.1/0.1/0.05`, and `λ = 0.5/0.5/0.4` across the three datasets.
- **Dataset-specific RL hyperparameters**: input max lengths are `128/512/1250`, generator learning rate is `1e-6`, retriever learning rate is `1e-7`, batch sizes are `12/8/2`, and templates per example are `3/3/2` for SQuAD split 1 / split 2 / NewsQA.

## Key Results

- **SQuAD split 1**: RAST achieves `Top-1 BLEU = 19.25`, `Oracle BLEU = 23.23`, `Pairwise BLEU = 48.91`, and `Overall BLEU = 9.14`, improving overall score over Composition Sampling (`7.21`) and CVAE (`5.99`) while also being more diverse than both.
- **SQuAD split 2**: RAST reaches `19.36 / 22.59 / 56.42 / 7.75` for Top-1 / Oracle / Pairwise / Overall BLEU, outperforming Composition Sampling on Top-1 (`15.94`) and Overall (`6.61`) with comparable Oracle BLEU.
- **NewsQA**: RAST reports `11.02 / 16.26 / 23.16 / 7.74`, giving the strongest overall result among listed baselines and a much lower Pairwise BLEU than CVAE (`41.37`) or Nucleus-T5 (`27.47`), indicating markedly higher diversity.
- **Reward trade-off**: when `λ` increases from `0.05` to `0.25` on SQuAD split 1, Pairwise BLEU drops from `74.7` to `35.18` while QA F1 decreases only from `85.16` to `78.2`, showing a favorable diversity-consistency trade-off.
- **Ablations**: removing clustering worsens SQuAD split 1 from `Overall BLEU = 9.14` to `5.88` and from `Pairwise BLEU = 48.91` to `61.06`; freezing end-to-end retriever training yields `Overall BLEU = 9.02`; querying with the unmasked question instead of the masked template yields `Pairwise BLEU = 54.09`.
- **Human evaluation**: on 50 SQuAD split 1 examples, RAST scores `3.36` on consistency and `2.36` on diversity versus Nucleus sampling's `3.00` and `1.78`; Fleiss' kappa ranges from `0.60` to `0.75`.

## Limitations

- QG evaluation is intrinsically difficult because multiple distinct questions can be valid; the paper relies on several automatic metrics and limited human evaluation rather than exhaustive human judgment.
- RL training is expensive and unstable relative to supervised learning; the reported RL stage still takes roughly `48`, `70`, and `120` hours on SQuAD split 1, SQuAD split 2, and NewsQA.
- The method is constrained by Transformer context length and does not solve long-context question generation beyond the configured maximum input lengths.
- Template retrieval remains noisy, which is why the approach needs template corruption during supervised initialization and clustering during RL exploration.
- The paper benchmarks against diverse-QG baselines available up to 2022-2023; because the domain is fast-moving, the strength of the comparison set may now be outdated.

## Concepts Extracted

- [[question-generation]]
- [[retrieval-augmented-generation]]
- [[style-transfer]]
- [[reinforcement-learning]]
- [[dense-passage-retrieval]]
- [[question-answering]]
- [[jaccard-similarity]]
- [[top-p-sampling]]
- [[self-critical-sequence-training]]
- [[diversity-driven-sampling]]
- [[sequence-to-sequence]]

## Entities Extracted

- [[qi-gou]]
- [[zehua-xia]]
- [[bowen-yu]]
- [[haiyang-yu]]
- [[fei-huang]]
- [[yongbin-li]]
- [[cam-tu-nguyen]]
- [[nanjing-university]]
- [[alibaba-group]]
- [[t5]]
- [[bert]]
- [[newsqa]]
- [[squad]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
