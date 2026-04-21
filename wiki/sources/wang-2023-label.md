---
type: source
subtype: paper
title: "Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning"
slug: wang-2023-label
date: 2026-04-20
language: en
tags: [icl, llm, interpretability, prompting, text-classification]
processed: true

raw_file: raw/papers/wang-2023-label/paper.pdf
raw_md: raw/papers/wang-2023-label/paper.md
bibtex_file: raw/papers/wang-2023-label/paper.bib
possibly_outdated: true

authors:
  - Lean Wang
  - Lei Li
  - Damai Dai
  - Deli Chen
  - Hao Zhou
  - Fandong Meng
  - Jie Zhou
  - Xu Sun
year: 2023
venue: EMNLP 2023
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.609
url: https://aclanthology.org/2023.emnlp-main.609
citation_key: wang2023label
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper argues that in-context learning in decoder-only large language models is organized around label words that act as anchors. Using saliency-based information-flow analysis, attention intervention, and correlation studies on GPT2-XL and GPT-J across SST-2, TREC, AGNews, and EmoContext, the authors show that shallow layers aggregate demonstration semantics into label tokens while deeper layers read those tokens to make predictions. Building on that hypothesis, they introduce anchor re-weighting to improve accuracy, Hiddenanchor context compression to speed up inference, and an anchor-distance analysis for diagnosing class confusion. The work is notable because it ties a mechanistic hypothesis to practical gains in effectiveness, efficiency, and interpretability under few-shot classification.

## Problem & Motivation

In-context learning is widely used for large language models because it adapts behavior through natural-language demonstrations instead of parameter updates, but its internal mechanism remains unclear. Prior analyses often focused on prompt order, formatting, or example selection rather than tracing how task information moves through the network. This paper targets that gap by asking whether a small subset of prompt tokens, namely the demonstration label words, play a privileged role in carrying class information from early computation to final prediction. The motivation is both scientific and practical: if such anchors exist, they can support better diagnostics, more efficient inference, and lightweight accuracy improvements.

## Method

- **Experimental setup**: analyze GPT2-XL (`1.5B`, `48` layers) and GPT-J (`6B`, `28` layers) on four classification datasets: [[sst-2]], [[trec]], [[ag-news]], and EmoContext. Main evaluation uses `1000` test examples, `1` demonstration per class, and averages over `5` random seeds.
- **Saliency-based information flow**: define attention saliency at layer `l` as `I_l = |Σ_h A_{h,l} ⊙ ∂L(x) / ∂A_{h,l}|`, averaged over heads, so `I_l(i,j)` measures the importance of information flowing from token `j` to token `i`.
- **Anchor-flow metrics**: quantify three pathways with `S_wp`, `S_pq`, and `S_ww`, where `S_wp` measures text-to-label flow, `S_pq` measures label-to-target flow, and `S_ww` is the remaining word-to-word baseline. The hypothesis is that `S_wp` dominates in shallow layers while `S_pq` dominates in deep layers.
- **Causal isolation test**: validate shallow-layer aggregation by zeroing attention from preceding tokens to label words, i.e. set `A_l(p,i) = 0` for `i < p`, then measure `Label Loyalty` and `Word Loyalty` against the unmodified model.
- **Deep-layer extraction analysis**: measure how well attention from the target position to label words predicts final outputs using per-layer `AUCROC_l`, then aggregate layer contributions with `R_l = Σ_{i=1}^l (AUCROC_i - 0.5) / Σ_{i=1}^N (AUCROC_i - 0.5)`.
- **Anchor re-weighting**: reinterpret label-word attention as a logistic-regression-like classifier, then rescale anchor attention with `Â(q,p_i) = exp(β_0^i) A(q,p_i)`. Train `β` on `4` auxiliary examples per class using Adam with `lr = 0.01`, `β_1 = 0.9`, `β_2 = 0.999`, batch size `1`, and `10` epochs.
- **Hiddenanchor compression**: cache per-layer hidden states of label words and formatting tokens, `H = {{h_l^i}_{i=1}^C}_{l=1}^N`, then concatenate those cached states during inference instead of replaying the full demonstrations.
- **Error diagnosis**: project anchor key vectors onto principal query-variation directions and define predicted class confusion with normalized distances between projected keys, linking anchor geometry to misclassification patterns.

## Key Results

- Across four datasets, deep-layer attention to label words becomes strongly predictive of the final decision: `AUCROC_l` approaches `0.8` in late layers for both GPT2-XL and GPT-J, while cumulative contribution `R_l` rises mainly in middle and deep layers.
- Isolating label words in the first `5` layers causes the largest disruption in both `Label Loyalty` and `Word Loyalty`; isolating the last `5` layers or random non-label words has much smaller effect, supporting shallow-layer information aggregation.
- Anchor re-weighting improves average accuracy from `51.90` for vanilla `1`-shot ICL and `46.87` for vanilla `5`-shot ICL to `68.64`. Per-dataset scores are `90.07` on SST-2, `60.92` on TREC, `81.94` on AGNews, and `41.64` on EmoContext.
- Hiddenanchor is the strongest compression baseline on both models. On GPT-J it reaches `89.06` label loyalty, `75.04` word loyalty, and `55.59` accuracy versus `56.82` for full ICL; on GPT2-XL it reaches `79.47`, `62.17`, and `45.04`.
- Compression yields `1.1x` to `2.9x` inference acceleration depending on demonstration length, with the largest gain on AGNews and stronger speedups on GPT-J than GPT2-XL.
- Anchor-distance analysis recovers the most confusing TREC class pair, `Description-Entity`, and also highlights high-confusion pairs such as `Entity-Abbreviation` and `Description-Abbreviation`.

## Limitations

- The study is restricted to classification tasks; it does not test generative settings where label words are less explicit.
- The analysis focuses on conventional in-context learning and leaves variants such as chain-of-thought prompting unexamined.
- Due to hardware limits, the largest model studied is GPT-J at `6B`; the paper does not verify whether the same anchor mechanism scales unchanged to much larger models.
- Hiddenanchor still loses some accuracy relative to full-context ICL, and the paper shows that preserving prompt formatting is necessary for compression to work well.
- The evidence is strong but still partial: saliency, intervention, and correlation support the anchor hypothesis, yet they do not constitute a full mechanistic proof of all in-context learning behavior.

## Concepts Extracted

- [[in-context-learning]]
- [[large-language-model]]
- [[information-flow]]
- [[label-anchor]]
- [[saliency-score]]
- [[attention-weight]]
- [[logistic-regression]]
- [[context-compression]]
- [[prompt-formatting]]
- [[text-classification]]

## Entities Extracted

- [[lean-wang]]
- [[lei-li-pku]]
- [[damai-dai]]
- [[deli-chen]]
- [[hao-zhou]]
- [[fandong-meng]]
- [[jie-zhou-tencent]]
- [[xu-sun]]
- [[peking-university]]
- [[tencent]]
- [[gpt2-xl]]
- [[gpt-j]]
- [[sst-2]]
- [[trec]]
- [[ag-news]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
