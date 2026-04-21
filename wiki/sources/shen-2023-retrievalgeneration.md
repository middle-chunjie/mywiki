---
type: source
subtype: paper
title: Retrieval-Generation Alignment for End-to-End Task-Oriented Dialogue System
slug: shen-2023-retrievalgeneration
date: 2026-04-20
language: en
tags: [task-oriented-dialogue, retrieval, generation, dialogue-systems, meta-knowledge]
processed: true
raw_file: raw/papers/shen-2023-retrievalgeneration/paper.pdf
raw_md: raw/papers/shen-2023-retrievalgeneration/paper.md
bibtex_file: raw/papers/shen-2023-retrievalgeneration/paper.bib
possibly_outdated: true
authors:
  - Weizhou Shen
  - Yingqi Gao
  - Canbin Huang
  - Fanqi Wan
  - Xiaojun Quan
  - Wei Bi
year: 2023
venue: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.514
url: https://aclanthology.org/2023.emnlp-main.514
citation_key: shen2023retrievalgeneration
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper studies end-to-end task-oriented dialogue under a retrieve-then-generate setup and argues that better retrieval does not automatically improve generation because retrieved KB entities are often highly homogeneous. It proposes MK-TOD, which aligns retrieval and generation in two ways: maximum marginal likelihood (MML) updates the retriever using response-generation signals, and retrieval-related meta knowledge helps the generator distinguish among similar entities. The meta knowledge covers retrieval order, confidence, and dialogue-history co-occurrence, and is injected through prefixes, prompts, or contrastive training. Across MultiWOZ 2.1, CamRest, and SMD, with T5 and ChatGPT backbones, the method improves response quality and makes generator performance track retriever quality more consistently.

## Problem & Motivation

End-to-end task-oriented dialogue systems avoid explicit belief-state annotations, but this also makes knowledge acquisition harder because the model must infer which KB entity matters directly from dialogue context. Prior retrieve-then-generate systems such as Q-TOD improve retrieval quality, yet the paper shows that response quality is only weakly correlated with retriever quality. The authors attribute this retrieval-generation misalignment to the homogeneity of retrieved entities: many candidates share nearly identical attribute-value structures, while the response generator is mainly trained on language tokens and may not learn to discriminate subtle KB differences. The goal is therefore not just better retrieval, but tighter coupling between retrieval signals and generation behavior.

## Method

- **Retriever-generator decomposition**: a BERT dual-encoder retriever encodes dialogue context `c_t` and each entity `e_i`, scores them by dot product `s_{t,i} = h_{c_t}^T h_{e_i}`, and returns top-`K` entities `E_t`. The generator then models `p(r_t | c_t, E_t; θ) = ∏_j p(r_{t,j} | r_{t,<j}, c_t, E_t; θ)`.
- **Generator training objective**: the base loss is negative log-likelihood, `L_NLL = -log p(r_t | c_t, E_t; θ)`.
- **Maximum marginal likelihood for retriever alignment**: because `L_NLL` cannot directly update the retriever, the paper optimizes `L_MML = -log Σ_{e_{t,i}∈E_t} q(e_{t,i}|c_t; φ) p(r_t | c_t, e_{t,i}; θ)`, with `q(e_{t,i}|c_t; φ) = exp(s_{t,i}) / Σ_j exp(s_{t,j})`. The joint loss is `L = αL_NLL + βL_MML`, using `α = 1`, `β = 1`.
- **Meta knowledge design**: each retrieved entity is annotated with retrieval order, retrieval confidence, and co-occurrence with dialogue history. Confidence is bucketed by score thresholds `(-∞, 0.4]`, `(0.4, 0.75]`, `(0.75, +∞)` in the main text, while the appendix gives token/prompt mappings for `< 0.25`, `[0.25, 0.75)`, and `>= 0.75`; the paper does not reconcile this discrepancy explicitly.
- **Meta knowledge implementations**: prefix maps metadata to discrete tokens such as `<2th-entity>`, `<mid-confidence>`, `<new-entity>`; prompt maps them to natural-language text such as `The top-2 recalled:` and `with middle confidence:`; contrastive learning compares entity-conditioned likelihood against a no-entity baseline.
- **Contrastive objective**: for positive retrieved entities `E_t*`, the model computes `d_{t,i} = log p(r_t | c_t, e_{t,i}; θ) / |r_t|` and baseline `d_t^- = log p(r_t | c_t; θ) / |r_t|`, then applies `L_CTR = Σ_i max(0, d_t^- - d_{t,i} + λ)` with `λ = 0.01`, giving `L = αL_NLL + βL_MML + γL_CTR`, `γ = 1`.
- **Negative entity training**: the generator is additionally exposed to the lowest-scoring entity `e_t^- ∉ E_t` with dedicated meta knowledge, improving discrimination during training but omitted at inference.
- **Backbones and implementation details**: retriever encoders are BERT; generators are T5 and ChatGPT (`gpt-3.5-turbo` via in-context learning). Training uses batch size `2` for T5-Base and `1` for T5-Large, gradient accumulation `32` / `64`, `1500` gradient steps, linear schedule, retriever and generator learning rates both `1e-4`, weight decay `0.01`, gradient clipping `0.01`, retriever max length `128`, context length `200`, entity length `100`, and output length `64`. Experiments run on a single `24 GB` RTX `3090`.
- **Retrieval settings**: large-scale KB sizes are `223` entities for MultiWOZ and `112` for CamRest; retrieved entity counts are `7` for T5-Base and `5` for T5-Large on large-scale KBs, and `6/6/8` on condensed MWOZ/CamRest/SMD for T5-Base.

## Key Results

- **Large-scale KBs, MWOZ**: `Ours_ctr (Large)` reaches `17.40` BLEU, `53.26` Entity F1, and `95.22` Recall@7; `Ours_prompt (Base)` also beats Q-TOD (T5-Large) on generation quality with `17.56` BLEU / `50.69` Entity F1 versus `15.52` / `46.74`.
- **Large-scale KBs, CamRest**: the best BLEU is `27.82` from `Ours_ctr (Large)`, while the best Entity F1 is `73.51` from `Ours_ctr (Base)`; both outperform Q-TOD's `21.44` BLEU / `63.88` Entity F1.
- **Condensed KBs**: on SMD, `Ours_ctr (Large)` achieves `25.43` BLEU and `73.31` Entity F1 versus Q-TOD (T5-Large) at `21.33` / `71.11`; on MWOZ, `Ours_prompt (ChatGPT)` lifts Entity F1 from `32.87` to `35.84`.
- **MML helps both retrieval and generation**: on large-scale MWOZ with prompt-based T5-Base, adding MML improves Recall@7 from `91.39` to `92.74` and Entity F1 from `50.41` to `50.69`; for contrastive training it improves BLEU from `14.78` to `15.96` and Entity F1 from `50.54` to `51.35`.
- **Negative entities help fine-tuned T5 but not ChatGPT**: for prefix-based T5-Base on large-scale MWOZ, adding negatives improves Entity F1 from `49.46` to `50.35` and Recall@7 from `90.24` to `92.51`, while analogous ChatGPT settings show no gain.
- **Behavioral analysis**: the authors report the retriever recalls `80.69%` of gold entities at top-1 on the MultiWOZ test set, and meta knowledge makes the generator prefer higher-ranked, higher-confidence entities more consistently than the baseline.

## Limitations

- MML requires computing generator likelihood for each retrieved entity, increasing training cost relative to plain NLL.
- The paper evaluates only a limited combination space of meta knowledge; it explicitly leaves combinations such as prompt plus contrastive learning or joint order-plus-co-occurrence analysis unexplored.
- The theoretical reason why meta knowledge improves task-oriented dialogue is not developed in depth; the evidence is primarily empirical.
- Prefix-style meta knowledge is weak for ChatGPT under few-shot in-context learning, suggesting that the approach is backbone-sensitive.
- CamRest is very small (`406` training dialogues), and the authors themselves note overfitting effects for larger generators.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[task-oriented-dialogue]]
- [[maximum-marginal-likelihood]]
- [[dense-retrieval]]
- [[dual-encoder]]
- [[meta-knowledge]]
- [[retrieval-generation-misalignment]]
- [[contrastive-learning]]
- [[in-context-learning]]
- [[negative-sampling]]
- [[co-occurrence]]
- [[prompt-engineering]]

## Entities Extracted

- [[weizhou-shen-sysu]]
- [[yingqi-gao]]
- [[canbin-huang]]
- [[fanqi-wan]]
- [[xiaojun-quan]]
- [[wei-bi]]
- [[sun-yat-sen-university]]
- [[tencent-ai-lab]]
- [[bert]]
- [[t5]]
- [[chatgpt]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
