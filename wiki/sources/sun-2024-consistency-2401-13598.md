---
type: source
subtype: paper
title: Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction
slug: sun-2024-consistency-2401-13598
date: 2026-04-20
language: en
tags: [relation-extraction, zero-shot-learning, synthetic-data, llm, document-level]
processed: true

raw_file: raw/papers/sun-2024-consistency-2401-13598/paper.pdf
raw_md: raw/papers/sun-2024-consistency-2401-13598/paper.md
bibtex_file: raw/papers/sun-2024-consistency-2401-13598/paper.bib
possibly_outdated: false

authors:
  - Qi Sun
  - Kun Huang
  - Xiaocui Yang
  - Rong Tong
  - Kun Zhang
  - Soujanya Poria
year: 2024
venue: WWW 2024
venue_type: conference
arxiv_id: 2401.13598
doi:
url: http://arxiv.org/abs/2401.13598
citation_key: sun2024consistency
paper_type: method

read_status: unread

domain: nlp
---

## Summary

This paper introduces GenRDK, a zero-shot document-level relation triplet extraction framework that uses large language models to synthesize long documents, entity sets, and relation triplets for unseen relation types, then denoises the generated labels before supervised fine-tuning. The core idea is to replace unavailable human annotations with a multi-step chain-of-retrieval prompting pipeline over ChatGPT, followed by a consistency-guided cross-document denoising module that fuses synthetic labels with pseudo labels predicted by a pre-denoising model. Using denoised synthetic data to fine-tune LLaMA2-13B-Chat, the method improves both ZeroDocRTE and ZeroDocRE on DocRED and Re-DocRED, showing that cross-document consistency can partially correct hallucinated or missing relational facts in LLM-generated supervision.

## Problem & Motivation

The paper targets zero-shot document-level relation triplet extraction, where systems must extract `(head entity, tail entity, unseen relation)` from multi-sentence documents without labeled training data for the target relation types. Prior zero-shot relation extraction work is mostly sentence-level or assumes gold entity pairs, which misses the harder setting where relations span sentences and entity detection must be solved jointly. The authors argue that modern LLMs can generate synthetic long-form documents containing unseen relations, but raw synthetic labels are noisy because hallucinations introduce false triplets and omit valid ones. GenRDK is proposed to turn LLM latent knowledge into usable training data while reducing this label noise.

## Method

- **Task setup**: split relation types into seen and unseen sets with `R = R_s ∪ R_u` and `R_s ∩ R_u = ∅`; training uses seen-label data plus synthetic unseen-label documents, while evaluation targets ZeroDocRTE and ZeroDocRE on unseen relations.
- **Chain-of-retrieval prompting**: for each unseen relation `r_i ∈ R_u`, ChatGPT first selects related relations `{r_ij}`, then generates a fictional document `d_ik` containing `r_i` and related relations, with `temperature = 1` to improve diversity.
- **Structured synthetic labeling**: after document generation, ChatGPT extracts entity set `E_k`, all relation triplets `{(e_s, e_o, r_l) | e_s, e_o ∈ E_k, r_l ∈ R}`, reasoning explanations `(e_s, e_o, r_l, a_c)`, supporting sentences `(e_s, e_o, r_l, h_p)`, and finally consolidated structured labels.
- **Pre-denoising model**: fine-tune LLaMA2-13B-Chat with [[low-rank-adaptation]] on seen data using randomly composed relation groups `R_s = [R_1, ..., R_m]`, optimizing samples of the form `\hat{M} <- Train(M, I, d_i^s, R_j, T_ij^s)` so the model can predict pseudo labels on synthetic unseen-relation documents.
- **Pseudo-label inference**: generate pseudo labels with `P_i = \hat{M}(I, d_i^u, R_u)`, then compare them against original synthetic labels to identify reliable cross-document relational facts.
- **Consistency-guided denoising**: build two cross-document knowledge graphs `KG_s` and `KG_p` from synthetic and pseudo labels, score each triplet with `s_ijk = F_ijk^s + F_ijk^p`, and prune low-consistency triplets using the per-relation threshold `η_k = \bar{s_ijk} - sqrt((1/(N_k^η - 1)) * Σ_l (s_ijk - \bar{s_ijk})^2)`.
- **Final extractor**: relabel the synthetic corpus with the denoised graph `KG_d`, filter synthetic documents lacking useful unseen triplets, and fine-tune the final model via `\tilde{M} <- Train(M, I, \hat{d}_i^{syn}, R_u, \hat{T}_i^{syn})`.
- **Training details**: ZeroDocRTE uses LLaMA2-13B-Chat with LoRA, learning rate `1e-6`, batch size `20`, on `4 × NVIDIA RTX A6000-48G`; ZeroDocRE uses a graph-based DocRE model with RoBERTa-large, AdamW, learning rate `3e-5`, `6%` warmup, and batch size `8`.

## Key Results

- **ZeroDocRTE, Re-DocRED test**: GenRDK reaches `13.1 ± 2.6` `F1` at `m = 5` and `8.2 ± 0.6` at `m = 10`, beating ChatGPT (`11.8 ± 3.8`, `8.1 ± 1.5`) and LLaMA2-13B-Chat (`8.7 ± 3.0`, `5.2 ± 0.8`).
- **ZeroDocRTE, DocRED test**: GenRDK reaches `14.2 ± 1.3` `F1` at `m = 5` and `9.4 ± 0.6` at `m = 10`, above ChatGPT (`11.2 ± 5.1`, `8.9 ± 2.3`) and LLaMA2-13B-Chat (`9.0 ± 1.8`, `5.5 ± 0.8`).
- **ZeroDocRE, Re-DocRED / DocRED test at `m = 5`**: GenRDK scores `41.3 ± 8.9` and `41.5 ± 8.7` `F1`, outperforming ChatGPT by `19.6` and `17.9` `F1`.
- **Denoising gains over raw synthetic data**: for ZeroDocRTE test at `m = 5`, GenRDK improves over CoR from `11.4 -> 13.1` on Re-DocRED and `12.1 -> 14.2` on DocRED.
- **Prompt design matters**: chain-of-retrieval improves ZeroDocRTE test `F1` from `9.04 -> 13.23` on Re-DocRED and `9.77 -> 13.38` on DocRED relative to a vanilla prompt; for ZeroDocRE, it improves from `42.45 -> 49.21` and `34.98 -> 48.30`.
- **Backbone robustness under denoising**: for the authors' RoBERTa-large DocRE system, denoised synthetic data improves test `F1` from `49.21 -> 51.88` on Re-DocRED and from `48.30 -> 51.31` on DocRED.

## Limitations

- Absolute ZeroDocRTE performance remains low, with best test `F1` only `13.1` on Re-DocRED and `14.2` on DocRED at `m = 5`, so the task is far from solved.
- The synthetic corpus is generated from ChatGPT and fictional documents, so distribution shift and hallucinated facts remain central risks even after denoising.
- The consistency score is frequency-based; rare but correct relational facts can be pruned if they do not recur often enough across synthetic documents.
- Evaluation is limited to DocRED and Re-DocRED splits with `m ∈ {5, 10}`, so transfer to other domains or relation schemas is untested.
- The pipeline depends on both strong LLM generation and an additional pre-denoising model, which increases complexity and cost relative to direct prompting baselines.

## Concepts Extracted

- [[document-level-relation-extraction]]
- [[relation-triplet-extraction]]
- [[zero-shot-learning]]
- [[large-language-model]]
- [[synthetic-data]]
- [[chain-of-retrieval]]
- [[knowledge-denoising]]
- [[knowledge-graph]]
- [[low-rank-adaptation]]
- [[parameter-efficient-fine-tuning]]

## Entities Extracted

- [[qi-sun]]
- [[kun-huang]]
- [[xiaocui-yang]]
- [[rong-tong]]
- [[kun-zhang]]
- [[soujanya-poria]]
- [[chatgpt]]
- [[llama-2]]
- [[openai]]
- [[docred]]
- [[re-docred]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
