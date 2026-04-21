---
type: source
subtype: paper
title: Structure-Aware Language Model Pretraining Improves Dense Retrieval on Structured Data
slug: li-2023-structureaware-2305-19912
date: 2026-04-20
language: en
tags: [dense-retrieval, structured-data, code-search, product-search, contrastive-learning]
processed: true
raw_file: raw/papers/li-2023-structureaware-2305-19912/paper.pdf
raw_md: raw/papers/li-2023-structureaware-2305-19912/paper.md
bibtex_file: raw/papers/li-2023-structureaware-2305-19912/paper.bib
possibly_outdated: true
authors:
  - Xinze Li
  - Zhenghao Liu
  - Chenyan Xiong
  - Shi Yu
  - Yu Gu
  - Zhiyuan Liu
  - Ge Yu
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2305.19912
doi:
url: http://arxiv.org/abs/2305.19912
citation_key: li2023structureaware
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper proposes SANTA, a dense retriever for structured data such as code and product descriptions. Instead of relying only on standard language-model pretraining, SANTA continuously pretrains T5-family encoders with two structure-aware objectives: Structured Data Alignment (SDA), which contrastively aligns structured records with paired natural-language text, and Masked Entity Prediction (MEP), which masks salient entities and asks the model to recover them. The method targets the modality gap between structured and unstructured text while also teaching models to focus on semantically important fields or identifiers. On code search and product search, SANTA delivers strong zero-shot retrieval and state-of-the-art fine-tuned performance, showing that structure-aware pretraining materially improves representation quality for structured retrieval.

## Problem & Motivation

Most dense retrievers are pretrained on unstructured natural language and therefore under-model the semantics carried by structure, such as code syntax, product attributes, and other semi-structured fields. That gap matters because structured retrieval tasks require matching free-form user queries to data whose meaning is distributed across schema-like tokens, identifiers, and attribute boundaries. Prior structure-aware pretraining mainly used masked language modeling, but the authors argue that masking alone is insufficient for retrieval-oriented representation learning. SANTA is motivated by two observations: structured documents often have naturally aligned unstructured text (for example, code documentation or product bullet points), and entities inside structured documents carry disproportionate semantic weight for downstream matching.

## Method

- SANTA follows the dense-retrieval formulation: encode query `q` and structured document `d` with T5-family models into `h_q = T5(q)` and `h_d = T5(d)` using the first decoder token representation, then score relevance with dot product `f(q, d) = sim(h_q, h_d)`.
- Retrieval finetuning minimizes the standard contrastive objective `L_DR = -log exp(f(q, d+)) / (exp(f(q, d+)) + sum_{d- in D-} exp(f(q, d-)))`, where negatives come from in-batch sampling or hard negatives.
- Continuous pretraining adds a structure-aware objective `L = L_SDA + L_MEP`, combining Structured Data Alignment (SDA) and Masked Entity Prediction (MEP).
- SDA pairs structured documents with naturally aligned passages `p` such as code documentation or product bullet points, then optimizes a contrastive loss `L_SDA` that pulls matched `<p, d+>` pairs together and pushes irrelevant structured documents away in the shared embedding space.
- MEP identifies entities in structured documents `X_d = {x_1, ent_1, x_2, ent_2, ..., ent_n}`, masks them as `X_d^mask = {x_1, <mask>_1, x_2, <mask>_2, ..., x_n}`, and trains the decoder to generate `Y_d = {<mask>_1, ent_1, ..., <mask>_n, ent_n}` with autoregressive loss `L_MEP`; repeated entities share the same special token.
- Pretraining data comes from CodeSearchNet and ESCI (large): code-documentation pairs for code retrieval and product-description/bullet-point pairs for product retrieval. Reported positive-pair counts include Python `429,596`, PHP `514,127`, Java `454,433`, and Product `331,590`; entity coverage ranges from `15.4%` to `28.8%` for code and `20.1%` for products.
- Downstream finetuning uses Adv for code search and ESCI (small) for product search. Evaluation metrics are `MRR@100` and `NDCG@100`.
- Initialization uses `CodeT5-base` for code search and `T5-base` for product search. Continuous training uses learning rates `5e-5` (code) and `1e-4` (product) for `6` epochs; in-batch finetuning uses `1e-5` for code over `6` epochs and `5e-5` for product over `60` epochs; hard-negative finetuning uses `1e-6` (code) and `1e-5` (product), batch size `16`, warmup proportion `0.1`, and negatives sampled from the top `100` retrieved candidates.

## Key Results

- In zero-shot evaluation, SANTA reaches code-search `MRR = 46.1`, beating `CodeRetriever = 34.7` and dramatically outperforming `CodeT5 = 0.03`; on product search it reaches `NDCG@100 = 76.38` (Two-C) and `77.14` (Four-C), versus `T5 = 70.21/71.25`.
- After finetuning, `SANTA (Hard Negative)` achieves code-search `MRR = 47.5` and product-search `NDCG@100 = 82.59/83.15`, exceeding `CodeRetriever = 43.0`, `CodeRetriever (AR2) = 46.9`, and `T5 = 79.77/80.46`.
- Ablations show SDA is the main driver of gains: zero-shot code-search `MRR` rises from `0.03` with baseline T5 to `45.01` with `T5 (w/ SDA)`, while adding MEP further improves SANTA to `46.08`.
- The entity-aware mask is materially better than random span masking on code retrieval: zero-shot `46.08 vs 35.88`, and fine-tuned `47.28 vs 42.11`.
- Appendix results show broader generalization: on CodeSearch, SANTA reports overall zero-shot `MRR = 70.9` vs `69.1` for CodeRetriever; on Adv, SANTA reports `48.1` zero-shot vs `34.7` for CodeRetriever and `48.6` after hard-negative finetuning vs `45.1`.

## Limitations

- The method depends heavily on the quality of naturally aligned structured-unstructured pairs; noisy alignment directly weakens the SDA objective.
- The pretraining setup uses task-specific aligned text sources, especially code descriptions and product bullet points, which may limit transfer to other structured domains.
- The empirical study is centered on retrieval; the paper explicitly leaves open whether SANTA also improves other downstream tasks such as code summarization or code generation.

## Concepts Extracted

- [[dense-retrieval]]
- [[structured-data]]
- [[structure-aware-pretraining]]
- [[structured-data-alignment]]
- [[masked-entity-prediction]]
- [[contrastive-learning]]
- [[code-search]]
- [[product-search]]
- [[hard-negative-mining]]
- [[zero-shot-retrieval]]

## Entities Extracted

- [[xinze-li]]
- [[zhenghao-liu]]
- [[chenyan-xiong]]
- [[shi-yu]]
- [[yu-gu-neu]]
- [[zhiyuan-liu]]
- [[ge-yu]]
- [[microsoft]]
- [[t5]]
- [[codet5]]
- [[codesearchnet]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
