---
type: source
subtype: paper
title: "RARE: RETRIEVAL AUGMENTED RETRIEVAL WITH IN-CONTEXT EXAMPLES"
slug: unknown-nd-rare-2410-20088
date: 2026-04-20
language: en
tags: [retrieval, dense-retrieval, in-context-learning, query-augmentation, embedding-models]
processed: true
raw_file: raw/papers/unknown-nd-rare-2410-20088/paper.pdf
raw_md: raw/papers/unknown-nd-rare-2410-20088/paper.md
bibtex_file: raw/papers/unknown-nd-rare-2410-20088/paper.bib
possibly_outdated: false
authors:
  - Atula Tejaswi
  - Yoonsang Lee
  - Sujay Sanghavi
  - Eunsol Choi
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2410.20088
doi:
url: https://openreview.net/pdf?id=6EkWIfvjj9
citation_key: unknownndrare
paper_type: method
read_status: unread
domain: ir
---

## Summary

RARe studies whether dense retrievers can benefit from in-context examples in the same way decoder-only large language models do. The core finding is negative for naive prompting but positive after training adaptation: simply prepending retrieved query-document exemplars at inference hurts existing embedders, while continued contrastive fine-tuning on augmented queries teaches the retriever to exploit those exemplars. The method retrieves semantically similar training queries with BM25, appends their paired positive documents to the target query, and keeps the standard dense retrieval objective unchanged. Across BeIR and RAR-b, RARe improves both LLM-backed and retriever-backed baselines, with especially strong gains on out-of-domain and reasoning-heavy tasks, while exposing a clear tradeoff between retrieval quality and substantial inference-time latency.

## Problem & Motivation

Retriever models are increasingly built from decoder-only LLM backbones or instruction-tuned embedders, but unlike autoregressive LLMs they do not naturally benefit from in-context examples at inference time. The paper shows that naively augmenting a retrieval query with example query-document pairs usually degrades retrieval quality, which suggests a mismatch between the pre-trained retriever's input distribution and the longer, demonstration-style query format. The motivation behind RARe is to preserve the practical appeal of in-context learning while adapting dense retrievers to exploit semantically similar exemplars for better task conditioning, especially under low-resource, cross-domain, and reasoning-oriented evaluation settings.

## Method

- **Baseline retriever setup**: encode query `q` and document `d` with a shared embedder `E(·)` into a common vector space, then rank by cosine similarity `cos(E(q), E(d))`. Training examples follow `(q, d+, d-)` with in-batch negatives under a standard contrastive objective.
- **Instruction-only query format**: prior work uses `q_inst = "Instruct: t_i; Query: q"` for task-aware retrieval, then optimizes a contrastive loss over `E(q_inst)`, `E(d+)`, `E(d-)`, and batch negatives.
- **RARe query augmentation**: retrieve `k = 5` nearest-neighbor queries from the same task-specific training pool using [[bm25]], collect their paired positive documents, and build `q_inst+ic = "Instruct: t_i; Query: q_1^ic; Document: d_1^ic+; ...; Query: q"`.
- **Training objective**: replace `q_inst` with `q_inst+ic` in the same contrastive loss, i.e. optimize `L_RARe = -log exp(cos(e_qinst+ic, e_d+)) / (exp(cos(e_qinst+ic, e_d+)) + exp(cos(e_qinst+ic, e_d-)) + Σ_n exp(cos(e_qinst+ic, e_n)))`.
- **Inference procedure**: for each test query, retrieve in-context examples from the task's train/dev pool; for BeIR datasets without such splits, use GenQ-style synthetic query-document pairs. For RAR-b, the paper wraps queries in brackets to preserve multi-segment structure such as `Start:` / `End:`.
- **LLM-backbone fine-tuning**: for Llama-3 retrieval training, use `4 × H100`, per-device batch size `8`, gradient accumulation `4`, LoRA rank `r = 32`, temperature `0.01`, learning rate `1e-4`, `100` warmup steps, document length `512`, query length `1024`, and a `70%` / `30%` mix of augmented versus plain queries.
- **Retriever-backbone fine-tuning**: for E5-Mistral-Instruct and LLM2Vec-Llama-3-8B-Supervised, use `8 × H100`, per-device batch size `32`, gradient accumulation `2`, `100K` public E5-mixture examples, learning rate `2e-4`, maximum length `1024`, warmup ratio `0.1`, one epoch, with LoRA `r = 16` for E5-Mistral-Instruct and `r = 4` for LLM2Vec to reduce overfitting.

## Key Results

- **Inference-only augmentation fails**: adding in-context examples at inference time without retraining generally lowers performance for SFR-Embedding-2-R, LLM2Vec-Llama-3-8B-Supervised, and E5-Mistral-7B-Instruct, with only marginal exceptions such as SciFact.
- **Training from LLM checkpoints helps**: RARe on Llama-3 raises average nDCG@10 from `38.68` to `40.99`, improving MS-MARCO / BeIR / RAR-b from `43.56 / 53.99 / 18.50` to `44.77 / 55.87 / 22.34`.
- **Competitive without synthetic prompts**: on Llama-3.1-Instruct, RARe reaches `40.88` average versus `39.94` for Promptriever, and improves RAR-b from `20.95` to `23.67` for a `+2.72` absolute gain.
- **Retriever checkpoints also benefit**: on E5-Mistral-Instruct, RARe improves BeIR All from `56.96` to `58.28` and RAR-b from `24.12` to `25.79`; on LLM2Vec-Llama-3-8B-Supervised, RARe improves BeIR All from `55.35` to `56.76` while RAR-b is roughly flat to slightly worse (`23.44` to `23.10`).
- **Exemplar selection matters**: with E5-Mistral-Instruct, increasing the number of in-context examples from `0` to `10` raises the average from `51.84` to `53.82`; retrieved examples outperform random ones, and higher top-1 exemplar similarity yields gains above `10%` on datasets such as NFCorpus and SciFact.
- **Prompt content matters**: the regular query-plus-positive-document format reaches `53.74` average, outperforming query-only (`51.39`), doc-only (`52.05`), shuffled formats (`51.88` / `52.36`), and the negative-document variant (`53.44`).
- **Latency is the main systems cost**: augmenting queries with examples increases total latency from `3.84s` to `153.76s` on NFCorpus (`40.04×`) and from `14.84s` to `287.90s` on FiQA2018 (`19.40×`), but the increase is smaller on large corpora such as DBPedia (`625.31s` to `755.67s`, `1.21×`).

## Limitations

- RARe requires accessible `(q, d+)` in-context example pools at inference time, which is not always available for new tasks or closed datasets.
- Longer augmented queries substantially increase embedding latency, especially on small corpora where query encoding dominates the pipeline.
- The paper evaluates only English retrieval tasks, so multilingual transfer remains untested.
- Performance gains depend on backbone and dataset; for some settings, such as ArguAna or LLM2Vec on RAR-b, more in-context information does not uniformly help.
- BM25 is used as a lightweight nearest-neighbor retriever for exemplar selection; stronger but more expensive example selectors are left unexplored.

## Concepts Extracted

- [[information-retrieval]]
- [[dense-retrieval]]
- [[in-context-learning]]
- [[contrastive-learning]]
- [[bm25]]
- [[query-augmentation]]
- [[query-expansion]]
- [[domain-generalization]]
- [[retrieval-augmented-generation]]

## Entities Extracted

- [[atula-tejaswi]]
- [[yoonsang-lee]]
- [[sujay-sanghavi]]
- [[eunsol-choi]]
- [[university-of-texas-at-austin]]
- [[seoul-national-university]]
- [[new-york-university]]
- [[beir]]
- [[rar-b]]
- [[ms-marco]]
- [[llama-3]]
- [[e5-mistral-instruct]]
- [[llm2vec-llama-3-8b-supervised]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
