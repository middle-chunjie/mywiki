---
type: source
subtype: paper
title: "FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions"
slug: weller-2024-followir-2403-15246
date: 2026-04-20
language: en
tags: [ir, retrieval, instruction-following, benchmark, reranking]
processed: true

raw_file: raw/papers/weller-2024-followir-2403-15246/paper.pdf
raw_md: raw/papers/weller-2024-followir-2403-15246/paper.md
bibtex_file: raw/papers/weller-2024-followir-2403-15246/paper.bib
possibly_outdated: false

authors:
  - Orion Weller
  - Benjamin Chang
  - Sean MacAvaney
  - Kyle Lo
  - Arman Cohan
  - Benjamin Van Durme
  - Dawn Lawrie
  - Luca Soldaini
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2403.15246
doi: 10.48550/arXiv.2403.15246
url: http://arxiv.org/abs/2403.15246
citation_key: weller2024followir
paper_type: benchmark

read_status: unread

domain: ir
---

## Summary

This paper argues that modern retrieval models still behave like query-only systems even when their backbones are large language models. It introduces FollowIR, a benchmark and training resource for instruction-following information retrieval built from real TREC relevance narratives rather than short synthetic prompts. The benchmark measures whether rankings change appropriately when annotator instructions are edited, and the paper proposes a paired metric, `p-MRR`, for that purpose. Across classical lexical baselines, embedding models, retrieval-tuned instruction models, API embeddings, and instruction-tuned LLM rerankers, most systems fail to follow long instructions and instead over-index on keyword overlap. The authors then train FollowIR-7B and show that targeted fine-tuning can improve both conventional retrieval quality and instruction sensitivity.

## Problem & Motivation

Neural IR models increasingly use LLM backbones, but most still accept only a short query and ignore the richer natural-language instructions that humans routinely use to define relevance. Existing instruction-aware retrieval work is limited by short, repetitive prompts and by benchmarks that do not explicitly test whether a model changes its ranking when the relevance definition changes. The paper targets this gap by asking whether IR models can follow long, realistic instructions with specificity, background detail, and negation, and by constructing an evaluation that isolates instruction following from ordinary relevance matching.

## Method

- **Benchmark construction**: FollowIR is built from three deeply judged TREC collections, using altered annotator narratives on top of TREC News 2021, TREC Common Core 2017, and TREC Robust 2004. The final evaluation set covers `32 + 20 + 52 = 104` altered-instruction queries, with average instruction lengths of `46.9`, `53.5`, and `75.2` words and average relevant documents per query of `19.2`, `32.7`, and `19.8` after re-annotation.
- **Annotation strategy**: expert annotators modify the original narratives to add extra inclusion criteria or explicit negation so that the new relevant set is mostly a subset of the original one. This keeps relabeling tractable because only originally relevant documents must be revisited.
- **Evaluation setup**: the task is cast as reranking rather than full-collection retrieval so every system can be evaluated on the same documents whose relevance may change. Documents are chunked into `400`-word passages with `200`-word overlap, and non-relevant candidates are pooled from `5` retrievers: [[bm25]], BGE-base, E5-base-v2, TART-Contriever, and INSTRUCTOR-XL. Passage scoring uses MaxP.
- **Pairwise metric**: the paper introduces `p-MRR` to score whether ranks move in the correct direction under the modified instruction. For a document with original rank `R_og` and new rank `R_new`, `p-MRR = MRR_og / MRR_new - 1` if `R_og > R_new`, else `1 - MRR_new / MRR_og`, then averages first within query and then across queries.
- **Model coverage**: experiments span no-instruction IR models, instruction-trained IR models, API embeddings, and instruction-tuned LLM rerankers, testing both standard retrieval metrics (`mAP`, `nDCG@5`) and the new pairwise metric.
- **Training data for teaching instruction following**: beyond the benchmark, the authors collect `1836` TREC query-narrative pairs from non-FollowIR tasks, prompt GPT-3.5-Turbo-1106 to synthesize about `2` relevant and `2` non-relevant documents per query, then filter those with [[mistral-7b-instruct-v0.2]] and balance to roughly `1800` training instances over about `1200` unique query-instruction pairs.
- **FollowIR-7B fine-tuning**: [[followir-7b]] is obtained by fine-tuning Mistral-7B-Instruct-v0.2 with [[lora]] in Llama-Factory for `8` epochs, tuning `q_proj`, `k_proj`, `v_proj`, and `o_proj` with rank `r = 8`, `alpha = 16`, `bfloat16`, batch size `32`, and learning rate `3e-5`.

## Key Results

- On the main FollowIR benchmark, most retrieval-specialized systems have negative average `p-MRR`: E5-base-v2 `-4.6`, Contriever `-3.5`, MonoBERT `-5.1`, BGE-base `-3.1`, OpenAI text-embedding-v3-large `-3.1`, and Cohere v3 English `-1.2`.
- Positive instruction-following behavior appears mainly in large or instruction-tuned rerankers: MonoT5-3B reaches average `p-MRR = +4.9`, Mistral-7B-Instruct reaches `+12.0`, and FollowIR-7B reaches the best average `+13.6`.
- FollowIR-7B also improves standard retrieval quality, achieving average score `23.9` versus `22.2` for Mistral-7B-Instruct, with per-dataset `p-MRR` of `+13.6` on Robust04, `+10.8` on News21, and `+16.3` on Core17.
- The fine-tuned model yields `+7.6%` relative improvement on standard IR metrics and `+13.3%` relative improvement on instruction-following over the original Mistral-7B-Instruct-v0.2 baseline.
- Ablations show that weaker models often treat instructions as keywords: on BEIR, replacing short instructions with keywords changes scores by roughly `±1` for many models, while stronger instruction-aware models such as E5-Mistral drop more when the instruction is reduced (`-5.1` on SciFact, `-6.5` on FiQA).

## Limitations

- The benchmark evaluates reranking rather than full retrieval, because full-collection retrieval would expose each model to a different set of changed-relevance documents.
- The released benchmark uses passage chunks under fair-use constraints instead of the full licensed corpora, so deployment conditions differ from the original TREC tasks.
- Both the original TREC judgments and the new altered-instruction annotations may contain residual labeling errors that the paper does not exhaustively audit.
- The FollowIR-7B training set depends on synthetic documents generated by GPT-3.5 and filtered by Mistral-7B-Instruct-v0.2, so improvements may partly inherit generator and filter biases.

## Concepts Extracted

- [[information-retrieval]]
- [[instruction-following]]
- [[dense-retrieval]]
- [[reranking]]
- [[relevance-judgment]]
- [[instruction-tuning]]
- [[text-embedding]]
- [[benchmark-dataset]]
- [[lora]]

## Entities Extracted

- [[orion-weller]]
- [[benjamin-chang]]
- [[sean-macavaney]]
- [[kyle-lo]]
- [[arman-cohan]]
- [[benjamin-van-durme]]
- [[dawn-lawrie]]
- [[luca-soldaini]]
- [[trec]]
- [[bm25]]
- [[mteb]]
- [[beir]]
- [[mistral-7b-instruct-v0.2]]
- [[followir-7b]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
