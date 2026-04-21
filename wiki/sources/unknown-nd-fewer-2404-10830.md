---
type: source
subtype: paper
title: Fewer Truncations Improve Language Modeling
slug: unknown-nd-fewer-2404-10830
date: 2026-04-20
language: en
tags: [llm, pretraining, truncation, hallucination, packing]
processed: true

raw_file: raw/papers/unknown-nd-fewer-2404-10830/paper.pdf
raw_md: raw/papers/unknown-nd-fewer-2404-10830/paper.md
bibtex_file: raw/papers/unknown-nd-fewer-2404-10830/paper.bib
possibly_outdated: false

authors:
  - Hantian Ding
  - Zijian Wang
  - Giovanni Paolini
  - Varun Kumar
  - Anoop Deoras
  - Dan Roth
  - Stefano Soatto
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2404.10830
doi:
url: https://arxiv.org/abs/2404.10830
citation_key: unknownndfewer
paper_type: method

read_status: unread

domain: llm
---

## Summary

This paper argues that the standard concatenate-then-split pipeline for decoder-only pretraining creates avoidable document truncations that remove grounding context and weaken context-faithful generation. It proposes Best-fit Packing, which first chunks only overlength documents and then packs intact chunks into fixed-length sequences via an optimized Best-Fit-Decreasing heuristic. The method preserves document integrity while remaining almost as token-efficient as concatenation, adding only negligible padding. Across text and code pretraining runs based on LLaMA-style 7B to 13B models, the authors report consistent gains on reading comprehension, natural language inference, context following, summarization faithfulness, and program synthesis, together with substantial reductions in closed-domain hallucination such as undefined-name code errors.

## Problem & Motivation

Modern LLM pretraining usually concatenates documents into one token stream and slices it into equal-length sequences to avoid padding. The paper argues that this formatting choice corrupts training data semantics: many documents that would fit inside the context window are still split across sequence boundaries, so later tokens are trained without the earlier spans that ground them. The authors hypothesize that this encourages spurious next-token prediction, degrades context awareness, and increases closed-domain hallucination. Their goal is therefore to keep each document intact whenever its length is `<= L`, while retaining nearly the same training efficiency as concatenation at billion-document scale.

## Method

- **Minimal chunking before packing**: given max sequence length `L`, each document is segmented into chunks with length `<= L`; only documents longer than `L` are truncated, which the paper treats as the irreducible minimum.
- **Packing objective**: the chunk set `C = {c_1, ..., c_N}` is partitioned into training sequences `S = {s_1, ..., s_M}` subject to `sum_{c in s_i} l(c) <= L`, with the objective of minimizing `M`, i.e. the number of packed sequences.
- **Best-Fit-Decreasing formulation**: after sorting chunks by descending length, each chunk is assigned to the feasible bin with the smallest remaining capacity. This casts sequence construction as an approximate [[bin-packing]] problem rather than raw stream slicing.
- **Optimized search structure**: because chunk lengths are integers in `[1, L]`, the implementation replaces generic `O(N log N)` sorting with count sort and tracks remaining capacities using a segment tree of size `O(L)`, reducing packing to roughly `O(N log L)` after linear-time preprocessing.
- **Analytical motivation**: in a toy stochastic process, the paper compares a model trained on full sequences with one trained on truncated sequences and shows the truncated model has strictly higher expected loss; for the full model the loss is `H(X_m | X_0) = -p log p - q log q`.
- **Training setup**: experiments use LLaMA-style decoder-only models at `13B` parameters for natural language and `7B` for code, with context lengths `2048` or `8192`, AdamW, learning rate `3e-4`, cosine decay, `3000` warmup steps, global batch size `2M` tokens, and `256 x A100` GPUs.
- **Attention handling in packed sequences**: when a packed training sequence contains multiple documents, cross-document attention is masked out; thanks to [[rotary-positional-embedding]], the setup keeps standard position ids without extra adjustment.
- **Data pipeline**: Best-fit Packing is implemented offline by tokenizing documents, running BFD over chunk lengths, and storing chunk indices; during training the system reconstructs each packed sequence on the fly, avoiding disk materialization of concatenated sequences.

## Key Results

- **Compactness is nearly identical to concatenation**: on RefinedWeb at `L = 2048`, packing adds only `6253` sequences over `2.6 x 10^8` total (`0.0024%`); on RefinedWeb at `L = 8192`, only `411` extra sequences over `6.5 x 10^7` (`0.00063%`); on The Stack at `L = 2048`, `1786` extra over `6.4 x 10^7` (`0.0028%`).
- **Packing runtime scales better than standard FFD**: on `1B` documents with `L = 2048`, optimized BFD takes `10,816s` versus `26,354s` for FFD; on `2B`, `22,244s` versus `55,074s`.
- **Reading comprehension improves across most datasets**: for the `13B, 2k` NL model, Natural Questions rises from `53.86%` to `61.61%`, SQuAD from `21.47%` to `24.44%`, and NarrativeQA from `60.38` to `63.91` F1.
- **Context-sensitive reasoning improves strongly**: the paper reports up to `+9.3%` relative improvement on natural language inference and up to `+16.8%` on context following; for example, MemoTrap at `2k` improves from `35.58%` to `41.56%`, and RTE from `55.74%` to `60.28%`.
- **Summarization becomes more faithful and instruction-following**: on CNN/DailyMail at `2k`, ROUGE-2 improves from `11.04` to `13.14`, SummaC from `32.77%` to `39.55%`, and the average summary length moves from `4.89` sentences toward the prompted `3` sentences.
- **Program synthesis benefits materially**: on HumanEval, `Pass@100` rises from `35.28%` to `40.57%`; on MBPP, `Pass@100` rises from `59.46%` to `62.93%`. Undefined-name hallucinations fall from `5.10%` to `2.41%` on HumanEval and from `9.52%` to `3.97%` on MBPP, a reduction of up to `58.3%`.

## Limitations

- The empirical study covers only a small set of model scales (`7B` and `13B`) and two data domains (natural language and code), so the claims are not yet validated for much larger frontier models or multimodal settings.
- Best-fit Packing requires an additional offline preprocessing stage and a chunk-index data pipeline, so it is not operationally free even if training-time FLOP efficiency is nearly unchanged.
- The hallucination analysis is focused on context-based or closed-domain failures; the method does not address knowledge-based hallucination when the required fact is absent from context.
- The packing algorithm remains heuristic: BFD has good empirical compactness here, but no per-dataset optimality guarantee, and its success partly depends on the real document-length distribution being favorable.

## Concepts Extracted

- [[best-fit-packing]]
- [[bin-packing]]
- [[document-truncation]]
- [[next-token-prediction]]
- [[closed-domain-hallucination]]
- [[context-awareness]]
- [[attention-mask]]
- [[rotary-positional-embedding]]
- [[natural-language-inference]]
- [[in-context-learning]]
- [[program-synthesis]]

## Entities Extracted

- [[hantian-ding]]
- [[zijian-wang]]
- [[giovanni-paolini]]
- [[varun-kumar]]
- [[anoop-deoras]]
- [[dan-roth]]
- [[stefano-soatto]]
- [[aws-ai-labs]]
- [[refinedweb]]
- [[the-stack]]
- [[llama]]
- [[natural-questions]]
- [[humaneval]]
- [[mbpp]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
