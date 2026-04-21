---
type: source
subtype: paper
title: "BatchPrompt: Accomplish more with less"
slug: lin-2023-batchprompt-2309-00384
date: 2026-04-20
language: en
tags: [llm, prompting, batching, efficiency, ensembling]
processed: true

raw_file: raw/papers/lin-2023-batchprompt-2309-00384/paper.pdf
raw_md: raw/papers/lin-2023-batchprompt-2309-00384/paper.md
bibtex_file: raw/papers/lin-2023-batchprompt-2309-00384/paper.bib
possibly_outdated: true

authors:
  - Jianzhe Lin
  - Maurice Diesendruck
  - Liang Du
  - Robin Abraham
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2309.00384
doi:
url: http://arxiv.org/abs/2309.00384
citation_key: lin2023batchprompt
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper studies how to amortize prompt overhead across many short NLP inputs by batching multiple inference examples into one LLM call. Naive batching hurts accuracy because answers depend on within-prompt position and growing context length, so the authors add Batch Permutation and Ensembling (BPE), which reorders each batch across voting rounds and aggregates predictions by majority vote, plus Self-reflection-guided EArly Stopping (SEAS), which stops voting once a sample receives stable confident predictions. On BoolQ, QQP, and RTE, BatchPrompt+BPE+SEAS at batch size `32` uses only `15.7%` of SinglePrompt's LLM calls while remaining competitive in accuracy, e.g. BoolQ `90.6% -> 90.9%`, QQP `87.2% -> 88.4%`, and RTE `91.5% -> 91.1%`, with large input-token savings.

## Problem & Motivation

Instruction-based prompting is inefficient when task instructions and few-shot demonstrations dominate the token budget while each inference example is short. The paper argues that SinglePrompt wastes token resources by repeating the same task specification for every item. Simply packing many items into one prompt is not enough: longer contexts degrade transformer performance, and the same sample can receive different predictions depending on its batch position and neighboring answers. The goal is therefore to preserve the token-efficiency benefits of batching while recovering enough label quality to make batched prompting practical for real LLM inference.

## Method

- **BatchPrompt baseline**: pack a batch `D = {d_1, d_2, ..., d_N}` into one indexed prompt so a single call predicts `N` labels instead of one. The same task description and few-shot demonstrations are amortized across the batch.
- **Batch Permutation and Ensembling (BPE)**: run `K` voting rounds with different permutations `S = {s_1, s_2, ..., s_K}` of the same batch and choose the final label for item `n` by `argmax_a sum_{k=1}^K 1(a_n^k = a)`. The core idea is that position-sensitive errors vary across permutations, while correct labels are more stable.
- **Context-aware voting view**: the paper further writes the vote as `argmax_a sum_{k=1}^K 1(a_n^k = a | prompt, a_1^k, ..., a_{n-1}^k)`, making explicit that autoregressive answers depend on previous generated outputs inside the same prompt.
- **Self-weighted majority voting**: ask the model to append either `"confident"` or `"not confident"` to each label. Votes then receive weight `w = 1` for confident outputs and `w = alpha` otherwise, with `alpha = 0.2`, yielding `argmax_a sum_{k=1}^K w_n 1(a_n^k = a | ...)`.
- **SEAS**: maintain active indices only for undecided examples. If the same sample receives the same answer with `"confident"` in two consecutive rounds (`k > 1`), remove it from later permutations and keep the current answer, shrinking the effective batch size over time.
- **Experimental setup**: evaluate on BoolQ, QQP, and RTE using validation subsets of `320`, `320`, and `277` examples respectively, with few-shot counts `4`, `4`, and `2`; temperature is fixed at `0`.
- **Models and hyperparameters**: GPT-4 uses batch sizes `16/32/64/160` except BoolQ without `160` due to the `32k` context limit; `gpt-3.5-turbo` uses `16/32` because of its `8k` context. Voting rounds are odd numbers `K in {1, 3, 5, 7, 9}`.
- **Prompt format detail**: batched items must be explicitly indexed as `"Input 0"`, `"Input 1"`, ...; the appendix reports that looser delimiter-only formatting can cause missing outputs.

## Key Results

- At batch size `32` with BPE+SEAS, the method uses only `15.7%` of SinglePrompt's LLM calls while keeping accuracy competitive: BoolQ `90.6% -> 90.9%`, QQP `87.2% -> 88.4%`, RTE `91.5% -> 91.1%`.
- The same configuration cuts input tokens by `72.6%` on BoolQ, `81.4%` on QQP, and `69.2%` on RTE; equivalently, it keeps `27.4%`, `18.6%`, and `30.8%` of SinglePrompt's input-token usage.
- Around `80%` of samples can terminate within two voting rounds under SEAS, which is why token growth slows sharply after round `3`.
- On GPT-4 BoolQ, larger batches hurt naive BatchPrompt badly at first (`72.8%` for batch size `64`, `1` vote), but BPE recovers much of the loss (`86.3%` at `9` votes).
- BPE can surpass SinglePrompt when the base model is strong: on RTE with GPT-4, the best non-SEAS result reaches `92.9%`, above the `91.4%` SinglePrompt baseline.
- In the appendix GSM8K side experiment, SinglePrompt reaches `94.9%`, while batch size `32` improves from `89.1%` at `1` vote to `92.0%` at `5` votes.

## Limitations

- The main evaluation is limited to three mostly binary classification tasks plus one small arithmetic side experiment; open-ended generation and summarization are explicitly left unresolved.
- Results rely on proprietary APIs (`gpt-3.5-turbo`, `GPT-4`) and relatively small validation subsets rather than full test sets.
- The paper optimizes only input tokens and LLM calls; generated output tokens are excluded from the efficiency accounting.
- SEAS improves efficiency but can slightly reduce peak accuracy on some settings, e.g. the paper notes an RTE best score of `92.9%` without SEAS versus `91.7%` with SEAS.
- The best batch size, vote count, and confidence weighting remain hand-tuned per task and model, so the method is not yet automatically adaptive.

## Concepts Extracted

- [[batch-prompting]]
- [[large-language-model]]
- [[few-shot-prompting]]
- [[in-context-learning]]
- [[self-consistency]]
- [[confidence-estimation]]
- [[early-stopping]]
- [[token-efficiency]]

## Entities Extracted

- [[jianzhe-lin]]
- [[maurice-diesendruck]]
- [[liang-du]]
- [[robin-abraham]]
- [[microsoft]]
- [[gpt-3-5-turbo]]
- [[gpt-4]]
- [[boolq]]
- [[qqp]]
- [[rte]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
