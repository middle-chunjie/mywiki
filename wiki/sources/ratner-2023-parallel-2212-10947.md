---
type: source
subtype: paper
title: Parallel Context Windows for Large Language Models
slug: ratner-2023-parallel-2212-10947
date: 2026-04-20
language: en
tags: [llm, in-context-learning, long-context, retrieval, question-answering]
processed: true
raw_file: raw/papers/ratner-2023-parallel-2212-10947/paper.pdf
raw_md: raw/papers/ratner-2023-parallel-2212-10947/paper.md
bibtex_file: raw/papers/ratner-2023-parallel-2212-10947/paper.bib
possibly_outdated: true
authors:
  - Nir Ratner
  - Yoav Levine
  - Yonatan Belinkov
  - Ori Ram
  - Inbal Magar
  - Omri Abend
  - Ehud Karpas
  - Amnon Shashua
  - Kevin Leyton-Brown
  - Yoav Shoham
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2212.10947
doi:
url: http://arxiv.org/abs/2212.10947
citation_key: ratner2023parallel
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces Parallel Context Windows (PCW), a training-free inference method for extending the usable context of decoder-only LLMs. PCW splits a long prompt into `B` parallel windows, reuses positional embeddings across those windows, and modifies the attention mask so context tokens attend only within their own window while task tokens attend to all windows. This lets the model read up to `B` times more supporting text without architectural retraining. Across GPT-2, LLaMA, and Jurassic-1 models, PCW is especially effective for in-context learning on tasks with many labels, improves information extraction, and helps retrieval-based QA by exposing the model to more candidate documents. Its benefits are weaker or negative when reasoning requires strong sequential dependence across windows, such as bridge-style multi-hop QA.

## Problem & Motivation

Off-the-shelf LLMs are bottlenecked by finite context windows, which cap how many in-context demonstrations or retrieved documents can be shown at inference time. Prior approaches for long inputs typically require specialized architectures or additional training, which limits applicability to production LLMs. This paper targets a narrower but pragmatic goal: extend the amount of accessible context for existing decoder-only models by changing only positional assignment and masking at inference time, so that more demonstrations or evidence documents can be exposed to the model without retraining.

## Method

- Let the original context window be `N`, let the task suffix use `T` tokens, and let the usable context per standard prompt be `C = N - T`. PCW uses `B` parallel context windows so the full input contains `B·C + T` tokens.
- **Positional embedding reuse:** for context positions, PCW reuses the first `C` position IDs in every window, with `p_i^{PCW} = p_{((i - 1) mod C) + 1}` for `1 <= i <= B C`; task tokens keep the trailing positions with `p_i^{PCW} = p_{i - (B - 1) C}` for `B C < i <= B C + T`.
- **Attention mask modification:** context tokens attend autoregressively only within their own window, `a^{b,b'}_{ii'} = 1` iff `1 <= i' <= i <= C` and `b = b'`; task tokens attend to all windows and prior task tokens, `a^{B+1,b'}_{ii'} = 1` iff `1 <= i' <= i <= N` and `b' in [B+1]`.
- The main ICL experiments use `B = 3`, which turns `n_max` in-context examples into `B × n_max` effective demonstrations. Inputs are greedily balanced across windows by length; the implementation uses left indentation and a single shared BOS token.
- To size the prompts, the paper sets `n_max = floor((N - T_max) / D_90)`, where `T_max` is the longest test example and `D_90` is the `90`th percentile training-example length after dropping the longest percentile to reduce outlier effects.
- Evaluation covers `9` pretrained models: GPT-2 Large (`0.75B`) and XL (`1.5B`), LLaMA (`6.7B`, `13B`, `32.5B`, `65.2B`), and Jurassic-1 Large (`7.5B`), Grande (`17B`), and Jumbo (`178B`). The standard setup samples `30` training sets and up to `250` test examples; for the three largest models this is reduced to `15` training sets and `125` test examples.
- Classification uses constrained greedy decoding to restrict outputs to valid labels; information extraction uses greedy decoding at temperature `0`. Retrieval QA uses BM25-retrieved Wikipedia documents, and HotpotQA uses `5` windows with `2` evidence documents per window in a zero-shot setup.

## Key Results

- On classification tasks with more than `6` labels, PCW yields average gains of `+7.4`, `+8.2`, and `+8.7` points for Jurassic-1 Large, Grande, and Jumbo, and `+2.3`, `+5.3`, `+6.7`, and `+7.1` for LLaMA `6.7B/13B/32.5B/65B`, respectively.
- Gains are strongest on large-label tasks: on BANKING77, Jurassic-1 Grande improves from `55.2` to `69.1` and Jumbo from `55.3` to `70.9`; on CLINIC150, Jumbo improves from `65.7` to `79.9`; on DBPedia, Jurassic-1 Grande improves from `92.5` to `97.3`.
- Improvement correlates with label-space size: for J1 models, the correlation between log number of classes and average gain is `r = 0.92` with slope `3.02`; the LLaMA correlation is also high at `r = 0.79`.
- PCW reduces variance across prompt samples: J1 average standard deviation drops from `3.9/3.4/3.9` to `3.1/2.3/2.6`, and LLaMA drops from `4.6/3.8/3.6/3.2` to `4.0/2.9/2.9/2.2`.
- Information extraction also improves: ATIS goes from `85.6` to `89.0` and `88.0` to `91.7`; SQuAD from `79.2` to `80.5` and `83.8` to `85.1`; adversarialQA from `43.0` to `44.6` and `46.4` to `47.4` for J1-Large and J1-Grande.
- On HotpotQA, PCW helps comparison questions (`15.3 -> 21.5` for J1-Large, `20.9 -> 28.7` for J1-Grande) but hurts bridge questions (`21.6 -> 16.5`, `27.1 -> 24.0`), indicating that its gains depend on cross-document independence.

## Limitations

- The number of windows `B` must be chosen in advance; appendix experiments suggest diminishing returns around `B = 5` to `7`, and the best `B` is task-dependent.
- PCW is less effective when reasoning across documents is sequentially dependent rather than parallelizable; the bridge-question drop on HotpotQA is the clearest failure mode.
- The method blocks attention between context windows, so it does not let evidence in one window condition the interpretation of another during context encoding.
- It is not uniformly beneficial on low-label or simpler tasks, where gains are often small and sometimes negative relative to vanilla ICL.
- PCW expands accessible context without retraining, but compute still grows with the number of processed windows and each window still retains standard within-window attention costs.

## Concepts Extracted

- [[parallel-context-windows]]
- [[context-window]]
- [[in-context-learning]]
- [[sparse-attention]]
- [[positional-embedding]]
- [[rotary-positional-embedding]]
- [[attention-mask]]
- [[retrieval-augmented-generation]]
- [[multihop-question-answering]]
- [[constrained-decoding]]
- [[decoder-only-language-model]]

## Entities Extracted

- [[nir-ratner]]
- [[yoav-levine]]
- [[yonatan-belinkov]]
- [[ori-ram]]
- [[inbal-magar]]
- [[omri-abend]]
- [[ehud-karpas]]
- [[amnon-shashua]]
- [[kevin-leyton-brown]]
- [[yoav-shoham]]
- [[ai21-labs]]
- [[jurassic-1]]
- [[llama]]
- [[natural-questions]]
- [[hotpotqa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
