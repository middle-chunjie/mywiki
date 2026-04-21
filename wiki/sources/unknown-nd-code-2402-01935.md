---
type: source
subtype: paper
title: Code Representation Learning at Scale
slug: unknown-nd-code-2402-01935
date: 2026-04-20
language: en
tags: [code, embeddings, pretraining, contrastive-learning, code-search]
processed: true

raw_file: raw/papers/unknown-nd-code-2402-01935/paper.pdf
raw_md: raw/papers/unknown-nd-code-2402-01935/paper.md
bibtex_file: raw/papers/unknown-nd-code-2402-01935/paper.bib
possibly_outdated: false

authors:
  - Dejiao Zhang
  - Wasi Ahmad
  - Ming Tan
  - Hantian Ding
  - Ramesh Nallapati
  - Dan Roth
  - Xiaofei Ma
  - Bing Xiang
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.01935
doi:
url: https://arxiv.org/abs/2402.01935
citation_key: unknownndcode1935
paper_type: method

read_status: unread

domain: nlp
---

## Summary

This paper introduces CodeSage, a family of encoder-only code representation models trained with a two-stage recipe over The Stack and large-scale text-code pairs. Stage I mixes identifier deobfuscation with a modified masked language modeling objective that abandons the standard 80-10-10 corruption scheme in favor of full masking, while Stage II applies bimodal contrastive learning with hard positives and hard negatives. The authors train 130M, 356M, and 1.3B parameter models on 237,961,548 code files and 75,389,347 summary-code pairs across nine languages. Across zero-shot code search and lightweight classification transfer, CodeSage consistently surpasses prior code encoders and often beats OpenAI Ada-002, especially on cross-lingual retrieval and obfuscation-sensitive natural-language-to-code search.

## Problem & Motivation

Prior encoder-based code models were typically capped at roughly 125M parameters and trained on comparatively small corpora such as CodeSearchNet, leaving large permissively licensed code corpora underused for general-purpose embeddings. The paper argues that scaling data alone is not sufficient: standard MLM corruption can damage both code syntax and mixed natural-language/programming-language semantics, while dropout-based unimodal contrastive learning does not scale well to very large pretraining sets. The goal is therefore to build a stronger off-the-shelf code encoder by pairing large-scale data with pretraining objectives that respect identifier structure and sequence-level semantic discrimination.

## Method

- **Stage I token-level pretraining**: for each example, sample identifier deobfuscation or random-mask MLM with equal probability `p = 0.5`. The MLM branch masks `15%` of tokens and replaces all selected tokens with `[MASK]`, explicitly avoiding BERT-style `80/10/10` corruption.
- **Identifier deobfuscation**: class names, function names, arguments, and variables are obfuscated into special tokens such as `` `c_i` ``, `` `f_i` ``, and `` `v_i` ``. The encoder must recover the original identifier tokens, forcing it to model code semantics, dependency structure, and natural-language cues from comments/docstrings.
- **Contrastive objective**: Stage II minimizes a symmetric loss over text-code pairs, with similarity `` `sim(h_i, h_j) = cos(h_i, h_j)` `` and temperature `` `tau = 0.05` ``. Hard-negative weights are scaled by `` `gamma_i^k = exp(sim(h_i,h_k)/tau) / sum_j exp(sim(h_i,h_j)/tau)` `` to emphasize nearby in-batch negatives.
- **Hard positives**: instead of contrasting docstrings with full functions, the code side removes function signatures and return statements so the model cannot solve alignment mostly through lexical overlap between summary text and function names.
- **Data pipeline**: Stage I uses The Stack over 9 languages with maximum sequence length `` `1024` `` and concatenation plus block attention; Stage II keeps only English summaries, filters summary lengths to `` `[3, 256]` `` tokens, removes URLs/HTML/doctags, and discards one-line functions.
- **Scale**: the corpus contains `237,961,548` files, `367,905,026` functions, and `75,389,347` usable summary-function pairs. Languages are Python, Java, JavaScript, TypeScript, C#, C, Ruby, Go, and PHP.
- **Model sizes**: CodeSage-small uses `6` layers, `8` heads, hidden size `` `1024` ``, and `130M` parameters; CodeSage-base uses `24` layers, `8` heads, hidden size `` `1024` ``, and `356M`; CodeSage-large uses `24` layers, `16` heads, hidden size `` `2048` ``, and `1.3B`.
- **Training hyperparameters**: Stage I runs `250,000` steps with dropout `` `0.1` ``, batch size `` `2048` ``, learning rate `` `3e-4` ``, and `5000` warmup steps. Stage II runs `20,000` steps with dropout `` `0.1` ``, batch size `` `8192` ``, learning rate `` `5e-6` ``, and `500` warmup steps.

## Key Results

- Zero-shot in-language Code2Code search average MAP rises from `21.17` for UnixCoder and `27.33` for OpenAI Ada-002 to `26.08` / `32.95` / `38.51` for CodeSage-small / base / large.
- On AdvTest NL2Code search, CodeSage reaches `41.28`, `49.08`, and `52.67` MRR, compared with `27.32` for UnixCoder and `38.08` for OpenAI Ada-002; this is the benchmark where identifier obfuscation matters most.
- On CoSQA, CodeSage-small obtains `49.92` MRR and already exceeds OpenAI Ada-002 at `44.23`; on CSN, CodeSage-large reaches `71.24` MRR, matching the best reported baseline in the main results table.
- For linear-head classification, CodeSage-large reaches macro-F1 `90.32` on complexity prediction and `24.42` on runtime error prediction, outperforming Ada-002 at `79.82` and `20.84`; CodeSage underperforms on defect detection in the frozen-encoder setup (`58.95` vs `62.56`).
- Under full end-to-end finetuning, CodeSage-large improves to macro-F1 `66.38` on defect detection, `96.20` on complexity prediction, and `49.25` on runtime error prediction.

## Limitations

- The paper is a preprint and the evidence is entirely from the authors' own large-scale training/evaluation stack, so external replication of the full recipe is expensive.
- Gains are strongest on retrieval and transfer benchmarks; the work does not study generative coding tasks directly because the model family is encoder-only.
- The method still depends on very large batches (`8192` for Stage II) and huge corpora, which makes the recipe substantially less accessible than prior 125M-scale encoders.
- Evaluation covers nine programming languages but not lower-resource languages, multi-file reasoning benchmarks, or downstream tool-use settings.
- In the frozen-encoder setting, the model still lags strong embedding baselines on defect detection, suggesting that its sequence representations are not uniformly better for all classification tasks.

## Concepts Extracted

- [[code-representation-learning]]
- [[masked-language-modeling]]
- [[identifier-deobfuscation]]
- [[contrastive-learning]]
- [[multimodal-contrastive-learning]]
- [[hard-negative-mining]]
- [[code-search]]
- [[cross-lingual-retrieval]]
- [[code-classification]]
- [[code-language-model]]

## Entities Extracted

- [[codesage]]
- [[dejiao-zhang]]
- [[wasi-uddin-ahmad-aws]]
- [[ming-tan]]
- [[hantian-ding]]
- [[ramesh-nallapati]]
- [[dan-roth]]
- [[xiaofei-ma]]
- [[bing-xiang]]
- [[aws-ai-labs]]
- [[the-stack]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
