---
type: source
subtype: paper
title: Compressing Context to Enhance Inference Efficiency of Large Language Models
slug: li-2023-compressing
date: 2026-04-20
language: en
tags: [llm, context-compression, long-context, inference-efficiency, self-information]
processed: true

raw_file: raw/papers/li-2023-compressing/paper.pdf
raw_md: raw/papers/li-2023-compressing/paper.md
bibtex_file: raw/papers/li-2023-compressing/paper.bib
possibly_outdated: true

authors:
  - Yucheng Li
  - Bo Dong
  - Chenghua Lin
  - Frank Guerin
year: 2023
venue: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.391
url: https://aclanthology.org/2023.emnlp-main.391
citation_key: li2023compressing
paper_type: method

read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

Li et al. propose Selective Context, a model-agnostic preprocessing method for compressing long prompts before large language model inference. Instead of changing the Transformer architecture, the method scores tokens with self-information from a smaller causal language model, merges them into lexical units such as phrases or sentences, and keeps only units above a percentile threshold. Evaluated on arXiv articles, BBC News, and ShareGPT conversations across summarization, question answering, reconstruction, and dialogue, the approach preserves output quality surprisingly well at moderate compression ratios. The headline result is that halving context length reduces inference memory by about 36% and per-token latency by about 32%, while causing only modest degradation in BERTScore and faithfulness.

## Problem & Motivation

Long documents and extended conversations are expensive for Transformer-based LLMs because attention cost grows quadratically with sequence length, and practical systems also face fixed context-window limits. The paper argues that many context spans are redundant for LLM inference: some are linguistically repetitive, while others restate knowledge already internalized during pretraining. Rather than redesigning the model or distilling new prompts, the authors target redundancy in the input itself and ask whether a compacted context can preserve enough evidence for downstream generation while materially lowering memory and latency costs.

## Method

- **Core idea**: Selective Context is a preprocessing pipeline that removes low-information spans before passing the prompt to a target LLM; the target model itself is unchanged, so the method is model-agnostic.
- **Token scoring**: for each token `x_i` in context `C = x_0, x_1, ..., x_n`, a base causal LM `M` computes self-information `I(x_i) = -log_2 P(x_i | x_0, x_1, ..., x_{i-1})`.
- **Aggregation to lexical units**: tokens are merged into lexical units `u` at token, phrase, or sentence granularity, and unit scores are summed by additivity: `I(u) = sum_{i=t}^{t+alpha} I(x_i)`.
- **Unit construction**: sentence-level units use NLTK sentence tokenization; phrase-level units use noun-phrase-style merging; the paper avoids verb-phrase merging because it can create overly long units.
- **Percentile filtering**: lexical units are ranked by self-information and retained if `I(u_i) >= I_p`, where `I_p = np.percentile([I(u_0), ..., I(u_k)], p)`. The retained units form compressed context `C'`.
- **Base scorers for experiments**: the OpenAI family uses a smaller GPT-3 Curie model to estimate self-information, while the LLaMA and Vicuna families use `LLaMA-7B` as the scorer.
- **Computation detail**: self-information is computed sentence by sentence instead of over the entire document because later spans otherwise tend to receive artificially lower scores.
- **Compression settings**: the evaluated reduction ratios are `0.2`, `0.35`, `0.5`, `0.65`, and `0.8`, with phrase-level filtering reported as the strongest default in most experiments.
- **Evaluation setup**: experiments cover `408` arXiv documents, `294` BBC News articles, and `470` ShareGPT conversations, using inputs capped below `2048` tokens and four tasks: reconstruction, [[summarization]], [[question-answering]], and conversation.

## Key Results

- At `ratio = 0.5`, the paper reports a `50%` context-cost reduction that yields about `36%` lower inference memory usage and `32%` lower per-token generation time; the case study shows `110.8 -> 76.3 ms/token`, plus a one-time selective-context construction cost of `46.1 ms`.
- Relative to full context under temperature `0.7`, average performance at `ratio = 0.2` drops only from `0.347 -> 0.295` BLEU, `0.571 -> 0.540` ROUGE-1, and `0.909 -> 0.902` BERTScore-F1.
- Even at `ratio = 0.5`, average BERTScore-F1 remains `0.887` against the full-context reference, and the abstract highlights only a `0.023` BERTScore drop and `0.038` faithfulness drop across four downstream applications.
- Against random deletion under greedy decoding, Selective Context is consistently better; at `ratio = 0.5`, it reaches `0.642` ROUGE-1 and `0.900` BERTScore-F1 versus the random baseline's `0.576` and `0.873`.
- In the manual faithfulness test on `1000` QA pairs with GPT-3.5, unfaithfulness is `0.038` at `ratio = 0.5` and `0.051` at `ratio = 0.65`; refusal counts rise from `4` at `0.5` to `19` at `0.65`, showing degradation is driven more by missing evidence than by hallucinated content.
- Phrase-level filtering outperforms token- and sentence-level variants, and human evaluation over `1150` summaries suggests Vicuna is more robust than base LLaMA under compressed context, while model scale alone does not show a clear robustness trend.

## Limitations

- Performance is sensitive to lexical-unit boundary quality; the paper relies on noun-phrase tokenization and explicitly notes the lack of a mature verb-phrase tokenizer.
- The pruning percentile is task- and context-dependent; the paper does not learn or predict an optimal threshold automatically.
- Original-context reconstruction degrades fastest as compression becomes aggressive, and ratios `0.65` and `0.8` look much less attractive than mild or moderate compression.
- The evaluation only uses inputs below `2048` tokens and, for arXiv, only the first two sections of each paper, so the results do not fully characterize ultra-long-context settings.
- References are generated from the same LLM family with full context rather than human gold annotations, which is pragmatic but may blur absolute task quality with similarity-to-full-context behavior.

## Concepts Extracted

- [[large-language-model]]
- [[context-compression]]
- [[selective-context]]
- [[self-information]]
- [[lexical-unit]]
- [[percentile-threshold]]
- [[context-window]]
- [[long-context-inference]]
- [[summarization]]
- [[question-answering]]
- [[faithfulness]]

## Entities Extracted

- [[yucheng-li]]
- [[bo-dong]]
- [[chenghua-lin]]
- [[frank-guerin]]
- [[university-of-surrey]]
- [[university-of-manchester]]
- [[gpt-3-5]]
- [[gpt-4]]
- [[llama]]
- [[vicuna]]
- [[sharegpt]]
- [[bbc-news]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
