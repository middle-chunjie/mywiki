---
type: source
subtype: paper
title: Improving language models by retrieving from trillions of tokens
slug: borgeaud-2022-improving-2112-04426
date: 2026-04-20
language: en
tags: [llm, retrieval, language-modeling, memory, pretraining]
processed: true
raw_file: raw/papers/borgeaud-2022-improving-2112-04426/paper.pdf
raw_md: raw/papers/borgeaud-2022-improving-2112-04426/paper.md
bibtex_file: raw/papers/borgeaud-2022-improving-2112-04426/paper.bib
possibly_outdated: true
authors:
  - Sebastian Borgeaud
  - Arthur Mensch
  - Jordan Hoffmann
  - Trevor Cai
  - Eliza Rutherford
  - Katie Millican
  - George van den Driessche
  - Jean-Baptiste Lespiau
  - Bogdan Damoc
  - Aidan Clark
  - Diego de Las Casas
  - Aurelia Guy
  - Jacob Menick
  - Roman Ring
  - Tom Hennigan
  - Saffron Huang
  - Loren Maggiore
  - Chris Jones
  - Albin Cassirer
  - Andy Brock
  - Michela Paganini
  - Geoffrey Irving
  - Oriol Vinyals
  - Simon Osindero
  - Karen Simonyan
  - Jack W. Rae
  - Erich Elsen
  - Laurent Sifre
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2112.04426
doi:
url: http://arxiv.org/abs/2112.04426
citation_key: borgeaud2022improving
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

The paper introduces RETRO, a retrieval-enhanced autoregressive language model that augments next-token prediction with chunk-level nearest neighbors retrieved from a massive external corpus. Instead of scaling only parameters, RETRO splits a `2048`-token context into `64`-token chunks, retrieves semantically similar chunks using frozen BERT embeddings, and injects them through a chunked cross-attention pathway. Using a retrieval database that scales from billions to `1.75T` evaluation tokens, the model shows persistent gains from `172M` to `7.5B` parameters and can match the broad language-modeling quality of much larger systems such as GPT-3 and Jurassic-1 on several settings. The work is a major early demonstration that semi-parametric memory can compete with pure parameter scaling.

## Problem & Motivation

The paper targets a scaling bottleneck in large language models: better performance usually requires larger models, more training compute, and more memorized knowledge in weights. The authors argue that these factors should be partially decoupled by giving an autoregressive LM direct access to a huge external text memory at inference and training time. Earlier retrieval-augmented LM methods either worked at token level, used much smaller databases, or were too expensive to update end-to-end during pre-training. RETRO is proposed as a scalable alternative that can exploit trillions of tokens while preserving autoregressive decoding and keeping cross-attention cost linear in the retrieved evidence rather than quadratic in all retrieved tokens.

## Method

- **Chunked autoregressive formulation**: split each sequence `X` into `l = n / m` chunks with `n = 2048` and `m = 64`; the log-likelihood of token `x_( (u-1)m + i )` conditions on previous tokens and only retrieved neighbors from earlier chunks, preserving causality.
- **Retrieval database**: each key-value entry stores `[N, F]`, where `N` is a neighbor chunk and `F` is its continuation; both have length `64`, so each retrieved item has `r = 128` tokens. Keys are averaged frozen BERT embeddings, and retrieval uses `d(C, N) = ||BERT(C) - BERT(N)||_2^2`.
- **Retriever and indexing**: approximate `k`-nearest-neighbor search is implemented with SCaNN; same-document neighbors are filtered during training to avoid trivial causal leakage. The paper reports roughly `10 ms` query time even for a `2T`-token database.
- **RETRO architecture**: a Transformer decoder interleaves standard LM blocks with RETRO blocks that apply chunked cross-attention to encoded neighbors. Retrieved neighbors are processed by a `2`-layer non-causal encoder with width `d' = 896`, conditioned on chunk activations from the decoder.
- **Cross-attention schedule**: RETRO blocks are inserted every `3` layers starting at layer `6`; for the smallest model this means layers `6`, `9`, and `12`. The first `m - 1` tokens that lack a previous chunk use identity in the CCA pathway.
- **Data and scaling setup**: training and retrieval use MassiveText with more than `5T` tokens; the default training retrieval database is `600B` tokens and the evaluation database is `1.75T` tokens. Tokenization uses SentencePiece with vocabulary size `128,000`.
- **Model sizes and training regime**: baselines / RETRO pairs are `132M -> 172M`, `368M -> 425M`, `1.309B -> 1.451B`, and `6.982B -> 7.532B` non-embedding parameters. Training uses `2` retrieved neighbors, while evaluation can scale to `40+` neighbors.
- **RETROfitting**: a pretrained Transformer can be converted into RETRO by freezing original weights and training only chunked cross-attention plus neighbor-encoder parameters, which are less than `10%` of weights for the `7B` model and require only `6M` sequences (`3%` of pre-training data).

## Key Results

- On C4, RETRO consistently beats the non-retrieval baseline across scale: `172M` improves from `0.98` to `0.82` bpb, and `7.5B` improves from `0.78` to `0.66` bpb.
- On LAMBADA, retrieval also helps across scale: the `7.5B` model rises from `0.69` to `0.73` top-1 accuracy.
- On Wikitext103, full-scale retrieval is dramatic: `7.5B` baseline perplexity `10.65` versus RETRO `2.22`; in the controlled Wikipedia-retrieval setting, RETRO reaches `18.46 / 18.97` valid/test perplexity versus the authors' kNN-LM `18.52 / 19.54` and baseline `21.53 / 22.96`.
- On Natural Questions, `RETRO 7.5B` reaches `45.5` exact match, well above the `30.4` closed-book `7B` baseline, though still below `FiD + Distill.` at `54.7`.
- The paper reports that a `7.5B` RETRO model is broadly comparable to GPT-3 and Jurassic-1 on the Pile despite using roughly `25x` fewer parameters; for example it improves over the authors' `7B` baseline on `github` from `0.420` to `0.199` bpb and on `books3` from `0.792` to `0.653`.
- Ablations show the retrieval pathway matters: on C4 with a `247M` model, RETRO gets `0.822` bpb versus `0.987` with no retrieval; using only neighbors gives `0.950`, only continuations `0.895`, and training with one neighbor worsens to `0.858`.

## Limitations

- The strongest gains on Wikitext103 and some web-like corpora are partly driven by residual train-test overlap; the paper explicitly warns that retrieval models exploit leakage more aggressively than baselines.
- RETRO can directly copy retrieved training text, which sharpens privacy, safety, and memorization concerns even if it helps factuality.
- Retrieval quality is dataset dependent: gains can be marginal on some hard subsets, such as `dm mathematics` where the `7.5B` RETRO model improves only slightly over the `7B` baseline (`1.177 -> 1.164` bpb), and it still trails Jurassic-1 on difficult subsets such as `ubuntu_irc`.
- The system requires a very large external index and storage budget: the paper cites `215GB` to index Wikipedia and about `93TB` for full MassiveText, even though this is still better than token-level retrieval alternatives.
- On downstream QA, RETRO remains behind stronger encoder-decoder retrieval systems such as FiD, suggesting that its decoder-centric architecture does not fully exploit retrieved evidence in all task regimes.

## Concepts Extracted

- [[retrieval-based-language-model]]
- [[semi-parametric-language-model]]
- [[chunked-cross-attention]]
- [[dense-retrieval]]
- [[nearest-neighbor-search]]
- [[sentence-embedding]]
- [[encoder-decoder-architecture]]
- [[large-language-model]]
- [[fine-tuning]]
- [[data-decontamination]]
- [[open-domain-question-answering]]

## Entities Extracted

- [[sebastian-borgeaud]]
- [[arthur-mensch]]
- [[jordan-hoffmann]]
- [[trevor-cai]]
- [[eliza-rutherford]]
- [[katie-millican]]
- [[george-van-den-driessche]]
- [[jean-baptiste-lespiau]]
- [[bogdan-damoc]]
- [[aidan-clark]]
- [[diego-de-las-casas]]
- [[aurelia-guy]]
- [[jacob-menick]]
- [[roman-ring]]
- [[tom-hennigan]]
- [[saffron-huang]]
- [[loren-maggiore]]
- [[chris-jones]]
- [[albin-cassirer]]
- [[andy-brock]]
- [[michela-paganini]]
- [[geoffrey-irving]]
- [[oriol-vinyals]]
- [[simon-osindero]]
- [[karen-simonyan]]
- [[jack-rae]]
- [[erich-elsen]]
- [[laurent-sifre]]
- [[deepmind]]
- [[retro]]
- [[bert]]
- [[scann]]
- [[massivetext]]
- [[sentencepiece]]
- [[c4]]
- [[the-pile]]
- [[wikitext-103]]
- [[natural-questions]]
- [[gpt-3]]
- [[jurassic-1]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
