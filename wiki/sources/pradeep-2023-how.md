---
type: source
subtype: paper
title: How Does Generative Retrieval Scale to Millions of Passages?
slug: pradeep-2023-how
date: 2026-04-20
language: en
tags: [generative-retrieval, dense-retrieval, scaling, ms-marco, synthetic-queries]
processed: true

raw_file: raw/papers/pradeep-2023-how/paper.pdf
raw_md: raw/papers/pradeep-2023-how/paper.md
bibtex_file: raw/papers/pradeep-2023-how/paper.bib
possibly_outdated: true

authors:
  - Ronak Pradeep
  - Kai Hui
  - Jai Gupta
  - Adam D. Lelkes
  - Honglei Zhuang
  - Jimmy Lin
  - Donald Metzler
  - Vinh Q. Tran
year: 2023
venue: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.83
url: https://aclanthology.org/2023.emnlp-main.83
citation_key: pradeep2023how
paper_type: benchmark

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper provides a large-scale empirical study of generative retrieval, asking which design choices remain useful once the corpus grows from roughly `10^5` documents to the full `8.8M`-passage MS MARCO collection. Using T5.1.1-based sequence-to-sequence models, the authors ablate document representations, synthetic queries, identifier schemes, PAWA decoding, constrained decoding, and consistency loss across NQ, TriviaQA, and three MS MARCO scales. Their central result is that synthetic queries are the only consistently important ingredient when scaling, while many sophisticated identifier or decoder modifications offer little value once compute is normalized. Even with model scaling up to `11B` parameters, full-corpus generative retrieval remains substantially behind strong dense dual encoders, indicating that million-scale generative retrieval is still an unsolved problem.

## Problem & Motivation

Generative retrieval promises to replace external dense or lexical indexes with a single sequence-to-sequence model that maps a query directly to a document identifier. Prior work showed promising results, but almost all evaluations were limited to corpora on the order of `100k` documents, leaving unclear whether the same modeling choices hold at realistic passage-retrieval scale. This paper targets that gap by stress-testing DSI-style retrieval on the full MS MARCO passage ranking corpus and by separating genuine method gains from gains that come only from extra parameters or compute.

## Method

- **Baseline formulation**: starts from [[differentiable-search-index]], which trains a seq2seq model on both indexing and retrieval tasks, mapping document content or relevant queries to a target [[document-identifier]].
- **Document representations**: compares `FirstP`, the first `64` tokens of a document, against `DaQ`, `10` random chunks of `64` tokens. On MS MARCO passages, FirstP and DaQ are effectively equivalent because passages fit within the input window.
- **Synthetic query generation**: adds [[synthetic-query-generation]] with `40` generated queries per passage for MS MARCO via docT5query, and `20` / `15` generated queries per document for NQ / TriviaQA. The model can then learn the retrieval mapping from synthetic queries to docids, reducing both coverage and distribution gaps.
- **Identifier design**: evaluates [[atomic-identifier]], [[naive-identifier]], and [[semantic-identifier]]. Atomic IDs treat each document as a single token; Naive IDs decode the original textual id string; Semantic IDs recursively cluster document embeddings with hierarchical `k`-means using `k = 10` and `c = 100` leaf capacity for MS MARCO.
- **2D semantic decoding**: for 2D Semantic IDs, PAWA replaces a fixed output projection `W ∈ R^{d × |V|}` with a position-aware projection `W^{pawa} ∈ R^{d × l × |V|}`, conditioning on both the output position and the decoded prefix.
- **Constrained decoding and regularization**: tests [[constrained-decoding]] with a trie over valid docids, and [[consistency-regularization]] with `L_reg = 1/2 [KL(p_{i,1} || p_{i,2}) + KL(p_{i,2} || p_{i,1})]` across two dropout masks; the latter was dropped from the final setup because it often diverged to `NaN`.
- **Backbone and training**: uses T5.1.1 in `t5x` / `seqio`, with maximum input length `128` on MS MARCO and `64` on NQ / TriviaQA. Base runs use batch size `512`, learning rate `1e-3`, dropout `0.1`, warmup `10k` steps (`100k` for Atomic IDs), and beam search with `40` beams for sequential identifiers.
- **Scale-up setting**: trains NQ100k, TriviaQA, and MSMarco100k for `1M` steps, and MSMarco1M / MSMarcoFULL up to `9M` steps or convergence. Hardware ranges from `8` TPUv4 chips at T5-Base to `64` / `128` chips for larger-scale models.

## Key Results

- On small corpora, the best configuration (`FirstP + DaQ + D2Q + labeled queries` with in-domain D2Q) reaches `70.7` Recall@1 on NQ100k and `90.0` Recall@5 on TriviaQA, setting a new SOTA on the NCI version of NQ.
- On MSMarco100k, `D2Q only + Atomic ID` reaches `80.3` MRR@10, and using all `100` sampled synthetic queries further improves this to `82.4`, close to GTR-Base at `83.2`.
- Synthetic queries dominate scaling behavior: moving from DSI-style labeled-query training to synthetic-query training boosts MSMarco100k from `23.9` to `77.7` MRR@10 for Naive IDs, and MSMarco1M from `12.4` to `48.2`.
- Full-corpus performance remains weak: on MSMarcoFULL, the best base-scale result is `24.2` MRR@10 with Atomic IDs, while base Naive IDs reach only `13.3` and base Semantic IDs `11.8`.
- After model scaling, the strongest full-corpus setting is `T5-XL` with Naive IDs at `26.7` MRR@10, outperforming T5-Base + Atomic IDs (`24.2`) despite using fewer total parameters than the `7.0B` Atomic-ID setup.
- Scaling is not monotonic: Naive-ID performance drops from `26.7` at `T5-XL (2.8B)` to `24.3` at `T5-XXL (11B)`, suggesting that larger parameter count alone does not solve generative retrieval at scale.

## Limitations

- The study is restricted to English retrieval benchmarks and mostly passage retrieval, so its conclusions may not transfer directly to multilingual or document-level settings.
- Even the strongest full-corpus generative retriever (`26.7` MRR@10) remains well below the reported GTR-Base dense retriever on MSMarcoFULL (`34.8`), so competitiveness at realistic scale is not achieved.
- Several sophisticated extensions, especially consistency loss, were unstable or ineffective; the paper therefore diagnoses failure modes more than it proposes a fundamentally better architecture.
- The work studies scaling under mostly fixed T5-style parameterizations and shared hyperparameter recipes, so it does not derive a principled scaling law or architecture for retrieval-specific memory demands.

## Concepts Extracted

- [[generative-retrieval]]
- [[dual-encoder-retrieval]]
- [[synthetic-query-generation]]
- [[document-identifier]]
- [[differentiable-search-index]]
- [[atomic-identifier]]
- [[naive-identifier]]
- [[semantic-identifier]]
- [[constrained-decoding]]
- [[consistency-regularization]]

## Entities Extracted

- [[ronak-pradeep]]
- [[kai-hui]]
- [[jai-gupta]]
- [[adam-lelkes]]
- [[honglei-zhuang]]
- [[jimmy-lin]]
- [[donald-metzler]]
- [[vinh-q-tran]]
- [[google-research]]
- [[university-of-waterloo]]
- [[ms-marco-passage-ranking]]
- [[natural-questions]]
- [[triviaqa]]
- [[t5]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
