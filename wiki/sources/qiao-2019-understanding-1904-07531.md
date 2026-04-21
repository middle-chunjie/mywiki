---
type: source
subtype: paper
title: Understanding the Behaviors of BERT in Ranking
slug: qiao-2019-understanding-1904-07531
date: 2026-04-20
language: en
tags: [bert, ranking, reranking, information-retrieval, neural-ranking]
processed: true

raw_file: raw/papers/qiao-2019-understanding-1904-07531/paper.pdf
raw_md: raw/papers/qiao-2019-understanding-1904-07531/paper.md
bibtex_file: raw/papers/qiao-2019-understanding-1904-07531/paper.bib
possibly_outdated: true

authors:
  - Yifan Qiao
  - Chenyan Xiong
  - Zhenghao Liu
  - Zhiyuan Liu
year: 2019
venue: arXiv
venue_type: preprint
arxiv_id: 1904.07531
doi:
url: http://arxiv.org/abs/1904.07531
citation_key: qiao2019understanding
paper_type: method

read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2019; re-verify against recent literature.

This paper analyzes how BERT behaves when adapted to ranking, comparing representation-only and interaction-heavy formulations on MS MARCO passage reranking and TREC Web Track ad hoc retrieval. The main result is that BERT is effective when query and document are encoded jointly, but weak when they are encoded separately and scored by vector similarity. On MS MARCO, simple joint encoding with a linear head strongly outperforms prior neural rerankers, while on ClueWeb/TREC it fails to beat feature-based learning to rank and trails click-pretrained Conv-KNRM. Attention and token-removal analyses suggest BERT relies on cross-sequence semantic matches, exact or near-exact query terms, and a small number of highly influential tokens, which helps explain the gap between QA-style passage ranking and classical ad hoc retrieval.

## Problem & Motivation

The paper asks whether BERT's pretraining on surrounding-context signals transfers cleanly to information retrieval, and if so, in what ranking formulation. Prior neural IR work had already distinguished interaction-based relevance modeling from representation-based embedding methods, while early BERT reranking results suggested large gains on MS MARCO. The authors therefore test multiple ways of applying BERT to ranking and study whether those gains generalize beyond QA-oriented passage reranking to TREC-style ad hoc document retrieval. A second motivation is interpretability: they want to understand what kinds of query-document matches BERT emphasizes and how those patterns differ from click-trained neural rankers.

## Method

- **Backbone**: all models fine-tune `BERT-Large` with `24` Transformer layers, hidden size `1024`, `16` attention heads, and about `340M` parameters.
- **Representation model**: `BERT (Rep)(q, d) = cos(q_cls^last, d_cls^last)`, where query and document are encoded separately and scored by cosine similarity between their final `[CLS]` embeddings.
- **Joint interaction model**: `BERT (Last-Int)(q, d) = w^T qd_cls^last`, where query and document are concatenated with `[SEP]` and scored from the final `[CLS]` embedding of the joint sequence.
- **Multi-layer interaction model**: `BERT (Mult-Int)(q, d) = Σ_k (w_mult^k)^T qd_cls^k`, combining `[CLS]` representations from all `24` layers to test whether intermediate layers add ranking signal.
- **Term-translation variant**: `s^k(q, d) = Mean_{i,j}(cos(relu(P^k q_i^k), relu(P^k d_j^k)))`, and `BERT (Term-Trans)(q, d) = Σ_k w_trans^k s^k(q, d)`. This adds a neural translation-style interaction layer on top of contextual token embeddings from each BERT layer.
- **Training**: all variants start from Google's pretrained `BERT-Large` and use binary classification loss for relevance; the authors also tried pairwise ranking loss and observed no meaningful difference.
- **Optimization**: Adam with learning rate `3e-6` for BERT-based rankers, except the projection layer in Term-Trans trained at `0.002`; on a typical GPU, maximum batch size is `1` and fine-tuning takes about `1` day to converge.
- **Data and evaluation**: MS MARCO passage reranking uses `1,010,916` training queries and is evaluated by `MRR@10`; ClueWeb09-B / TREC Web Track uses `200` judged queries, `10`-fold cross-validation, top-`100` SDM candidates, and reports `NDCG@20` and `ERR@20`.
- **Behavior analysis**: the authors inspect attention patterns on `100` random MS MARCO Dev queries, compare attention allocated to markers, stopwords, and regular words, and measure token influence by removing one non-stopword from each candidate passage.

## Key Results

- On MS MARCO Dev, `BERT (Last-Int)` reaches `MRR@10 = 0.3367`, beating `Conv-KNRM = 0.2474`, `K-NRM = 0.2100`, and `LeToR = 0.1946`; on MS MARCO Eval it reaches `0.3590` versus `0.2472` for Conv-KNRM and `0.1905` for LeToR.
- `BERT (Rep)` is ineffective as a retrieval embedding model: `MRR@10 = 0.0432` on MS MARCO Dev and `0.0153` on Eval, close to random compared with joint-query-document BERT.
- More complicated interaction variants do not beat the simple one: `BERT (Mult-Int)` gets `0.3060 / 0.3287` on MS MARCO Dev/Eval, and `BERT (Term-Trans)` gets `0.3310 / 0.3561`, both below `BERT (Last-Int)`.
- On ClueWeb09-B, BERT does not solve ad hoc retrieval: `BERT (Last-Int)` reports `NDCG@20 = 0.2407` and `ERR@20 = 0.1649`, below `LeToR` on NDCG (`0.2681`) and below click-pretrained `Conv-KNRM (Bing)` on both metrics (`0.2872`, `0.1814`).
- In the attention analysis, removing marker tokens such as `[CLS]` and `[SEP]` reduces MRR by about `15%`, while removing stopwords has essentially no effect despite attracting substantial attention mass.
- Token-removal analysis shows BERT produces more extreme scores than Conv-KNRM, often near `0` or `1`, and a small number of exact or paraphrastic query-related terms dominate the final ranking decision.

## Limitations

The study only covers reranking settings with pre-generated candidate sets rather than end-to-end retrieval, so its conclusions do not establish how BERT would behave as a first-stage retriever. The strongest positive results are concentrated on MS MARCO, a QA-oriented benchmark whose relevance patterns are closer to sequence matching than classical ad hoc search, so transfer claims are limited. The behavioral analyses use only `100` sampled MS MARCO queries and mostly qualitative examples, which constrains statistical depth. The paper also does not explore longer-document handling beyond candidate truncation, and it leaves open whether deeper models pretrained directly on click or relevance signals could close the TREC gap.

## Concepts Extracted

- [[information-retrieval]]
- [[ad-hoc-retrieval]]
- [[reranking]]
- [[sequence-to-sequence]]
- [[transformer]]
- [[pretrained-language-model]]
- [[pretraining]]
- [[fine-tuning]]
- [[interaction-based-ranking]]
- [[representation-based-ranking]]
- [[question-answering]]

## Entities Extracted

- [[yifan-qiao]]
- [[chenyan-xiong]]
- [[zhenghao-liu-tsinghua]]
- [[zhiyuan-liu]]
- [[tsinghua-university]]
- [[microsoft-research]]
- [[bert]]
- [[ms-marco-passage-ranking]]
- [[clueweb09-b]]
- [[trec-web-track]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
