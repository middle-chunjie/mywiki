---
type: source
subtype: paper
title: Learning Transformer Programs
slug: friedman-2023-transformer-2306-01128
date: 2026-04-20
language: en
tags: [transformer, interpretability, mechanistic-interpretability, program-induction, nlp]
processed: true

raw_file: raw/papers/friedman-2023-transformer-2306-01128/paper.pdf
raw_md: raw/papers/friedman-2023-transformer-2306-01128/paper.md
bibtex_file: raw/papers/friedman-2023-transformer-2306-01128/paper.bib
possibly_outdated: true

authors:
  - Dan Friedman
  - Alexander Wettig
  - Danqi Chen
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2306.01128
doi:
url: https://arxiv.org/abs/2306.01128
citation_key: friedman2023transformer
paper_type: method

read_status: unread

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper proposes Transformer Programs, a constrained Transformer family whose weights can be deterministically converted into discrete Python programs. The key idea is to enforce a disentangled residual stream, discrete variable selection, one-to-one predicate matrices, hard attention, and lookup-table-style feed-forward modules so that each learned component corresponds to a human-readable operation. The method is trained with Gumbel-Softmax relaxations over discrete structure choices and evaluated on toy in-context learning, seven RASP algorithmic tasks, named entity recognition, and text classification. It reaches perfect accuracy on the toy induction task, exceeds 99% accuracy on five of seven short RASP tasks, beats a unigram baseline on CoNLL-2003 NER, and remains competitive with standard Transformers while being substantially easier to inspect and debug.

## Problem & Motivation

Post-hoc analysis of Transformer models can reveal partial patterns, but the paper argues that these analyses remain labor-intensive and may still fail to provide a faithful algorithmic account of model behavior. The authors therefore invert the standard interpretability workflow: rather than explaining a black-box Transformer after training, they constrain the architecture so the learned computation already lies in a discrete, interpretable subspace. The goal is not just qualitative transparency, but exact conversion into executable code that can be inspected with ordinary debugging and static-analysis tools.

## Method

- **Disentangled residual stream**: input embeddings encode named discrete variables in orthogonal subspaces; if the residual stream contains `m` categorical variables of cardinality `k`, then `x in {0,1}^{N x mk}` and each module writes a new variable to a dedicated address, yielding an attention-only output size of `` `(2 + L x H) x k` ``.
- **Discrete variable reading**: each projection is parameterized by a one-hot gate over available variables, so an attention head selects fixed key/query/value variables via `` `pi_K, pi_Q, pi_V in {0,1}^m` `` rather than arbitrary dense projections.
- **Categorical attention as predicates**: each head learns a binary predicate matrix `` `W_predicate in {0,1}^{k x k}` `` with each row summing to `1`; the score matrix is `` `S = x W_Q W_predicate (x W_K)^T` ``, making attention equivalent to an interpretable query-to-key rule table.
- **Hard attention for categorical outputs**: to keep outputs discrete, the head uses `` `A_i = One-hot(argmax_j S_{i,j})` ``; when no key matches it defaults to the beginning-of-sequence token, and when multiple keys match it selects the closest one through `selectclosest`.
- **Discrete optimization via relaxation**: the model optimizes distributions over gates and predicate rows with Gumbel-Softmax reparameterization, annealing temperature from `` `tau = 3.0` `` to `` `0.01` ``; after training it takes `` `theta* = argmax p(theta | Phi)` `` and decompiles the result to Python predicate functions plus library routines.
- **Extended modules**: larger tasks add factored categorical word embeddings, numerical attention with output `` `sum_j S_{i,j} var[j]` `` to support counting, and feed-forward modules that read `` `l = 2` `` variables and are converted to lookup tables by enumerating all discrete inputs.
- **Experimental settings**: the toy in-context task uses `2` layers, `1` categorical head per layer, variable cardinality `` `k = 10` ``, batch size `` `512` ``, learning rate `` `0.05` ``, and `250` epochs; RASP tasks grid-search `` `L in {2,3}` ``, `` `H in {4,8}` ``, `` `M in {2,4}` `` with equal categorical/numerical heads; NER uses CoNLL-2003 with `300`-d GloVe initialization, batch size `` `32` ``, variable cardinality `` `32` ``, and embedding-variable search over `` `{2,4,8}` ``.

## Key Results

- **Toy in-context learning**: a `2`-layer attention-only Transformer Program with `1` head per layer achieves `100%` accuracy on the held-out synthetic induction task.
- **RASP tasks**: the method exceeds `99%` accuracy on `5/7` tasks; the exceptions are Double-Histogram at `98.40` and Most-Freq at `75.69`.
- **Specific best RASP models**: Reverse reaches `99.79`, Histogram `100.0`, Sort `99.83`, Dyck-1 `99.30`, and Dyck-2 `99.09`.
- **Named entity recognition**: on CoNLL-2003, the best Transformer Program achieves `81.0` precision, `71.8` recall, and `76.1` F1, outperforming a unigram baseline (`66.2` F1) and also exceeding the compared standard Transformer (`70.5` F1).
- **Text classification**: Transformer Programs score `85.6` on TREC, `77.9` on MR, `93.0` on Subj, and `90.8` on AG News, competitive with standard Transformers (`88.6`, `77.2`, `92.3`, `91.7` respectively).
- **Scaling evidence**: on longer-input, larger-vocabulary RASP settings, Transformer Programs degrade more sharply than standard Transformers, with the hardest setting showing an accuracy gap of over `50%`.

## Limitations

- The method is optimization-sensitive: the paper's Reverse-task case study shows that manually initializing attention read/predicate components can enable learning, whereas randomly initializing even one such component can derail training.
- Performance deteriorates on longer sequences and larger vocabularies, suggesting that the learned discrete programs do not yet scale robustly beyond the short synthetic settings used in the main RASP experiments.
- The architectural restrictions that make the model interpretable also reduce expressivity, especially the one-to-one categorical predicate matrix, hard attention, and bounded numerical mechanisms.
- The resulting programs are inspectable but not always simple; the paper explicitly notes that larger learned programs can still be brittle, heuristic, and non-intuitive rather than compact human-style algorithms.

## Concepts Extracted

- [[transformer]]
- [[self-attention]]
- [[mechanistic-interpretability]]
- [[transformer-programs]]
- [[rasp]]
- [[hard-attention]]
- [[gumbel-softmax]]
- [[in-context-learning]]
- [[named-entity-recognition]]

## Entities Extracted

- [[dan-friedman]]
- [[alexander-wettig]]
- [[danqi-chen]]
- [[princeton-university]]
- [[tracr]]
- [[conll-2003]]
- [[glove]]
- [[pytorch]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
