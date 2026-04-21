---
type: source
subtype: paper
title: Bottleneck-Minimal Indexing for Generative Document Retrieval
slug: du-2024-bottleneckminimal-2405-10974
date: 2026-04-20
language: en
tags: [generative-retrieval, document-retrieval, indexing, information-theory, k-means]
processed: true
raw_file: raw/papers/du-2024-bottleneckminimal-2405-10974/paper.pdf
raw_md: raw/papers/du-2024-bottleneckminimal-2405-10974/paper.md
bibtex_file: raw/papers/du-2024-bottleneckminimal-2405-10974/paper.bib
possibly_outdated: false
authors:
  - Xin Du
  - Lixin Xiu
  - Kumiko Tanaka-Ishii
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2405.10974
doi:
url: http://arxiv.org/abs/2405.10974
citation_key: du2024bottleneckminimal
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper reframes generative document retrieval (GDR) through information bottleneck theory instead of treating indexing as a document-only clustering problem. It formalizes indexing as a tradeoff between compact document codes `I(X;T)` and query predictability `I(T;Q)`, then derives a bottleneck-minimal indexing (BMI) objective whose practical approximation clusters documents by the mean BERT embeddings of their associated queries rather than by document embeddings. Using hierarchical `k`-means over query-derived representations plus an NCI retrieval model built on T5, BMI consistently outperforms hierarchical `k`-means, locality-sensitive hashing, and random indexing on NQ320K and MARCO Lite. The empirical bottleneck curves also show that better indexing corresponds to lower conditional distortion `I(X;Q|T)` for a given index rate.

## Problem & Motivation

Prior GDR systems usually assign document identifiers by partitioning document space alone, for example via hierarchical `k`-means on BERT document embeddings or by using textual substrings. The paper argues that this is theoretically incomplete because retrieval quality depends on how well document IDs preserve the query distribution, not just document geometry. Its motivation is therefore to replace distortion-minimal indexing with bottleneck-minimal indexing, where an index should be compact but still transmit the information needed for mapping queries to the correct documents.

## Method

- **Information-bottleneck objective**: optimize indexing through `\mathcal{L}(p(T|X)) = I(X;T) - \beta I(T;Q)`, equivalently `I(X;T) + \beta I(X;Q|T) + const` under the Markov condition `T \leftrightarrow X \leftrightarrow Q`.
- **Stationary solution**: the optimal assignment satisfies `p^*(T|X) \propto p^*(T)\exp(-\beta \mathrm{KL}[p(Q|X)\|p(Q|T)])`, so good IDs should preserve query-conditioned distributions rather than only document embeddings.
- **BMI definition**: define an indexing function `f: x \mapsto t` and score it by the likelihood `p(\mathcal{X}, \mathcal{Q} | f) = \prod_{x \in \mathcal{X}} p^*(X=x \mid T=f(x))`.
- **Gaussian approximation**: assume `p(Q|X=x) \sim \mathcal{N}(\mu_{Q|x}, \Sigma)` and `p(Q|T=t) \sim \mathcal{N}(\mu_t, \sigma^2 I)`. Under this assumption, minimizing the BMI objective reduces to `k`-means clustering.
- **Query-aware representation**: estimate each document representation as `\mu_{Q|x} \approx \mathrm{mean}(\mathrm{BERT}(\mathcal{Q}_x))` instead of the document embedding `\mu_x`. Here `\mathcal{Q}_x` combines `RealQ`, `GenQ`, and `DocSeg`.
- **Index construction**: run hierarchical `k`-means over `{ \mu_{Q|x} }`, use alphabet `V = [1,2,\ldots,30]`, and choose the minimum ID length `m` such that `|V|^m \ge |\mathcal{X}|`.
- **Query generation details**: for NQ320K use `15` `GenQ` queries per document; for MARCO Lite use `5`. For both datasets, add `10-12` random `DocSeg` spans per document with about `60` words each.
- **Baselines**: compare against hierarchical random indexing (HRI), hierarchical `k`-means indexing (HKmI), and locality-sensitive hashing indexing (LSHI). For LSH, Boolean codes are remapped every fifth bit into `V = [1,2,\ldots,32]`, and collisions are resolved with extra HKmI digits.
- **Retriever architecture**: use the NCI sequence-to-sequence setup with a Transformer encoder initialized from T5 checkpoints, a PAWA decoder with `4` Transformer layers, and model scales from `T5-mini` to `T5-base`.
- **Bottleneck estimation**: estimate `I(X;T)` and `I(X;Q|T)` by training models to predict ID prefixes of length `l = 2, 3, m`, which permits controlled ID collisions and traces empirical bottleneck curves.

## Key Results

- On **NQ320K with T5-base**, BMI reaches `66.69` Rec@1, `86.17` Rec@10, `93.23` Rec@100, and `73.91` MRR@100, beating HKmI at `65.43/85.20/92.64/72.73`.
- On **MARCO Lite with T5-base**, BMI reaches `45.20` Rec@1 and `55.47` MRR@100, improving over HKmI's `41.48` Rec@1 and `52.57` MRR@100.
- The gains are largest for smaller models: with **T5-mini**, BMI improves Rec@1 over HKmI by `+7.06` on NQ320K (`48.49` vs `41.43`) and `+6.45` on MARCO Lite (`13.54` vs `7.09`).
- In the MARCO Lite ablation, `GenQ + RealQ + DocSeg` outperforms `GenQ + RealQ` by `+5.77` Rec@1 for T5-base (`45.20` vs `39.43`), showing that document segments add information missing from short generated queries.
- With improved GenQ queries from fine-tuned docT5query, BMI reaches `67.8` Rec@1 on NQ320K, exceeding other static-index methods such as NCI (`66.9`) and approaching learned textual-index systems like GENRET (`68.1`) and NOVO (`69.3`).

## Limitations

- The practical derivation depends on Gaussian assumptions for `p(Q|X=x)` and `p(Q|T=t)`; the paper explicitly notes that multi-modal query distributions would require more sophisticated clustering than hierarchical `k`-means.
- BMI depends on accurately estimating `\mu_{Q|x}` from `RealQ`, `GenQ`, and `DocSeg`; low-quality or duplicate generated queries can weaken the index quality.
- The method still uses static numeric IDs and does not surpass the best learned textual-index methods in the final NQ320K comparison.
- Experiments are limited to NQ320K and MARCO Lite with the NCI/T5 backbone, so the evidence for cross-lingual or multimodal settings remains theoretical rather than empirical.

## Concepts Extracted

- [[generative-retrieval]]
- [[document-identifier]]
- [[information-bottleneck]]
- [[rate-distortion-theory]]
- [[mutual-information]]
- [[k-means-clustering]]
- [[locality-sensitive-hashing]]
- [[vector-quantization]]
- [[sequence-to-sequence]]
- [[transformer]]

## Entities Extracted

- [[xin-du]]
- [[lixin-xiu]]
- [[kumiko-tanaka-ishii]]
- [[bert]]
- [[t5]]
- [[doc-t5query]]
- [[natural-questions]]
- [[ms-marco]]
- [[bottleneck-minimal-indexing]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
