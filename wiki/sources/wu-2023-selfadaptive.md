---
type: source
subtype: paper
title: "Self-Adaptive In-Context Learning: An Information Compression Perspective for In-Context Example Selection and Ordering"
slug: wu-2023-selfadaptive
date: 2026-04-20
language: en
tags: [in-context-learning, few-shot-learning, prompt-selection, prompt-ordering, information-compression]
processed: true

raw_file: raw/papers/wu-2023-selfadaptive/paper.pdf
raw_md: raw/papers/wu-2023-selfadaptive/paper.md
bibtex_file: raw/papers/wu-2023-selfadaptive/paper.bib
possibly_outdated: true

authors:
  - Zhiyong Wu
  - Yaoxiang Wang
  - Jiacheng Ye
  - Lingpeng Kong
year: 2023
venue: "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)"
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.acl-long.79
url: "https://aclanthology.org/2023.acl-long.79"
citation_key: wu2023selfadaptive
paper_type: method

read_status: unread
read_date:
rating:

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper introduces [[self-adaptive-in-context-learning]], an instance-level formulation of [[in-context-learning]] that selects and orders demonstrations separately for each test input rather than reusing one corpus-level prompt organization. The authors cast demonstration organization as an NP-hard combinatorial search problem and propose a two-stage select-then-rank pipeline: semantic retrieval narrows the candidate pool, then a [[minimal-description-length]] objective ranks sampled organizations by expected code length under the model's predictive distribution. Using GPT2-XL on eight NLP benchmarks, the resulting TopK+MDL system reaches 60.16 average accuracy, roughly 40% relative improvement over common random organization and 17% over TopK+LocalE, while analyses highlight label bias, substantial ranking headroom, and a clear efficiency-effectiveness trade-off.

## Problem & Motivation

ICL performance is highly sensitive to which demonstrations are shown and in what order, yet common practice still relies on random or corpus-level prompt organization. Existing search methods usually optimize one shared prompt using a validation set, which introduces majority bias and cannot tailor example organization to individual test inputs. The paper therefore asks whether each sample can adaptively choose its own demonstration set and permutation without access to labeled validation data, and whether an information-theoretic objective can provide a principled unsupervised ranking signal.

## Method

- **Problem formulation**: for a test sample `(\mathbf{x}, y)`, the prompt context `c` is an ordered set of `k` in-context examples; self-adaptive ICL searches for the organization `c*` that maximizes downstream accuracy for each instance.
- **Two-stage framework**: a selection module first reduces the combinatorial search space, then a ranking module scores candidate organizations. This converts an NP-hard global search into a tractable select-then-rank pipeline.
- **Selection stage**: the main instantiation uses [[nearest-neighbor-search]] over [[semantic-similarity]] to retrieve TopK candidates; the experiments set the retrieval pool to `30` examples per test input. The paper also studies VoteK and [[determinantal-point-process]] selection.
- **Ranking objective**: the ranking score follows an [[information-compression]] view of learning and seeks `c* = argmin_c L_theta(y | c, x) + L(theta)`. Because `L(theta)` is constant across organizations, ranking only depends on expected label codelength.
- **MDL approximation**: with Shannon-Huffman coding, `L_theta(y | c, x) = -log_2 p(y | c, x)`. Since the true label is unknown at ranking time, the paper replaces it with the model expectation `-\mathbb{E}_{p(y_i | c, x)} log_2 p(y_i | c, x)`, making the practical objective an entropy-like MDL criterion.
- **Search budget**: even `10` candidates and `8` demonstrations yield about `A_10^8 ≈ 1.8M` organizations, so the implementation randomly samples `10` organizations for ranking instead of exhaustive enumeration.
- **Experimental setup**: unless otherwise stated, experiments use `[[gpt2-xl]]` (`1.5B` parameters), `8` in-context examples, `3` random seeds, HuggingFace datasets/models, and eight benchmarks spanning sentiment classification, NLI, topic classification, and commonsense QA.

## Key Results

- **Overall average**: TopK+MDL reaches `60.16` average accuracy across `8` datasets, outperforming TopK (`58.14`), TopK+LocalE (`51.36`), corpus-level GlobalE (`49.75`), and prompting (`39.32`).
- **Relative gains**: the method delivers about `40%` relative improvement over the common random corpus-level organization baseline (`42.78 -> 60.16`) and `17%` over TopK+LocalE (`51.36 -> 60.16`).
- **Dataset-level wins**: it achieves `91.51` on [[sst-2]], `58.77` on [[snli]], `46.56` on [[mnli]], `61.43` on QNLI, `42.47` on [[trec]], `87.94` on [[ag-news]], and `53.15` on CommonsenseQA, beating or matching the strongest non-majority baselines on most tasks.
- **MDL correlates with accuracy**: lower average MDL aligns with better performance, e.g. on SST-2 TopK+MDL gets `0.6810` MDL with `91.51` accuracy versus TopK `0.6861` / `83.91`; on TREC it gets `5.4496` / `42.47` versus TopK `5.5618` / `40.80`.
- **Analysis findings**: increasing ranking window size from `2` toward `50` steadily improves accuracy but multiplies inference cost; gains also persist in few-shot retrieval pools from `16` to `1024` candidates and across OPT scales from `350M` to `175B`.

## Limitations

- The method trades efficiency for effectiveness because each test instance requires scoring multiple sampled organizations instead of one fixed prompt.
- Performance depends on having a sufficiently rich retrieval set; when the available annotated pool is small, the gains over TopK shrink noticeably.
- The current ranking stage samples only `10` organizations, so it remains a coarse approximation to the optimal permutation/selection search.
- The strong TopK selector partly drives the final gains, but TopK itself degrades on tasks with large label spaces such as multi-choice QA, limiting transfer to broader settings.
- The study focuses on classification-style NLP tasks and does not test generative tasks with effectively unbounded output spaces.

## Concepts Extracted

- [[self-adaptive-in-context-learning]]
- [[in-context-learning]]
- [[few-shot-learning]]
- [[demonstration-selection]]
- [[nearest-neighbor-search]]
- [[semantic-similarity]]
- [[minimal-description-length]]
- [[information-compression]]
- [[determinantal-point-process]]
- [[prompt-order-sensitivity]]
- [[large-language-model]]

## Entities Extracted

- [[zhiyong-wu]]
- [[yaoxiang-wang]]
- [[jiacheng-ye]]
- [[lingpeng-kong]]
- [[shanghai-artificial-intelligence-laboratory]]
- [[xiamen-university]]
- [[university-of-hong-kong]]
- [[gpt2-xl]]
- [[sst-2]]
- [[snli]]
- [[mnli]]
- [[trec]]
- [[ag-news]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
