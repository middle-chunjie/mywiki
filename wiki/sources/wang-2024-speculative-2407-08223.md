---
type: source
subtype: paper
title: "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting"
slug: wang-2024-speculative-2407-08223
date: 2026-04-20
language: en
tags: [rag, retrieval, llm, question-answering, inference]
processed: true
raw_file: raw/papers/wang-2024-speculative-2407-08223/paper.pdf
raw_md: raw/papers/wang-2024-speculative-2407-08223/paper.md
bibtex_file: raw/papers/wang-2024-speculative-2407-08223/paper.bib
possibly_outdated: false
authors:
  - Zilong Wang
  - Zifeng Wang
  - Long T. Le
  - Huaixiu Steven Zheng
  - Swaroop Mishra
  - Vincent Perot
  - Yuwei Zhang
  - Anush Mattapalli
  - Ankur Taly
  - Jingbo Shang
  - Chen-Yu Lee
  - Tomas Pfister
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2407.08223
doi:
url: http://arxiv.org/abs/2407.08223
citation_key: wang2024speculative
paper_type: method
read_status: unread
domain: retrieval
---

## Summary

This paper proposes Speculative RAG, a retrieval-augmented generation framework that splits RAG into parallel drafting by a smaller specialist model and single-pass verification by a larger generalist model. Retrieved documents are clustered into perspectives, then mixed into diverse low-token subsets so each draft sees less redundant evidence than standard long-context RAG. The drafter outputs both an answer candidate and a rationale, while the verifier ranks draft-rationale pairs with language-model probabilities rather than additional tuning. Across TriviaQA, MuSiQue, PopQA, PubHealth, and ARC-Challenge, the method improves both accuracy and latency over standard RAG, Self-RAG, and CRAG, with the largest reported gain being `+12.97` accuracy points and `-50.83%` latency on PubHealth.

## Problem & Motivation

Standard RAG often concatenates many retrieved documents into one long prompt, which increases latency and makes reasoning over redundant or weakly ordered evidence difficult. Prior attempts to improve RAG commonly rely on iterative refinement, self-critique tokens, or extra instruction tuning of the main model, which either raises deployment cost or slows inference. This paper targets a more practical tradeoff: preserve grounding quality while reducing long-context burden by letting a smaller specialist model draft answers from compact evidence subsets and a stronger generalist model only verify the drafts.

## Method

- **Task decomposition**: Speculative RAG decomposes RAG into a specialist drafter `\mathcal{M}_{\text{Drafter}}` and a generalist verifier `\mathcal{M}_{\text{Verifier}}`, selecting the final answer with `\hat{A} = \arg\max_{\alpha_j} \rho_j`.
- **Problem setup**: each example is `(Q, D, A)`, where `Q` is the query, `D = {d_1, ..., d_n}` are retrieved documents, and `A` is the target answer; the system samples `m` subsets of size `k` from `D`.
- **Instruction-tuned drafter**: the drafter is fine-tuned to jointly produce an answer and rationale by maximizing `` `\mathbb{E}_{(Q,A,D,E)} \log P_{\mathcal{M}_{\text{Drafter}}}(A, E \mid Q, D)` ``, where `E` is a synthesized rationale distilled from stronger LMs.
- **Multi-perspective sampling**: retrieved documents are embedded with an instruction-aware encoder conditioned on `Q`, clustered with K-Means into `k` groups, and each draft subset samples one document per cluster to reduce redundancy while preserving topical diversity.
- **Parallel drafting**: for subset `\delta_j = {d_{j_1}, ..., d_{j_k}}`, the drafter generates answer draft `\alpha_j` and rationale `\beta_j` in parallel from `` `Q, d_{j_1}, ..., d_{j_k} -> \alpha_j, \beta_j` ``.
- **Draft confidence**: the drafter-side reliability score is `` `\rho_{\text{Draft}, j} = P(\beta_j \mid Q, d_{j_1}, ..., d_{j_k}) + P(\alpha_j \mid Q, d_{j_1}, ..., d_{j_k}, \beta_j)` ``.
- **Verifier scoring**: the verifier ranks each draft-rationale pair with self-consistency and self-reflection scores, `` `\rho_{\text{Self-contain}} = P(\alpha, \beta \mid Q)` `` and `` `\rho_{\text{Self-reflect}} = P("Yes" \mid Q, \alpha, \beta, R)` ``, where `R` is a prompt such as "Do you think the rationale supports the answer?"
- **Final score**: the overall selection score is `` `\rho_j = \rho_{\text{Draft}, j} \cdot \rho_{\text{SC}, j} \cdot \rho_{\text{SR}, j}` ``, computed from one forward pass over prompt `[Q, \alpha, \beta, R, "Yes"]`.
- **Experimental configuration**: the main drafter is Mistral `7B`; verifiers are Mistral `7B` or Mixtral `8x7B`; embeddings use InBedder-Roberta; inference uses vLLM with greedy decoding (`temperature = 0`).
- **Hyperparameters**: for TriviaQA, PopQA, PubHealth, and ARC-Challenge, they retrieve top `10` documents and use `m = 5`, `k = 2`; for MuSiQue, they retrieve top `15` documents and use `m = 10`, `k = 6`.
- **Training details**: the drafter is trained on `40,059` instruction-following examples augmented with retrieved documents and rationales, using Mistral `7B` as the base model and `16` Nvidia A100-SXM4-40GB GPUs.

## Key Results

- With Mixtral `8x7B` as verifier plus the `7B` drafter, accuracy reaches `74.24` on TriviaQA, `31.57` on MuSiQue, `57.54` on PopQA, `76.60` on PubHealth, and `80.55` on ARC-Challenge.
- Relative to Mixtral-Instruct `8x7B` standard RAG, the method improves accuracy by `+0.33` on TriviaQA, `+2.15` on MuSiQue, `+3.86` on PopQA, `+12.97` on PubHealth, and `+2.14` on ARC-Challenge.
- The drafter alone already beats most baselines in the main table, scoring `71.11`, `27.89`, `56.40`, `75.58`, and `74.49` across the five benchmarks.
- Using rationale-only verification is both faster and at least as strong as verifying against original documents: on TriviaQA it gets `74.24` accuracy with `1.93s` latency, versus `74.08` and `2.13s` when replacing rationale with documents.
- Compared with the strongest standard-RAG baseline, latency drops by up to `11.90%` on TriviaQA, `15.07%` on MuSiQue, `44.31%` on PopQA, `50.83%` on PubHealth, and `22.77%` on ARC-Challenge.
- Multi-perspective sampling matters: removing clustering drops TriviaQA by `1.22` and PubHealth by `1.22`, while forcing all sampled documents from one cluster drops `1.88` and `2.23` respectively.

## Limitations

- The method assumes a specialized drafter that is instruction-tuned with rationale supervision, so deployment still requires extra data curation and model training even if the verifier is off-the-shelf.
- Efficiency gains partly depend on parallel drafter endpoints; the reported latency reductions may shrink in resource-constrained serving setups.
- The paper focuses on answer selection after retrieval rather than improving retrieval itself, so failure cases from missing evidence are not fundamentally resolved.
- Multi-perspective clustering is less helpful for bridge-style multi-hop questions, which the appendix notes may require explicit bridge-entity identification or dynamic search.
- Experiments are centered on five QA/verification benchmarks and do not test broader generation settings such as citation-heavy long-form answers.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[speculative-decoding]]
- [[instruction-tuning]]
- [[rationale-generation]]
- [[lost-in-the-middle]]
- [[long-context-reasoning]]
- [[k-means-clustering]]
- [[dense-retrieval]]

## Entities Extracted

- [[zilong-wang]]
- [[zifeng-wang]]
- [[long-le]]
- [[huaixiu-steven-zheng]]
- [[swaroop-mishra]]
- [[vincent-perot]]
- [[yuwei-zhang]]
- [[anush-mattapalli]]
- [[ankur-taly]]
- [[jingbo-shang]]
- [[chen-yu-lee]]
- [[tomas-pfister]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
