---
type: source
subtype: paper
title: Chain-of-Retrieval Augmented Generation
slug: wang-2025-chainofretrieval-2501-14342
date: 2026-04-20
language: en
tags: [rag, retrieval, multi-hop-qa, test-time-scaling, grounding]
processed: true
raw_file: raw/papers/wang-2025-chainofretrieval-2501-14342/paper.pdf
raw_md: raw/papers/wang-2025-chainofretrieval-2501-14342/paper.md
bibtex_file: raw/papers/wang-2025-chainofretrieval-2501-14342/paper.bib
possibly_outdated: false
authors:
  - Liang Wang
  - Haonan Chen
  - Nan Yang
  - Xiaolong Huang
  - Zhicheng Dou
  - Furu Wei
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2501.14342
doi: 10.48550/arXiv.2501.14342
url: http://arxiv.org/abs/2501.14342
citation_key: wang2025chainofretrieval
paper_type: method
read_status: unread
domain: retrieval
---

## Summary

This paper proposes CoRAG, a retrieval-augmented generation framework that explicitly trains an LLM to retrieve, decompose, and reformulate queries step by step before producing the final answer. Instead of relying on a single retrieval call or pure prompting, the method constructs intermediate retrieval chains with rejection sampling and fine-tunes a base model on next sub-query, sub-answer, and final-answer prediction. At inference time, CoRAG can scale test-time compute through longer chains, best-of-`N` sampling, or tree search. The resulting `8B` model substantially improves multi-hop QA, often by more than `10` EM points over strong baselines, and reaches new state-of-the-art scores on nearly all KILT hidden-test tasks despite using a comparatively modest open model backbone.

## Problem & Motivation

Standard [[retrieval-augmented-generation]] systems usually do one retrieval step before generation, so overall quality is bottlenecked by the retriever's first-pass recall. This is especially problematic for [[multihop-question-answering]], where the right next query often depends on evidence uncovered in earlier steps. The paper argues that in-context prompting or distillation alone is insufficient for this setting and instead trains the model to perform iterative retrieval, reasoning, and [[query-reformulation]] explicitly. A second motivation is controllability: by exposing retrieval-chain length and decoding strategy as knobs, the system can trade off quality against [[test-time-scaling]] cost at inference time.

## Method

- **Retrieval-chain construction**: for each training example with query `Q` and answer `A`, the method samples a chain of sub-queries and sub-answers `(Q_{1:L}, A_{1:L})`, where each `Q_i = LLM(Q, Q_{<i}, A_{<i})` and each `A_i = LLM(Q_i, D_{1:k}^{(i)})` after retrieving top-`k` documents.
- **Chain scoring via [[rejection-sampling]]**: candidate chains are ranked by the final-answer likelihood `log P(A | Q, Q_{1:L}, A_{1:L})`, and the best chain is kept to augment QA-only data with intermediate supervision.
- **Training objective**: CoRAG jointly optimizes `L_sub_query = -log P(Q_i | Q, Q_{<i}, A_{<i})`, `L_sub_answer = -log P(A_i | Q_i, D_{1:k}^{(i)})`, and `L_final_answer = -log P(A | Q, Q_{1:L}, A_{1:L}, D_{1:k})` under standard next-token prediction.
- **Backbone and data**: the base model is [[llama-3-1-8b-instruct]]; rejection sampling uses [[e5-mistral]] for KILT retriever training and `E5-large` for intermediate retrieval over the `36M`-passage KILT Wikipedia corpus.
- **Training hyperparameters**: full-parameter fine-tuning for `1` epoch, max sequence length `3072`, learning rate `5e-6` on multi-hop QA and `1e-5` on KILT, batch size `256` / `1024`, warmup `100` steps, and sample ratio `0.2` for both sub-query and sub-answer tasks.
- **Data-generation hyperparameters**: up to `16` sampled chains per instance, maximum chain length randomly selected from `[1, 5]`, sub-query temperature `0.7`, sub-answer temperature `0`, top-`5` retrieved passages per sub-query, and early termination when the sub-answer matches the gold answer or average conditional log-likelihood exceeds `-0.05`.
- **Inference strategies**: greedy decoding, best-of-`N` chain sampling with penalty `log P("No relevant information found" | chain)`, and a [[breadth-first-search]] tree-search variant that expands candidate states, rolls out descendants, and keeps the state with the lowest average penalty.
- **Tree-search hyperparameters**: expansion size `4`, rollout count `2`, rollout depth capped at `2`, and repeated sub-queries are discarded to avoid degenerate loops.
- **Stopping variant**: an auxiliary stop head predicts whether the current chain prefix is sufficient; decoding is constrained to `"Yes"` / `"No"` and the stop policy can be biased by logit adjustment.

## Key Results

- On [[2wikimultihopqa]], CoRAG-8B improves from `55.1` EM / `60.7` F1 for a fine-tuned RAG baseline to `72.5` EM / `77.3` F1 with `L = 10`, best-of-`8`.
- On [[hotpotqa]], the best reported CoRAG setting reaches `56.3` EM / `69.8` F1 versus `50.3` / `63.5` for the fine-tuned baseline and `45.2` / `57.3` for Search-o1-32B.
- On [[bamboogle]], CoRAG reaches `54.4` EM / `68.3` F1 at `L = 10`, best-of-`8`, exceeding the fine-tuned baseline by `13.6` EM.
- On [[musique]], CoRAG improves from `17.4` EM / `28.1` F1 to `30.9` EM / `42.4` F1, a gain of `13.5` EM over the same-backbone baseline.
- On the KILT hidden test set, the model posts `93.9` on AIDA, `88.2` on WnWi, `76.7` on WnCw, `88.0` on T-REx, `87.2` on zsRE, `63.1` on NQ, `60.6` on HoPo, `88.3` on TriviaQA, and `93.1` on FEVER.
- Retrieval quality also improves: for example, HotpotQA `R@10` rises from `59.1` to `72.1`, 2WikiMultiHopQA from `54.9` to `81.4`, Bamboogle from `31.2` to `59.2`, and MuSiQue from `29.0` to `47.1`.

## Limitations

The method is evaluated mainly on short-answer knowledge-intensive tasks, so its behavior on long-form grounded generation remains unclear. Performance gains depend strongly on the quality of sampled retrieval chains; iterative self-training gives mixed results, while stronger teachers such as [[gpt-4o]] still help. The paper's compute analysis ignores retrieval latency/cost and treats prompt and generated tokens equally, so the reported scaling frontier is only approximate. Tree search is substantially more expensive than greedy or best-of-`N` decoding and is not explored deeply. The system also relies on a fixed Wikipedia corpus, which hurts datasets like [[bamboogle]] when fresher evidence is required.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[multihop-question-answering]]
- [[query-reformulation]]
- [[test-time-scaling]]
- [[rejection-sampling]]
- [[dense-retrieval]]
- [[bi-encoder]]
- [[approximate-nearest-neighbor-search]]
- [[breadth-first-search]]
- [[exact-match]]
- [[large-language-model]]

## Entities Extracted

- [[liang-wang-microsoft]]
- [[haonan-chen]]
- [[nan-yang]]
- [[xiaolong-huang]]
- [[zhicheng-dou]]
- [[furu-wei]]
- [[microsoft-research]]
- [[renmin-university-of-china]]
- [[llama-3-1-8b-instruct]]
- [[e5-mistral]]
- [[rankllama]]
- [[2wikimultihopqa]]
- [[hotpotqa]]
- [[bamboogle]]
- [[musique]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
