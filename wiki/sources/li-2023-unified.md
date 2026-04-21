---
type: source
subtype: paper
title: Unified Demonstration Retriever for In-Context Learning
slug: li-2023-unified
date: 2026-04-20
language: en
tags: [in-context-learning, demonstration-retrieval, dense-retrieval, ranking, multitask]
processed: true

raw_file: raw/papers/li-2023-unified/paper.pdf
raw_md: raw/papers/li-2023-unified/paper.md
bibtex_file: raw/papers/li-2023-unified/paper.bib
possibly_outdated: true

authors:
  - Xiaonan Li
  - Kai Lv
  - Hang Yan
  - Tianyang Lin
  - Wei Zhu
  - Yuan Ni
  - Guotong Xie
  - Xiaoling Wang
  - Xipeng Qiu
year: 2023
venue: ACL 2023
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.acl-long.256
url: https://aclanthology.org/2023.acl-long.256
citation_key: li2023unified
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper proposes Unified Demonstration Retriever (UDR), a single retriever for in-context learning across more than 30 tasks from 13 task families, instead of training one demonstration retriever per task. UDR uses a bi-encoder conditioned on task instructions, converts LM feedback into a unified listwise ranking signal, and iteratively mines better candidates from each task's full training set. This lets one model absorb supervision from classification, generation, semantic parsing, code summarization, and other workloads while remaining parameter-efficient. Across broad evaluations, UDR consistently outperforms Random, BM25, SBERT, Instructor, DR-Target, and EPR, and further shows stable transfer to unseen datasets, different inference LMs, different demonstration orders, and varying demonstration budgets.

## Problem & Motivation

Demonstration retrieval strongly affects in-context learning performance, but prior approaches either rely on generic retrievers such as BM25 or SBERT, or train task-specific retrievers with bespoke supervision. The first class is portable but heuristic; the second can be stronger but is hard to scale because every new task needs its own signal design, parameters, and deployment path. UDR is motivated by turning many tasks' supervision into one shared formulation: rank candidate demonstrations according to how much they help a language model predict the gold answer, then train one multitask retriever to imitate that ranking signal across heterogeneous NLP tasks.

## Method

- **Retriever architecture**: UDR uses a task-aware bi-encoder with separate query and demonstration encoders, scoring a query `x` and candidate `z` by `sim(x, z) = E_q(concat(I_i, x))^T E_d(concat(I_i, z))`, where `I_i` is the task instruction for task `T_i`.
- **Encoders**: `E_q` and `E_d` are initialized from two separate `BERT-base-uncased` encoders with CLS pooling; the full retriever has about `220M` parameters.
- **Unified LM feedback**: for each training example `(x, y)` and candidate list `Z`, the scoring LM ranks candidates by the conditional likelihood of the gold label or output. Generation tasks use `s_gen(z_j) = p_G(y | z_j, x)`, while classification and multi-choice tasks use normalized label probability `s_cls(z_j) = p_G(y | z_j, x) / sum_{y' in Y} p_G(y' | z_j, x)`.
- **Listwise ranking objective**: UDR optimizes a LambdaRank-style loss `L_rank = sum w * log(1 + exp(sim(x, z_j) - sim(x, z_i)))` with pair weight `w = max(0, 1 / r(z_i) - 1 / r(z_j))`, so higher-ranked demonstrations are pulled above lower-ranked ones.
- **In-batch negative training**: it adds `L_ib = -log(exp(sim(x, z*)) / sum_{z in Z_batch} exp(sim(x, z)))`, where `z*` is the rank-1 candidate, and combines losses as `L = lambda * L_rank + (1 - lambda) * L_ib`.
- **Multitask sampling**: batches are sampled task-wise with `p(T_i) = q_i^alpha / sum_j q_j^alpha`, using `alpha = 0.5` to reduce bias toward high-resource tasks.
- **Iterative candidate mining**: instead of fixing candidates by target overlap, UDR repeatedly mines `Z* = top-K_{z in D} sim(x, z)` over the full task dataset, uses the LM to rescore them, and thereby discovers both stronger positives and harder negatives.
- **Training hyperparameters**: Appendix B reports `optimizer = AdamW`, `learning_rate = 1e-4`, `warmup_steps = 500`, `batch_size = 128`, `lambda = 0.8`, `iterations = 3`, `K = 50` scored candidates, and `l = 8` sampled training candidates. Training uses `8` NVIDIA `A100-80GB` GPUs, `30` epochs before mining, then `10` epochs per iteration, for about `8` days end-to-end.
- **Inference**: each task's training set is encoded once with `E_d`, approximate search is done with FAISS, classification and multi-choice use `L = 8` demonstrations, generation uses the largest `L` satisfying `sum_i |z_i| + |x_test| + |y| <= C`, and final prediction is produced by greedy decoding.

## Key Results

- On classification and multi-choice tasks, UDR reaches `73.2` overall, outperforming EPR (`68.8`), Instructor (`63.2`), SBERT (`61.6`), and BM25 (`57.7`).
- On generation tasks, UDR reaches `30.9` overall, beating EPR (`27.7`) and BM25 (`24.2`), with large gains on Java code summarization (`25.2` vs. `17.4` for EPR) and E2E data-to-text (`32.6` vs. `29.3`).
- UDR improves hard semantic parsing datasets over EPR, including BREAK `35.2` vs. `31.9`, MTOP `66.8` vs. `64.4`, and SMCalFlow `60.4` vs. `54.3`.
- On selected classification datasets, gains can be substantial: Yelp `61.7` vs. `49.6` for EPR, SNLI `83.6` vs. `74.0`, and CR `82.6` vs. `65.7`.
- Ablations show the main gains come from the proposed training components: removing rank loss drops average performance from `58.4` to `55.7`, and removing self-guided candidate mining drops it to `56.5`.
- UDR transfers beyond its scoring LM: on SMCalFlow, it scores `64.7` with Text-Davinci-003 vs. `58.9` for EPR and `55.0` for BM25; on unseen datasets it reaches Twitter `56.8`, QNLI `74.4`, Ruby `19.6`, and JavaScript `21.6`, about `10` points better than BM25/SBERT on average.

## Limitations

- The paper only initializes UDR from `BERT-base-uncased`; it does not test whether stronger encoders such as RoBERTa or DeBERTa would materially change the results.
- UDR remains a black-box dense retriever, so the paper does not explain why certain demonstrations are judged informative or how to make retrieval behavior transparent.
- Training scores demonstrations independently with the LM, but inference conditions on a sequence of demonstrations, so interaction effects among retrieved examples are not explicitly modeled.
- The method is computationally heavy for a retrieval paper, requiring `8` A100-80GB GPUs and roughly `8` days for the full scoring-and-training pipeline.
- Because UDR learns from LM feedback, it may inherit the bias profile of the scoring language model.

## Concepts Extracted

- [[in-context-learning]]
- [[demonstration-retrieval]]
- [[bi-encoder]]
- [[task-instruction]]
- [[listwise-ranking]]
- [[iterative-candidate-mining]]
- [[multi-task-learning]]
- [[dense-retrieval]]
- [[in-batch-negatives]]

## Entities Extracted

- [[xiaonan-li]]
- [[kai-lv]]
- [[hang-yan]]
- [[tianyang-lin]]
- [[wei-zhu]]
- [[yuan-ni]]
- [[guotong-xie]]
- [[xiaoling-wang]]
- [[xipeng-qiu]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
