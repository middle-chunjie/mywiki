---
type: source
subtype: paper
title: Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering
slug: shi-2024-generatethenground
date: 2026-04-20
language: en
tags: [rag, multi-hop-qa, grounding, distillation, llm]
processed: true

raw_file: raw/papers/shi-2024-generatethenground/paper.pdf
raw_md: raw/papers/shi-2024-generatethenground/paper.md
bibtex_file: raw/papers/shi-2024-generatethenground/paper.bib
possibly_outdated: false

authors:
  - Zhengliang Shi
  - Shuo Zhang
  - Weiwei Sun
  - Shen Gao
  - Pengjie Ren
  - Zhumin Chen
  - Zhaochun Ren
year: 2024
venue: "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)"
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2024.acl-long.397
url: "https://aclanthology.org/2024.acl-long.397"
citation_key: shi2024generatethenground
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

The paper proposes GenGround, a retrieval-augmented multi-hop QA framework that inverts the standard retrieve-then-read order: an LLM first decomposes a complex question into a simpler sub-question and produces an immediate answer from parametric knowledge, then grounds that question-answer pair in retrieved evidence to revise mistakes before continuing to the next hop. This design aims to exploit both the internal world knowledge of LLMs and external corpora while reducing the damage from noisy retrieval. The method also introduces instructional grounding distillation (IDG), where ChatGPT-generated grounding trajectories supervise a smaller Mistral-7B student. Across four MHQA benchmarks, GenGround consistently beats strong baselines such as ReAct, DSPy, RetGen, and SearChain.

## Problem & Motivation

Standard retrieval-augmented multi-hop QA pipelines usually follow retrieve-then-read: decompose a question, retrieve documents, and answer from the retrieved evidence. The paper argues this pattern has two coupled weaknesses. First, answer quality is bounded by retriever recall, so potentially useful parametric knowledge in the LLM is underused. Second, retrieved sets contain irrelevant or plausible-but-wrong passages, and injecting them directly into reasoning can push the model onto an incorrect chain. The motivation behind GenGround is therefore to let the LLM first exploit its own deductive and world knowledge to draft a stepwise answer, then use retrieval selectively as a verification-and-revision mechanism rather than as the only source of reasoning.

## Method

- **Iterative formulation**: for a multi-hop question `Q`, GenGround alternates answer deduction and grounding over hops `i`, maintaining context `H_i = {(q_j, \\tilde{a}_j) | j < i}` until a final answer is reached.
- **Answer deduction**: the backbone model `\\mathcal{M}_\\theta` receives instruction `I_A`, the original question `Q`, and prior context `H_i`, then generates both a sub-question and an immediate answer: `q_i, a_i = \\mathcal{M}_\\theta(I_A, Q, H_i)`.
- **Retrieval step**: the method retrieves evidence with the deduced sub-question rather than the original question, `D_i = Retrieval(q_i)`, so retrieval is aligned to the current hop.
- **Instructional knowledge grounding**: GenGround asks the LLM to cite the most relevant evidence for `(q_i, a_i)` and revise the answer, `\\tilde{a}_i = \\mathcal{M}_\\theta(I_G, Q, q_i, a_i, D_i)`. If no relevant evidence is found, the model emits an `Empty` signal and keeps the current answer as backup.
- **Context update**: each revised trajectory is folded back into reasoning with `H_{i+1} = H_i \\cup {(q_i, \\tilde{a}_i)}`, so later hops condition on grounded intermediate answers instead of raw generations.
- **Batch grounding**: rather than exposing the LLM to all retrieved documents at once, the grounding phase scans them in batches of size `b`; the implementation uses `b = 3` and retrieves top-`10` documents. Grounding stops early once supporting evidence is found, which reduces noise and unnecessary token consumption.
- **Instructional grounding distillation (IDG)**: to transfer the grounding behavior into a smaller model, the authors synthesize `50k` single-hop examples from [[natural-questions]], pair each question with a smaller-model immediate answer and a ChatGPT-generated revision trajectory, and optimize a student with language-model loss `\\mathcal{L}_G = -\\log P_\\theta(\\tilde{a} | I_G, \\{\\tilde{d}\\} \\cup D)`.
- **Implementation details**: the main backbone is `gpt-3.5-turbo` with decoding temperature `0`; the main retriever is [[colbertv2]] with top-`10` retrieval, with [[bm25]] and Google Search used in analysis; the student model is [[mistral-7b]], trained with DeepSpeed ZeRO, learning rate `5e-5`, weight decay `0.01`, and `18` hours on `3 x NVIDIA A100-PCIE-80GB`.

## Key Results

- On [[hotpotqa]], GenGround reaches `F1 = 52.26`, `Acc = 47.27`, `Acc† = 55.73`, beating DSPy (`47.80 / 42.43 / 50.07`) and ReAct (`40.70 / 33.10 / 37.12`).
- On [[musique]], it achieves `F1 = 27.36`, `Acc = 20.24`, `Acc† = 24.77`, clearly above DSPy (`20.11 / 13.40 / 17.40`) and SearChain (`- / 17.07 / 20.45` on reported metrics).
- On [[2wiki-multihopqa]], it scores `F1 = 50.21`, `Acc = 45.61`, `Acc† = 48.58`, outperforming DSPy (`44.77 / 43.43 / 45.43`) and RetGen (`36.00 / 42.17 / 45.21`).
- On [[strategyqa]], it reaches `Acc = 77.12`, above DSPy (`71.78`), RetGen (`73.42`), and slightly above SearChain (`76.95`).
- With [[mistral-7b]] as backbone, the distilled version averages `32.69` Acc across HQA/MQA/WQA, improving over vanilla GenGround prompting (`29.45`) and all reported baselines (`28.31` DSPy, `25.63` GRG w/ decomposition, `24.13` RetGen, `26.53` SearChain).
- Changing retrievers still leaves the method on top: average Acc is `40.32` with [[bm25]] and `46.87` with Google Search, both better than the corresponding baselines in Table 5.
- Ablations show each component matters: without deduction, HotpotQA Acc drops by `6` and StrategyQA Acc by `10`; without grounding, metrics also fall sharply; without batch grounding, HotpotQA F1 drops from `52.26` to `47.27` and Acc from `47.27` to `45.03`.
- Analysis reports a `53.2%` overall success rate on sampled HotpotQA cases, composed of `28.7%` directly correct generations and `24.5%` incorrect-then-corrected cases, while the error rate after grounding is only `5.6%`.

## Limitations

- The first step depends on the LLM being able to generate a useful immediate answer; if the initial answer is poor or off-manifold for the task, later grounding has less to repair.
- The framework assumes complex questions can be decomposed into simpler sub-questions, but decomposition itself remains a hard unresolved problem.
- Grounding can only correct mistakes when retrieved documents actually contain the necessary evidence; missing evidence or misleading evidence still limits performance.
- The evaluation is concentrated on Wikipedia-backed MHQA benchmarks, so transfer to other domains, corpora, or non-QA tasks is not established here.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[multihop-question-answering]]
- [[question-decomposition]]
- [[knowledge-grounding]]
- [[batch-grounding]]
- [[hallucination]]
- [[dense-retrieval]]
- [[instruction-tuning]]
- [[knowledge-distillation]]
- [[chain-of-thought]]
- [[synthetic-data]]
- [[parametric-knowledge]]

## Entities Extracted

- [[zhengliang-shi]]
- [[shuo-zhang]]
- [[weiwei-sun]]
- [[shen-gao]]
- [[pengjie-ren]]
- [[zhumin-chen]]
- [[zhaochun-ren]]
- [[shandong-university]]
- [[university-of-electronic-science-and-technology-of-china]]
- [[leiden-university]]
- [[hotpotqa]]
- [[musique]]
- [[2wiki-multihopqa]]
- [[strategyqa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
