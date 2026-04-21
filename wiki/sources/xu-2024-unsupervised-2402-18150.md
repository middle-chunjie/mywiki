---
type: source
subtype: paper
title: Unsupervised Information Refinement Training of Large Language Models for Retrieval-Augmented Generation
slug: xu-2024-unsupervised-2402-18150
date: 2026-04-20
language: en
tags: [rag, llm, unsupervised-training, zero-shot, robustness]
processed: true

raw_file: raw/papers/xu-2024-unsupervised-2402-18150/paper.pdf
raw_md: raw/papers/xu-2024-unsupervised-2402-18150/paper.md
bibtex_file: raw/papers/xu-2024-unsupervised-2402-18150/paper.bib
possibly_outdated: false

authors:
  - Shicheng Xu
  - Liang Pang
  - Mo Yu
  - Fandong Meng
  - Huawei Shen
  - Xueqi Cheng
  - Jie Zhou
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.18150
doi:
url: http://arxiv.org/abs/2402.18150
citation_key: xu2024unsupervised
paper_type: method
read_status: unread
domain: llm
---

## Summary

The paper reframes retrieval-augmented generation as an information refinement problem: instead of merely conditioning on retrieved passages, a large language model should refine them into outputs that are more concise, accurate, and complete. To realize that behavior without task-specific labels, the authors propose InFO-RAG, an unsupervised training method built from Wikipedia sentence windows and three synthetic retrieval scenarios: direct extraction, correction/completion under corrupted evidence, and answer generation from only semantically related context. The method trains LLaMA-2 with LoRA and preserves the decoder-only prefix-LM format. Across 11 datasets spanning QA, slot filling, dialogue, language modeling, and code generation, InFO-RAG improves zero-shot RAG performance by `9.39%` relative points on average, also helping in-context learning and robustness to retrieval quality shifts.

## Problem & Motivation

The paper argues that standard decoder-only pre-training creates a mismatch for RAG. During pre-training, the model minimizes the negative log-likelihood of the entire input sequence, so retrieved passages are treated as just another prefix segment rather than as external evidence that should be selectively trusted, corrected, or ignored. This becomes problematic when retrieved texts are long, noisy, incomplete, or wrong: the model may overlook useful facts, copy misleading statements, or fail to combine retrieved content with parametric knowledge. Prior prompt-only or task-specific supervised fixes either do not update this underlying capability or risk hurting generalization. The authors therefore target a general, low-cost, unsupervised way to train LLMs to use retrieved text more intelligently across heterogeneous zero-shot RAG tasks.

## Method

- **Information-refiner view**: define the LLM in RAG as an information refiner that should produce "positive information gain," i.e. outputs more concise, accurate, and complete than the retrieved input.
- **Unsupervised sample construction**: from each Wikipedia document `d`, sample `k = 15` consecutive sentences `S = [s_1, ..., s_k]`; choose one sentence `s_l`, keep the first `1/3` to `2/3` of its tokens as prefix `s_l^p`, and predict the remainder `s_l^t`.
- **Prefix-LM training form**: optimize next-token prediction for `s_l^t` conditioned on retrieved context and prefix, written as `` `p(s_l^t) = p_theta([R(s_l^p); s_l^p])` ``.
- **Scenario 1 / Select and Copy**: use all sentences in `S` as retrieved texts, so `` `p(s_l^t) = p_theta([S; s_l^p])` ``. Because the answer sentence is already present, the model learns to extract the relevant span from complex retrieved context.
- **Scenario 2 / Correct and Complete**: score token informativeness by comparing layerwise next-word distributions. For token `s_i^[a]`, compute `` `d_j(s_i^[a] | s_i^<a) = softmax(W H_j^[a])` ``, then use the maximum Jensen-Shannon divergence against the last layer, `` `O_i^[a] = argmax_{j in J} JSD(d_N || d_j)` ``, over candidate layers `J = {0, ..., N/2}`. Select the top `50%` most informative tokens, perturb `30%` of them, then apply `50%` `[MASK]`, `40%` random replacement, and `10%` unchanged to simulate incomplete, incorrect, and correct evidence.
- **Scenario 3 / Contextual Stimulation**: remove `s_l` from the retrieved set and train with only related context, i.e. `` `p(s_l^t) = p_theta([S - {s_l}; s_l^p])` ``, forcing the model to activate parametric knowledge using semantically relevant but non-answer-bearing retrieval.
- **Multi-task schedule**: mix the three tasks during training, with Select-and-Copy taking `20%` of batches and Correct-and-Complete plus Contextual-Stimulation each taking `40%`.
- **Model adaptation**: train LLaMA-2 `7B` and `13B` plus chat variants using LoRA on `4` A100 GPUs for `5K` steps with learning rate `1e-5`; per-GPU batch size is `4` for `7B` and `2` for `13B`.
- **Evaluation setup**: for ODQA, slot filling, and language modeling, use [[colbertv2]] over a Wikipedia corpus of `21,015,324` passages and provide Top-`5` retrieved passages; for LFQA, dialogue, and multi-hop QA, use dataset-provided distractor passages.

## Key Results

- Across 11 datasets and 7 tasks, InFO-RAG improves average zero-shot RAG performance by `9.39%` relative points. The strongest model, LLaMA-2-13B-chat, rises from `43.23` to `46.55` overall; LLaMA-2-13B improves from `36.86` to `41.04`.
- On LLaMA-2-13B-chat, ODQA and slot-filling improve from `50.36 -> 54.04` on [[natural-questions]], `45.47 -> 51.07` on [[webquestions]], `62.53 -> 65.39` on `T-REx`, and `56.81 -> 59.05` on Zero-Shot RE.
- Multi-hop and generation-heavy tasks also improve: HotpotQA `61.23 -> 61.91`, MuSiQue `47.06 -> 47.93`, WikiText `60.52 -> 63.92`, Python CodeBLEU `22.34 -> 31.98`, and Java CodeBLEU `30.96 -> 38.12`.
- Fine-grained scenario analysis shows better use of imperfect retrieval. On LLaMA-2-13B-chat, NQ "replace" rises `30.72 -> 33.85`, WebQ "no-ans." jumps `5.47 -> 11.25`, and zsRE "replace" increases `16.58 -> 25.02`.
- In-context learning benefits instead of interfering: on NQ, InFO-RAG moves from `43.36` at `0` examples to `47.75` at `12` examples, while the baseline fluctuates and only reaches `44.32`; on WebQ the same setting improves `43.20 -> 47.86`.
- The method also strengthens an external open-retrieval framework: SearChain + InFO-RAG improves HotpotQA `31.21 -> 33.04`, MuSiQue `11.27 -> 12.10`, T-REx `64.58 -> 66.95`, and zsRE `58.91 -> 60.72`.
- Additional analysis shows the gains are not just from extra Wikipedia training: on LLaMA-2-13B-chat, plain training on Wikipedia yields `42.92` overall versus `46.55` for InFO-RAG. MMLU without RAG stays close to baseline (`54.8` vs `54.3` for `13B`), indicating limited catastrophic forgetting.

## Limitations

- The paper only trains and evaluates `7B` and `13B` LLaMA-2 variants; it does not establish whether the same unsupervised recipe scales to larger or newer LLMs.
- The training data are synthesized from Wikipedia sentence windows, which is convenient and cheap but only an approximation of real retrieval failures in web or enterprise RAG.
- Most experiments use a fixed retriever/setup per task family, so the method's sensitivity to stronger retrievers, different chunking schemes, or modern long-context models is not fully explored.
- Improvements on some tasks are modest relative to the headline average, especially on already-strong settings such as HotpotQA, MuSiQue, ELI5, and Wizard of Wikipedia.
- Scenario 1 alone can overfit and hurt performance, so the final behavior depends on careful multi-task balancing rather than a single simple objective.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[information-refinement]]
- [[positive-information-gain]]
- [[prefix-language-modeling]]
- [[zero-shot-learning]]
- [[in-context-learning]]
- [[lora]]
- [[parameter-efficient-fine-tuning]]
- [[catastrophic-forgetting]]
- [[open-domain-question-answering]]
- [[late-interaction-model]]
- [[retrieval-robustness]]

## Entities Extracted

- [[shicheng-xu]]
- [[liang-pang]]
- [[mo-yu]]
- [[fandong-meng]]
- [[huawei-shen]]
- [[xueqi-cheng]]
- [[jie-zhou-tencent]]
- [[chinese-academy-of-sciences]]
- [[tencent]]
- [[llama-2]]
- [[colbertv2]]
- [[wikipedia]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
