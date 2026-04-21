---
type: source
subtype: paper
title: "Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation"
slug: ning-2024-skeletonofthought-2307-15337
date: 2026-04-20
language: en
tags: [llm, prompting, inference-efficiency, parallel-decoding, routing]
processed: true

raw_file: raw/papers/ning-2024-skeletonofthought-2307-15337/paper.pdf
raw_md: raw/papers/ning-2024-skeletonofthought-2307-15337/paper.md
bibtex_file: raw/papers/ning-2024-skeletonofthought-2307-15337/paper.bib
possibly_outdated: false

authors:
  - Xuefei Ning
  - Zinan Lin
  - Zixuan Zhou
  - Zifu Wang
  - Huazhong Yang
  - Yu Wang
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2307.15337
doi:
url: http://arxiv.org/abs/2307.15337
citation_key: ning2024skeletonofthought
paper_type: method

read_status: unread

domain: llm
---

## Summary

Skeleton-of-Thought (SoT) is a prompting-based inference method that reduces end-to-end latency by decomposing a long answer into a short skeleton and then expanding each point in parallel. Instead of changing the model, hardware, or serving stack, the paper exploits the instruction-following ability of large language models to make their outputs structurally parallelizable. The method works for both API-only models through parallel API calls and local open-source models through batched decoding. Across 12 LLMs, SoT reaches up to `2.39x` speed-up while often preserving, and sometimes improving, answer quality. The paper further introduces SoT-R, which uses either GPT-4 prompting or a fine-tuned RoBERTa router to trigger SoT only on questions whose answers can be expanded as independent points.

## Problem & Motivation

The paper targets high inference latency in large language models, arguing that sequential autoregressive decoding is a third major bottleneck beyond model size and quadratic attention cost. The authors observe that many real-world questions admit a planned answer structure: humans often draft an outline first, then elaborate each part. SoT operationalizes that intuition by turning one long sequential response into several shorter, independent continuations that can be generated in parallel. The motivation is pragmatic: reduce user-visible latency for long-form assistant responses without retraining, architecture changes, or specialized hardware, while preserving answer quality on categories such as knowledge, generic, common-sense, roleplay, and counterfactual questions.

## Method

- **Two-stage SoT pipeline**: given a question `q`, the model first receives a skeleton prompt `T^s(q)` and returns a skeleton response `R^s` with `B` points; each point `R_b^s` is then expanded independently with a point-expanding prompt `T^{pe}(q, R^s, b, R_b^s)`, and the final answer is the concatenation of `{R_b^{pe}}_{b=1}^B`.
- **Skeleton prompt design**: the prompt asks for `3-10` numbered points, each only `3-5` words, and supplies a partial answer `1.` to bias formatting. For non-GPT-4 models, the authors add two demonstrations; point extraction is handled with a simple regular expression over numbered list items.
- **Point expansion constraints**: each point-expanding prompt tells the model to continue only one point in `1-2` sentences and avoid writing other points. For Claude and GPT-4, the authors remove the phrase "very shortly" because those models already follow the brevity instruction well.
- **Parallel execution modes**: API-based models run multiple point-expanding requests concurrently; local open-source models left-pad point-expanding prompts and decode them as a batch, exploiting the fact that decoding is weight-I/O bound and weakly sensitive to batch size.
- **Latency formulation**: the profiling-based latency estimator is `T(l_i, l_o, B) = t~_B^P(l_i) + sum_{k=l_i}^{l_i+l_o-1} t_B^D(k)`, with SoT stage latencies `L^s(l_i^s, l_o^s) = T(l_i^s, l_o^s, 1)` and `L^{pe}(l_i^{pe}, l_o^{pe}, B) = T(l_i^{pe}, l_o^{pe}, B)`.
- **Efficiency rationale**: on an NVIDIA A100, the decoding phase dominates latency because weights must be reloaded for each generated token; Table 5 reports LLaMA-7B prefill/decode latency of `40 ms / 2735 ms` and GPU performance of `43 / 0.31 TFLOPS`, motivating batch-parallel point expansion.
- **Router extension (SoT-R)**: a prompting router asks GPT-4 to choose among classes `A/B/C`, where only `A` triggers SoT. A trained router formulates routing as sequence classification with `roberta-base` (`120M` parameters), `AdamW`, weight decay `0.01`, learning rate warmup over the first `1%` of steps to `5e-5`, linear decay, `2` epochs, batch size `32`, max length `512`, Tversky loss with `alpha = 0.7`, `beta = 0.3`, and label smoothing `epsilon = 0.2`.
- **Implementation details**: open-source experiments rely on FastChat conversation templates; for multi-round chat, only the original question and final aggregated answer are kept in history so SoT does not inflate future prefilling cost.

## Key Results

- In the motivating example, SoT reduces latency from `22 s` to `12 s` on Claude (`1.83x`) and from `43 s` to `16 s` on Vicuna-33B V1.3 (`2.69x`) on one NVIDIA A100.
- Across `12` evaluated models, SoT achieves more than `2x` average speed-up on `8/12` models, with a maximum reported average speed-up of `2.39x`.
- On the five categories where answer quality remains strong (`knowledge`, `generic`, `common-sense`, `roleplay`, `counterfactual`), SoT improves speed by `1.89x-2.33x` in the main evaluation; actual batch tests on open-source models report `2.15x-2.50x` on those categories.
- Overall answer quality remains competitive: SoT is not worse than normal generation in around `60%` of cases, with strict win rates of `45.8%` under FastChat and `29.5%` under LLMZoo general metrics.
- SoT tends to improve diversity and relevance, but degrades coherence and immersion on average according to LLMZoo.
- The trained router is cheap to run: average router latency on Vicuna-80 is `0.04 s` versus `0.65 s` for the GPT-4 prompting router, and the full fine-tuning finishes in about `2 minutes` on one A100.
- Human suitability annotation marks `37/80` Vicuna-80 questions, `58/218` WizardLM questions, and `371/1030` LIMA training examples as suitable for SoT.
- Peak memory overhead of SoT-R stays below `1.11x` across evaluated Vicuna-80 models and categories.

## Limitations

- SoT assumes answer points can be expanded independently, so it performs poorly on tasks that require step-by-step dependencies, especially math, coding, fermi, and some writing questions.
- The method increases prompt and prefilling overhead substantially: for suitable Vicuna-80 questions, Stage-1 plus Stage-2 prefilling is `60.46x` normal for ChatGPT-3.5, `85.79x` for Claude, and `89.20x` for GPT-4 in token count.
- The current formulation ignores dependencies between points; the paper explicitly proposes graph-structured extensions as future work.
- Quality evaluation depends on LLM judges and prompt choices, and the authors did not run human evaluation because SoT answers have recognizable formatting that could bias raters.
- Latency gains are most attractive when concurrent load is unsaturated; under saturated serving workloads, throughput and overhead trade-offs become more important.

## Concepts Extracted

- [[skeleton-of-thought]]
- [[large-language-model]]
- [[parallel-decoding]]
- [[autoregressive-decoding]]
- [[batched-decoding]]
- [[prompt-engineering]]
- [[inference-latency]]
- [[routing-mechanism]]
- [[sequence-classification]]
- [[data-centric-optimization]]
- [[llm-as-a-judge]]
- [[key-value-cache]]

## Entities Extracted

- [[xuefei-ning]]
- [[zinan-lin]]
- [[zixuan-zhou]]
- [[zifu-wang]]
- [[huazhong-yang]]
- [[yu-wang]]
- [[tsinghua-university]]
- [[microsoft-research]]
- [[ku-leuven]]
- [[infinigence-ai]]
- [[gpt-4]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
