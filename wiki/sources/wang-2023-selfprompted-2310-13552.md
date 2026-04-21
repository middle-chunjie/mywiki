---
type: source
subtype: paper
title: Self-prompted Chain-of-Thought on Large Language Models for Open-domain Multi-hop Reasoning
slug: wang-2023-selfprompted-2310-13552
date: 2026-04-20
language: en
tags: [llm, reasoning, multi-hop-qa, prompting, open-domain-qa]
processed: true

raw_file: raw/papers/wang-2023-selfprompted-2310-13552/paper.pdf
raw_md: raw/papers/wang-2023-selfprompted-2310-13552/paper.md
bibtex_file: raw/papers/wang-2023-selfprompted-2310-13552/paper.bib
possibly_outdated: true

authors:
  - Jinyuan Wang
  - Junlong Li
  - Hai Zhao
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.13552
doi:
url: http://arxiv.org/abs/2310.13552
citation_key: wang2023selfprompted
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper defines open-domain multi-hop reasoning (ODMR) as answering multi-hop questions without a provided supporting corpus, and proposes Self-prompted Chain-of-Thought (SP-CoT) to generate high-quality reasoning demonstrations automatically. SP-CoT has three parts: an LLM-only pipeline that generates 2-hop QA quadruplets and composes them into 2-4 hop ODMR datasets, an adaptive in-context demonstration sampler based on Sentence-BERT embeddings plus k-means clustering, and an inference prompt that reuses sampled multi-step explanations as chain-of-thought exemplars. Evaluated on ComplexWebQuestions, HotpotQA, 2WikiMultiHopQA, and MuSiQue with ChatGPT, InstructGPT, and several 13B instruction-tuned models, SP-CoT beats prior LLM-only baselines and often yields large gains over zero-shot prompting, especially on smaller models.

## Problem & Motivation

Existing open-domain QA work mostly targets single-hop questions, while multi-hop QA benchmarks often assume a fixed retrieved article set that is unavailable in real deployment. The paper formalizes this gap as ODMR: multi-hop question answering in an open-domain setting with explicit reasoning steps but no candidate corpus. Prior CoT prompting methods also leave a gap. Manual CoTs are expensive, static, and hard to diversify; automated approaches such as Zero-shot-CoT and Auto-CoT are more scalable but do not sufficiently control the quality of generated intermediate reasoning. The authors therefore seek an LLM-only pipeline that can generate diverse, higher-quality multi-hop CoT demonstrations and then reuse them adaptively during inference.

## Method

- **Task framing**: define [[open-domain-multi-hop-reasoning]] as open-domain multi-step QA with explicit rationales, positioned between [[open-domain-question-answering]] and [[multihop-question-answering]].
- **Stage 1, self-generation of 2-hop QA quadruplets**: manually specify `29` topics, ask the LLM to produce keywords `k_1`, generate a Wiki-style passage `p_1`, and derive `(q_1, a_1, e_1)` where the answer is a named entity and the explanation contains the answer (`a_1 in e_1`).
- **Answer extraction and validation**: use spaCy and NLTK to extract named-entity candidates from `p_1`; require the model to answer the generated question again and accept the pair only when the checked answer `a'_1 = a_1`.
- **Second hop construction**: set `k_2 = a_1`, filter infeasible entity types (`QUANTITY`, `ORDINAL`, `CARDINAL`, `PERCENT`, `MONEY`, `DATE`, `TIME`), generate `p_2`, then build `(q_2, a_2, e_2)` under constraints `a_1 in q_2`, `a_2 != a_1`, and no cyclic reuse of blocked entities.
- **Stage 2, composition into ODMR**: connect 2-hop pairs using the composability rule that an intermediate answer is a named entity and appears in the next-hop question; compose `2-4` hop chains over `6` reasoning graph types while enforcing shortcut-avoidance and acyclicity constraints.
- **Duplication and binary-question control**: filter overly similar chains using a preset duplication degree; for each reasoning type, sample `10%` chains for positive binary reformulation and `10%` for negative reformulation.
- **Multi-hop question generation**: recursively replace an intermediate answer in the next question with `[q_i]`, then prompt the LLM with `4` manual demonstrations to rewrite the result into a natural multi-hop question.
- **Stage 3, adaptive demonstration sampling**: embed questions with [[sentence-bert]], cluster them with [[k-means-clustering]], and for a test question retrieve the most similar question from each cluster to form `n` in-context demonstrations.
- **Reasoning-chain prompt assembly**: concatenate hop explanations as `Step 1: ... Step 2: ...` and use them as [[chain-of-thought]] exemplars for [[in-context-learning]]; the default number of demonstrations is `8`.
- **Inference setup**: main RQ1 experiments use [[chatgpt]] (`gpt-3.5-turbo-0301`), while transfer experiments use [[instructgpt]], [[alpaca-13b]], [[vicuna-13b]], and [[wizardlm-13b]]; temperature is usually `0`, except generation-heavy substeps where it is set to `1.0`.

## Key Results

- On ChatGPT, SP-CoT reaches average `28.8` EM / `36.0` F1 across [[musique]], [[hotpotqa]], [[2wikimultihopqa]], and [[complexwebquestions]], beating Auto-CoT (`22.6` / `29.6`) by `+6.2` EM and `+6.4` F1 on average.
- Dataset-wise on ChatGPT, SP-CoT reports `14.5/22.6` on MuSiQue, `33.2/42.9` on HotpotQA, `30.1/34.7` on 2WikiMultiHopQA, and `37.5/43.6` on ComplexWebQuestions (EM/F1).
- Relative to GENREAD, SP-CoT improves the average score from `26.5/33.2` to `28.8/36.0`; the largest gap is on MuSiQue, where EM rises from `8.6` to `14.5`.
- On [[instructgpt]], mean EM improves from `22.9` zero-shot to `33.1` with SP-CoT (`+10.2`).
- On smaller `13B` models, SP-CoT nearly doubles zero-shot performance in several cases: [[alpaca-13b]] `8.9 -> 17.4`, [[vicuna-13b]] `12.6 -> 21.5`, and [[wizardlm-13b]] `10.9 -> 22.5` mean EM.
- Demonstration-count analysis shows gains from `2` to `8` in-context examples, while `10` demonstrations add no further improvement; the paper therefore fixes the default at `8`.
- For intermediate reasoning quality on MuSiQue, SP-CoT recovers roughly `50%` of intermediate answers and is judged by GPT-4 to be clearer, more concise, more comprehensible, and more direct than Zero-shot-CoT and Auto-CoT.

## Limitations

- The method depends strongly on the base model's instruction-following ability; the paper reports that GPT-NeoX could not be improved by Zero-shot-CoT, Auto-CoT, or SP-CoT.
- The generated ODMR datasets may still contain incorrect QA pairs or explanations because the pipeline relies on LLM self-generation; double-checking reduces but does not eliminate these errors.
- Manual-CoT remains competitive and sometimes stronger, especially when high-quality cherry-picked demonstrations are available, so SP-CoT is not a strict upper bound on prompting performance.
- The approach is LLM-only and closed-book at inference time, so its ceiling is still constrained by the model's parametric knowledge rather than explicit retrieval.

## Concepts Extracted

- [[open-domain-multi-hop-reasoning]]
- [[multihop-question-answering]]
- [[open-domain-question-answering]]
- [[large-language-model]]
- [[chain-of-thought]]
- [[self-prompting]]
- [[in-context-learning]]
- [[demonstration-selection]]
- [[zero-shot-cot]]
- [[auto-cot]]

## Entities Extracted

- [[jinyuan-wang]]
- [[junlong-li]]
- [[hai-zhao]]
- [[shanghai-jiao-tong-university]]
- [[chatgpt]]
- [[instructgpt]]
- [[alpaca-13b]]
- [[vicuna-13b]]
- [[wizardlm-13b]]
- [[hotpotqa]]
- [[2wikimultihopqa]]
- [[complexwebquestions]]
- [[musique]]
- [[sentence-bert]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
