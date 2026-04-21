---
type: source
subtype: paper
title: Making Retrieval-Augmented Language Models Robust to Irrelevant Context
slug: unknown-nd-making-2310-01558
date: 2026-04-20
language: en
tags: [retrieval, llm, question-answering, robustness, nli]
processed: true

raw_file: raw/papers/unknown-nd-making-2310-01558/paper.pdf
raw_md: raw/papers/unknown-nd-making-2310-01558/paper.md
bibtex_file: raw/papers/unknown-nd-making-2310-01558/paper.bib
possibly_outdated: true

authors:
  - Ori Yoran
  - Tomer Wolfson
  - Ori Ram
  - Jonathan Berant
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.01558
doi:
url: https://arxiv.org/abs/2310.01558
citation_key: unknownndmaking
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper studies retrieval-augmented language models under noisy retrieval and asks for a stronger robustness criterion: retrieval should help when evidence is relevant and should not degrade answers when evidence is irrelevant. Across five open-domain QA benchmarks, the authors show that top-1 retrieval can improve single-hop and explicit multi-hop tasks but can hurt implicit reasoning, while random passages are highly distracting. They test two remedies: an NLI-based back-off that rejects unsupported retrieved evidence, and a retrieval-robust fine-tuning recipe that mixes relevant and irrelevant contexts during training. With automatically generated decompositions and only `1000` single-hop or `500` multi-hop questions, the fine-tuned model preserves robustness under noisy retrieval while improving accuracy when retrieval is useful.

## Problem & Motivation

Retrieval-augmented LMs are attractive for open-domain QA because they can use external evidence instead of relying only on parametric memory, but real retrievers return noisy or off-target passages. The paper shows that irrelevant context can distract the model, corrupt intermediate decomposition steps, and trigger wrong answers even when the base model would answer correctly without retrieval. The target behavior is therefore retrieval robustness: use evidence when it helps, but ignore it when it is irrelevant, especially in multi-hop settings where early mistakes cascade.

## Method

- **Formalization**: standard LM decoding is `p_LM = ∏_{i=1}^{n} p_θ(x_i | x_{<i})`, while retrieval-augmented decoding is `p_RALM = ∏_{i=1}^{n} p_θ(x_i | R_C(x_{<i}); x_{<i})`, where `R_C(.)` retrieves evidence from corpus `C`.
- **Task setting**: the paper studies in-context RALMs for open-domain QA. Single-hop questions directly predict the answer from `[r_q; q]`; multi-hop questions interleave Self-Ask-style intermediate questions, intermediate answers, and retrieval at each step.
- **NLI back-off baseline**: run the model once with retrieval and once without retrieval; accept the retrieval-augmented output only if an NLI model judges every generated question-answer pair to be entailed by the retrieved evidence.
- **NLI model details**: the entailment model is BART-Large trained on [[mnli]] with `407M` parameters; a generated pair is accepted when entailment probability is `>= 0.5`, otherwise the system falls back to the no-retrieval answer.
- **Single-hop data generation**: create training examples `D = {([r_q; q], a)}` from gold QA pairs. Relevant context is top-1 retrieval; irrelevant context is either a low-ranked result or a random top-1 passage from another question.
- **Multi-hop data generation**: create examples `D = {([r_x; x], y)}` where `x` is the prior decomposition prefix, `r_x` is retrieved context for all steps, and `y` is the next correct intermediate question, intermediate answer, or final answer.
- **Decomposition generation**: use GPT-3 `code-davinci-002` with the no-retrieval Self-Ask prompt to produce decompositions. For implicit datasets, sample `5` decompositions per question (`1` greedy, `4` at temperature `0.7`) and keep the greedy one only when all samples reach the same correct answer.
- **Training setup**: fine-tune [[llama-2-13b]] with [[parameter-efficient-fine-tuning]] via QLoRA for `5` epochs, learning rate `2e-4`, linear schedule, on a single GPU; each model trains in at most `3.5` hours.
- **Training mixture**: the robust model samples top-1, low-ranked, or random evidence with equal probability at each step, so the model learns both to exploit good retrieval and to ignore bad retrieval.
- **Evaluation setup**: use Google Search via SerpAPI and [[colbertv2]] over `2018` Wikipedia. Benchmarks are [[natural-questions]], 2WikiMQA, [[bamboogle]], [[strategyqa]], and Fermi; evaluation uses `500` dev examples per dataset except Fermi, which uses all `286` official problems.

## Key Results

- Prompted [[llama-2-13b]] with top-1 Google retrieval improves NQ from `29.6` to `41.0`, 2WikiMQA from `32.0` to `56.0`, and Bamboogle from `47.4` to `68.0`, but hurts StrategyQA from `65.6` to `62.1` and Fermi from `27.7` to `27.4`.
- The robust fine-tuned model SA-RetRobust beats SA-NoRet by `+2.7`, `+2.4`, and `+2.4` average points under top-1, low-ranked, and random Google retrieval, respectively; against SA-Ret@1 the gains are `+0.2`, `+0.4`, and `+3.2`.
- Under low-ranked Google retrieval, SA-RetRobust reaches `37.9` on NQ and `45.0` on 2WikiMQA, which is `+3.8` and `+2.8` above the models trained and evaluated without retrieval (`34.1` and `42.2`), while staying within `1.2` points of the no-retrieval baseline on StrategyQA (`68.6` vs `69.8`).
- Under random Google retrieval, SA-RetRobust remains robust: NQ improves from `31.1` to `33.8`, 2WikiMQA from `40.4` to `41.6`, and StrategyQA from `66.6` to `69.9`, whereas prompted retrieval models often lose more than `10` points.
- On NQ, prompted [[llama-2-70b]] reaches `38.4` without retrieval and at best `42.4` with top-1 retrieval, while the trained `13B` SA-RetRobust model reaches `45.7`.
- NLI analysis shows many helpful retrieval cases have low entailment probability: for NQ top-1 retrieval, `32.6%` of examples are low-entailment yet still gain `+0.11`; for 2WikiMQA top-1, `83.0%` are low-entailment with `+0.12`.
- Error analysis shows that when retrieval causes failure, the "Wrong" category dominates `65%`-`85%` of cases; for NQ with low-ranked passages, the wrong entity appears in the retrieved passage `77%` of the time.

## Limitations

- The NLI filter is conservative: it prevents many harmful cases but also discards relevant evidence that would have improved accuracy.
- The training pipeline depends on automatically generated decompositions; manual inspection still finds only about `90%` decomposition correctness on StrategyQA, though 2WikiMQA exceeds `95%`.
- Finetuning is demonstrated in a low-data regime (`1000` single-hop questions or `500` multi-hop questions) and only on three training datasets, so broader generalization is not established.
- Fermi and Bamboogle are evaluation-only because they have fewer than `300` examples, limiting evidence for robustness under task transfer.
- The method assumes access to a strong retriever and, for data generation, a strong teacher LLM plus gold QA supervision.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[retrieval-robustness]]
- [[natural-language-inference]]
- [[open-domain-question-answering]]
- [[question-decomposition]]
- [[in-context-learning]]
- [[few-shot-prompting]]
- [[multi-hop-reasoning]]
- [[self-consistency]]
- [[parameter-efficient-fine-tuning]]

## Entities Extracted

- [[ori-yoran]]
- [[tomer-wolfson]]
- [[ori-ram-tau]]
- [[jonathan-berant]]
- [[tel-aviv-university]]
- [[allen-institute-for-ai]]
- [[llama-2-13b]]
- [[colbertv2]]
- [[natural-questions]]
- [[strategyqa]]
- [[bamboogle]]
- [[mnli]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
