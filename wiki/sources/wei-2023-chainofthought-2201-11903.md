---
type: source
subtype: paper
title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
slug: wei-2023-chainofthought-2201-11903
date: 2026-04-20
language: en
tags: [llm, prompting, reasoning, chain-of-thought, few-shot]
processed: true

raw_file: raw/papers/wei-2023-chainofthought-2201-11903/paper.pdf
raw_md: raw/papers/wei-2023-chainofthought-2201-11903/paper.md
bibtex_file: raw/papers/wei-2023-chainofthought-2201-11903/paper.bib
possibly_outdated: true

authors:
  - Jason Wei
  - Xuezhi Wang
  - Dale Schuurmans
  - Maarten Bosma
  - Brian Ichter
  - Fei Xia
  - Ed H. Chi
  - Quoc V. Le
  - Denny Zhou
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2201.11903
doi:
url: http://arxiv.org/abs/2201.11903
citation_key: wei2023chainofthought
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces [[chain-of-thought-prompting]] as a few-shot prompting format that replaces plain input-output exemplars with triples `⟨input, chain of thought, output⟩`, letting sufficiently large [[large-language-model]]s externalize intermediate reasoning before answering. Across arithmetic, commonsense, and symbolic tasks, the method yields large gains over standard prompting without finetuning, with the effect appearing only at sufficiently large scale. The strongest result is on GSM8K, where PaLM `540B` with `8` chain-of-thought exemplars reaches `56.9%` accuracy and `58.6%` with an external calculator, surpassing prior finetuned systems. The paper's central claim is that multi-step reasoning can be elicited by prompt design alone, but only once model scale is high enough for coherent reasoning traces to emerge.

## Problem & Motivation

Standard few-shot prompting often underperforms on tasks that require multi-step reasoning, even when model scale is increased. Prior rationale-based approaches typically require supervised rationale data or finetuning, which is expensive and task-specific. The paper asks whether one can unlock reasoning behavior in off-the-shelf language models by changing the prompt format alone: instead of showing only question-answer pairs, show intermediate natural-language reasoning steps that demonstrate how an answer is derived. This would preserve the flexibility of [[in-context-learning]] while avoiding the annotation and retraining costs of rationale-supervised pipelines.

## Method

- **Prompt format**: replace standard few-shot exemplars `(x, y)` with triples `⟨x, c, y⟩`, where `c` is a natural-language reasoning trace. Operationally, the test prompt is a sequence `{(x_i, c_i, y_i)}_{i=1}^k` followed by a new input `x_*`, and the model is expected to continue with a chain of thought plus final answer.
- **Core intervention**: the method is purely prompt-based. No gradient updates, finetuning, or task-specific retraining are applied to the evaluated models.
- **Arithmetic setup**: use `k = 8` manually written chain-of-thought exemplars for GSM8K, SVAMP, ASDiv, and MAWPS; use `k = 4` exemplars for AQuA because it is multiple-choice.
- **Commonsense setup**: manually compose few-shot chain-of-thought exemplars for CSQA, StrategyQA, Date Understanding, Sports Understanding, and SayCan; BIG-bench tasks use the first `10` evaluation examples as exemplars because they lack training sets.
- **Symbolic setup**: create few-shot chain-of-thought prompts for two toy tasks, last-letter concatenation and coin flip, then evaluate both in-domain and OOD length generalization by increasing the number of steps beyond those seen in exemplars.
- **Models evaluated**: GPT-family models from `350M` to `175B`, LaMDA from `422M` to `137B`, PaLM at `8B`, `62B`, and `540B`, plus UL2 `20B` and Codex (`code-davinci-002`).
- **Decoding**: results are primarily obtained with greedy decoding; LaMDA results are averaged over `5` random seeds with different exemplar orders, while most other runs use a single exemplar order to limit compute.
- **Ablations**: compare full chain-of-thought prompting against `equation-only`, `variable-compute-only`, and `reasoning-after-answer` variants to isolate whether improvements come from natural-language intermediate reasoning rather than simply extra tokens or answer-adjacent explanations.
- **Post-hoc calculator analysis**: for arithmetic tasks only, an external calculator is optionally applied to equations extracted from model-generated reasoning chains to quantify how much residual error comes from arithmetic execution rather than task understanding.

## Key Results

- On GSM8K, PaLM `540B` improves from `17.9%` with standard prompting to `56.9%` with chain-of-thought prompting, and to `58.6%` with an external calculator; this exceeds the prior best finetuned result of `55.0%`.
- On other arithmetic tasks with PaLM `540B`, chain-of-thought prompting improves SVAMP from `69.4%` to `79.0%`, AQuA from `25.2%` to `35.8%`, and MAWPS from `79.2%` to `93.3%`; ASDiv also rises from `72.1%` to `73.9%`.
- The gains are scale-dependent: for PaLM `8B`, GSM8K drops from `4.9%` to `4.1%`, while for PaLM `62B` it jumps from `9.6%` to `29.9%`, and for `540B` to `56.9%`, supporting the paper's [[emergent-ability]] claim.
- On commonsense reasoning, PaLM `540B` with chain-of-thought prompting reaches `77.8%` on [[strategyqa]] versus `68.6%` standard prompting and `95.4%` on Sports Understanding versus an `84%` unaided sports enthusiast baseline.
- On symbolic reasoning, PaLM `540B` reaches `99.4%` on in-domain two-word last-letter concatenation, `94.8%` on OOD four-word concatenation, and `100.0%` on in-domain coin flip, while standard prompting largely fails on OOD settings (`0.2%` for OOD concatenation and `49.3%` for OOD coin flip).
- Manual analysis of `50` correct GSM8K outputs from LaMDA `137B` found `49/50` chains of thought logically and mathematically correct; among `50` incorrect outputs, `46%` were almost correct aside from minor mistakes and `54%` had major semantic or coherence errors.

## Limitations

- The method is not effective at small scale; the main benefits appear only around `~100B+` parameters, making deployment costly.
- Few-shot chain-of-thought exemplars are manually authored, which is cheap for prompting but would become expensive if scaled to supervised finetuning.
- Generated reasoning traces are not guaranteed to be faithful or correct; models can still reach correct answers through flawed reasoning or produce plausible but invalid chains.
- Performance improvements are uneven across tasks: gains are small on some easier tasks and minimal on CSQA, so chain-of-thought prompting is not a universal fix.
- The paper studies prompting only and does not resolve whether models are truly reasoning internally or merely exploiting learned textual patterns.

## Concepts Extracted

- [[chain-of-thought-prompting]]
- [[few-shot-prompting]]
- [[in-context-learning]]
- [[large-language-model]]
- [[emergent-ability]]
- [[arithmetic-reasoning]]
- [[commonsense-reasoning]]
- [[symbolic-reasoning]]
- [[prompt-engineering]]

## Entities Extracted

- [[jason-wei]]
- [[xuezhi-wang]]
- [[dale-schuurmans]]
- [[maarten-bosma]]
- [[brian-ichter]]
- [[fei-xia]]
- [[ed-chi]]
- [[quoc-le]]
- [[denny-zhou]]
- [[google-research]]
- [[google-brain]]
- [[gsm8k]]
- [[strategyqa]]
- [[big-bench]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
