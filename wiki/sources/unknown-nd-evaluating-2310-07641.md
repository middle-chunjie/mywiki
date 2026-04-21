---
type: source
subtype: paper
title: Evaluating Large Language Models at Evaluating Instruction Following
slug: unknown-nd-evaluating-2310-07641
date: 2026-04-20
language: en
tags: [llm-evaluation, instruction-following, meta-evaluation, prompting, benchmark]
processed: true

raw_file: raw/papers/unknown-nd-evaluating-2310-07641/paper.pdf
raw_md: raw/papers/unknown-nd-evaluating-2310-07641/paper.md
bibtex_file: raw/papers/unknown-nd-evaluating-2310-07641/paper.bib
possibly_outdated: true

authors:
  - Zhiyuan Zeng
  - Jiatong Yu
  - Tianyu Gao
  - Yu Meng
  - Tanya Goyal
  - Danqi Chen
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.07641
doi:
url: https://arxiv.org/abs/2310.07641
citation_key: unknownndevaluating
paper_type: benchmark
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper introduces LLMBar, a 419-instance meta-evaluation benchmark for testing whether LLM-based evaluators can correctly prefer outputs that better follow instructions instead of outputs that merely sound more polished. The benchmark combines a filtered Natural split with a 319-instance Adversarial split designed so the wrong answer often has attractive superficial qualities such as better tone or formatting. Across GPT-4, ChatGPT, PaLM2, LLaMA-2-70B-Chat, and Falcon-180B-Chat evaluators, the study finds large capability gaps and substantial failure on adversarial instruction-following judgments. Even the best GPT-4-based setup remains below expert humans, while prompting strategies such as explicit rules, self-generated metrics, reference outputs, and swap-based synthesis materially improve robustness.

## Problem & Motivation

Prior LLM-as-evaluator meta-evaluation sets often rely on noisy or subjective human preferences from open-ended comparisons, making it hard to tell whether a strong evaluator is actually detecting objective qualities such as instruction following. The paper argues that this is especially problematic because models and evaluators can be biased toward superficial qualities like verbosity, polish, or engaging tone. It therefore constructs a benchmark where one output is objectively better with respect to the instruction and uses that benchmark to measure how reliably different evaluator models and prompts recover the correct preference.

## Method

- **Benchmark definition**: each instance is a tuple `` `(I, O_1, O_2, p)` `` where `I` is the instruction, `O_1` and `O_2` are candidate outputs, and `p in {1, 2}` marks the objectively preferred output.
- **Dataset composition**: LLMBar contains `419` instances total, with `100` in the Natural set and `319` in the Adversarial set; the Adversarial split further contains `134` Neighbor, `92` GPTInst, `47` GPTOut, and `46` Manual instances.
- **Natural-set construction**: candidate pairs are sampled from AlpacaFarm and LLMEval2, then heavily filtered and manually modified so each retained pair has an objective preference on instruction following rather than a subjective style preference.
- **Neighbor construction**: given instruction `I`, retrieve a similar but sufficiently different `I'` from the same source dataset using `` `I' = arg max_{I'' in D, sim(I, I'') < epsilon} sim(I, I'')` ``, where `sim(.)` is cosine similarity from INSTRUCTOR; then pair an `I`-following output `O_1` with an `I'`-conditioned but superficially stronger output `O_2`.
- **Other adversarial subsets**: GPTInst asks GPT-4 to generate a related instruction `I'`; GPTOut asks GPT-4 to generate a superficially good but unhelpful output; Manual adds author-written adversarial examples inspired by the automatic pipelines.
- **Adversarial filtering**: each candidate is scored by `4` ChatGPT-based evaluators under both orders `(O_1, O_2)` and `(O_2, O_1)`, yielding `8` preference labels; candidates are kept only when the majority does not align with the expected gold label, then manually checked for objectivity and correctness.
- **Evaluator setup**: the study evaluates GPT-4 (`gpt-4-0613`), ChatGPT (`gpt-3.5-turbo-0613`), PaLM2 (`text-bison-001`), LLaMA-2-70B-Chat, and Falcon-180B-Chat with temperature `0` for proprietary APIs and greedy decoding for open models.
- **Prompting strategies**: baselines include Vanilla, CoT, Reference, and ChatEval; new strategies include Rules, self-generated Metrics with at most `3` instruction-specific questions, and Swap/Swap+CoT, which query both presentation orders and synthesize contradictory judgments.
- **Evaluation protocol**: every pair is judged twice with swapped output order; the paper reports accuracy and positional agreement, and separately samples `80` benchmark items for expert-human agreement measurement.

## Key Results

- LLMBar achieves high expert agreement: `94%` overall on a sampled `80` instances, with `90%` on Natural and `95%` on Adversarial, substantially above previously reported human agreement for FairEval (`71.7%`), LLMEval2 (`80%`), and MT-Bench (`63%`).
- GPT-4-based evaluators are strongest, but still imperfect: Vanilla reaches `73.1%` average accuracy on the Adversarial split, while `Metrics+Reference*` raises this to `82.8%`; expert humans are still at `95%`.
- Prompting changes matter materially: for GPT-4, overall average accuracy increases from `77.2%` with Vanilla to `85.4%` with `Metrics+Reference*`, while `Swap+CoT*` pushes overall positional agreement to `97.5%`.
- ChatGPT-based evaluators are fragile on adversarial cases: `Metrics+Reference*` reaches only `38.9%` average adversarial accuracy, and several settings are near or below chance on hard subsets.
- PaLM2 is materially stronger than ChatGPT, LLaMA-2, and Falcon on this benchmark: `Metrics+Reference*` obtains `69.5%` adversarial accuracy and `72.9%` overall average accuracy.
- Reward models also underperform: AlpacaFarm `reward-model-sim` and `reward-model-human` reach only `31.1%` and `38.0%` average adversarial accuracy, with `38.4%` and `44.4%` overall average accuracy respectively.
- Falcon-180B-Chat exhibits severe positional bias; with `CoT*`, its overall positional agreement drops to just `12.0%`, motivating the swap-based correction strategy.

## Limitations

- The benchmark is manually curated and relatively small at `419` instances, so it may not fully reflect the diversity or frequency distribution of real instruction-following failures.
- LLMBar focuses on single-turn instruction following, not multi-turn dialogue, factual correctness, toxicity, or other alignment properties the paper itself identifies as important.
- The GPTOut subset may advantage GPT-4-based evaluators because the adversarial outputs are produced by GPT-4.
- Adversarial filtering is performed against ChatGPT-based evaluators, which likely makes the benchmark especially challenging for ChatGPT-family judges.
- Some prompt interventions trade off different desiderata: CoT often hurts adversarial accuracy even when swap-based methods improve positional agreement.

## Concepts Extracted

- [[instruction-following]]
- [[large-language-model]]
- [[llm-evaluator]]
- [[meta-evaluation-benchmark]]
- [[prompt-engineering]]
- [[adversarial-filtering]]
- [[position-bias]]
- [[chain-of-thought-prompting]]
- [[reward-model]]
- [[reinforcement-learning-from-human-feedback]]

## Entities Extracted

- [[zhiyuan-zeng-tsinghua]]
- [[jiatong-yu]]
- [[tianyu-gao]]
- [[yu-meng-uiuc]]
- [[tanya-goyal]]
- [[danqi-chen]]
- [[tsinghua-university]]
- [[princeton-university]]
- [[university-of-illinois-urbana-champaign]]
- [[llmbar]]
- [[gpt-4]]
- [[chatgpt]]
- [[llama-2-70b-chat]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
