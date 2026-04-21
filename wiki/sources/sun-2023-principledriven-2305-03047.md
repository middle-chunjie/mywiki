---
type: source
subtype: paper
title: Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision
slug: sun-2023-principledriven-2305-03047
date: 2026-04-20
language: en
tags: [llm-alignment, self-alignment, synthetic-data, ai-assistant, safety]
processed: true

raw_file: raw/papers/sun-2023-principledriven-2305-03047/paper.pdf
raw_md: raw/papers/sun-2023-principledriven-2305-03047/paper.md
bibtex_file: raw/papers/sun-2023-principledriven-2305-03047/paper.bib
possibly_outdated: true

authors:
  - Zhiqing Sun
  - Yikang Shen
  - Qinhong Zhou
  - Hongxin Zhang
  - Zhenfang Chen
  - David Cox
  - Yiming Yang
  - Chuang Gan
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2305.03047
doi:
url: https://arxiv.org/abs/2305.03047
citation_key: sun2023principledriven
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces SELF-ALIGN, a four-stage pipeline for aligning a base LLM into an assistant without relying on large human-labeled SFT or RLHF corpora. Starting from LLaMA-65B, the method combines topic-guided self-instruct data generation, a small hand-written set of 16 behavioral principles plus 5 in-context exemplars, principle engraving via LoRA fine-tuning on self-generated responses, and a final verbose-cloning stage based on context distillation. The resulting assistant, Dromedary, uses fewer than 300 lines of human supervision yet reports strong gains over LLaMA and Alpaca on TruthfulQA and HHH Eval. The paper is notable because it frames alignment as rule-conditioned self-bootstrapping from scratch, while also honestly documenting residual hallucination, context-length limits, and a "verbose tax" tradeoff.

## Problem & Motivation

Recent assistant LLMs were largely aligned through supervised fine-tuning and reinforcement learning from human feedback, both of which require substantial annotation budgets and inherit quality, diversity, and bias issues from human or teacher supervision. The paper asks whether a strong base model can be aligned from scratch with dramatically less human input by using a compact rule set and the model's own generative ability. The motivation is to reduce supervision cost, avoid dependence on proprietary teacher models such as ChatGPT, and explore an alternative "first align, then follow" paradigm for open-source AI assistants.

## Method

- **Base setup**: start from `LLaMA-65B` and define a four-stage pipeline: topic-guided self-instruct, principle-driven self-alignment, principle engraving, and verbose cloning.
- **Topic-Guided Red-Teaming Self-Instruct**: begin with `175` manually written self-instruct seeds plus `20` adversarial instruction types, for `195` human-written prompt lines overall; the model expands these into `267,597` open-domain prompts and `99,121` topic-guided red-teaming prompts.
- **Self-instruct decoding**: for both self-instruct generators, the maximum generation length is `384` new tokens with nucleus sampling `top-p = 0.98` and temperature `t = 1.0`.
- **Principle-Driven Self-Alignment**: provide `16` human-written principles and `5` in-context learning exemplars that demonstrate an "internal-thought-then-answer" procedure; the principles plus exemplars occupy about `1800` LLaMA tokens.
- **Principle-conditioned response generation**: for the self-alignment stage, each prompt is answered with up to `256` new tokens using `top-p = 0.9` and temperature `t = 0.5`, allowing the model to pick relevant rules and produce either direct answers or principled refusals.
- **Filtered training data**: after low-quality-response filtering, the paper retains `191,628` self-instruct query-response pairs and `67,250` red-teaming pairs, for a total of `258,878` training examples.
- **Principle engraving**: fine-tune only the LoRA weights in the multi-head attention modules for `1` epoch with batch size `768`, maximum sequence length `512`, max learning rate `4e-4`, and a `335`-step schedule with `100` warm-up steps followed by linear decay.
- **Verbose cloning / context distillation**: prompt the principle-engraved teacher to produce verbose outputs for topic-guided queries, generating `358,777` additional responses with maximum generation length `512`, `top-p = 0.7`, temperature `t = 0.3`, and a repetition penalty.
- **Final verbose model training**: fine-tune the base model again for `1` epoch on the verbose dataset, still with LoRA-only attention updates, using batch size `768`, maximum sequence length `768`, max learning rate `4e-4`, and about `465` training steps.
- **Evaluation protocol**: benchmark generation defaults to temperature `0.7`; the paper compares Dromedary against LLaMA, Alpaca, Vicuna, ChatGPT, GPT-4, and Anthropic-LM style baselines on TruthfulQA, HHH Eval, and GPT-4-judged Vicuna benchmark questions.

## Key Results

- **Annotation efficiency**: Dromedary uses fewer than `300` human-written lines, compared with `77K` annotations for InstructGPT and `52K` teacher-generated examples for Alpaca.
- **TruthfulQA multiple choice**: the paper reports a new state-of-the-art `MC1 = 69`, outperforming the compared baselines including GPT-4.
- **TruthfulQA generation**: Dromedary non-verbose reaches `0.74` truthful and `0.57` truthful-times-informative; the final verbose model reaches `0.72 / 0.61`, both ahead of `LLaMA-65B` at `0.57 / 0.53` and reproduced `Alpaca-65B` at `0.47 / 0.47`.
- **HHH Eval overall**: Dromedary non-verbose scores `0.85` and the final model `0.83`, compared with `0.77` for `LLaMA-65B` and `0.79` for reproduced `Alpaca-65B`, while remaining below `ChatGPT` at `0.87`.
- **HHH harmlessness**: both Dromedary variants reach `0.91` on the harmless subset, versus `0.71` for `LLaMA-65B` and `0.76` for `Alpaca-65B`.
- **Open-ended evaluation**: on GPT-4-judged Vicuna benchmark questions, Dromedary surpasses Text-Davinci-003 and Alpaca, but still trails ChatGPT and Vicuna.

## Limitations

- The `16` principles are exploratory and author-defined; the paper explicitly notes they are not a final or globally valid specification of aligned behavior.
- Principle-driven prompting requires all rules and exemplars in context during data generation, costing roughly `1800` tokens and limiting scalability under shorter context windows.
- Dromedary inherits the base model's dated and incomplete knowledge, which the paper links to inaccurate or stale answers for post-2021 questions.
- The final verbose-cloning stage introduces a reported "verbose tax": generation quality improves, but multiple-choice trustworthiness degrades relative to the non-verbose model.
- The paper documents persistent failures to strictly follow its own principles, including hallucinated biographies and indirect answers to real-time questions.
- On open-ended chatbot comparisons, Dromedary remains weaker than ChatGPT and ChatGPT-distilled Vicuna, so the method does not close the frontier gap.

## Concepts Extracted

- [[self-align]]
- [[self-instruct]]
- [[in-context-learning]]
- [[context-distillation]]
- [[constitutional-ai]]
- [[supervised-fine-tuning]]
- [[reinforcement-learning-from-human-feedback]]
- [[large-language-model]]
- [[instruction-following]]

## Entities Extracted

- [[zhiqing-sun]]
- [[yikang-shen]]
- [[qinhong-zhou]]
- [[hongxin-zhang]]
- [[zhenfang-chen]]
- [[david-cox]]
- [[yiming-yang]]
- [[chuang-gan]]
- [[carnegie-mellon-university]]
- [[ibm-research]]
- [[umass-amherst]]
- [[llama-65b]]
- [[dromedary]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
