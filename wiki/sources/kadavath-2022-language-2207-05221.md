---
type: source
subtype: paper
title: Language Models (Mostly) Know What They Know
slug: kadavath-2022-language-2207-05221
date: 2026-04-20
language: en
tags: [llm, calibration, self-evaluation, honesty, uncertainty]
processed: true
raw_file: raw/papers/kadavath-2022-language-2207-05221/paper.pdf
raw_md: raw/papers/kadavath-2022-language-2207-05221/paper.md
bibtex_file: raw/papers/kadavath-2022-language-2207-05221/paper.bib
possibly_outdated: true
authors:
  - Saurav Kadavath
  - Tom Conerly
  - Amanda Askell
  - Tom Henighan
  - Dawn Drain
  - Ethan Perez
  - Nicholas Schiefer
  - Zac Hatfield-Dodds
  - Nova DasSarma
  - Eli Tran-Johnson
  - Scott Johnston
  - Sheer El-Showk
  - Andy Jones
  - Nelson Elhage
  - Tristan Hume
  - Anna Chen
  - Yuntao Bai
  - Sam Bowman
  - Stanislav Fort
  - Deep Ganguli
  - Danny Hernandez
  - Josh Jacobson
  - Jackson Kernion
  - Shauna Kravec
  - Liane Lovitt
  - Kamal Ndousse
  - Catherine Olsson
  - Sam Ringer
  - Dario Amodei
  - Tom Brown
  - Jack Clark
  - Nicholas Joseph
  - Ben Mann
  - Sam McCandlish
  - Chris Olah
  - Jared Kaplan
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2207.05221
doi:
url: http://arxiv.org/abs/2207.05221
citation_key: kadavath2022language
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

This paper studies whether pretrained language models can estimate both whether a specific answer is correct and whether they know how to answer a question at all. Anthropic evaluates calibration on multiple-choice and true/false tasks, defines `P(True)` for judging sampled answers, and trains a value-head predictor `P(IK)` for "I know" on top of the base model. Across models from `800M` to `52B`, letter-labeled formatting, few-shot prompting, and comparison against multiple `T = 1` samples substantially improve calibration and self-evaluation. The strongest models generalize partially across trivia, Lambada, arithmetic, code, and GSM8k, and `P(IK)` rises when relevant documents or hints are inserted into context, suggesting useful but still fragile model self-knowledge.

## Problem & Motivation

The paper is motivated by honesty in AI systems: if a model cannot tell when it is likely to be wrong, it cannot safely decide when to answer, abstain, defer, or seek evidence. The authors therefore separate two related questions. First, can a model judge whether a concrete proposed answer is true in the world? Second, can a model estimate whether it knows the answer before committing to one? They study these questions because calibration, truthfulness, and self-knowledge are prerequisites for more reliable large language model behavior, especially under distribution shift and in settings where context can change what the model is able to answer.

## Method

- Evaluate pretrained language models of `800M`, `3B`, `12B`, and `52B` parameters, using the same architecture family as Anthropic's earlier models and pretraining length `850B` tokens.
- For multiple-choice calibration, present explicit lettered options and read the model's option probabilities; for answer verification, reformulate the task as `P(True) = p(answer is true | question, proposed answer)`.
- Improve self-evaluation by showing `5` brainstormed `T = 1` samples for the same question before asking the model to score one candidate answer as true or false; the best-calibrated setting is `20-shot`.
- Define question-level self-knowledge as `P(IK) = p(I know | question)` and train it with an added value head read from the final token of the input sequence.
- Construct `P(IK)` training labels by sampling `30` answers at `T = 1` for each question, duplicating `(Q, IK)` and `(Q, IDK)` examples according to how many samples are correct, and optimizing cross-entropy on these hard labels as a proxy for a soft target.
- Finetune the full model plus value head for `P(IK)` with batch size `7680` and learning rate `1/3` of the original pretraining rate; use `10-shot` prompting for TriviaQA, Lambada, Arithmetic, and GSM8k, and `0-shot` for Python function synthesis.
- Measure calibration with `ECE = (1 / N) sum_i |y_i - x_i|`, `RMSCE = sqrt((1 / N) sum_i (y_i - x_i)^2)`, and `Brier = (1 / N) sum_i (p_i - c_i)^2`.
- Probe robustness by testing out-of-distribution transfer from TriviaQA to Lambada, arithmetic, GSM8k, HumanEval, and Python synthesis, and by checking whether `P(IK)` changes when relevant source documents or hints are inserted into the context.
- For RLHF policies, show that apparent miscalibration can be partly repaired with post-hoc temperature scaling at `T = 2.5`.

## Key Results

- The paper's strongest context-use example shows a `52B` model raising `P(IK)` from `18%` to `78%` when a relevant Wikipedia passage is prepended to a TriviaQA-style question.
- In the model-specific self-knowledge test on TriviaQA, model A assigns average `P(IK) = 0.463` to questions only A answers correctly versus model B's `0.409`; on questions only B answers correctly, A gives `0.408` while B gives `0.477`.
- In the crossed training experiment, testing on model-B ground truth favors starting from model B over model A: `AUROC / Brier = 0.8717 / 0.1443` versus `0.8460 / 0.1582`.
- The cross-model analysis isolates `650` questions only model A gets right and `826` only model B gets right, supporting the claim that `P(IK)` captures model-specific knowledge rather than only raw task difficulty.
- For self-evaluation, the paper reports that `20-shot` true/false evaluation with `5` comparison samples produces the best Brier scores across the compared `P(True)` setups, and that filtering to answers with `P(True) > 0.5` yields substantially higher conditional accuracy than unfiltered `T = 1` samples.
- For GSM8k, the task is hard enough that the authors only evaluate the `52B` model and sometimes binarize ground-truth `P(IK)` at threshold `0.25` rather than `0.5`.

## Limitations

- Zero-shot self-evaluation is poorly calibrated and often collapses toward probabilities near `50%`, so the method depends heavily on prompt format and few-shot context.
- `P(IK)` transfers nontrivially across tasks, but out-of-distribution calibration remains weak, especially on Lambada where the model becomes systematically underconfident.
- Comparison samples help short-answer tasks more than long-form tasks such as HumanEval and GSM8k, where answer validity is harder to judge from a single verbalized trace.
- The study focuses mostly on pretrained language models that imitate human text; it does not resolve harder honesty questions where models know things humans do not or where truth diverges from typical human statements.
- RLHF policies are initially miscalibrated enough that extra temperature tuning is required, so the results do not automatically carry over to aligned chat-style systems.

## Concepts Extracted

- [[large-language-model]]
- [[calibration]]
- [[self-evaluation]]
- [[self-knowledge]]
- [[confidence-estimation]]
- [[few-shot-prompting]]
- [[temperature-scaling]]
- [[out-of-distribution-generalization]]
- [[truthfulness]]
- [[reinforcement-learning-from-human-feedback]]

## Entities Extracted

- [[saurav-kadavath]]
- [[tom-conerly]]
- [[amanda-askell]]
- [[tom-henighan]]
- [[dawn-drain]]
- [[ethan-perez]]
- [[nicholas-schiefer]]
- [[zac-hatfield-dodds]]
- [[nova-dassarma]]
- [[eli-tran-johnson]]
- [[scott-johnston]]
- [[sheer-el-showk]]
- [[andy-jones]]
- [[nelson-elhage]]
- [[tristan-hume]]
- [[anna-chen]]
- [[yuntao-bai]]
- [[sam-bowman]]
- [[stanislav-fort]]
- [[deep-ganguli]]
- [[danny-hernandez]]
- [[josh-jacobson]]
- [[jackson-kernion]]
- [[shauna-kravec]]
- [[liane-lovitt]]
- [[kamal-ndousse]]
- [[catherine-olsson]]
- [[sam-ringer]]
- [[dario-amodei]]
- [[tom-brown]]
- [[jack-clark]]
- [[nicholas-joseph]]
- [[ben-mann]]
- [[sam-mccandlish]]
- [[chris-olah]]
- [[jared-kaplan]]
- [[anthropic]]
- [[big-bench]]
- [[mmlu]]
- [[triviaqa]]
- [[gsm8k]]
- [[humaneval]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
