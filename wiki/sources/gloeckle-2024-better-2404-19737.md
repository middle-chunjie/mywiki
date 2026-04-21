---
type: source
subtype: paper
title: Better & Faster Large Language Models via Multi-token Prediction
slug: gloeckle-2024-better-2404-19737
date: 2026-04-20
language: en
tags: [llm, training, decoding, efficiency, code-generation]
processed: true

raw_file: raw/papers/gloeckle-2024-better-2404-19737/paper.pdf
raw_md: raw/papers/gloeckle-2024-better-2404-19737/paper.md
bibtex_file: raw/papers/gloeckle-2024-better-2404-19737/paper.bib
possibly_outdated: false

authors:
  - Fabian Gloeckle
  - Badr Youbi Idrissi
  - Baptiste Rozière
  - David Lopez-Paz
  - Gabriel Synnaeve
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2404.19737
doi:
url: http://arxiv.org/abs/2404.19737
citation_key: gloeckle2024better
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

The paper replaces standard next-token pretraining with multi-token prediction (MTP), asking a shared Transformer trunk to predict the next `n` future tokens in parallel through `n` independent heads. The authors argue that this makes training more sample-efficient and better aligned with downstream autoregressive generation. Across code, natural-language, and synthetic reasoning experiments, MTP improves the strongest gains on generative tasks and at larger scales: 13B code models solve `17%` more MBPP problems and `12%` more HumanEval problems than next-token baselines, while 4-token models can also unlock self-speculative decoding for up to `3x` faster inference. The paper further argues that MTP promotes induction and longer-horizon reasoning by forcing representations to encode information useful beyond the immediate next token.

## Problem & Motivation

Next-token prediction with teacher forcing is locally supervised: the model only needs to be correct about the immediate next token, even when downstream utility depends on longer-range consequences. The authors argue this encourages large language models to overfit short-range patterns and underweight hard semantic choice points, producing poor sample efficiency and a mismatch between teacher-forced training and autoregressive inference. They therefore ask whether one can keep the same causal language-modeling setup, but augment pretraining so each hidden state must already contain information useful for several future positions. The goal is stronger generative performance, better reasoning behavior, and faster decoding without training a separate draft model.

## Method

- **Objective**: replace next-token loss `L_1 = -\sum_t \log P_\theta(x_{t+1} \mid x_{t:1})` with multi-token loss `L_n = -\sum_t \log P_\theta(x_{t+n:t+1} \mid x_{t:1})`.
- **Factorization**: implement `L_n` with a shared trunk `f_s` that produces a contextual state `z_{t:1}`, then `n` independent prediction heads `f_{h_i}` plus a shared unembedding `f_u`, so `P_\theta(x_{t+i} \mid x_{t:1}) = softmax(f_u(f_{h_i}(f_s(x_{t:1}))))`.
- **Architecture**: the default MTP design uses parallel Transformer heads on top of a shared Transformer trunk. For compute-matched comparisons, when adding `n - 1` head layers, the authors remove `n - 1` layers from the trunk so total parameter count stays fixed.
- **Memory-efficient training**: instead of materializing all `n` vocab-logit tensors simultaneously, the implementation runs head forward/backward passes sequentially after the shared-trunk forward pass, accumulating trunk gradients. This reduces peak activation memory from `O(nV + d)` to `O(V + d)`.
- **Inference**: the simplest use keeps only the `i = 1` head and performs standard autoregressive decoding. The extra heads can also be reused for [[speculative-decoding]] via self-speculative/blockwise verification, without a separate draft model.
- **Alternative heads**: Appendix B compares linear, causal, anticausal, and parallel head organizations. The parallel Transformer-head design is the strongest and most stable of the tested variants.
- **Training hyperparameters**: all runs use Adam with `β1 = 0.9`, `β2 = 0.95`, decoupled weight decay `0.1`, linear warmup plus cosine decay, and gradient clipping `1.0` except CodeContests finetuning (`0.1`). Typical large-scale code runs use `peak_lr = 3e-4`, `context_length = 4096`, `warmup_steps = 2000`, `25,000` steps for `~209.7B` tokens; the byte-level `7B` run uses `context_length = 8192`.
- **Experimental settings**: model scaling studies cover `0.3B` to `13B`; code experiments use `n ∈ {1,2,4,6,8}` for token models and `n ∈ {1,8,16,32}` for byte models; natural-language runs compare `n = 1, 2, 4`.

## Key Results

- **Scale matters**: on code models trained from scratch, MTP is weak for small models but wins at scale; for `13B`, MBPP `pass@1` improves from `26.0` to `30.5` and HumanEval `pass@1` from `14.1` to `15.8` when moving from `n = 1` to `n = 4`.
- **Best 7B token model**: with `200B` code tokens, `n = 4` is best overall, improving MBPP from `30.0/53.8/73.7` to `33.8/55.9/76.9` at `pass@1/10/100`, and HumanEval from `22.8/36.4/62.0` to `24.0/40.1/66.1`.
- **Byte-level gains**: on `313B` bytes of code, `n = 8` lifts MBPP `pass@1` from `19.3` to `32.3`, HumanEval `pass@1` from `18.1` to `21.8`, and APPS/Intro `pass@1` from `0.1` to `1.2`.
- **Inference speed**: self-speculative decoding with a `7B` 4-token model reaches `3.05x` relative speedup on code with `3.50` tokens/forward, and `2.74x` / `2.67x` on Wikipedia/books. Byte-level `n = 8` reaches `6.39x`.
- **Multiple epochs**: after `1T` code tokens (`4` epochs), `n = 4` still improves MBPP `pass@1` from `40.7` to `43.1` and HumanEval `pass@100` from `83.0` to `86.2`.
- **Finetuning**: a `7B` model pretrained with `n = 4` and finetuned on [[codecontests]] outperforms a next-token-pretrained baseline across `pass@k`; stripping the extra heads and finetuning with `n' = 1` is actually the best finetuning variant.
- **Natural language**: on standard multiple-choice NLP benchmarks, `n = 2` is roughly on par and `n = 4` regresses, but on eight summarization datasets the average ROUGE-L `F1` improves by `+0.51` (`n = 2`) and `+0.46` (`n = 4`) for `200B`-token models.

## Limitations

- Gains are not universal: for small code models (`0.3B`, `0.6B`) MTP is worse than next-token prediction, so the method is not a free win at all scales.
- On natural-language choice benchmarks, `n = 4` regresses and even `n = 2` is mostly parity, so benefits are concentrated on generative rather than standard discriminative evaluations.
- The optimal prediction horizon is domain-dependent: `n = 4` is best for 32k-token code models, but byte-level models prefer `n = 8`, and APPS/Intro sometimes prefers `n = 6`.
- The reported training-time story is "no meaningful overhead" at the method level, but the actual implementation still shows slight slowdowns (`1.02x` to `1.22x`) because of imperfect FSDP overlap.
- Appendix D reports that directly finetuning an existing Llama 2 model with 4-token prediction does not reliably help, suggesting MTP is more effective as a pretraining objective than as a drop-in post hoc finetuning loss.
- The full experimental campaign is expensive: roughly `500K` GPU hours on A100-80GB and H100 hardware.

## Concepts Extracted

- [[large-language-model]]
- [[autoregressive-language-model]]
- [[multi-token-prediction]]
- [[teacher-forcing]]
- [[speculative-decoding]]
- [[byte-level-tokenization]]
- [[induction-head]]
- [[algorithmic-reasoning]]
- [[mutual-information]]

## Entities Extracted

- [[fabian-gloeckle]]
- [[badr-youbi-idrissi]]
- [[baptiste-roziere]]
- [[david-lopez-paz]]
- [[gabriel-synnaeve]]
- [[meta-ai]]
- [[mbpp]]
- [[humaneval]]
- [[apps-benchmark]]
- [[codecontests]]
- [[gsm8k]]
- [[xformers]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
