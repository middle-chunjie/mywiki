---
type: source
subtype: paper
title: Self-Consistency Improves Chain of Thought Reasoning in Language Models
slug: wang-2023-selfconsistency-2203-11171
date: 2026-04-20
language: en
tags: [llm, reasoning, prompting, decoding, chain-of-thought]
processed: true

raw_file: raw/papers/wang-2023-selfconsistency-2203-11171/paper.pdf
raw_md: raw/papers/wang-2023-selfconsistency-2203-11171/paper.md
bibtex_file: raw/papers/wang-2023-selfconsistency-2203-11171/paper.bib
possibly_outdated: true

authors:
  - Xuezhi Wang
  - Jason Wei
  - Dale Schuurmans
  - Quoc Le
  - Ed H. Chi
  - Sharan Narang
  - Aakanksha Chowdhery
  - Denny Zhou
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2203.11171
doi:
url: http://arxiv.org/abs/2203.11171
citation_key: wang2023selfconsistency
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper proposes self-consistency, a test-time decoding strategy for [[chain-of-thought-prompting]] that samples multiple diverse [[reasoning-path]]s from one [[large-language-model]] and predicts the answer with the strongest agreement after [[answer-aggregation]]. Instead of trusting a single [[greedy-decoding]] trace, the method marginalizes latent rationales and returns `arg max_a Σ_i 1(a_i = a)` over sampled final answers. Across UL2-20B, LaMDA-137B, PaLM-540B, and GPT-3, the authors use up to `m = 40` sampled outputs per question and show large gains on arithmetic, commonsense, symbolic, and several standard NLP tasks. The paper established self-consistency as a simple training-free way to trade extra inference for better reasoning accuracy.

## Problem & Motivation

Chain-of-thought prompting had already shown that intermediate rationales can improve reasoning, but the standard decoding choice was still a single greedy trajectory. The authors argue this is brittle because many problems admit multiple valid solution paths, while an incorrect early decision can lock greedy decoding into a wrong final answer. They therefore recast reasoning as a latent-variable decoding problem: sample diverse candidate rationales, then trust the answer that remains stable across those rationales. The goal is to improve reasoning without additional supervision, verifiers, rerankers, or fine-tuning.

## Method

- **Core decoding rule**: sample `m` candidate pairs `(r_i, a_i)`, where `r_i` is a latent reasoning path and `a_i` is the parsed final answer, then choose the most consistent answer with `arg max_a Σ_i 1(a_i = a)`.
- **Alternative scoring studied**: the paper also evaluates weighted aggregation using `P(r_i, a_i | prompt, question) = exp^{(1/K) Σ_k log P(t_k | prompt, question, t_<k)}` and finds normalized weighted sum is close to unweighted majority voting, while weighted average is much worse.
- **Prompting setup**: all experiments stay in few-shot mode with no training or fine-tuning; arithmetic tasks use `8` manually written CoT exemplars, while commonsense tasks use `4-7` exemplars with handwritten rationales.
- **Sampling hyperparameters**: UL2-20B and LaMDA-137B use [[temperature-sampling]] with `T = 0.5` plus [[top-k-sampling]] with `k = 40`; PaLM-540B uses `T = 0.7, k = 40`; GPT-3 uses `T = 0.7` without top-`k` truncation.
- **Evaluation protocol**: main results average over `10` runs, each run sampling `40` outputs independently from the decoder; answers are parsed from the final segment after phrases like "The answer is ...".
- **Comparative analyses**: the paper compares self-consistency against greedy CoT, sample-and-rank, beam search, prompt-order ensembles, multi-prompt ensembles, equation-only reasoning, and zero-shot CoT.

## Key Results

- **Aggregation itself matters**: on PaLM-540B, Table 1 shows greedy decoding reaches `56.5` on GSM8K, while unweighted majority vote reaches `74.4`; it also improves AQuA `35.8 -> 48.3`, SVAMP `79.0 -> 86.6`, and ARC-challenge `85.2 -> 88.7`.
- **Arithmetic gains are often large**: with PaLM-540B, GSM8K improves `56.5 -> 74.4` (`+17.9`), AQuA `35.8 -> 48.3` (`+12.5`), and SVAMP `79.0 -> 86.6` (`+7.6`); with GPT-3 code-davinci-002, GSM8K improves `60.1 -> 78.0`, AQuA `39.8 -> 52.0`, and SVAMP `75.8 -> 86.8`.
- **Commonsense and symbolic reasoning also benefit**: on PaLM-540B, StrategyQA improves `75.3 -> 81.6`, ARC-challenge `85.2 -> 88.7`, Letter(4) `65.8 -> 70.8`, and Coinflip(4) `88.2 -> 91.2`; on GPT-3 code-davinci-001, ARC-challenge improves `43.1 -> 53.7`.
- **Self-consistency helps even when CoT hurts**: on PaLM-540B, Table 5 reports ANLI-R1 `68.8 -> 78.5`, e-SNLI `81.0 -> 88.4`, RTE `79.1 -> 86.3`, BoolQ `74.2 -> 78.4`, and HotpotQA `28.9/39.8 -> 33.8/44.6` (EM/F1).
- **Diversity beats deterministic search/ensembles**: on UL2-20B AQuA with `40` paths, beam search gives `10.2`, self-consistency with beam search `24.2`, and self-consistency with sampling `26.9`; on LaMDA-137B GSM8K, self-consistency with `40` sampled paths reaches `27.7` versus `19.2` for `40` prompt permutations.

## Limitations

- The method increases inference cost because it requires multiple sampled decodes instead of one pass; the paper suggests starting with `5` or `10` paths when compute is limited.
- It assumes tasks have a fixed or parseable final answer set; the approach is less direct for open-ended generation unless a reliable agreement metric is available.
- Benefits depend on having both model capability and path diversity: smaller models still remain weak on some tasks, e.g. UL2-20B on GSM8K only improves `4.1 -> 7.3`, and Letter(4) stays at `0.0`.
- Sampled rationales can still be nonsensical or factually wrong even when the final answer improves, so the method does not solve faithfulness or grounding.

## Concepts Extracted

- [[self-consistency]]
- [[chain-of-thought-prompting]]
- [[greedy-decoding]]
- [[few-shot-prompting]]
- [[reasoning-path]]
- [[answer-aggregation]]
- [[majority-voting]]
- [[temperature-sampling]]
- [[top-k-sampling]]
- [[large-language-model]]

## Entities Extracted

- [[xuezhi-wang]]
- [[jason-wei]]
- [[dale-schuurmans]]
- [[quoc-le]]
- [[ed-chi]]
- [[sharan-narang]]
- [[aakanksha-chowdhery]]
- [[denny-zhou]]
- [[google-research]]
- [[google-brain]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
