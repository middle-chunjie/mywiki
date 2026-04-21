---
type: source
subtype: paper
title: Learning Performance-Improving Code Edits
slug: shypula-2024-performanceimproving-2302-07867
date: 2026-04-20
language: en
tags: [llm, code-optimization, benchmarking, code-generation, program-analysis]
processed: true

raw_file: raw/papers/shypula-2024-performanceimproving-2302-07867/paper.pdf
raw_md: raw/papers/shypula-2024-performanceimproving-2302-07867/paper.md
bibtex_file: raw/papers/shypula-2024-performanceimproving-2302-07867/paper.bib
possibly_outdated: false

authors:
  - Alexander Shypula
  - Aman Madaan
  - Yimeng Zeng
  - Uri Alon
  - Jacob Gardner
  - Milad Hashemi
  - Graham Neubig
  - Parthasarathy Ranganathan
  - Osbert Bastani
  - Amir Yazdanbakhsh
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2302.07867
doi:
url: http://arxiv.org/abs/2302.07867
citation_key: shypula2024performanceimproving
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

The paper introduces PIE, a benchmark for learning performance-improving code edits from human `C++` submissions, and uses it to study whether large code models can optimize programs beyond compiler-level tuning. The core contribution is not only the `77,967`-pair dataset but also a deterministic gem5-based evaluation stack that avoids noisy speedup estimates from commodity hardware. On top of PIE, the authors test instruction prompting, chain-of-thought, retrieval-based few-shot prompting, performance-conditioned generation, and self-play-based synthetic augmentation. Data-driven adaptation is decisive: the best GPT-3.5 model with self-play reaches `6.86x` mean speedup at `Best@8`, exceeds the sampled human average of `3.66x`, and slightly surpasses the fastest human submission upper bound on the benchmark (`9.64x` vs. `9.56x`).

## Problem & Motivation

High-level program optimization remains difficult to automate because it requires semantic understanding of code, algorithmic reasoning, and reliable performance measurement. Existing compiler optimizations do not address many source-level changes such as selecting better algorithms, replacing data structures, or rewriting I/O patterns. The paper argues that progress has been slowed by two bottlenecks: the lack of an open dataset of genuine performance-improving edits, and the unreliability of runtime measurements on real hardware.

The authors therefore frame program optimization as a learning problem over human revision trajectories. They also treat deterministic performance evaluation as a first-class requirement, showing that even identical programs can appear to improve by `1.12x` on average under conventional benchmarking due to variance alone.

## Method

- **PIE construction**: build slow-fast pairs from chronological accepted `C++` submissions in CodeNet. Keep only pairs satisfying `((time(y_i) - time(y_j)) / time(y_i)) > 10%`.
- **Dataset scale**: `77,967` train pairs from `1,474` problems, `2,544` validation pairs from `77` problems, and `978` test pairs from `41` problems.
- **Correctness filtering**: retain only accepted programs and reject generated outputs if any unit test fails. Median test counts are `82.5` (train), `75` (val), and `104` (test) after augmentation with AlphaCode-derived cases.
- **Performance measurement**: relabel runtimes with gem5 full-system simulation instead of noisy hardware timing; the pipeline performs more than `42.8` million simulations using the Verbatim Intel Skylake configuration.
- **Prompting baselines**: evaluate instruction prompting, fixed few-shot prompting with `2` sampled `slow -> fast` exemplars, and chain-of-thought prompting.
- **Dynamic retrieval prompting**: embed programs with CodeBERTScore models trained for `C++`, retrieve nearest neighbors with FAISS, and assemble prompts from the top `K` examples; ablations use `K ∈ {1, 2, 4}` and the preferred setting is `K = 4`.
- **High-quality subset**: define an HQ training split of `4,085` pairs by keeping high-speedup examples and at most `4` pairs per problem, averaging `2.77` pairs per problem.
- **Performance-conditioned generation**: assign each target solution a score tag in `{1, ..., 10}` based on deciles of achievable performance for that problem; at inference, request the maximum tag `10/10`.
- **Synthetic augmentation / self-play**: generate `10,000` candidate programs with GPT-3.5, keep `6,553` outside PIE splits, group them into `3,314` equivalence sets, then retain `1,485` optimized pairs with at least `5x` speedup and no more than `3` semantic duplicates.
- **Training details**: fine-tune CODELLAMA `7B` and `13B` with AdamW, batch size `32`, learning rate `1e-5`; full-data and performance-conditioned runs use `1` epoch and take `24-36` hours on `8 x 48GB` GPUs.
- **Inference protocol**: evaluate `Best@1` and `Best@8` with temperature `0.7`; compile generated programs with GCC `9.3.0`, `C++17`, and `-O3`.

## Key Results

- Hardware benchmarking is unreliable: on `500` identical-program pairs, HYPERFINE reports mean apparent speedup `1.12x`, standard deviation `0.36`, and top-`5%` false speedups of `1.91x`.
- Best prompting baseline: GPT-3.5 with CoT reaches `43.05%` optimized, `1.60x` average speedup, and `91.72%` correctness at `Best@8`.
- Best retrieval model: GPT-4 with dynamic retrieval (`K = 4`) reaches `76.07%` optimized, `3.93x` speedup, and `95.71%` correctness at `Best@8`.
- Best open fine-tuned model: CODELLAMA `13B` with performance conditioning reaches `66.56%` optimized, `5.65x` speedup, and `70.96%` correctness at `Best@8`.
- Best overall model: GPT-3.5 fine-tuned on HQ plus self-play data reaches `87.63%` optimized, `6.86x` speedup, and `95.09%` correctness at `Best@8`.
- Human comparison: the sampled human reference averages `3.66x` speedup; with larger sampling, the model's fastest generation reaches `9.64x` versus `9.56x` for the fastest human submission pool.
- Edit analysis on `120` optimized pairs finds algorithmic changes (`34.15%`), I/O edits (`26.02%`), data-structure changes (`21.14%`), and miscellaneous optimizations (`18.70%`).
- Error analysis of GPT-3.5 + self-play shows `12.51%` syntax/type failures, about `50.51%` incorrect-but-compiling outputs across error buckets, `9.93%` slower correct outputs, `9.40%` unchanged outputs, and `10.75%` faster-but-below-threshold outputs.

## Limitations

- The benchmark only covers competitive-programming-style `C++` tasks, so generalization to larger software systems, other languages, or maintenance settings is unresolved.
- Correctness is approximated by available test cases; the appendix notes `10/120` inspected speedups that may be partly spurious because reduced array-allocation constants were not fully challenged by tests.
- Strong optimization still trades off against correctness: self-play improves coverage and speedup but can reduce precision, and many failures come from broken semantics rather than weak speed gains.
- gem5 improves determinism but is computationally expensive, requiring tens of millions of simulations and making data creation harder to scale.
- Parameter-efficient adaptation remains weak here: LoRA results stay far below full fine-tuning, suggesting the task may require broader parameter updates than lightweight adapters can provide.

## Concepts Extracted

- [[large-language-model]]
- [[code-language-model]]
- [[code-optimization]]
- [[benchmark-dataset]]
- [[benchmark-reliability]]
- [[few-shot-prompting]]
- [[chain-of-thought-prompting]]
- [[retrieval-based-few-shot-prompting]]
- [[performance-conditioned-generation]]
- [[self-play]]
- [[synthetic-data-augmentation]]

## Entities Extracted

- [[alexander-shypula]]
- [[aman-madaan]]
- [[yimeng-zeng]]
- [[uri-alon]]
- [[jacob-gardner]]
- [[milad-hashemi]]
- [[graham-neubig]]
- [[parthasarathy-ranganathan]]
- [[osbert-bastani]]
- [[amir-yazdanbakhsh]]
- [[pie-dataset]]
- [[codenet]]
- [[gem5]]
- [[code-llama]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
