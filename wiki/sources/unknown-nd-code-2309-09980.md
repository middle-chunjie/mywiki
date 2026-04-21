---
type: source
subtype: paper
title: Code Representation Pre-training with Complements from Program Executions
slug: unknown-nd-code-2309-09980
date: 2026-04-20
language: en
tags: [code-intelligence, code-representation-learning, fuzzing, pretraining, software-engineering]
processed: true

raw_file: raw/papers/unknown-nd-code-2309-09980/paper.pdf
raw_md: raw/papers/unknown-nd-code-2309-09980/paper.md
bibtex_file: raw/papers/unknown-nd-code-2309-09980/paper.bib
possibly_outdated: true

authors:
  - Jiabo Huang
  - Jianyu Zhao
  - Yuyang Rong
  - Yiwen Guo
  - Yifeng He
  - Hao Chen
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2309.09980
doi:
url: https://arxiv.org/abs/2309.09980
citation_key: unknownndcode9980
paper_type: method

read_status: unread

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper proposes FuzzPretrain, a code representation pre-training framework that augments static code or AST inputs with dynamic information extracted from program executions. It fuzzes 1.2M CodeNet submissions to collect input-output test cases, concatenates those cases with code during pre-training, and trains with three objectives: masked language modeling for static structure, dynamic information matching for code-test consistency, and dynamic information distillation so runtime semantics remain encoded even when test cases are absent at inference. Built on top of CodeBERT and UniXcoder, FuzzPretrain substantially improves code-search retrieval and transfers to clone detection, defect detection, and text-to-code search, suggesting that executable behavior complements syntax for more discriminative code embeddings.

## Problem & Motivation

Existing code pre-training methods mostly learn from source text or static syntactic structures such as ASTs and CFG-like abstractions. That captures lexical and structural regularities, but it does not directly encode whether two implementations exhibit the same functionality or whether a small code change alters runtime behavior. The paper targets this gap by using executions as a weak but explicit semantic signal: if programs are defined by what they do on inputs, then test-case input-output mappings provide information about functionality that static views miss. The practical challenge is to exploit this signal during pre-training without requiring test cases at downstream inference time.

## Method

- **Training corpus**: collect `1.2M` C/C++/Python/Java submissions from CodeNet and fuzz each program to synthesize multiple test cases; C/C++/Java programs are instrumented before compilation, while Python uses a modified interpreter to expose execution behavior.
- **Input construction**: represent code as `T^s` and concatenated test cases as `T^d`; the model consumes `T = T^s ⊕ T^d` with separator tokens and an `<EOS>` suffix. Each test case is rendered in natural language as `Input is: INPUT; Output is: OUTPUT`.
- **Backbones**: instantiate the framework on CodeBERT and UniXcoder, both using a `12`-layer Transformer with about `125M` parameters; CodeBERT uses the `<BOS>` embedding as the code-level vector, while UniXcoder averages token embeddings.
- **Static Information Modeling (SIM)**: apply MLM on code tokens only, masking `15%` of tokens with the standard `80/10/10` corruption rule and optimizing `L_SIM = -Σ log p(x_i | X̃^s)`.
- **Dynamic Information Matching (DIM)**: sample either matched test cases `T^d` or unmatched negatives `T^{d-}`, predict a binary label `y ∈ {0,1}`, and optimize `L_DIM = -y log p(y|x̃) - (1-y) log(1-p(y|x̃))` after pooling with `g(f_θ(T))` and a linear projection.
- **Dynamic Information Distillation (DID)**: distill holistic code-plus-execution semantics into code-only embeddings using a momentum encoder `f_θ̂`, cosine similarity `h(·,·)`, temperature `t = 0.07`, momentum `m = 0.999`, and `l^n = 2^16` queued negatives in `L_DID`.
- **Training schedule**: alternate SIM, DIM, and DID during pre-training for `10K` steps with Adam and learning rate `2e-5`; batch size is `2048` / `1024` and max length `512` / `1024` for CodeBERT / UniXcoder, reserving `400` / `800` tokens for code or AST and the remainder for test cases.
- **Inference behavior**: discard test cases after pre-training and keep only the main encoder `f_θ`, so downstream tasks use code-only representations despite having learned from executions.

## Key Results

- On code search, FuzzCodeBERT improves overall mAP from `10.05` to `16.13` over the code-only MLM baseline (`+6.08`), while FuzzUniXcoder improves from `10.27` to `30.22` over the AST-only MLM baseline (`+19.95`).
- Relative to stronger static baselines, FuzzCodeBERT beats CodeBERT (`4.94 -> 16.13`) and FuzzUniXcoder beats UniXcoder (`20.45 -> 30.22`) on overall code-search mAP.
- In unseen domains, FuzzCodeBERT reaches `93.0` mAP on clone detection, `64.1%` accuracy on defect detection, and `69.1` MRR on text-to-code search; FuzzUniXcoder reaches `92.2`, `64.5%`, and `70.7`, respectively.
- Against state-of-the-art baselines, FuzzCodeBERT outperforms GraphCodeBERT on clone detection (`93.0` vs. `85.2`) and text search (`69.1` vs. `68.4`), while FuzzUniXcoder yields the best text-search score in the table at `70.7`.
- Removing either dynamic component hurts substantially: on code search, FuzzUniXcoder drops from `30.22` to `12.79` without DIM and to `5.44` without DID, showing both matching and distillation are critical.
- Pre-training remains practical at moderate scale, taking about `12` hours with code inputs and `20` hours with AST inputs on `8` Nvidia V100 GPUs.

## Limitations

- The pre-training corpus is restricted to OJ-style CodeNet programs, so the learned benefits may depend on competitive-programming distributions and may transfer imperfectly to broader software repositories.
- The method requires compilation or execution instrumentation plus fuzzing during pre-training, which adds nontrivial engineering cost and limits applicability on code bases without runnable environments.
- Text-code performance is still constrained because CodeNet lacks aligned natural-language descriptions; the paper explicitly notes that richer text-code-test corpora could yield larger gains.
- The authors report no improvement on code-generation tasks, suggesting the global execution-aware objectives do not directly help token-level generation.
- Because the paper is from `2023` in a fast-moving code-LLM area, comparisons against more recent execution-aware and retrieval-augmented code models should be re-checked.

## Concepts Extracted

- [[code-representation-learning]]
- [[fuzzing]]
- [[masked-language-modeling]]
- [[contrastive-learning]]
- [[dynamic-program-analysis]]
- [[static-program-analysis]]
- [[program-execution]]
- [[abstract-syntax-tree]]
- [[code-search]]
- [[code-clone-detection]]
- [[defect-detection]]
- [[code-intelligence]]

## Entities Extracted

- [[jiabo-huang]]
- [[jianyu-zhao]]
- [[yuyang-rong]]
- [[yiwen-guo]]
- [[yifeng-he-uc-davis]]
- [[hao-chen]]
- [[tencent-security-big-data-lab]]
- [[uc-davis]]
- [[afl-plus-plus]]
- [[codenet]]
- [[codebert]]
- [[unixcoder]]
- [[poj-104]]
- [[devign]]
- [[cosqa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
