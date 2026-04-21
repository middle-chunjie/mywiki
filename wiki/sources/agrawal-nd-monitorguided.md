---
type: source
subtype: paper
title: Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context
slug: agrawal-nd-monitorguided
date: 2026-04-20
language: en
tags: [code-llm, static-analysis, constrained-decoding, code-completion, repository-level]
processed: true

raw_file: raw/papers/agrawal-nd-monitorguided/paper.pdf
raw_md: raw/papers/agrawal-nd-monitorguided/paper.md
bibtex_file: raw/papers/agrawal-nd-monitorguided/paper.bib
possibly_outdated: true

authors:
  - Lakshya A Agrawal
  - Aditya Kanade
  - Navin Goyal
  - Shuvendu K. Lahiri
  - Sriram K. Rajamani
year: 2023
venue: NeurIPS 2023
venue_type: conference
arxiv_id: 2306.10763
doi:
url: https://arxiv.org/abs/2306.10763
citation_key: agrawalndmonitorguided
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces monitor-guided decoding (MGD), a decoding-time interface between frozen code language models and repository-scale static analysis. Instead of expanding the prompt with cross-file context, a stateful monitor watches generation, triggers at predefined program points such as object dereferences, queries static analysis over the full repository and dependencies, and converts valid continuations into token masks over the LM vocabulary. The main instantiation enforces type-consistent identifier dereferences for Java method completion. On PRAGMATICCODE and DOTPROMPTS, MGD consistently improves compilation and agreement-with-ground-truth metrics across CodeGen, SantaCoder, and text-davinci-003, while also composing well with prompt augmentation and fill-in-the-middle decoding. The paper further argues that the same interface generalizes to richer semantic constraints such as argument arity, typestates, and session protocols.

## Problem & Motivation

Code LMs perform well when local file context is sufficient, but they frequently hallucinate identifiers, APIs, and semantic actions that depend on types or artifacts defined elsewhere in the repository or external libraries. This failure mode is especially severe for repository-level completion, where cross-file types, build-generated bindings, and API contracts are not directly visible in the prompt. Prior work typically injects extra retrieved context into the prompt or modifies the model architecture, which increases input length and still leaves the LM to infer which constraints matter. The paper asks whether IDE-style static analysis can be exposed directly at decoding time so that code generation remains locally conditioned by the LM while semantic consistency is enforced against the global repository context.

## Method

- **Framework**: define a monitor for property `φ` as `M_φ = (A_φ, s_0, S, pre, update, maskgen)`, where static analysis `A_φ` is invoked when the trigger `pre` fires and its suggestions become the monitor state.
- **Joint decoding rule**: if the monitor is idle, sample from the LM normally; otherwise sample from `softmax(ℓ ⊕ m)`, where `ℓ = L_θ(· | x_1, ..., x_n; p)` and mask `m = maskgen(s, V)` zeros out invalid next tokens by resetting logits to `-K`.
- **Trigger condition for the main monitor**: fire when the partial program ends in an object dereference like `obj.` so the next generated identifier must be a field or method accessible from inferred type `T`.
- **Static analysis scope**: `A_φ(x_1, ..., x_n; C)` runs over partial code plus repository context `C`, including imported files, external libraries, class hierarchies, and build-generated artifacts, to infer the type of `obj` and enumerate valid members.
- **Subtoken-aware masking**: for every vocabulary token `t ∈ V`, set `m[t] = 1` if `t` is a prefix of any suggested identifier; also allow tokens matching `w · E · Σ*` so that identifier-ending symbols like `(` or `,` can terminate generation legally.
- **State update**: after sampling token `x_{n+1}`, revert to `s_0` if an end-of-identifier symbol was emitted; otherwise prune every candidate in the current suggestion set that is not prefixed by `x_{n+1}`, and strip the emitted prefix from surviving candidates.
- **Composition**: multiple monitors can be combined by taking the product of their state spaces, enabling joint guidance for dereferences, argument counts, enum cases, typestates, or session-type protocols.
- **Datasets**: PRAGMATICCODE contains `100` buildable Java repositories; DOTPROMPTS contains `1,420` methods and `10,538` dereference prompts extracted from them.
- **Models**: evaluate CodeGen-{`350M`, `2B`, `6B`}, SantaCoder-`1.1B`, and `text-davinci-003`, all without retraining.
- **Decoding / prompting hyperparameters**: nucleus sampling with `top-p = 0.95`, `n = 6` samples, prompt budget `1536`, generation budget `512`, total context window `2048`; temperatures are `0.2`, `0.4`, `0.6`, and `0.8` with one sample at `0.2` and `0.4`, and two samples each at `0.6` and `0.8`.
- **Prompt variants**: standard left-truncated local context, `classExprTypes` prompt augmentation with `20%` token budget reserved for type-definition files, RLPG augmentation, and SantaCoder fill-in-the-middle (FIM).

## Key Results

- On standard prompts at `score@6`, SantaCoder improves from `CR 59.97` to `73.03`, `NIM 82.40` to `88.42`, `ISM 38.14` to `40.69`, and `PM 32.10` to `34.25` with MGD.
- `text-davinci-003` also benefits from MGD: `CR 62.66 -> 74.26`, `NIM 86.18 -> 91.19`, `ISM 44.97 -> 47.33`, and `PM 38.77 -> 39.94`.
- The smallest CodeGen model becomes competitive with much larger models: CodeGen-`350M` with MGD reaches `CR 65.37`, surpassing vanilla `text-davinci-003` on compilation rate.
- Prompt augmentation and MGD are complementary: `SC-RLPG-MGD` reaches `CR 78.14`, `NIM 89.89`, `ISM 44.47`, and `PM 37.97`, outperforming both RLPG alone and standard MGD.
- FIM and MGD are also complementary: `SC-FIM-MGD` reaches `CR 80.19`, `NIM 89.89`, `ISM 44.50`, and `PM 37.91`; `SC-FIM-classExprTypes-MGD` pushes `NIM` to `90.42`.
- For the hardest identifier-complexity bucket `[4, 18)`, MGD improves next-identifier match by `21.79%` to `27.91%` relative over the corresponding base models.
- The generalizability microbenchmark covers `3` languages (Java, C#, Rust), `4` coding scenarios, and `2` richer analysis families (typestate and session-type constraints), showing that MGD is not limited to Java dereference prediction.

## Limitations

- The approach depends on static analysis for partial and incomplete programs, which the authors explicitly describe as heuristic, potentially imprecise, and incomplete.
- MGD improves semantic validity proxies such as compilation and identifier agreement, but it does not guarantee functional correctness, invariant satisfaction, or full behavioral correctness; testing and human review remain necessary.
- The large-scale evaluation is centered on Java repository-level method completion and one main monitor for type-consistent dereferences; broader claims rely on a small `10`-example microbenchmark.
- Decoding incurs non-trivial overhead: for CodeGen-`6B`, the reported mean inference time rises from `22.57s` to `41.34s`, an `83.16%` slowdown.
- Some richer scenarios are only fully solved when multiple monitors are combined, which suggests single-property monitoring can still leave residual errors such as wrong argument counts or invalid follow-up dereferences.

## Concepts Extracted

- [[monitor-guided-decoding]]
- [[static-analysis]]
- [[repository-level-context]]
- [[code-completion]]
- [[constrained-decoding]]
- [[language-server-protocol]]
- [[logit-masking]]
- [[prompt-augmentation]]
- [[type-consistency]]

## Entities Extracted

- [[lakshya-agrawal]]
- [[aditya-kanade]]
- [[navin-goyal]]
- [[shuvendu-lahiri]]
- [[sriram-rajamani]]
- [[microsoft-research]]
- [[pragmaticcode]]
- [[dotprompts]]
- [[santacoder]]
- [[codegen]]
- [[text-davinci-003]]
- [[multilspy]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
