---
type: source
subtype: paper
title: To Code, or Not To Code? Exploring Impact of Code in Pre-training
slug: unknown-nd-code
date: 2026-04-20
language: en
tags: [llm-pretraining, code-data, data-mixture, reasoning, code-generation]
processed: true

raw_file: raw/papers/unknown-nd-code/paper.pdf
raw_md: raw/papers/unknown-nd-code/paper.md
bibtex_file: raw/papers/unknown-nd-code/paper.bib
possibly_outdated: false

authors:
  - Viraat Aryabumi
  - Yixuan Su
  - Raymond Ma
  - Adrien Morisot
  - Ivan Zhang
  - Acyr Locatelli
  - Marzieh Fadaee
  - Ahmet Üstün
  - Sara Hooker
year: 2024
venue: OpenReview
venue_type: preprint
arxiv_id:
doi:
url: https://openreview.net/pdf?id=zSfeN1uAcx
citation_key: unknownndcode3239
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

This paper studies whether code in the pretraining mixture improves non-code abilities of decoder-only LLMs. The authors run controlled ablations over initialization strategy, code proportion, code quality, synthetic code, model scale, and cooldown stage using `470M` and `2.8B` parameter transformers. Across natural-language reasoning, world-knowledge QA, code benchmarks, and LLM-as-a-judge win-rates, they find that code is not only useful for code generation but also materially improves general downstream behavior when mixed into training in the right way. The strongest overall recipe uses balanced code-text pretraining, text-heavy continued pretraining that still retains some code, and a cooldown phase that also includes code. The study also shows that higher-quality synthetic code has outsized benefits relative to its token share.

## Problem & Motivation

The paper asks a specific data-mixture question: why do general-purpose LLMs often benefit from code in pretraining even when their intended use is not code generation? Prior practice in systems such as PaLM, Gopher, BLOOM, and Llama includes code in the mixture, but the effect of code on non-code tasks had mostly been anecdotal or studied only in narrow settings. The authors therefore isolate code as a pretraining variable and evaluate whether its benefits depend on training stage, proportion, quality, or scale. The motivation is practical as much as scientific: if code genuinely improves reasoning and open-ended generation, then preserving or improving code data becomes a concrete pretraining-design lever.

## Method

- **Model family**: decoder-only [[decoder-only-transformer]] models trained with a standard LM objective at `470M` and `2.8B` parameters; architecture uses parallel attention layers, SwiGLU, no dense-layer biases, and [[byte-pair-encoding]] with vocabulary size `256,000`.
- **Optimization**: AdamW with batch size `512`, cosine learning-rate schedule, warmup `1325` steps, and maximum sequence length `8192`.
- **Core data setup**: natural-language text comes from [[slimpajama]], filtered to remove GitHub and StackExchange, leaving `503B` text tokens; main web code comes from [[the-stack]] after filtering, with a reported `139B`-token code subset; additional sources include `3.2B` formally verified synthetic-code tokens and `21.4B` code-adjacent tokens from GitHub commits, Jupyter notebooks, and StackExchange.
- **Initialization ablations**: compare text-only pretraining (`400B` text), balanced-only pretraining (`200B` text + `200B` code), balanced-initialized text models, and code-initialized text models. For `code -> text`, the initializer is trained on `200B` code tokens with an `80%` code / `20%` markup mixture.
- **[[continual-pretraining]] recipe**: after initialization, text-heavy continuation uses `200B` more tokens and keeps `10%` code in the mixture for `balanced -> text` and `code -> text` to avoid a full distribution shift.
- **Code-proportion sweep**: train six models for `200B` tokens with code proportions `{0%, 25%, 50%, 75%, 90%, 100%}` to test the trade-off between non-code and code performance.
- **Code-quality ablations**: replace portions of web code with markup-style data (`20%`), code-adjacent data (`15%`), or synthetic code (`10%`) to test whether code quality and structure matter beyond sheer code volume.
- **[[pretraining-cooldown]] stage**: run a final `40B`-token cooldown (`10%` of the pretraining budget) that up-weights high-quality text, math, code, and instruct-style data; compare no-cooldown, cooldown without code, and cooldown with `20%` code while switching from cosine decay to linear annealing with final learning rate `1e-6`.
- **Evaluation**: world knowledge uses TriviaQA and Natural Questions Open; [[natural-language-reasoning]] averages `11` benchmarks across QA, NLI, sentence completion, coreference, and ARC-Easy; code uses [[humaneval]] and [[mbpp]] pass@1; open-ended generation is measured by [[llm-as-a-judge]] win-rates on Dolly-200 English with Cohere Command-R+ as judge.
- **Infrastructure**: all models are trained in FAX on TPU v5e; the paper reports `64` pretraining runs total, with each `200B`-token run costing `4,736` TPU-chip-hours at `470M` and `13,824` TPU-chip-hours at `2.8B`.

## Key Results

- Relative to text-only pretraining, the paper's best configuration reports up to `+8.2%` on [[natural-language-reasoning]], `+4.2%` on [[world-knowledge]], `+6.6%` win-rate improvement on open-ended generation, and a `12x` boost on code benchmarks.
- In the initialization study, `code -> text` and `balanced -> text` outperform the text-only baseline on NL reasoning by `8.8%` and `8.2%` respectively; `balanced -> text` also gives the best world-knowledge result, improving `4.1%` over text-only.
- The full recipe table shows `balanced -> text` plus cooldown reaching `54.9` reasoning, `10.9` world knowledge, `32.9` NL average, `5.8` code, and `23.9` total average, the strongest overall non-code profile in the paper.
- In the code-proportion sweep, `25%` code / `75%` text gives the best reasoning result from-scratch, improving `3.4%` over `0%` code, while `100%` code causes an `18.3%` reasoning drop and an `86%` world-knowledge drop relative to the text-only model.
- Code performance increases roughly linearly with more code: the `100%` code model is `2.6x` better on code benchmarks than the `25%` code model, while the `0%` code model drops to `0` average pass@1.
- Data quality matters sharply: adding only `10%` synthetic code improves NL reasoning by `9.0%` and code by `44.9%` over the web-code-only baseline; in continual pretraining, `balanced+synth -> text` improves over `balanced -> text` by `2%` on reasoning and `35%` on code.
- Cooldown with code improves over the no-cooldown balanced-text model by `3.6%` on reasoning, `10.1%` on world knowledge, `20%` on code, and reaches `52.3%` win-rate versus `48.2%` for cooldown without code.

## Limitations

- The paper explicitly does not study safety effects, so it leaves open whether more code in pretraining changes harmfulness, alignment, or misuse behavior.
- The largest model size is `2.8B`; the authors argue trends should transfer upward, but that extrapolation is still unverified in this study.
- Code evaluation is narrow, relying on Python-centric function-completion benchmarks ([[humaneval]] and [[mbpp]]), so broader software-engineering effects are not tested directly.
- The synthetic-code source is proprietary, which limits exact reproducibility of the strongest quality-based ablations.
- World-knowledge measurement is relatively small, using only TriviaQA and Natural Questions Open, so some knowledge-transfer claims rest on a narrow benchmark slice.

## Concepts Extracted

- [[language-model-pretraining]]
- [[continual-pretraining]]
- [[pretraining-cooldown]]
- [[data-mixture]]
- [[data-quality]]
- [[synthetic-data]]
- [[code-generation]]
- [[natural-language-reasoning]]
- [[world-knowledge]]
- [[llm-as-a-judge]]
- [[decoder-only-transformer]]
- [[byte-pair-encoding]]

## Entities Extracted

- [[viraat-aryabumi]]
- [[yixuan-su]]
- [[raymond-ma]]
- [[adrien-morisot]]
- [[ivan-zhang]]
- [[acyr-locatelli]]
- [[marzieh-fadaee]]
- [[ahmet-ustun]]
- [[sara-hooker]]
- [[cohere]]
- [[slimpajama]]
- [[the-stack]]
- [[humaneval]]
- [[mbpp]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
