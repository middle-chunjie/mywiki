---
type: source
subtype: paper
title: Efficient Training of Language Models to Fill in the Middle
slug: bavarian-2022-efficient-2207-14255
date: 2026-04-20
language: en
tags: [llm, infilling, code, pretraining, evaluation]
processed: true

raw_file: raw/papers/bavarian-2022-efficient-2207-14255/paper.pdf
raw_md: raw/papers/bavarian-2022-efficient-2207-14255/paper.md
bibtex_file: raw/papers/bavarian-2022-efficient-2207-14255/paper.bib
possibly_outdated: true

authors:
  - Mohammad Bavarian
  - Heewoo Jun
  - Nikolas Tezak
  - John Schulman
  - Christine McLeavey
  - Jerry Tworek
  - Mark Chen
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2207.14255
doi:
url: https://arxiv.org/abs/2207.14255
citation_key: bavarian2022efficient
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

This paper studies how to endow causal decoder LMs with [[fill-in-the-middle]] capability by transforming a fraction of training documents into prefix-suffix-middle examples separated by sentinel tokens, without changing the underlying [[transformer]] architecture. Across code and natural-language pretraining runs from `50M` to `6.9B` parameters, the authors argue that FIM is effectively "for free": models retain standard [[autoregressive-language-model]] behavior while gaining strong infilling performance. The paper also turns the topic from an isolated augmentation trick into a training recipe by ablating FIM rate, PSM/SPM formatting, context-level versus document-level augmentation, and span selection strategy. A central empirical message is that [[sampling-based-evaluation]] is more informative than [[perplexity]] for judging practical infilling quality.

## Problem & Motivation

Standard left-to-right decoder models can only condition on a prefix, which makes them awkward for code editing, docstring completion, import insertion, and other workflows where both left and right context are available. Prior infilling-capable systems often relied on different architectures or specialized objectives, but the dominant large-scale model family was still causal decoding. The paper asks whether infilling can be added to that family cheaply, without sacrificing the strong generative and scaling properties that made autoregressive LMs attractive in the first place.

## Method

- Apply [[fill-in-the-middle]] augmentation to each document with probability `p`, splitting raw text uniformly at random into `prefix`, `middle`, and `suffix`, then reordering as `prefix, suffix, middle` or `suffix, prefix, middle`.
- Use sentinel tokens `<PRE>`, `<SUF>`, and `<MID>` so the tokenized PSM example becomes `` `<PRE> ∘ Enc(prefix) ∘ <SUF> ∘ Enc(suffix) ∘ <MID> ∘ Enc(middle)` ``; inference prompts omit the middle and sample until `<EOT>`.
- Retain loss on prefix, suffix, and middle tokens rather than masking parts of the example; the authors argue this preserves the original autoregressive learning signal and is important for the "FIM-for-free" effect.
- Compare two formatting modes: PSM (`prefix, suffix, middle`) and SPM (`suffix, prefix, middle`), with a default joint mixture that allocates half of the total FIM rate to each format.
- Compare [[document-level-fim]] versus [[context-level-fim]] augmentation. Context-level FIM first packs documents to the training context, then applies augmentation inside each chunk to avoid broken prefix/suffix fragments after packing.
- Study middle-span selection by choosing cut points at the line, token, or character level; the recommended setting is [[character-level-span-selection]] for robustness to mid-token boundaries.
- Evaluate on standard AR benchmarks plus code infilling benchmarks derived from [[humaneval]], including single-line, multi-line, [[random-span-infilling]], and a smaller random-span-infilling-light variant.
- Main code models follow GPT-3/Codex-style decoder stacks with [[relative-positional-encoding]]-style relative attention, context length `2048`, and scales from `50M` to `6.9B` parameters trained for up to `100B` tokens.

## Key Results

- Across models from `50M` to `6.9B` parameters, adding FIM during pretraining preserves left-to-right capability up to a FIM rate of `0.9`; degradation appears only at `1.0` FIM rate in AR loss.
- The infilling benchmark suite is much larger than vanilla HumanEval: `1033` single-line tasks, `5815` multi-line tasks, and `1640` random-span tasks.
- In Table 1, joint PSM+SPM training at FIM rate `0.9` reaches SPM-mode pass rates of `0.622` single-line, `0.305` multi-line, and `0.420` random-span, improving over the joint `0.5` setting (`0.595`, `0.293`, `0.379`).
- In Table 2, character-level span selection dominates on random-span infilling with pass rate `0.321`, versus `0.102` for token-level and `0.015` for line-level span selection, while remaining competitive on line-based benchmarks.
- In Table 4, the `6.9B` FIM90 model outperforms the `6.9B` FIM50 model on all three main infilling benchmarks: `0.751` vs `0.730` single-line, `0.441` vs `0.406` multi-line, and `0.551` vs `0.521` random-span.
- Finetuning is inefficient: among `16` finetuning configurations, only the most aggressive setup (`50B` finetuning tokens, FIM rate `0.9`, learning-rate multiplier `1.0`) catches up to the pretrained FIM baseline.

## Limitations

- The paper does not prove that FIM is universally free; it only shows no regression on the evaluated AR losses and benchmark suite, leaving open the possibility of unmeasured regressions.
- The study focuses on single-slot infilling, not multi-slot or structured editing settings that may matter in real IDE workflows.
- Natural-language infilling remains qualitatively weaker than code infilling, and the paper does not provide a comparably strong automatic evaluation setup for open-ended text.
- FIM sampling still suffers from stopping failures: missing `<EOT>` can produce overlong middles, suffix mismatch, or repetition.
- Several practical recommendations are benchmark-driven on internal model/data mixtures, so exact optima may shift for newer tokenizers, architectures, or post-training regimes.

## Concepts Extracted

- [[fill-in-the-middle]]
- [[autoregressive-language-model]]
- [[infilling]]
- [[context-level-fim]]
- [[document-level-fim]]
- [[random-span-infilling]]
- [[character-level-span-selection]]
- [[sampling-based-evaluation]]
- [[perplexity]]
- [[large-language-model]]
- [[relative-positional-encoding]]

## Entities Extracted

- [[mohammad-bavarian]]
- [[heewoo-jun]]
- [[nikolas-tezak]]
- [[john-schulman]]
- [[christine-mcleavey]]
- [[jerry-tworek]]
- [[mark-chen]]
- [[openai]]
- [[humaneval]]
- [[codex]]
- [[incoder]]
- [[code-davinci-002]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
