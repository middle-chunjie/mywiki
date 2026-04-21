---
type: source
subtype: paper
title: Measuring and Narrowing the Compositionality Gap in Language Models
slug: press-2023-measuring-2210-03350
date: 2026-04-20
language: en
tags: [llm, reasoning, prompting, multi-hop-qa, compositionality]
processed: true

raw_file: raw/papers/press-2023-measuring-2210-03350/paper.pdf
raw_md: raw/papers/press-2023-measuring-2210-03350/paper.md
bibtex_file: raw/papers/press-2023-measuring-2210-03350/paper.bib
possibly_outdated: true

authors:
  - Ofir Press
  - Muru Zhang
  - Sewon Min
  - Ludwig Schmidt
  - Noah A. Smith
  - Mike Lewis
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2210.03350
doi:
url: http://arxiv.org/abs/2210.03350
citation_key: press2023measuring
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper studies whether large language models can compose facts they appear to know individually, and formalizes the failure mode as the compositionality gap: the rate of incorrect 2-hop answers among cases where both constituent sub-questions are answered correctly. To measure this, the authors build Compositional Celebrities, an `8.6k`-question benchmark with `17` 2-hop templates that combine common facts in unlikely ways. Across GPT-3 and InstructGPT variants, single-hop accuracy rises with scale but the gap stays near `40%`, suggesting scaling improves factual recall more than composition. The paper then introduces self-ask, a structured prompting method that explicitly asks and answers follow-up questions, and shows that it improves multi-hop QA further when coupled with a search engine.

## Problem & Motivation

The paper targets a central ambiguity in LLM evaluation: strong question answering may come from memorizing isolated facts rather than composing them into unseen answers. The authors therefore focus on 2-hop questions whose supporting facts are individually common but jointly unlikely to have appeared verbatim in pretraining. This setup is intended to separate factual recall from compositional reasoning. They further argue that naive direct prompting under-allocates test-time computation to hard compositional questions, motivating prompts that let the model explicitly decompose and reason through sub-problems before answering.

## Method

- **Compositionality gap metric**: define the gap as the fraction of compositional questions answered incorrectly among cases where both sub-questions are answered correctly, i.e. `gap = #(2-hop wrong and both 1-hop right) / #(both 1-hop right)`.
- **Compositional Celebrities (CC)**: automatically generate `8.6k` 2-hop questions from celebrity birth facts and country/year attributes, covering `17` categories such as birthplace-to-capital and birth-year-to-literature-Nobel-winner.
- **CC evaluation protocol**: evaluate GPT-3-family models with category-specific `2-shot` prompts for both 1-hop and 2-hop questions; compare regular GPT models and InstructGPT variants across model scales from roughly `1B` to `175B` parameters.
- **Self-ask prompting**: use a structured prompt with scaffolds such as `Are follow up questions needed here:`, `Follow up:`, `Intermediate answer:`, and `So the final answer is:` so the model explicitly decomposes a question into sub-questions before answering.
- **Search-augmented self-ask**: when the LM emits a follow-up question, stop generation after `Intermediate answer:`, query a search engine with the full sub-question, insert the returned answer back into the prompt, and resume generation without changing model weights or prompt format.
- **Additional benchmarks**: test on Bamboogle (`125` handwritten 2-hop questions), a `1.2k` subset of 2WikiMultiHopQA dev, and `1252` 2-hop MuSiQue dev questions; use few-shot prompts, with `4` examples for the 2WikiMultiHopQA/Bamboogle setting.
- **Efficiency comparison**: compare self-ask to least-to-most prompting on 2WikiMultiHopQA and MuSiQue using average generated tokens, reporting `569` vs `844` on 2Wiki and `663` vs `1020` on MuSiQue.

## Key Results

- On CC, `davinci-002` answers `45.4%` of 2-hop questions correctly but about `80%` of sub-questions correctly, yielding a large residual compositionality gap.
- Across GPT-3 and InstructGPT scale variants, the compositionality gap remains roughly `40%` instead of shrinking with model size.
- On CC, when the worse of the two correct sub-question answers has perplexity between `1.000` and `1.002`, 2-hop accuracy reaches `81.1%`; when it is between `1.232` and `6.738`, 2-hop accuracy drops to `42.6%`.
- On Bamboogle with `davinci-002`, direct prompting scores `17.6`, chain-of-thought `46.4`, self-ask `57.6`, and self-ask + search `60.0` accuracy.
- On 2WikiMultiHopQA, direct prompting scores `25.4`, chain-of-thought `29.8`, self-ask `30.0`, and self-ask + search `40.1` exact match.
- On MuSiQue, direct prompting scores `5.6`, chain-of-thought `12.6`, self-ask `13.8`, and self-ask + search `15.2` exact match.
- Compared with least-to-most on later 2Wiki/MuSiQue runs, self-ask is similar or better in accuracy (`35.5` vs `29.0` on 2Wiki; `16.3` vs `16.8` on MuSiQue) while using over `30%` fewer generated tokens.

## Limitations

- The empirical claim about a roughly constant gap is only tested on models between about `1B` and `175B` parameters; larger post-2023 models could behave differently.
- The evaluation is centered on English 2-hop QA, so the conclusions may not transfer to other reasoning regimes such as arithmetic, logic, semantic parsing, or multilingual settings.
- CC is intentionally synthetic and template-based, which is useful for measurement but may underrepresent the variability of real user questions.
- The search-augmented setup relies on an external search engine API and measures answer accuracy rather than full evidence quality or citation faithfulness.
- The paper does not train or fine-tune models, so improvements come from prompting and tool use rather than stronger underlying reasoning representations.

## Concepts Extracted

- [[compositionality-gap]]
- [[self-ask]]
- [[chain-of-thought]]
- [[question-decomposition]]
- [[multihop-question-answering]]
- [[open-domain-question-answering]]
- [[few-shot-prompting]]
- [[prompt-engineering]]
- [[large-language-model]]
- [[compositional-generalization]]

## Entities Extracted

- [[ofir-press]]
- [[muru-zhang]]
- [[sewon-min]]
- [[ludwig-schmidt]]
- [[noah-a-smith]]
- [[mike-lewis]]
- [[university-of-washington]]
- [[mosaicml]]
- [[meta-ai]]
- [[allen-institute-for-ai]]
- [[gpt-3]]
- [[compositional-celebrities]]
- [[bamboogle]]
- [[2wiki-multihopqa]]
- [[musique]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
