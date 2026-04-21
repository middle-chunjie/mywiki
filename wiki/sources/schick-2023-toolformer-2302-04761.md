---
type: source
subtype: paper
title: "Toolformer: Language Models Can Teach Themselves to Use Tools"
slug: schick-2023-toolformer-2302-04761
date: 2026-04-20
language: en
tags: [llm, tool-use, self-supervision, zero-shot, nlp]
processed: true

raw_file: raw/papers/schick-2023-toolformer-2302-04761/paper.pdf
raw_md: raw/papers/schick-2023-toolformer-2302-04761/paper.md
bibtex_file: raw/papers/schick-2023-toolformer-2302-04761/paper.bib
possibly_outdated: true

authors:
  - Timo Schick
  - Jane Dwivedi-Yu
  - Roberto Dessì
  - Roberta Raileanu
  - Maria Lomeli
  - Luke Zettlemoyer
  - Nicola Cancedda
  - Thomas Scialom
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2302.04761
doi:
url: http://arxiv.org/abs/2302.04761
citation_key: schick2023toolformer
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

Toolformer trains a `6.7B` GPT-J language model to decide when to call external APIs, which tool to invoke, and what arguments to pass, without supervised tool-use traces. The core idea is to let the model propose candidate API calls inside raw pretraining text, execute them, and keep only calls that reduce future-token loss under a weighted language-model objective. The resulting finetuning corpus augments CCNet with useful calls to a question-answering system, Wikipedia search, calculator, machine translation model, and calendar API. This yields large zero-shot gains on factual completion, arithmetic, open-domain QA, and temporal reasoning while preserving ordinary language-model perplexity when tool use is disabled.

## Problem & Motivation

Large language models are strong at in-context and zero-shot generalization, but they remain weak at precise arithmetic, up-to-date factual lookup, low-resource language understanding, and temporal grounding. Prior tool-use methods either require substantial human supervision or teach tool invocation only in task-specific prompting setups. Toolformer targets a more general regime: learn tool use self-supervised from plain text so the model can decide autonomously when a tool is worth calling, without losing its base language-modeling ability.

## Method

- **Base setup**: start from GPT-J with `6.7B` parameters and a subset of CCNet `\mathcal{C}`; every tool must accept and return plain text so calls can be inserted inline with special markers `` `<API>` ``, `` `</API>` ``, and `` `->` ``.
- **API-call representation**: model an API call as `c = (a_c, i_c)`, where `a_c` is the API name and `i_c` its textual input. The linearized forms are `` `e(c) = <API> a_c(i_c) </API>` `` and `` `e(c, r) = <API> a_c(i_c) -> r </API>` ``.
- **Sampling candidate calls**: for each prompt-annotated sequence `P(x), x_1, ..., x_n`, compute `` `p_i = p_M(<API> | P(x), x_{1:i-1})` ``. Keep positions with `p_i > \tau_s`, cap to top `k`, and sample up to `m` concrete calls per position.
- **Loss-based filtering**: for weights `w_t`, define weighted future-token loss `` `L_i(z) = - \sum_{j=i}^n w_{j-i} \log p_M(x_j | z, x_{1:j-1})` ``. Compare `` `L_i^+ = L_i(e(c_i, r_i))` `` against `` `L_i^- = min(L_i(\epsilon), L_i(e(c_i, \epsilon)))` `` and keep a call only if `` `L_i^- - L_i^+ >= \tau_f` ``.
- **Weighting function**: use `` `\tilde{w}_t = max(0, 1 - 0.2 t)` `` and normalize to `w_t`, which biases retained calls toward information that helps near-future prediction.
- **Tool suite**: question answering uses Atlas (Atlas-large for data creation; Atlas-xxl at inference), Wikipedia search uses a `[[bm25]]` retriever over KILT Wikipedia, the calculator supports the four basic arithmetic operators with rounding to two decimals, machine translation uses `600M` NLLB with fastText language detection, and the calendar API returns the current date.
- **Sampling hyperparameters**: default settings are `\tau_s = 0.05`, `\tau_f = 1.0`, `k = 5`, `m = 5`; for calculator and MT, the paper uses `\tau_s = 0.0`, `k = 20`, `m = 10`, and `\tau_f = 0.5` because useful calls are rarer.
- **Data heuristics**: calculator candidates are drawn from numeric-heavy documents; MT is restricted to non-English spans surrounded by English context; calendar examples are approximated from dates extracted from URLs.
- **Finetuning**: build the augmented corpus `\mathcal{C}^*` by interleaving retained calls with original text, then finetune with standard LM loss using batch size `128`, learning rate `1e-5`, linear warmup over the first `10%`, max sequence length `1024`, ZeRO-3, `8 x A100 40GB`, BF16, and up to `2k` training steps.
- **Inference**: decode normally until the model expects a tool result after `` `->` ``; to encourage tool use, emit `` `<API>` `` whenever it is among the top `k = 10` next tokens, while restricting decoding to at most one API call per input.

## Key Results

- At filtering threshold `\tau_f = 1.0`, the final augmented corpus contains `18,526` QA examples, `60,974` Wikipedia-search examples, `994` calculator examples, `20,587` calendar examples, and `1,034` MT examples.
- On LAMA subsets, Toolformer reaches `33.8` on SQuAD, `11.5` on Google-RE, and `53.5` on T-REx, beating the best GPT-J-family baseline by `11.7`, `5.2`, and `18.6` points; on these examples it uses the QA tool `98.1%` of the time.
- On math benchmarks, Toolformer scores `40.4` on ASDiv, `29.4` on SVAMP, and `44.0` on MAWPS, versus `14.0`, `10.0`, and `19.8` for GPT-3 (`175B`); it invokes the calculator on `97.9%` of examples.
- On open-domain QA, Toolformer improves GPT-J from `18.5/12.8/43.9` to `26.3/17.7/48.8` on WebQS, Natural Questions, and TriviaQA, with Wikipedia search used on `99.3%` of cases, though GPT-3 remains stronger on NQ and TriviaQA.
- On temporal tasks, Toolformer reaches `16.3` on TEMPLAMA and `27.3` on DATESET; the DATESET gain is driven by calendar calls on `54.8%` of examples, whereas the calendar is used on only `0.2%` of TEMPLAMA examples.
- Standard LM quality is preserved when tools are disabled: perplexity is `10.3` on WikiText and `10.5` on held-out CCNet, matching GPT-J+CC after finetuning on plain CCNet.

## Limitations

- Toolformer cannot learn chained tool use because calls for different tools are generated independently, so the finetuning data contains no multi-step tool compositions.
- The model cannot interact with tools iteratively; for search-like tools, it cannot browse multiple results or reformulate a failed query.
- Tool invocation is prompt-sensitive: small wording changes can affect whether the model decides to call an API.
- The data-generation pipeline is sample-inefficient for some tools; processing over a million documents still yields only a few thousand useful calculator examples.
- The decoding policy ignores tool-dependent inference cost, so it does not optimize for latency or monetary cost when deciding to call an API.

## Concepts Extracted

- [[large-language-model]]
- [[self-supervised-learning]]
- [[in-context-learning]]
- [[zero-shot-learning]]
- [[retrieval-augmented-generation]]
- [[arithmetic-reasoning]]
- [[temporal-reasoning]]
- [[information-retrieval]]

## Entities Extracted

- [[timo-schick]]
- [[jane-dwivedi-yu]]
- [[roberto-dessi]]
- [[roberta-raileanu]]
- [[maria-lomeli]]
- [[luke-zettlemoyer]]
- [[nicola-cancedda]]
- [[thomas-scialom]]
- [[meta-ai]]
- [[gpt-j]]
- [[bm25]]
- [[natural-questions]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
