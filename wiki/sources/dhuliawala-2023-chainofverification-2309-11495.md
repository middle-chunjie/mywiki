---
type: source
subtype: paper
title: Chain-of-Verification Reduces Hallucination in Large Language Models
slug: dhuliawala-2023-chainofverification-2309-11495
date: 2026-04-20
language: en
tags: [llm, hallucination, verification, prompting, factuality]
processed: true

raw_file: raw/papers/dhuliawala-2023-chainofverification-2309-11495/paper.pdf
raw_md: raw/papers/dhuliawala-2023-chainofverification-2309-11495/paper.md
bibtex_file: raw/papers/dhuliawala-2023-chainofverification-2309-11495/paper.bib
possibly_outdated: true

authors:
  - Shehzaad Dhuliawala
  - Mojtaba Komeili
  - Jing Xu
  - Roberta Raileanu
  - Xian Li
  - Asli Celikyilmaz
  - Jason Weston
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2309.11495
doi:
url: http://arxiv.org/abs/2309.11495
citation_key: dhuliawala2023chainofverification
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper proposes Chain-of-Verification (CoVe), an inference-time prompting pipeline for reducing hallucination in [[large-language-model]] outputs without retrieval or external tools. Starting from a baseline answer, the same model plans verification questions, answers them independently, and then regenerates a final response conditioned on the discovered agreements or inconsistencies. The authors compare joint, two-step, factored, and factor+revise execution variants using Llama 65B on list generation from [[wikidata]] and [[quest]], closed-book QA on [[multispanqa]], and biography generation scored by [[factscore]]. The main empirical claim is that separating verification from the original draft reduces copied hallucinations and yields better precision, F1, and factual precision than few-shot, instruction-tuned, and chain-of-thought baselines, though errors and added inference cost remain.

## Problem & Motivation

Large language models often generate plausible but false factual statements, especially on tail facts and long-form generations where exposure bias can compound mistakes. The paper asks whether the model can use its own latent knowledge more reliably by decomposing a broad answer into targeted factual checks. The motivation is pragmatic: many wrong long-form outputs contain claims that the same model can answer correctly when each claim is queried independently, so a structured self-verification loop may suppress hallucinations without retraining or tool augmentation.

## Method

- **Overall CoVe pipeline**: execute `4` stages with the same base LLM: `(1)` baseline response, `(2)` verification planning, `(3)` verification execution, and `(4)` final verified response.
- **Verification planning**: condition on the original query and draft answer, then generate free-form verification questions rather than templates; the paper uses few-shot demonstrations and notes this could also be done zero-shot with a strong enough instruction follower.
- **Joint execution**: plan and answer verifications in one left-to-right prompt, which is simple but lets verification answers attend to the original draft and potentially repeat its hallucinations.
- **Two-step execution**: split planning and answering into separate prompts so verification answers condition only on the generated questions, not on the original draft.
- **Factored execution**: answer each verification question in its own prompt, removing cross-question interference and allowing parallel batching; for list tasks, the generated comma-separated questions are parsed into separate prompts.
- **Factor+revise execution**: add an explicit cross-check step that classifies each original fact as `CONSistent`, `INCONSistent`, or `PARTIALLY CONSISTENT` before regeneration.
- **Prompting setup**: use greedy decoding throughout; Llama 65B is the main base model, and the prompt templates shown for biography generation use `3` few-shot examples per stage.
- **Evaluation setup**: test on `56` Wikidata list questions, `55` QUEST-derived wiki-category list questions with `8` answers each, `418` closed-book MultiSpanQA questions, and long-form biography generation evaluated with FACTSCORE.

## Key Results

- **Wikidata lists**: precision improves from `0.17` with Llama 65B few-shot to `0.36` with CoVe two-step; average hallucinated entities drop from `2.95` to `0.68`.
- **Wiki-Category lists**: precision rises from `0.12` with few-shot to `0.22` with CoVe factored, while positives stay close (`0.55` to `0.52`) and negatives shrink substantially.
- **MultiSpanQA**: F1 improves by about `23%`, from `0.39` to `0.48`, with precision `0.40 -> 0.50` and recall `0.38 -> 0.46`.
- **Biography generation**: FACTSCORE increases from `55.9` with few-shot Llama 65B to `71.4` with factor+revise, while average facts decrease from `16.6` to `12.3`.
- **Against stronger baselines**: on FACTSCORE, CoVe factor+revise (`71.4`) exceeds ChatGPT (`58.7`), PerplexityAI retrieval-based (`61.6`), and InstructGPT (`41.1`) in the reported setting.
- **Verification design matters**: on the Wiki-Category task, rule-based verification questions reach only `0.16` precision under factored execution, versus `0.22` for model-generated open questions; yes/no verification questions also underperform open-ended ones (`0.19` vs `0.22`).

## Limitations

- CoVe reduces but does not eliminate hallucinations; final answers can still contain false or misleading claims.
- The study targets directly stated factual inaccuracies, not broader failure modes such as flawed reasoning, opinionated claims, or normative judgments.
- Factored and factor+revise variants require multiple additional prompts, increasing token usage and inference cost relative to baseline generation.
- The method does not use retrieval or other external tools, so its ceiling is bounded by what the base model already knows.
- The reported gains are tied to a specific prompting setup and base models from 2023, so transfer to newer models should be re-validated.

## Concepts Extracted

- [[chain-of-verification]]
- [[hallucination-mitigation]]
- [[self-verification]]
- [[fact-verification]]
- [[few-shot-learning]]
- [[prompt-engineering]]
- [[chain-of-thought]]
- [[long-form-generation]]
- [[instruction-tuning]]
- [[retrieval-augmented-generation]]

## Entities Extracted

- [[shehzaad-dhuliawala]]
- [[mojtaba-komeili]]
- [[jing-xu]]
- [[roberta-raileanu]]
- [[xian-li]]
- [[asli-celikyilmaz]]
- [[jason-weston]]
- [[meta-ai]]
- [[eth-zurich]]
- [[llama-65b]]
- [[llama-2-70b-chat]]
- [[wikidata]]
- [[multispanqa]]
- [[quest]]
- [[factscore]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
