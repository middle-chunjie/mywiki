---
type: source
subtype: paper
title: "Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models"
slug: weller-2024-promptriever-2409-11136
date: 2026-04-20
language: en
tags: [retrieval, dense-retrieval, prompting, instruction-tuning, synthetic-data]
processed: true

raw_file: raw/papers/weller-2024-promptriever-2409-11136/paper.pdf
raw_md: raw/papers/weller-2024-promptriever-2409-11136/paper.md
bibtex_file: raw/papers/weller-2024-promptriever-2409-11136/paper.bib
possibly_outdated: false

authors:
  - Orion Weller
  - Benjamin Van Durme
  - Dawn Lawrie
  - Ashwin Paranjape
  - Yuhao Zhang
  - Jack Hessel
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2409.11136
doi: 10.48550/arXiv.2409.11136
url: http://arxiv.org/abs/2409.11136
citation_key: weller2024promptriever
paper_type: method

read_status: unread

domain: ir
---

## Summary

Promptriever is a dense bi-encoder retriever trained to treat natural-language instructions as per-query relevance specifications rather than as a fixed dataset prefix. Starting from the `491k`-query MS MARCO training set, the paper adds a synthetic instruction corpus and a new class of instruction negatives, where a passage is query-relevant but becomes non-relevant once an instruction is attached. The resulting model retains standard retrieval quality while becoming substantially more controllable: it improves instruction-following metrics on FollowIR and InstructIR, is markedly more robust to prompt phrasing, and can even gain average BEIR performance through zero-shot prompt selection. The paper's main claim is that promptability, previously associated with language models, can be transplanted into dense retrieval through instruction-aware training data.

## Problem & Motivation

Standard retrieval systems usually collapse query-document relevance into one static semantic similarity score, which makes control coarse: users often need keyword tinkering, filters, or multiple search iterations to express nuanced relevance conditions. Prior "instruction-tuned" retrievers typically prepend the same task description to every query in a dataset, so they do not really support instance-level control. The paper targets this gap by asking whether a retriever can preserve the natural-language steerability of its LLM backbone after IR training, letting a user refine relevance per query through free-form prompts such as exclusions, temporal constraints, persona-specific needs, or extra background conditions.

## Method

- **Backbone and retrieval form**: Promptriever is a [[bi-encoder]] [[dense-retrieval]] model built on an LLM backbone, primarily `LLaMA-2 7B`, so query and passage are encoded separately and scored without query-document cross-attention.
- **Base corpus**: training starts from `tevatron-msmarco-aug`, which contains roughly `491k` MS MARCO queries, one positive passage, and `30` hard negatives per query, matching the RepLLaMA setup.
- **Instruction generation**: for each `(query, positive passage)` pair, `Llama-3-70B-Instruct` generates a per-instance instruction that refines relevance while attempting to keep the original positive passage valid. Instruction styles vary across negation, persona, and background, and across short to very long formats.
- **Positive-passages safeguard**: a `FollowIR-7B` cross-encoder filters generated instructions; about `15%` of instructions make the original positive passage invalid, so those positives are replaced with a generated instruction-compatible positive passage.
- **Instruction negatives**: the paper introduces [[instruction-negative]] examples where a passage is still query-positive but instruction-negative. `gpt-4o-2024-05-13` generates `1` instruction-positive passage and `3` instruction-negative candidates per `(query, instruction)` pair, and the candidates are filtered again with `FollowIR-7B`.
- **Filtering quality**: on the triplet relevance-filtering task, average human-human agreement is `75%` over `N = 32`, while average human-model agreement with the filter is `84%` over `N = 64`, which the paper treats as sufficient for large-scale automatic filtering.
- **Training recipe**: Promptriever keeps RepLLaMA hyperparameters for apples-to-apples comparison: LoRA rank `r = 32`, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `down_proj`, `up_proj`, `gate_proj`, `bfloat16`, EOS pooling, normalization, temperature `0.01`, learning rate `1e-4`, `1` epoch, `100` warmup steps, train group size `16`, and effective batch size `128`.
- **Length settings**: the main hyperparameter deviation from RepLLaMA is query max length `304` instead of `32` to accommodate long instructions; passage length is `256` in training, and inference uses length `512` for both query and passage.
- **Joint data mixture**: the final model trains on both the original MS MARCO retrieval data and the new instruction-augmented data, using all valid instruction negatives plus sampled remaining hard negatives from the RepLLaMA dataset.
- **Prompt-time evaluation**: to test promptability directly, the paper selects from `10` generic prompts using `10` validation examples per BEIR dataset when a dev/train split exists, treating prompting as a zero-shot hyperparameter search over natural-language relevance criteria.

## Key Results

- On FollowIR and InstructIR, Promptriever improves over `RepLLaMA` by `+14.3` `p-MRR` on average and `+3.1` average `nDCG/MAP`, reaching `p-MRR = +11.2` and `nDCG = 92.1` on InstructIR.
- On InstructIR robustness, it improves `Robustness@10` by `+12.9` over RepLLaMA (`63.1` vs `50.2`) and reduces BEIR prompt variance by `44%` relative to RepLLaMA.
- On standard zero-shot BEIR without prompts, Promptriever is essentially tied with RepLLaMA (`55.0` vs `54.9` average `nDCG@10`); with the best prompt it rises to `56.4`, a `+1.4` gain, while RepLLaMA slightly drops to `54.8`.
- On the promptable BEIR setting, Promptriever improves over its no-prompt baseline on `12/13` datasets and ties on the last; prompt selection using a tiny dev set recovers the best prompt in `6/7` applicable cases.
- The ablation study shows incremental gains from each recipe component: adding real instructions raises `p-MRR` from `-0.9`/`-3.0`-range lexical baselines to `+5.7`, adding instruction negatives lifts it further to `+8.8`, and the full joint model reaches `+11.2`.
- The method generalizes beyond the LLaMA-2 backbone: with the same recipe, Mistral v1 reaches `p-MRR = +11.8`, Llama `3.1` reaches `+11.3`, and Llama `3.1 Instruct` reaches BEIR `57.2` with prompts.

## Limitations

The paper leaves several important questions open. It does not explore in-context learning or example-based prompting for retrieval, only imperative zero-shot prompts. The synthetic corpus may contain factual errors or social biases despite automatic filtering, and the paper explicitly calls for deeper audits. Prompt effectiveness is still not well explained: the authors' error-analysis attempts with BERT and bag-of-words classifiers fail to beat a majority baseline, so the mechanism behind helpful prompts remains unclear. There is also an internal reporting inconsistency: Table 3 and Table 7 appear to swap the MS MARCO scores of RepLLaMA and Promptriever, so the standard-retrieval comparison should be read as "roughly comparable" rather than as a single definitive number.

## Concepts Extracted

- [[information-retrieval]]
- [[dense-retrieval]]
- [[bi-encoder]]
- [[instruction-following]]
- [[instruction-tuning]]
- [[instruction-negative]]
- [[hard-negative]]
- [[synthetic-data]]
- [[prompt-engineering]]
- [[zero-shot-prompting]]
- [[retrieval-robustness]]
- [[relevance-judgment]]

## Entities Extracted

- [[orion-weller]]
- [[benjamin-van-durme]]
- [[dawn-lawrie]]
- [[ashwin-paranjape]]
- [[yuhao-zhang]]
- [[jack-hessel-samaya]]
- [[johns-hopkins-university]]
- [[samaya-ai]]
- [[ms-marco]]
- [[beir]]
- [[followir-7b]]
- [[repllama]]
- [[llama-2-7b]]
- [[gpt-4o]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
