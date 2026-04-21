---
type: source
subtype: paper
title: "InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining"
slug: unknown-nd-instructretroinstruction-2310-07713
date: 2026-04-20
language: en
tags: [llm, retrieval, instruction-tuning, pretraining, qa]
processed: true

raw_file: raw/papers/unknown-nd-instructretroinstruction-2310-07713/paper.pdf
raw_md: raw/papers/unknown-nd-instructretroinstruction-2310-07713/paper.md
bibtex_file: raw/papers/unknown-nd-instructretroinstruction-2310-07713/paper.bib
possibly_outdated: true

authors:
  - Boxin Wang
  - Wei Ping
  - Lawrence McAfee
  - Peng Xu
  - Bo Li
  - Mohammad Shoeybi
  - Bryan Catanzaro
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.07713
doi:
url: https://arxiv.org/abs/2310.07713
citation_key: unknownndinstructretroinstruction
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper scales retrieval-augmented pretraining to `48B` parameters and then applies instruction tuning to test whether retrieval improves the resulting decoder's zero-shot generalization. Starting from a `43B` GPT model pretrained on `1.1T` tokens, the authors continue training on another `100B` tokens with Retro-style retrieval over a `1.2T`-token datastore containing `19B` chunks of length `64`. The resulting Retro 48B is then instruction-tuned on `128K` conversational samples, with the Retro encoder disabled during supervised fine-tuning so the decoder can learn to operate without reliable retrieved neighbors. Across QA and summarization benchmarks, InstructRetro consistently beats an instruction-tuned GPT baseline, and the decoder-only `43B` ablation remains close to the full `48B` model.

## Problem & Motivation

Prior retrieval-augmented language models improved perplexity and factuality, but they remained much smaller than frontier decoder-only LLMs and had not been combined seriously with instruction tuning. The paper asks whether retrieval-augmented pretraining still matters at `40B+` scale, whether its gains survive later supervised alignment, and whether the extra retrieval-specific encoder is still necessary once the decoder has already absorbed retrieval-aware representations. The motivating hypothesis is that continued pretraining with retrieval yields a stronger decoder for knowledge-intensive tasks than plain GPT pretraining followed by the same instruction-tuning recipe.

## Method

- **Foundation model scaling**: start from a pretrained GPT `43B` decoder trained on `1.1T` tokens, then convert it into Retro `48B`; the extra parameters come from Retro-specific modules, adding only about `10%` overhead beyond the shared decoder backbone.
- **Retro architecture**: keep a two-layer bidirectional Retro encoder with the same hidden size as the decoder, and inject retrieved evidence through chunk-wise cross-attention inside the autoregressive transformer.
- **Chunked retrieval setup**: split each sequence into chunks of size `m = 64` with maximum context length `n = 4096`; for chunk `C_i`, retrieve neighbors from the previous chunk `C_{i-1}` to preserve causality, using top-`k = 2` nearest chunks.
- **Retrieval database**: build a key-value datastore from the full `1.2T`-token English corpus, excluding a `1%` validation holdout; the datastore contains about `19B` chunks, keyed by BERT embeddings and valued by the original token chunks.
- **ANN index**: use [[faiss]] with `2^22` IVF centroids, `HNSW32`, optimized product quantization, and query-time settings `nprobe = 4096`, `efSearch = 32`; the paper reports about `4 ms/query` averaged per chunk on a DGX-A100 node.
- **Continued pretraining ("Retro-fitting")**: unlike the original Retro-fitting recipe, unfreeze the pretrained decoder and jointly train all parameters on an extra `100B` tokens (`~25M` samples at sequence length `4096`); for the `43B` run, Appendix A reports `lr = 9e-6`, `min_lr = 9e-7`, cosine decay, batch size `768`, and `32.5k` steps with Adam `beta1 = 0.9`, `beta2 = 0.95`.
- **Instruction tuning**: mix `128K` conversations from SODA, ELI5, Self-Instruct, Unnatural Instructions, FLAN/CoT data, OpenAssistant, Dolly, private dialogue data, and samples from the pretraining corpus; optimize only the last assistant response using teacher forcing.
- **Instruction-tuning hyperparameters**: fine-tune GPT-fitting `43B` and Retro `48B` with batch size `128`, learning rate `5e-6`, weight decay `0.01`, `1000` steps, and Adam `beta1 = 0.9`, `beta2 = 0.98`.
- **Gate ablation**: during pretraining set the Retro encoder gate to `1`; during instruction tuning and inference, set it to `0` when reliable retrieved neighbors are unavailable, effectively freezing the encoder/cross-attention path and updating only the decoder backbone.
- **Downstream RAG evaluation**: for open-domain QA, retrieve top-`k = 5` passages with task-specific retrievers such as DPR or DRAGON+ and concatenate them into the prompt; compare encoder-on `48B` and encoder-off `43B` inference.

## Key Results

- **Compute efficiency**: Retro `48B` uses `69,995` GPU hours for the extra `100B`-token stage versus `53,329` GPU hours for GPT `43B`, which is `31%` overhead on the continued-pretraining stage but only `2.58%` overhead relative to full `1.2T`-token GPT pretraining.
- **Perplexity scaling**: continued retrieval-augmented pretraining beats both pretrained GPT and GPT-fitting across all reported scales, and the paper claims Retro achieves perplexity comparable to GPT models with roughly `4x` more parameters.
- **Short-form QA / reading comprehension**: decoder-only InstructRetro `43B` improves the average score by about `7%` over InstructGPT-RAG `43B`; examples include NQ `37.0 -> 38.9` EM, NewsQA `52.4 -> 57.4` F1, SQuAD 2.0 `70.7/64.3 -> 75.6/69.3` F1/EM, and NarrativeQA `53.9 -> 60.0` F1.
- **Long-form QA**: average relative gain is about `10%`; doc2dial improves `32.87 -> 35.74`, Car #1 `58.18 -> 63.52`, Car #2 `50.88 -> 57.49`, and IT Doc `31.40 -> 34.08` F1.
- **Summarization**: average relative gain is about `16%`; GovReport improves `12.59 -> 17.46`, SummScreenFD `10.43 -> 10.93`, and QMSum `15.06 -> 15.61` on the reported ROUGE geometric mean.
- **Decoder-only surprise**: InstructRetro `43B` (encoder bypassed) is nearly tied with full InstructRetro `48B`, e.g. NQ `38.9` vs `38.6` and doc2dial `35.74` vs `35.95`, suggesting most of the downstream benefit resides in the improved decoder.
- **RAG ablation**: on NQ, InstructRetro rises from `21.8` EM without RAG to `38.9` with RAG; on TriviaQA, it rises from `54.5` to `65.6`.

## Limitations

- The paper omits the exact hidden sizes, layer counts, and attention-head counts of the `43B/48B` backbones, so replication still depends on external implementation details.
- Several long-form QA evaluations rely on proprietary datasets (two car-manual sets and one IT-documentation set), limiting reproducibility and cross-paper comparison.
- Instruction tuning explicitly disables the Retro encoder because the authors lack reliable retrieval-augmented instruction data, so the method does not fully test end-to-end retrieval-conditioned alignment.
- The strongest claims are zero-shot QA and summarization gains; the paper does not study RLHF, tool use, safety alignment, or broad interactive evaluation.
- The authors themselves note persistent risks around misinformation, privacy leakage from retrieved corpora, and amplification of training-data biases.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[instruction-tuning]]
- [[continual-pretraining]]
- [[approximate-nearest-neighbor-search]]
- [[chunked-cross-attention]]
- [[decoder-only-language-model]]
- [[cross-attention]]
- [[dense-retrieval]]
- [[autoregressive-language-model]]
- [[large-language-model]]
- [[supervised-fine-tuning]]
- [[teacher-forcing]]

## Entities Extracted

- [[boxin-wang]]
- [[wei-ping]]
- [[lawrence-mcafee]]
- [[peng-xu]]
- [[bo-li]]
- [[mohammad-shoeybi]]
- [[bryan-catanzaro]]
- [[nvidia]]
- [[retro]]
- [[faiss]]
- [[bert]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
