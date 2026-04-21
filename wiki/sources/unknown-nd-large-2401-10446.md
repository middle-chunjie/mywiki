---
type: source
subtype: paper
title: Large Language Models are Efficient Learners of Noise-Robust Speech Recognition
slug: unknown-nd-large-2401-10446
date: 2026-04-20
language: en
tags: [llm, asr, speech-recognition, generative-error-correction, noise-robustness]
processed: true

raw_file: raw/papers/unknown-nd-large-2401-10446/paper.pdf
raw_md: raw/papers/unknown-nd-large-2401-10446/paper.md
bibtex_file: raw/papers/unknown-nd-large-2401-10446/paper.bib
possibly_outdated: false

authors:
  - Yuchen Hu
  - Chen Chen
  - Chao-Han Huck Yang
  - Ruizhe Li
  - Chao Zhang
  - Pin-Yu Chen
  - Eng Siong Chng
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2401.10446
doi:
url: https://arxiv.org/abs/2401.10446
citation_key: unknownndlarge0446
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper studies noise-robust automatic speech recognition through generative error correction (GER) with large language models rather than conventional rescoring. It introduces RobustGER, which extracts a language-space noise embedding from the diversity of an ASR `N = 5` best list and injects it into LLaMA-Adapter prompts as a denoising signal, avoiding the instability of directly conditioning on audio features. The method combines utterance-level SBERT sentence differences, token-level edit-style embeddings, and audio-to-language knowledge distillation via mutual information neural estimation (MINE). On the proposed Robust HyPoradise benchmark, built from 113,525 training pairs across CHiME-4, VoiceBank-DEMAND, NOIZEUS, LibriSpeech-FreeSound, and RATS, RobustGER reaches up to 53.9% relative word error rate reduction and remains effective with limited training data.

## Problem & Motivation

Standard LM rescoring only reorders ASR candidates and leaves much of the error-correction capacity in the `N`-best list unused. Recent LLM-based GER can map the entire hypotheses list to the ground-truth transcript, but its gains degrade substantially under noisy conditions because the source audio noise still corrupts the hypotheses. A straightforward fix is to inject audio noise representations into LLM finetuning, but the paper argues that this suffers from a cross-modality gap that destabilizes adaptation. The authors therefore ask whether noise can be represented inside language space itself, using the diversity of noisy ASR hypotheses as a proxy for acoustic corruption, and whether that representation can teach an LLM to denoise before generating the final transcript.

## Method

- **Benchmark formulation**: the paper extends GER to noisy ASR, learning a hypotheses-to-transcription mapping `Y = M_H2T(Y_N)` with autoregressive loss `L_H2T = sum_t -log P_theta(y_t* | y_<t>*, Y_N)`.
- **Overall RobustGER pipeline**: noisy speech is decoded by ASR into an `N = 5` best list `Y_N = {Y_1, ..., Y_5}`; the model extracts a language-space noise embedding `E_LN` from that list and performs denoised generation as `Y = M_H2T(Y_N; -E_LN)`.
- **Utterance-level noise embedding**: each hypothesis is encoded with SBERT and pairwise sentence differences are concatenated as `E_LN^utt = Concat[E_sbert(Y_i) - E_sbert(Y_j)]_(i>j)`, capturing global semantic diversity caused by noise.
- **Token-level noise embedding**: hypotheses are forced-aligned with zero padding and an edit-style representation is computed as `E_edit(Y_i^ali, Y_j^ali) = sum_t [E_sbert(y_i_t^ali) - E_sbert(y_j_t^ali)]`; the final token component is `E_LN^tok = Concat E_edit(Y_i^ali, Y_j^ali)_(i>j)`.
- **Combined language-space representation**: the denoising signal is `E_LN = [E_LN^utt; E_LN^tok]`, where SBERT `all-MiniLM-L6-v2` provides `D_sbert = 384` dimensional embeddings. The paper's ablations indicate the token-level component is the more important one.
- **Audio noise distillation**: to inject real acoustic noise information into `E_LN`, the method maximizes mutual information between language embeddings and noisy Whisper embeddings using MINE, with `I_Theta(X; Z) = sup_theta E[P_XZ][psi_theta] - log E[P_X P_Z][exp(psi_theta)]`. Positive pairs use noisy audio `E_ASR(X_n)` and negative pairs use clean audio `E_ASR(X_c)`.
- **Adapter-based denoising injection**: RobustGER uses LLaMA-Adapter and subtracts a projected noise embedding from the adapter prompt, `G_l^dn = G_l - g_l^dn * T_omega(E_LN)`. Prompt length is `U = N(N-1) = 20`, and tunable layers are `L = H - 1`.
- **Training setup**: models include LLaMA-2-7B/13B, LLaMA-7B, and Falcon-7B. Finetuning uses AdamW for `2` epochs, batch size `4` with accumulation `8` (effective batch `32`), weight decay `0.02`, warmup `20%` of an epoch, learning rate `1e-2` on CHiME-4 and `5e-3` on larger datasets, MINE learning rate at `10%` of the LLM rate, and distillation weight `lambda = 0.5`. Inference uses temperature `0.2` with greedy decoding.

## Key Results

- **Main CHiME-4 result (LLaMA-2-7B)**: average WER drops from `12.8` (ASR baseline) to `5.9` with RobustGER, a `53.9%` relative reduction, outperforming LM rescoring (`12.2`), vanilla GER (`6.9`), and direct audio denoising (`6.7`).
- **Per-split CHiME-4 gains**: `test-real` improves from `12.6` to `5.6` (`55.6%`), `test-simu` from `15.4` to `8.2` (`46.8%`), `dev-real` from `10.6` to `4.1` (`61.3%`), and `dev-simu` from `12.4` to `5.8` (`53.2%`).
- **Cross-model robustness**: on CHiME-4 average WER, RobustGER reaches `6.1` with LLaMA-7B, `6.6` with Falcon-7B, and `5.3` with LLaMA-2-13B (`58.6%` relative reduction), showing the approach transfers across multiple LLM backbones.
- **Generalization across noise conditions**: on clean evaluation subsets the method still helps, reducing WER from `3.0` to `2.1` (`30.0%`) on LS-FreeSound clean data and from `1.3` to `0.7` (`46.2%`) on clean VoiceBank-DEMAND.
- **Data efficiency**: on CHiME-4, using about `5k` training pairs still yields `6.3` average WER versus `5.9` with the full `9.6k`; even `2k` pairs gives `7.2`, close to vanilla GER's `6.9`.
- **Ablations**: MINE is the best distillation strategy, improving CHiME-4 average WER from `6.3` with language denoising alone to `5.9`, better than contrastive learning (`6.1`) and teacher-student distillation (`6.3`).

## Limitations

- The gains are uneven across datasets and noise types: the paper reports much stronger improvements on CHiME-4 than on harder conditions such as RATS, where average WER only improves from `45.7` to `43.6`.
- Denoising can mislead generation when contextual reasoning should dominate, as in the failure case where RobustGER changes a ground-truth `Ms` to `Miss` because the denoising signal overweights phonetic similarity.
- The authors note that CHiME-4 may be unusually favorable because its transcripts contain many finance-related terms that LLMs can correct more easily, so some headline gains may not transfer cleanly to other domains.
- The method depends on parallel noisy and clean speech for the audio-distillation stage and requires several coupled components (ASR encoder, SBERT extractor, adapter tuner, MINE), which increases system complexity.

## Concepts Extracted

- [[large-language-model]]
- [[automatic-speech-recognition]]
- [[generative-error-correction]]
- [[language-space-noise-embedding]]
- [[audio-noise-distillation]]
- [[knowledge-distillation]]
- [[mutual-information-neural-estimation]]
- [[parameter-efficient-fine-tuning]]
- [[word-error-rate]]
- [[beam-search]]

## Entities Extracted

- [[yuchen-hu]]
- [[chen-chen-ntu]]
- [[chao-han-huck-yang]]
- [[ruizhe-li]]
- [[chao-zhang]]
- [[pin-yu-chen]]
- [[eng-siong-chng]]
- [[sentence-bert]]
- [[llama-2]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
