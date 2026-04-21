---
type: concept
title: Automatic Speech Recognition
slug: automatic-speech-recognition
date: 2026-04-20
updated: 2026-04-20
aliases: [ASR, 自动语音识别]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Automatic Speech Recognition** (自动语音识别) — the task of converting spoken audio into text transcripts, often with beam search decoding that yields multiple candidate hypotheses.

## Key Points

- The paper studies noise-robust ASR rather than clean-speech transcription, focusing on settings where acoustic corruption makes decoding uncertain.
- A Whisper Large-V2 recognizer with beam size `50` is used to generate the raw `N`-best hypotheses that RobustGER later corrects.
- The method keeps the top `N = 5` distinct ASR hypotheses after deduplication and treats them as the input structure for downstream GER.
- Noise severity is inferred indirectly from how diverse the ASR hypotheses become under different acoustic conditions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2401-10446]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2401-10446]].
