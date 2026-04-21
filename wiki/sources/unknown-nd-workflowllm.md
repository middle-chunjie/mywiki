---
type: source
subtype: paper
title: "WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models"
slug: unknown-nd-workflowllm
date: 2026-04-20
language: en
tags: [agents, workflow-orchestration, tool-use, synthetic-data, benchmark]
processed: true
raw_file: raw/papers/unknown-nd-workflowllm/paper.pdf
raw_md: raw/papers/unknown-nd-workflowllm/paper.md
bibtex_file: raw/papers/unknown-nd-workflowllm/paper.bib
possibly_outdated: false
authors:
  - Shengda Fan
  - Xin Cong
  - Yuepeng Fu
  - Zhong Zhang
  - Shuyan Zhang
  - Yuanwei Liu
  - Yesai Wu
  - Yankai Lin
  - Zhiyuan Liu
  - Maosong Sun
year: 2024
venue: OpenReview
venue_type: preprint
arxiv_id:
doi:
url: https://openreview.net/pdf/61b261072d2e135d1cbe1c150891ef1b3f631413.pdf
citation_key: unknownndworkflowllm
paper_type: method
read_status: unread
domain: agents
---

## Summary

WorkflowLLM studies how to improve large language models on workflow orchestration, where a model must generate long API-based programs with branches and loops rather than short sequential tool calls. The paper builds WorkflowBench, a supervised fine-tuning corpus with `106,763` workflow instances spanning `1,503` APIs, `83` applications, and `28` categories, by combining `14,771` collected Apple Shortcuts workflows with `91,992` synthesized examples. Fine-tuning Llama-3.1-8B on this corpus produces WorkflowLlama, which reaches `39.3` CodeBLEU and `76.9%` Pass Rate on in-distribution unseen-instruction evaluation, and `35.1` CodeBLEU with `70.4%` Pass Rate on unseen APIs. The paper also shows transfer to the external planning benchmark T-Eval, where WorkflowLlama achieves `77.5` F1.

## Problem & Motivation

The paper is motivated by a gap between real-world workflow automation and current LLM-based agents. Existing agentic process automation systems can usually handle only small workflows with limited action counts and mostly sequential logic, whereas real Apple Shortcuts workflows average `70.4` actions and include nested branches and loops. The authors argue that unlocking workflow orchestration requires both larger-scale, more realistic data and training objectives that expose models to executable-looking code, API documentation, and hierarchical reasoning signals.

## Method

- **Overall framework**: WorkflowLLM is a data-centric pipeline with three stages: data collection, query expansion, and workflow generation, followed by supervised fine-tuning of Llama-3.1-8B into WorkflowLlama.
- **Collected seed data**: the authors crawl and filter `14,771` high-quality shortcuts from RoutineHub and related sources, covering `28` categories. Each shortcut is paired with title, description, iCloud source, and API metadata such as parameter names, types, defaults, and return values.
- **Workflow transcription**: property-list shortcut code is converted into an AST and then into Python-style workflow code via pre-order traversal. Each workflow is represented as `w = {Q, D, P, A}`, where `Q` is the task query, `D` API documentation, `P` the task plan, and `A` annotated Python-like actions.
- **Hierarchical thought generation**: ChatGPT generates three levels of supervision in a bottom-up manner: action-level comments `c_i`, workflow-level task plans `P`, and high-level user queries `Q`. This hierarchy is intended to improve reliability versus directly generating coarse task descriptions.
- **Query expansion**: to synthesize harder and more diverse tasks, the method samples `1-5` applications and their APIs, with roughly `⌊n/2⌋` APIs drawn from Apple's built-in actions and the rest from third-party apps. ChatGPT then generates new workflow queries conditioned on API docs, demonstrations, and target categories.
- **Workflow generation and filtering**: an annotator model fine-tuned on collected shortcuts generates workflows for synthesized queries. ChatGPT refines generated plans and code, and rule-based filtering removes samples that omit code, ignore provided APIs, or violate parameter constraints.
- **Final corpus and model training**: the synthetic set contributes `91,992` instances, producing a final WorkflowBench size of `106,763`. Both the annotator and WorkflowLlama are fine-tuned for `3` epochs with AdamW, peak learning rate `2e-5`, warm-up ratio `0.1`, batch size `32`, and maximum sequence length `8192`.
- **Evaluation**: CodeBLEU is computed as a weighted sum of BLEU, weighted n-gram match, AST match, and data-flow match with weights `0.1, 0.1, 0.4, 0.4`. A ChatGPT-based evaluator measures Pass Rate, and its labels agree with human evaluation on `268/330 = 81.2%` of sampled cases.

## Key Results

- WorkflowBench contains `106,763` instances, `1,503` APIs, `83` applications, and `28` workflow categories; the training split averages `78.5` actions and nested depth `2.7`.
- On unseen instructions (ID), WorkflowLlama reaches `39.3` overall CodeBLEU and `76.9%` Pass Rate, outperforming GPT-4o with ICL (`30.2`, `67.5%`) and vanilla Llama-3.1-8B (`24.6`, `33.0%`).
- On unseen APIs (OOD), WorkflowLlama still achieves `35.1` CodeBLEU and `70.4%` Pass Rate, compared with GPT-4o with ICL at `30.0` and `57.6%`.
- On T-Eval PLAN, WorkflowLlama scores `77.5` F1, improving over Llama-3.1-8B (`68.2`) and exceeding larger open-source baselines such as Qwen-72B (`73.4`).
- Ablations show both natural-language thoughts and synthetic data matter: removing task plans drops CodeBLEU from `39.3` to `38.2`, removing comments drops it to `38.1`, removing both drops it to `37.4`, and removing synthetic data drops it to `37.3`.

## Limitations

- The data source is concentrated on Apple Shortcuts, so API coverage and workflow styles may not transfer cleanly to broader enterprise, web, or mobile ecosystems.
- Evaluation is static rather than execution-based; the authors do not run workflows end-to-end because of registration, permissions, and API drift issues.
- The main workflow-orchestration setup bypasses API selection by directly providing the correct APIs as input, so the reported gains isolate orchestration but do not fully measure open-ended tool retrieval.
- Fine-tuning improves compliance and planning quality but can still introduce redundant actions in generated workflows, as shown in the case study.
- Bibliographic metadata in the local `paper.bib` is incomplete, so venue-level publication status is less certain than the technical content extracted from the paper itself.

## Concepts Extracted

- [[large-language-model]]
- [[workflow-orchestration]]
- [[agentic-process-automation]]
- [[robotic-process-automation]]
- [[tool-learning]]
- [[synthetic-data]]
- [[query-expansion]]
- [[fine-tuning]]
- [[in-context-learning]]
- [[out-of-distribution-generalization]]
- [[chain-of-thought-prompting]]

## Entities Extracted

- [[shengda-fan]]
- [[xin-cong]]
- [[yuepeng-fu]]
- [[zhong-zhang]]
- [[shuyan-zhang]]
- [[yuanwei-liu]]
- [[yesai-wu]]
- [[yankai-lin]]
- [[zhiyuan-liu]]
- [[maosong-sun]]
- [[workflowbench]]
- [[workflowllama]]
- [[apple-shortcuts]]
- [[routinehub]]
- [[t-eval]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
