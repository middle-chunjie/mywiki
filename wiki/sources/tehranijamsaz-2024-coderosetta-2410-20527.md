---
type: source
subtype: paper
title: "CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"
slug: tehranijamsaz-2024-coderosetta-2410-20527
date: 2026-04-20
language: en
tags: [code-translation, parallel-programming, cuda, fortran, transformer]
processed: true

raw_file: raw/papers/tehranijamsaz-2024-coderosetta-2410-20527/paper.pdf
raw_md: raw/papers/tehranijamsaz-2024-coderosetta-2410-20527/paper.md
bibtex_file: raw/papers/tehranijamsaz-2024-coderosetta-2410-20527/paper.bib
possibly_outdated: false

authors:
  - Ali TehraniJamsaz
  - Arijit Bhattacharjee
  - Le Chen
  - Nesreen K. Ahmed
  - Amir Yazdanbakhsh
  - Ali Jannesari
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2410.20527
doi:
url: http://arxiv.org/abs/2410.20527
citation_key: tehranijamsaz2024coderosetta
paper_type: method

read_status: unread

domain: llm
---

## Summary

CodeRosetta is an unsupervised encoder-decoder Transformer for code translation between general-purpose languages and high-performance computing extensions, targeting `C++ ↔ CUDA` and `Fortran ↔ C++`. The paper extends prior unsupervised code translation with structure-aware pretraining and code-specific corruption: cross-language masked language modeling, AST Entity Recognition, denoising auto-encoding with weighted token dropping and language-specific token insertion, back translation, and optional synthetic-data fine-tuning from large closed-source LLMs. The final `~0.8B`-parameter model improves over BabelTower on `C++ → CUDA` by `+2.9` BLEU, `+1.72` CodeBLEU, and `+6.05` compilation-accuracy points, reaching `98.85%` compilable outputs after lightweight post-processing. It also reaches `65.93` CodeBLEU on `Fortran → C++`, outperforming much larger general and code-specialized LLM baselines.

## Problem & Motivation

Parallel-code translation is harder than ordinary code translation because paired corpora are scarce and the target languages encode nontrivial execution semantics such as kernels, thread hierarchies, and device-specific primitives. Prior unsupervised systems such as [[transcoder]] do not explicitly model these parallel constructs, while task-specific systems such as [[babeltower]] depend on language-specific metrics that do not generalize cleanly across paradigms. CodeRosetta is motivated by the need for a bidirectional, metric-agnostic model that can learn structural and semantic differences between related languages from mostly unpaired code, while remaining small enough to train and deploy more cheaply than very large closed-source LLMs.

## Method

- **Backbone**: a `12`-layer encoder-decoder [[transformer]] with `12` attention heads per layer, hidden size `d_model = 1536`, and a pretrained [[byte-pair-encoding]] tokenizer initialized from UniXcoder; total size is `~0.8B` parameters.
- **Cross-language MLM**: pretrains on mixed-language corpora with whole-word masking rather than subtoken masking, using `15%` masking, learning rate `8e-5`, and `100` epochs to capture shared syntax and semantics across `C++`, CUDA, and Fortran.
- **AER pretraining**: introduces [[abstract-syntax-tree-entity-recognition]] after MLM. ASTs are parsed with [[tree-sitter]], then tokens are labeled with structural tags such as identifier, function, type identifier, primitive type, number literal, pointer/reference, and constant; AER uses learning rate `5e-6` for `10` epochs.
- **Decoder initialization**: copies encoder weights into the decoder before sequence-to-sequence training so the decoder starts from structure-aware code representations instead of random initialization.
- **DAE corruption**: uses [[denoising-autoencoding]] with masking `15%`, weighted token dropping `25%`, and language-specific token insertion `15%`; the corruption ratio increases by `2.5%` per epoch until a capped maximum.
- **Keyword-aware noise**: weighted dropping prioritizes language-specific reserved words such as `blockIdx`, `threadIdx`, `blockDim`, and `__global__`, forcing the model to reconstruct HPC-specific semantics rather than memorizing local syntax.
- **Language-boundary control**: inserts foreign-language tokens and prepends a special `` `<LANG>` `` token so the decoder learns to reject invalid cross-language tokens and stay within the requested target language.
- **Unsupervised refinement**: alternates [[back-translation]] with DAE so forward and reverse translation loops supply weak supervision without collapsing into identity mappings; DAE+BT training uses learning rate `5e-5` for `20` epochs.
- **Optional distillation**: performs fine-tuning on roughly `5,000` valid synthetic `C++ ↔ CUDA` pairs distilled from `100,000` GPT-4 and Gemini generations, using learning rate `5e-5` for `10` epochs.
- **Data and compute**: trains on `243,008` unpaired `C++/CUDA` files and `474,856` unpaired `C++/Fortran` files plus `282` paired `C++/Fortran` fine-tuning examples, with AdamW, batch size `16`, gradient accumulation `2`, mixed precision, and `4 ×` [[nvidia-a100]] `80GB` GPUs.

## Key Results

- `C++ → CUDA`: CodeRosetta reaches `76.90` BLEU, `78.84` CodeBLEU, `81.05` ChrF, `82.12` ROUGE-L, and `98.85%` compilation accuracy after post-processing.
- Against BabelTower on `C++ → CUDA`, it improves by `+2.90` BLEU, `+1.72` CodeBLEU, and `+6.05` compilation-accuracy points (`98.85%` vs. `92.80%`).
- Against GPT-4 on `C++ → CUDA`, it improves from `46.98` to `76.90` BLEU and from `64.45` to `78.84` CodeBLEU while slightly improving compilation accuracy (`96.10%` to `98.85%`).
- On `Fortran → C++`, fine-tuned CodeRosetta reaches `65.93` CodeBLEU, beating fine-tuned StarCoder (`61.30`) and greatly exceeding GPT-4 (`19.21`) and Gemini-Pro (`18.91`).
- Runtime evaluation on `30` translated CUDA kernels reports approximately `93%` functional correctness.
- Ablations show each component matters: removing MLM drops BLEU from `76.90` to `52.12`; removing AER drops CodeBLEU from `78.84` to `75.55`; removing special DAE noises drops CodeBLEU to `73.22`; removing back translation drops CodeBLEU to `73.18`; removing synthetic-data fine-tuning drops CodeBLEU to `71.21`.

## Limitations

- The paper studies only two translation settings, `C++ ↔ CUDA` and `Fortran ↔ C++`, so broader generalization to other HPC languages and APIs is unproven.
- Compilation accuracy still depends on lightweight post-processing; many failures come from missing type definitions, missing initializations, or unmatched braces rather than fully robust generation.
- The synthetic-data stage is optional but depends on expensive teacher models, and filtering `100,000` prompts down to `~5,000` useful pairs suggests low raw yield.
- Evaluation for `Fortran → C++` uses a very small test set of `33` programs, which limits confidence in broad deployment claims.
- The current AER tag inventory is still narrow; the paper explicitly notes future expansion toward richer control-flow and parallel-programming primitives.

## Concepts Extracted

- [[code-translation]]
- [[masked-language-modeling]]
- [[abstract-syntax-tree]]
- [[abstract-syntax-tree-entity-recognition]]
- [[denoising-autoencoding]]
- [[back-translation]]
- [[encoder-decoder-architecture]]
- [[transformer]]
- [[byte-pair-encoding]]
- [[synthetic-data]]

## Entities Extracted

- [[ali-tehranijamsaz]]
- [[arijit-bhattacharjee]]
- [[le-chen]]
- [[nesreen-k-ahmed]]
- [[amir-yazdanbakhsh]]
- [[ali-jannesari]]
- [[google-deepmind]]
- [[transcoder]]
- [[tree-sitter]]
- [[babeltower]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
