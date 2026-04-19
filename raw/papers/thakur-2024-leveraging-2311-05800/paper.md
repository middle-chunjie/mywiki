Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval
====================================================================================================

Nandan Thakur†§, Jianmo Ni†♡, Gustavo Hernández Ábrego◆  
John Wieting♡, Jimmy Lin§, Daniel Cer†◆  
  
◆Google Research, ♡Google DeepMind, §University of Waterloo  
∗Work done while Nandan was a student researcher at Google Research.†Correspondence to: Nandan Thakur <nandan.thakur@uwaterloo.ca>, Jianmo Ni <jianmon@google.com>, Daniel Cer <cer@google.com>.

###### Abstract

Dense retrieval models have predominantly been studied for English, where models have shown great success, due to the availability of human-labeled training pairs.
However, there has been limited success for multilingual retrieval so far, as training data is uneven or scarcely available across multiple languages.
Synthetic training data generation is promising (e.g., InPars or Promptagator), but has been investigated only for English. Therefore, to study model capabilities across both cross-lingual and monolingual retrieval tasks, we develop SWIM-IR, a synthetic retrieval training dataset containing 33 (high to very-low resource) languages for training multilingual dense retrieval models without requiring any human supervision.
To construct SWIM-IR, we propose SAP (summarize-then-ask prompting), where the large language model (LLM) generates a textual summary prior to the query generation step. SAP assists the LLM in generating informative queries in the target language.
Using SWIM-IR, we explore synthetic fine-tuning of multilingual dense retrieval models and evaluate them robustly on three retrieval benchmarks: XOR-Retrieve (cross-lingual), XTREME-UP (cross-lingual) and MIRACL (monolingual).
Our models, called SWIM-X, are competitive with human supervised dense retrieval models, e.g., mContriever, finding that SWIM-IR can cheaply substitute for expensive human-labeled retrieval training data.111Our dataset and trained models are available at: [github.com/google-research-datasets/swim-ir](https://github.com/google-research-datasets/swim-ir "")

Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval

  
Nandan Thakur††thanks: ∗Work done while Nandan was a student researcher at Google Research.†Correspondence to: Nandan Thakur <nandan.thakur@uwaterloo.ca>, Jianmo Ni <jianmon@google.com>, Daniel Cer <cer@google.com>. †§, Jianmo Ni†♡, Gustavo Hernández Ábrego◆John Wieting♡, Jimmy Lin§, Daniel Cer†◆◆Google Research, ♡Google DeepMind, §University of Waterloo

| Dataset | Q Gen. | Cross. | Mono. | # L | Domain | # Train |
| --- | --- | --- | --- | --- | --- | --- |
| NeuCLIR | Human | EN$\rightarrow$L | L$\rightarrow$L | 3 | News (hc4) | $\boldsymbol{\times}$ |
| MKQA | Human | L$\rightarrow$EN | $\boldsymbol{\times}$ | 26 | Wikipedia | 10K |
| mMARCO | Translate | $\boldsymbol{\times}$ | L$\rightarrow$L | 13 | MS MARCO | 533K |
| Mr.TyDI | Human | $\boldsymbol{\times}$ | L$\rightarrow$L | 11 | Wikipedia | 49K |
| MIRACL | Human | $\boldsymbol{\times}$ | L$\rightarrow$L | 18 | Wikipedia | 726K |
| JH-POLO | GPT-3 | EN$\rightarrow$L | $\boldsymbol{\times}$ | 3 | News (hc4) | 78K |
| SWIM-IR | PaLM 2 | L$\rightarrow$EN | L$\rightarrow$L | 33 | Wikipedia | 28M |

*Table 1: Existing datasets contain only up to a few thousand training pairs, as scaling human-labeled annotations is expensive and cumbersome. In our work, we construct SWIM-IR, a “synthetic” multilingual dataset with 28 million LLM-generated training pairs across 33 languages; (Q Gen.) denotes the query generation task; (Cross. and Mono.) denotes the retrieval task and query$\rightarrow$document language pair; (# L and # Train) denotes the language count and training pairs available.*

<img src='x1.png' alt='Refer to caption' title='' width='332' height='73' />

*Figure 1: An illustration of SAP (Summarize-then-Ask Prompting) versus standard prompting for English query generation on English Wikipedia. SAP assists the large language model (LLM) in improving multilingual query generation by identifying the relevant sections of the input passage (highlighted in red) using extractive summarization (yellow box) as an intermediate reasoning step.*

1 Introduction
--------------

Dense retrieval models have demonstrated impressive performance in ad-hoc information retrieval (IR), e.g., web search, outperforming traditional retrieval systems such as BM25 *(Karpukhin et al., [2020](#bib.bib23 ""); Lin et al., [2021](#bib.bib28 ""); Ni et al., [2022](#bib.bib38 ""); Neelakantan et al., [2022](#bib.bib36 ""), *inter alia*)*.
A major reason for the success lies in the availability of large-scale supervised training datasets in English, such as MS MARCO *Nguyen et al. ([2016](#bib.bib37 ""))* or NQ *Kwiatkowski et al. ([2019](#bib.bib25 ""))*, and coupled with effective training strategies, such as custom hard-negative mining *Xiong et al. ([2021](#bib.bib58 "")); Lin et al. ([2023](#bib.bib29 ""))*, or teacher distillation *Hofstätter et al. ([2021](#bib.bib18 "")); Ren et al. ([2021](#bib.bib43 ""))*.

However, there is limited exploration for dense retrieval models in multilingual retrieval222Throughout the paper, we use “multilingual retrieval” to collectively denote both cross-language, i.e., cross-lingual and within language, i.e., monolingual retrieval tasks. due to the uneven and low distribution of supervised training data for languages apart from English *Reimers and Gurevych ([2020](#bib.bib42 "")); Ruder ([2022](#bib.bib45 "")); Feng et al. ([2022](#bib.bib13 "")); Wieting et al. ([2023](#bib.bib57 ""))*.
Collecting human annotations for training data generation is not scalable, as it is cumbersome to search and hire native speakers, check their language proficiency, and teach them. Additionally, human annotators are expensive, requiring large annotation budgets for creating a sufficient amount of human-labeled training pairs (cf.[Figure 5](#S6.F5 "Figure 5 ‣ 6 Background and Related Work ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")).

Recently, Large Language Models (LLMs) such as GPT-3 *Brown et al. ([2020](#bib.bib7 ""))*, LLAMA-2 *Touvron et al. ([2023](#bib.bib52 ""))*, or PaLM 2 *Anil et al. ([2023](#bib.bib1 ""))* have demonstrated impressive performance on a wide range of natural language tasks *Chang et al. ([2023](#bib.bib8 "")); Zhang et al. ([2023a](#bib.bib61 ""))*. An efficient alternative to human annotation is generating synthetic queries produced by prompting an LLM, only assuming access to a set of unlabeled passages *Bonifacio et al. ([2022](#bib.bib5 "")); Dai et al. ([2023](#bib.bib10 ""))*. However, exploration in prior research work was limited to English.
Multilingual question generation is a complex task *Wang et al. ([2021](#bib.bib54 ""))*. The task requires understanding of semantic mappings of words across languages, similar in machine translation *Forcada ([2002](#bib.bib14 "")); Tan et al. ([2019](#bib.bib49 "")); Zhu et al. ([2023](#bib.bib65 ""))*. As illustrated in [Figure 1](#S0.F1 "Figure 1 ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), standard prompting techniques can lead to the LLM generating extractive or uninformative333Uninformative denotes a query that can be easily answered using the first (or last) few words in the passage. queries.

<img src='x2.png' alt='Refer to caption' title='' width='332' height='50' />

*Figure 2: An illustration of cross-lingual SWIM-IR dataset construction procedure. (1) Sample $N$ passages from English Wikipedia using stratified sampling for each target language out of a total of $L$ languages; (2) Feed a single input passage along with few-shot exemplars to the LLM with SAP (summarize-then-ask prompting); (3 \& 4) Parse the LLM output to receive the synthetic query in target language (above in Bengali); (5) Fine-tune a multilingual dense retrieval model (SWIM-X) with training data combined for all languages, i.e., $N$$\times$$L$ pairs.*

To improve the quality of the generated query, we propose SAP (Summarize-then-Ask Prompting), where we prompt the LLM to break down the question generation into two stages: (i) summary extraction which identifies the relevant information from the long input passage and extracts the sentences as the summary, and (ii) query generation which generates a multilingual query relevant for the input passage, using the extracted summary generated as an intermediate step. SAP highlights the relevant information within a passage (occasionally long) and produces difficult (i.e., informative) queries in the target language.

In our work, we utilize PaLM 2 *Anil et al. ([2023](#bib.bib1 ""))* for multilingual question generation. The generated query paired with the original passage from Wikipedia is to construct the SWIM-IR (Synthetic WIkipedia-based Multilingual IR) dataset.
SWIM-IR spans 33 diverse languages, including high and very-low resource languages. SWIM-IR provides synthetic training pairs for improving dense retrieval models without requiring any human supervision. SWIM-IR is one of largest multilingual synthetic dataset, providing 28 million training pairs (cf. [Table 1](#S0.T1 "Table 1 ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")).

In our work, we fine-tune synthetic-only multilingual (monolingual and cross-lingual) dense retrieval models, called SWIM-X, using mT5 (base) *Xue et al. ([2021](#bib.bib60 ""))* and SWIM-IR. We compare SWIM-X with models trained with human supervision by changing only the training dataset while keeping other, i.e., model parameters and training settings unchanged. We evaluate three standard multilingual retrieval benchmarks (two cross-lingual and one monolingual).
On XOR-Retrieve *Asai et al. ([2021a](#bib.bib2 ""))*, SWIM-X outperforms the previous best-supervised baseline (mContriever-X), by 7.1 points at Recall@5kt. On MIRACL *Zhang et al. ([2023c](#bib.bib64 ""))*, a monolingual retrieval benchmark, SWIM-X is inferior to the mContriever baseline (trained with four hard negatives) by 9.0 points at nDCG@10, which shows room for future improvements. On XTREME-UP *Ruder et al. ([2023](#bib.bib46 ""))*, a challenging benchmark with 20 underrepresented Indo-European languages, SWIM-X greatly elevates model performance in low-resource languages by 11.7 points at MRR@10.

In summary, the contributions of our work are as follows:
(1) We develop SWIM-IR, a synthetic multilingual IR dataset containing 28 million training pairs for 33 languages to cheaply substitute for expensive human-labeled training data. (2) We introduce SAP using PaLM 2 *Anil et al. ([2023](#bib.bib1 ""))* to improve multilingual question generation. (3) We fine-tune dense retrieval models only on synthetic data (SWIM-IR). Our models outperform supervised counterparts on XOR-Retrieve and XTREME-UP while remaining competitive on MIRACL.

2 SWIM-IR Dataset Overview
---------------------------

In our dataset overview, we describe the design formulation of SAP for multilingual query generation (§[2.1](#S2.SS1 "2.1 SAP Design Formulation ‣ 2 SWIM-IR Dataset Overview ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")), data construction details (§[2.2](#S2.SS2 "2.2 SWIM-IR Dataset Construction ‣ 2 SWIM-IR Dataset Overview ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")), and statistics and analysis (§[2.3](#S2.SS3 "2.3 Dataset Statistics and Human Validation ‣ 2 SWIM-IR Dataset Overview ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")).

### 2.1 SAP Design Formulation

Multilingual question generation is not a trivial task as it requires a deep understanding of the passage content and its own translations across different languages *Wang et al. ([2021](#bib.bib54 ""))*. Passages are often lengthy and contain information about multiple topics. Using the complete passage can even hallucinate models by generating non-meaningful queries, which affects retrieval performance *Gospodinov et al. ([2023](#bib.bib17 ""))*.

To break down the task complexity of multilingual question generation for improving question quality, we implement summarize-then-ask prompting (SAP). As illustrated in [Figure 1](#S0.F1 "Figure 1 ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), we identify the relevant information within a passage by asking the LLM to generate an extractive summary and use it as an intermediate step for generating informative queries. The procedure can be split into two stages, which we describe below.

(i) Summary extraction. The LLM constructs an extractive summary $e_{s}$ of the input passage $p_{s}$, where $s$ denotes the source language. The summary captures the most relevant information from within a passage (which occasionally may be long) acting as a useful intermediate signal for the LLM to generate a multilingual question in the later stage. We denote the first stage as $e_{s}\=\mathrm{LLM}(p_{s};\theta^{1},\cdots,\theta^{k})$, where ($\theta^{1},\cdots,\theta^{k}$) denotes the $k$ few-shot prompt exemplars444Multilingual query generation requires few-shot prompt exemplars. As our experiments show in (§[4](#S4 "4 Ablation Studies ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")), zero-shot prompting often generates unparseable outputs with PaLM 2. containing the passage, summary in the source language $s$ and the query in the target language $t$.555In our work, we did not use abstractive summarization, as LLMs have notoriously been shown to hallucinate and generate factual inconsistencies in their output generations *Maynez et al. ([2020](#bib.bib34 "")); Liu et al. ([2023](#bib.bib30 ""))*.

(ii) Query Generation. Next, the LLM combines the summary $e_{s}$ as an intermediate step, with the original input passage $p_{s}$, highlighting the relevant information required for composing the query ($q_{t}$) in the target language $t$. We denote this stage as $q_{t}\=\mathrm{LLM}(e_{s},p_{s};\theta^{1},\cdots,\theta^{k})$, where extractive summary $e_{s}$, input passage $p_{s}$ and $k$-shot exemplars come from the first stage.

### 2.2 SWIM-IR Dataset Construction

For constructing SWIM-IR, we only require an unlabeled corpus of passages. We can construct either cross-lingual or monolingual training pairs. We provide an overview of cross-lingual generation procedure is shown in [Figure 2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"). Prompt examples are shown in Appendix (§[C.3](#A3.SS3 "C.3 SWIM-IR Prompts ‣ Appendix C SWIM-IR Extra Material ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")).

Cross-lingual. The goal is to generate a query in target language $t$ from an input passage in English (source language $s$).
We use a stratified sampling algorithm (for more details, refer to §[E.4](#A5.SS4 "E.4 Stratified Sampling Strategy for SWIM-IR ‣ Appendix E Additional Technical Details ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") in Appendix) to
sample a maximum of one million passages for each target language $t$ from the English Wikipedia corpus used in XOR-Retrieve *Clark et al. ([2020](#bib.bib9 "")); Asai et al. ([2021a](#bib.bib2 ""))* or XTREME-UP *Ruder et al. ([2023](#bib.bib46 ""))*.
Next, we construct five prompt exemplars in English, where we generate the summaries and queries in English. Next, Google Translate666Google Translate: [translate.google.com](https://translate.google.com/ "") is used for translating the exemplar queries in English to other languages. Finally, we explain our question generation task as an instruction in our prompt, the language, and the 5-shot exemplars and provide the prompt as an input to the LLM *Anil et al. ([2023](#bib.bib1 ""))* with SAP. The LLM outputs the summary in English and the multilingual query in the target language $t$.

Monolingual. The goal is to generate a query in the same language as the input passage ($s\=t$). We follow the setting identical to the cross-lingual task. We first sample one million passages (if available) for each language-specific Wikipedia corpus available in MIRACL *Zhang et al. ([2023c](#bib.bib64 ""))*. For 16 out of the 18 languages, MIRACL contains a training split. we carefully select three training pairs as our prompt exemplars.777As language-specific passages consume more tokens, e.g., Telugu, to save computational budget, we rely only on 3-shot exemplars (instead of 5) for the monolingual task. For two languages with no training data: German (de) and Yoruba (yo), we ourselves create our prompt exemplars. Next, for generating exemplar summaries in target language, we use Google Bard.888Google Bard: [bard.google.com](https://bard.google.com/ "") Finally, in the prompt we explain our question generation task, the language, and our 3-shot exemplars with SAP. The LLM outputs both the summary and query in the same language.

| Lang. (ISO) | fluency ($\uparrow$) | | | adequacy ($\uparrow$) | | | language ($\uparrow$) | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rating ($\rightarrow$) | 0 | 1 | 2 | 0 | 1 | 2 | 0 | 1 | 2 |
| English (en) | 2% | 3% | 95% | 2% | 13% | 85% | 0% | 0% | 100% |
| Spanish (es) | 1% | 10% | 89% | 14% | 12% | 74% | 1% | 0% | 99% |
| Chinese (zh) | 7% | 19% | 74% | 7% | 30% | 63% | 0% | 0% | 100% |
| Hindi (hi) | 12% | 5% | 83% | 6% | 19% | 75% | 0% | 0% | 100% |
| Bengali (bn) | 6% | 4% | 90% | 10% | 14% | 76% | 1% | 0% | 99% |

*Table 2: Human validation statistics on SWIM-IR. Annotators (native speakers) evaluate the query quality on a three-level rating scale (0/1/2) measured for (i) fluency, (ii) adequacy and (iii) language.*

### 2.3 Dataset Statistics and Human Validation

SWIM-IR is a synthetic retrieval dataset that spans 33 different languages, including both cross- and monolingual training pairs. All queries in SWIM-IR are synthetic and LLM-generated. For our LLM choice, we use PaLM 2 *Anil et al. ([2023](#bib.bib1 ""))* with small size (S) for query generation. Detailed statistics, including the amount of training pairs for each language can be found in [Table 7](#A7.T7 "Table 7 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval").

Content Filtering. LLMs have been shown to generate undesirable content, particularly under conditions that prime the model with material targeted at drawing out any negative patterns or associations in the model’s training data *Gehman et al. ([2020](#bib.bib16 "")); Bender et al. ([2021](#bib.bib4 ""))*. We originally hoped that sampled Wikipedia passages would provide almost entirely safe material for prompting LLMs. However, for each combination of query-passage languages within SWIM-IR, we discovered that between 6–10% of the pairs contained sensitive subjects and adult content (i.e., weapons; violence and abuse; accidents and disasters; death and tragedy; war and conflict). We used the Google Cloud Natural Language content classification categories999[cloud.google.com/natural-language/docs/categories](https://cloud.google.com/natural-language/docs/categories "") to identify and remove pairs when either the original sampled passage or the resulting LLM generated query has a content classification of either /Adult or any of the /Sensitive Subjects labels.

Human validation. We conduct an annotation study to evaluate the quality of generated queries in the SWIM-IR dataset for a subset of the languages.101010Finding native speakers for all of the 33 languages, who are willing to annotate is cumbersome and expensive. We evaluate each query across a three-level rating scale measuring fluency, adequacy and language.
From [Table 2](#S2.T2 "Table 2 ‣ 2.2 SWIM-IR Dataset Construction ‣ 2 SWIM-IR Dataset Overview ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), we find that the generated quality of English is found best. At least 86% of the query generated have found to be adequate and 88% to be fluent (ratings 1 and 2) across all evaluated languages, denoting queries available in the SWIM-IR dataset are fluent and adequate. For more details on our human validation study including results, please refer to Appendix (§[D](#A4 "Appendix D Human Validation ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")).

3 Experiments
-------------

| Model | PLM | PT | Finetune | Recall@5kt | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | (Datasets) | Avg. | Ar | Bn | Fi | Ja | Ko | Ru | Te |
| Existing Supervised Baselines (Prior work) | | | | | | | | | | | |
| Dr. DECR Li et al. ([2022](#bib.bib27 "")) | XLM-R | WikiM | NQ + XOR∗ | 73.1 | 70.2 | 85.9 | 69.4 | 65.1 | 68.8 | 68.8 | 83.2 |
| mDPR Asai et al. ([2021a](#bib.bib2 "")) | mBERT | — | XOR | 50.2 | 48.9 | 60.2 | 59.2 | 34.9 | 49.8 | 43.0 | 55.5 |
| mBERT + xQG Zhuang et al. ([2023](#bib.bib66 "")) | mBERT | — | XOR | 53.5 | 42.4 | 54.9 | 54.1 | 33.6 | 52.3 | 33.8 | 52.5 |
| Google MT + DPR Asai et al. ([2021a](#bib.bib2 "")) | BERT | — | NQ | 69.6 | 69.6 | 82.2 | 62.4 | 64.7 | 68.8 | 60.8 | 79.0 |
| OPUS MT + DPR Asai et al. ([2021a](#bib.bib2 "")) | BERT | — | NQ | 50.6 | 52.4 | 62.8 | 61.8 | 48.1 | 58.6 | 37.8 | 32.4 |
| Zero-shot baselines (English-only supervision) | | | | | | | | | | | |
| mContriever | mT5 | mC4 | — | 38.9 | 35.9 | 33.9 | 43.6 | 34 | 35.1 | 45.1 | 44.5 |
| mDPR (En) | mT5 | — | MS MARCO | 39.3 | 34.3 | 35.5 | 45.2 | 40.2 | 36.5 | 43.9 | 39.5 |
| mContriever (En) | mT5 | mC4 | MS MARCO | 44.0 | 37.5 | 38.2 | 50.6 | 41.1 | 37.2 | 49.8 | 53.8 |
| Supervised Baselines (Cross-lingual supervision) | | | | | | | | | | | |
| mDPR-X | mT5 | — | XOR | 53.6 | 51.5 | 63.5 | 52.5 | 45.6 | 52.3 | 43.0 | 66.8 |
| mContriever-X | mT5 | mC4 | XOR | 55.3 | 52.1 | 68.1 | 54.5 | 47.7 | 50.5 | 50.2 | 64.3 |
| mDPR-X | mT5 | — | MS MARCO + XOR | 58.2 | 55.3 | 70.1 | 56.7 | 49.8 | 55.8 | 50.6 | 69.3 |
| mContriever-X | mT5 | mC4 | MS MARCO + XOR | 59.6 | 54.7 | 73.4 | 57.0 | 53.1 | 56.5 | 51.5 | 71.0 |
| Synthetic Baselines (Our work) | | | | | | | | | | | |
| SWIM-X (500K) | mT5 | — | SWIM-IR | 59.0 | 54.0 | 67.4 | 59.2 | 52.7 | 55.1 | 54.4 | 70.2 |
| SWIM-X (500K) | mT5 | mC4 | SWIM-IR | 63.0 | 57.0 | 71.1 | 61.8 | 56.8 | 60.7 | 63.3 | 70.2 |
| SWIM-X (7M) | mT5 | — | SWIM-IR | 65.1 | 57.9 | 75.0 | 65.6 | 59.3 | 58.9 | 64.6 | 74.4 |
| SWIM-X (7M) | mT5 | mC4 | SWIM-IR | 66.7 | 61.2 | 77.0 | 65.0 | 62.2 | 62.8 | 65.4 | 73.5 |

*Table 3: Experimental results showing Recall@5kt for cross-lingual retrieval on XOR-Retrieve dev *Asai et al. ([2021a](#bib.bib2 ""))*; (PLM) denotes the pretrained language model; (PT) denotes the pretraining dataset; (∗) Dr.DECR is fine-tuned in a complex training setup across more datasets ($\mathsection$[3.2](#S3.SS2 "3.2 Experimental Methods ‣ 3 Experiments ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")); WikiM denotes WikiMatrix *Schwenk et al. ([2021](#bib.bib48 ""))*; XOR denotes XOR-Retrieve; SWIM-X (ours) is fine-tuned on 500K and 7M synthetic data.*

### 3.1 Datasets and Metrics

XOR-Retrieve *Asai et al. ([2021a](#bib.bib2 ""))* is a cross-lingual open retrieval training and evaluation task within TyDi-QA *Clark et al. ([2020](#bib.bib9 ""))*.
XOR-Retrieve contains 15K human annotated relevant passage-query pairs in the training set with one hard negative and 2K passage-answer pairs in the dev set.
The corpus $C$ contains 18.2M passages with a maximum of 100 word tokens from the English Wikipedia. The queries are multilingual and cover seven languages.
We evaluate our models using recall at m kilo-tokens, i.e., Recall@mkt, which computes the fraction of queries for which the minimal answer is contained within the top $m$ thousand tokens of the retrieved passages.
Following prior work in *Asai et al. ([2021a](#bib.bib2 ""))*, we evaluate our models at Recall@5kt and Recall@2kt.

MIRACL *Zhang et al. ([2023c](#bib.bib64 ""))* is a monolingual open retrieval evaluation task containing 18 languages.
MIRACL was developed on top of Mr.TyDi *Zhang et al. ([2021](#bib.bib62 ""))*, and covers more languages and provides denser judgments by human annotators.
The test set is not publicly released, hence in this paper we evaluate using the dev set.
The training set contains 88,288 pairs, with the exception of Yoruba (yo) and German (de) which do not have any training data available. The authors also provide labeled hard negatives for the training query-passage pairs.
The dev set contains around 13,495 query-passage pairs.
The corpus $C$ in MIRACL are language-specific Wikipedia articles with various sizes starting from smallest, Yoruba (yo) with 49K passages, till the largest, English (en) with 39.2M passages. Following prior work in *Zhang et al. ([2023c](#bib.bib64 ""))* and *Kamalloo et al. ([2023](#bib.bib22 ""))*, we evaluate our models at nDCG@10 and Recall@100.

| Model | Avg. | ar | bn | en | es | fa | fi | fr | hi | id | ja | ko | ru | sw | te | th | zh | de | yo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing Supervised Baselines (Prior work) | | | | | | | | | | | | | | | | | | | |
| BM25 | 38.5 | 48.1 | 50.8 | 35.1 | 31.9 | 33.3 | 55.1 | 18.3 | 45.8 | 44.9 | 36.9 | 41.9 | 33.4 | 38.3 | 49.4 | 48.4 | 18.0 | 22.6 | 40.6 |
| mDPR | 41.8 | 49.9 | 44.3 | 39.4 | 47.8 | 48.0 | 47.2 | 43.5 | 38.3 | 27.2 | 43.9 | 41.9 | 40.7 | 29.9 | 35.6 | 35.8 | 51.2 | 49.0 | 39.6 |
| Hybrid | 56.6 | 67.3 | 65.4 | 54.9 | 64.1 | 59.4 | 67.2 | 52.3 | 61.6 | 44.3 | 57.6 | 60.9 | 53.2 | 44.6 | 60.2 | 59.9 | 52.6 | 56.5 | 37.4 |
| Cohere-API | 54.2 | 66.7 | 63.4 | 50.1 | 50.7 | 48.4 | 67.5 | 44.3 | 57.3 | 50.5 | 51.6 | 54.6 | 47.7 | 54.3 | 63.8 | 60.6 | 38.9 | 41.4 | 62.9 |
| Zero-shot baselines (English-only supervision) | | | | | | | | | | | | | | | | | | | |
| mDPR (En) | 39.8 | 49.7 | 50.1 | 35.4 | 35.3 | 39.3 | 48.2 | 31.3 | 37.4 | 35.6 | 38.9 | 44.1 | 36.1 | 33.8 | 49.2 | 50.6 | 34.7 | 32.1 | 34.4 |
| mContriever (En) | 37.8 | 49.1 | 48.4 | 32.7 | 33.3 | 37.1 | 48.4 | 27.0 | 35.9 | 32.7 | 34.1 | 40.2 | 35.1 | 44.5 | 46.2 | 45.0 | 27.5 | 29.7 | 33.7 |
| Supervised Baselines (Monolingual supervision) | | | | | | | | | | | | | | | | | | | |
| mDPR-X | 39.6 | 52.8 | 57.1 | 30.2 | 24.7 | 37.6 | 46.1 | 26.4 | 27.8 | 37.3 | 42.9 | 38.3 | 34.9 | 53.7 | 68.4 | 58.2 | 34.9 | 19.2 | 22.2 |
| mContriever-X | 55.4 | 66.4 | 68.4 | 44.2 | 42.8 | 48.9 | 65.2 | 46.2 | 45.0 | 45.8 | 56.8 | 58.8 | 51.2 | 67.7 | 79.0 | 70.7 | 49.4 | 42.3 | 48.4 |
| Synthetic Baselines (Our work) | | | | | | | | | | | | | | | | | | | |
| SWIM-X (180K) | 46.4 | 60.2 | 57.1 | 34.7 | 33.4 | 36.3 | 40.6 | 64.3 | 33.0 | 39.5 | 40.8 | 43.3 | 49.7 | 40.0 | 55.9 | 56.3 | 63.3 | 50.2 | 36.5 |

*Table 4:  Experimental results for monolingual retrieval on MIRACL dev *Zhang et al. ([2023c](#bib.bib64 ""))*. All scores denote nDCG@10; (Hyb.) denotes Hybrid retriever with ranked fusion of three retrievers: mDPR, mColBERT and BM25; BM25, mDPR and Hybrid scores taken from *Zhang et al. ([2023c](#bib.bib64 ""))*; Cohere-API is used as a reranker on top of 100 BM25 results, taken from *Kamalloo et al. ([2023](#bib.bib22 ""))*. SWIM-X (ours) is fine-tuned on 180K synthetic data.*

XTREME-UP *Ruder et al. ([2023](#bib.bib46 ""))* contains diverse information-access and user-centric tasks focused on under-represented languages. In this paper, we evaluate cross-lingual retrieval task containing 5,280 query-passage pairs in the training set. The corpus $C$ contains 112,426 passages sampled from TyDi-QA *Clark et al. ([2020](#bib.bib9 ""))*. The test set contains 10,705 query-passage pairs for evaluation. The cross-language retrieval for QA task contains 20 under-represented Indic languages. Following prior work in *Ruder et al. ([2023](#bib.bib46 ""))*, we evaluate our models at MRR@10.

### 3.2 Experimental Methods

Baselines. Following common practice across all datasets, we evaluate three broad range of baselines: (1) Zero-shot: where the model is fine-tuned only for English training data such as MS MARCO *Nguyen et al. ([2016](#bib.bib37 ""))* or NQ *Kwiatkowski et al. ([2019](#bib.bib25 ""))*. (2) Gold FT: models denoted by “X” (model-X) are fine-tuned on language-specific human labeled, i.e., gold training data. (3) Synthetic FT: models denoted by “SWIM-X” are fine-tuned without any gold training data, relying only on our SWIM-IR training data. Additionally, we also report the amount of synthetic pairs used, e.g., 500K for training a SWIM-X (500K) model.

Model Choices. For our dense retrieval models, we adapt DPR *Karpukhin et al. ([2020](#bib.bib23 ""))* to the multilingual setting.
Next, we include mContriever *Izacard et al. ([2022](#bib.bib19 ""))* which adopts additional pre-training with contrastive loss based on unsupervised data prepared from mC4*Xue et al. ([2021](#bib.bib60 ""))*.

Existing Baselines. For XOR-Retrieve, we include Dr. DECR *Li et al. ([2022](#bib.bib27 ""))*, a cross-lingual ColBERT *Khattab and Zaharia ([2020](#bib.bib24 ""))* fine-tuned on large amounts of supervised data in a computationally expensive setup of knowledge distillation with English ColBERTv2 *Santhanam et al. ([2022](#bib.bib47 ""))*. xQG *Zhuang et al. ([2023](#bib.bib66 ""))* involving cross-language query generation and concatenating the queries along with the passage representation. We include two-stage translation baselines, adding Google Translate and Opus-MT from *Asai et al. ([2021a](#bib.bib2 ""))*. For MIRACL, we include the official BM25, mDPR and Hybrid (combining BM25, mDPR and mColBERT) baseline available in *Zhang et al. ([2023c](#bib.bib64 ""))*, and the Cohere-API model is used as a reranker on top of 100 BM25 results available in *Kamalloo et al. ([2023](#bib.bib22 ""))*.

Implementation Details. We replicate the mContriever and mDPR zero-shot baselines by initializing from a multilingual T5-base checkpoint *Xue et al. ([2021](#bib.bib60 ""))* and fine-tune on the English MS MARCO dataset, in a setup similar to *Ni et al. ([2022](#bib.bib38 ""))*. Similarly, mContriever-X and mDPR-X have been additionally fine-tuned on training split available for each dataset. For additional technical details on supervised baselines, please refer to Appendix (§[E.2](#A5.SS2 "E.2 Baseline FT Models ‣ Appendix E Additional Technical Details ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")). During the mContriever pre-training, we set the batch size to 8192,
learning rate to $1e^{-3}$ and pre-train for 600K steps with mC4 *Xue et al. ([2021](#bib.bib60 ""))*. For additional details on pretraining, refer to Appendix (§[E.1](#A5.SS1 "E.1 mContriever Pretraining ‣ Appendix E Additional Technical Details ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")).
Next, for our synthetic baselines, we pretrain on mC4 and fine-tune on SWIM-IR training pairs with in-batch negatives with contrastive loss *van den Oord et al. ([2018](#bib.bib53 ""))*. During fine-tuning, we set the batch size to $4096$, learning rate to $1e^{-}3$ and fine-tune for about 5K to 50K steps, depending upon the amount of training data available.
In all our experiments, we use the PaLM 2 Small *Anil et al. ([2023](#bib.bib1 ""))* to generate the cross-language multilingual queries due to its rather low-cost and quick inference. For additional hyperparameter choices and training details, refer to Appendix (§[E.3](#A5.SS3 "E.3 Synthetic FT models ‣ Appendix E Additional Technical Details ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")). For all our experiments, we use the T5X Retrieval framework for training and evaluation *Ni et al. ([2022](#bib.bib38 ""))*.

### 3.3 Experimental Results

XOR-Retrieve. [Table 3](#S3.T3 "Table 3 ‣ 3 Experiments ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") shows that SWIM-X (7M), fine-tuned on 7M synthetic pairs (max. of 1M per language) outperforms the best FT model, mContriever-X, by 7.1 points on Recall@$5kt$. Without mC4 pre-training, our SWIM-X (7M) performance drops by only 1.6 points. We also evaluate SWIM-X (500k), a limited-budget baseline fine-tuned on 500k training pairs, that outperforms mContriever-X by 3.6 points. Few existing baselines outperform SWIM-X, however, the comparison is not fair, as Dr. DECR is a multilingual ColBERT *Khattab and Zaharia ([2020](#bib.bib24 ""))* model, which is computationally expensive at runtime *Thakur et al. ([2021](#bib.bib51 ""))* and Google MT + DPR rely on a powerful Google Translate system for translation.

MIRACL. [Table 4](#S3.T4 "Table 4 ‣ 3.1 Datasets and Metrics ‣ 3 Experiments ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") shows that SWIM-X (180K) model is competitive on MIRACL. SWIM-X (180K) outperforms the best zero-shot model, by 6.6 points on nDCG@10. However, SWIM-X is unable to outperform mContriever-X, fine-tuned on about 90K human-labeled training pairs with up to four hard negatives available in MIRACL. Whereas SWIM-X have not been optimized with hard-negatives. Few existing baselines outperform SWIM-X, however the comparison is not fair, as the Hybrid baseline relies on information based on aggregation of three models, and for Cohere, model information is unknown. We leave as future work to explore optimized training techniques, including better sampling of hard negatives with SWIM-IR, in contrast to current synthetic models trained only with in-batch negatives.

| Model | Avg. | as | bho | brx | gbm | gom | gu | hi | hne | kn | mai | ml | mni | mr | mwr | or | pa | ps | sa | ta | ur |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot baselines (English-only supervision) | | | | | | | | | | | | | | | | | | | | | |
| mDPR (En) | 6.3 | 2.6 | 6.4 | 0.4 | 7.2 | 1.3 | 8.6 | 13.3 | 5.2 | 10.4 | 6.4 | 12.3 | 0.2 | 8.9 | 5.8 | 0.4 | 6.0 | 5.6 | 5.2 | 10.2 | 10.0 |
| mContriever (En) | 7.9 | 7.9 | 3.2 | 7.8 | 0.3 | 9.7 | 2.2 | 11.1 | 15.2 | 8.2 | 10.6 | 8.6 | 15.6 | 0.4 | 10.7 | 8.5 | 1.1 | 10.3 | 3.3 | 5.7 | 12.9 |
| Supervised Baselines (Cross-lingual supervision) | | | | | | | | | | | | | | | | | | | | | |
| mDPR-X | 8.4 | 6.7 | 9.9 | 4.8 | 10.0 | 8.7 | 8.8 | 9.1 | 9.4 | 9.0 | 10.0 | 10.5 | 4.8 | 7.8 | 9.6 | 6.9 | 8.6 | 7.4 | 8.5 | 8.1 | 9.1 |
| mContriever-X | 12.4 | 9.8 | 15.7 | 6.7 | 14.0 | 11.7 | 13.3 | 15.5 | 13.9 | 13.6 | 13.9 | 16.9 | 6.5 | 12.0 | 13.8 | 7.5 | 13.4 | 9.8 | 12.4 | 13.0 | 14.1 |
| mContriever-X♡ | 13.5 | 11.6 | 15.4 | 8.0 | 16.9 | 12.3 | 15.2 | 16.7 | 15.7 | 14.7 | 15.6 | 17.4 | 7.0 | 14.2 | 14.7 | 9.1 | 13.2 | 10.1 | 14.8 | 12.1 | 14.9 |
| Synthetic Baselines (Our work) | | | | | | | | | | | | | | | | | | | | | |
| SWIM-X (120K)MT | 26.1 | 25.2 | 29.5 | 2.1 | 30.8 | 22.1 | 31.5 | 35.8 | 31.5 | 28.7 | 32.2 | 34.6 | 2.2 | 32.7 | 27.7 | 14.8 | 30.7 | 21.0 | 28.2 | 30.6 | 29.2 |
| SWIM-X (120K) | 25.2 | 24.4 | 27.7 | 4.3 | 28.3 | 25.4 | 29.4 | 32.4 | 28.8 | 30.1 | 31.8 | 34.4 | 5.1 | 30.7 | 25.7 | 15.8 | 29.6 | 20.6 | 26.1 | 27.9 | 26.1 |

*Table 5:  Experimental results for cross-lingual retrieval on XTREME-UP test *Ruder et al. ([2023](#bib.bib46 ""))*. (♡) denotes the mContriever-X model fine-tuned without MS MARCO *Nguyen et al. ([2016](#bib.bib37 ""))*; Two variants of SWIM-X considered, both fine-tuned on 120K synthetic data: (1) SWIM-X (120K)MT fine-tuned using Google Translate, i.e., translated prompt exemplars for 15 languages, whereas (2) SWIM-X (120K) is fine-tuned using prompt exemplars sampled from XTREME-UP training split for all languages.*

XTREME-UP. [Table 5](#S3.T5 "Table 5 ‣ 3.3 Experimental Results ‣ 3 Experiments ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") shows that SWIM-X (120K, 6K each language) was trained by randomly selecting 5 exemplars from the XTREME-UP training dataset (human-labeled queries) for all 20 languages, whereas the MT variant was trained on 5 XOR-Retrieve exemplars containing translated summaries and queries using Google Translate for 15 languages.111111We were unable to translate our prompt exemplars for 5 languages due to language unavailability in Google Translate: Boro (brx), Garhwali (gbm), Chattisgarhi (hne) and Marwari (mwr). Manipuri (mni) is available in Google Translate in “Meitei” script instead of the “Bengali-Assamese” script present in the XTREME-UP dataset. SWIM-X (120K) MT outperforms the best supervised baseline, mContriever-X (FT without MS MARCO) by a huge margin of 12.6 points on MRR@10. The SWIM-X model with XTREME-UP training pairs as prompt exemplars perform minimally worse than translated prompts by 0.9 points at MRR@10.
Interestingly, none of the models perform well on two extremely low-resource languages, Boro (brx) and Manipuri (mni). We evaluate whether is it likely due to a tokenizer issue in mT5 *Xue et al. ([2022](#bib.bib59 ""))*, so we conduct an ablation with a language-independent ByT5-base (§[4](#S4 "4 Ablation Studies ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")), finding that the mT5 tokenizer is not the cause of the performance drop in extremely low-resource language settings.

### 3.4 SAP versus Standard Prompting

We evaluate how does the quality of the generated query using our technique SAP compare against standard few-shot for downstream retrieval performance on XOR-Retrieve.
We additionally ablate across different LLM sizes, to observe correlations in XOR-Retrieve model performance with change in LLM size. To ensure consistency, we adopt the experimental setup utilized in SWIM-X (500k).
Our results are shown in [Figure 3](#S4.F3 "Figure 3 ‣ 4 Ablation Studies ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") (Left), we infer two insights: (i) Increase in LLM size provides diminishing gains in SWIM-X model performance on XOR-Retrieve, PaLM-2 (S) provides the best trade-off in terms of performance and inference speed. (ii) SAP technique outperforms standard prompting by at least 0.6 points Recall@5kt for all PaLM-2 generators on XOR-Retrieve, where the maximum improvements are observed by up to 3.2 points Recall@5kt for model sizes (S) or smaller. We hypothesize that PaLM 2 (model sizes $>$ S) are inherently able to generate coherent questions, leading to diminishing improvements with SAP versus few-shot standard prompting.

### 3.5 How much synthetic data to generate?

We analyse the optimal value of synthetic training data for training SWIM-X models. [Figure 5](#S6.F5 "Figure 5 ‣ 6 Background and Related Work ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") depicts the relative improvement in SWIM-X on XOR-Retrieve, with the performance (gradually increasing) starting to saturate after 500K synthetic pairs. The first observation is that with only 2K pairs, the SWIM-X (2K) achieves 49.1 Recall@5kt on XOR-Retrieve, which outperforms the best zero-shot (English-only) baseline. The break-even point occurs at around 200K synthetic pairs, where the SWIM-X (250K) model achieves 60.5, outperforming the best supervised baseline of mContriever-X achieving 59.6 Recall@5kt.

### 3.6 Indo-European Language Transferability

We investigate language transfer capabilities of synthetic data generated with SWIM-IR on Indic (Indo-European language family). We fine-tune separate SWIM-X models individually for eight languages and evaluate on XTREME-UP. From [Figure 4](#S4.F4 "Figure 4 ‣ 4 Ablation Studies ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), we observe that models fine-tuned for Konkani (gom) or Hindi (hi) transfer best on all languages in XTREME-UP (rows 3\&4), whereas Tamil (ta) transfers worst (row 8). Assamese (as), Konkani (gom), Odia (or), Pashto (pa) and Sanskrit (sa) have the lowest zero-shot capabilities with SWIM-X, where in-language synthetic data is found crucial for improvement in MRR@10. Hindi (hi), Kannada (kn), Malayalam (ml), Gujarati (gu) show good zero-shot transfer capabilities with all individual fine-tuned Indic languages.

4 Ablation Studies
------------------

K-shot prompt exemplars. We investigate the number of K-shot prompt exemplars required by PaLM 2 and how does the cross-lingual performance vary with K.121212We limit K \= 5, as it fits within the 4096 tokens in context length. Adding more exemplars require longer PaLM 2 contexts which increases the computational cost significantly. From [Figure 3](#S4.F3 "Figure 3 ‣ 4 Ablation Studies ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") (right), we observe a linear improvement in Recall@5kt with increase in K. Best Recall@5kt is observed with K \= 5. Our SAP technique cannot perform well zero-shot (i.e., K \= 0) due to the complex nature of the multilingual question generation task which requires a few examples for PaLM 2 to understand the query generation task.

<img src='x3.png' alt='Refer to caption' title='' width='189' height='202' />

<img src='x4.png' alt='Refer to caption' title='' width='136' height='213' />

*Figure 3: (Left) SAP (Summarize-then-Ask Prompting) (green) versus standard prompting (red) for various PaLM 2 model sizes. (Right) Varying K-shot prompt exemplars. All SWIM-X models are fine-tuned on 500K synthetic data and evaluated on XOR-Retrieve.*

ByT5 tokenizer and Query Replacement. We evaluate whether the low-performance of SWIM-X models on low-resource languages in XTREME-UP which can be attributed towards language-based tokenization. We reproduce our SWIM-X models, with a ByT5-base *Xue et al. ([2022](#bib.bib59 ""))* model as backbone, which contains a language independent tokenizer extension. From our results in [Table 6](#S6.T6 "Table 6 ‣ 6 Background and Related Work ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), ByT5 models achieve a lower MRR@10 on average on XTREME-UP, in contrast to mT5-base. Additionally, the performance on both mni and brx do not improve with ByT5.
Further, we evaluate the impact of human-generated versus LLM-generated query on XTREME-UP. We conduct an ablation, where we replace all human-generated queries in the training split with synthetic queries generated using PaLM 2. From [Table 6](#S6.T6 "Table 6 ‣ 6 Background and Related Work ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), we observe that the performance only drops by 2.0 points on MRR@10. This shows human-generated queries are better in quality, however, SWIM-X can be fine-tuned effectively with synthetic generated queries, by marginally dropping in performance.

<img src='x5.png' alt='Refer to caption' title='' width='332' height='170' />

*Figure 4: Heatmap showing MRR@10 denoting language-based transfer ability of SWIM-X (120K) across Indo-European languages available in XTREME-UP *Ruder et al. ([2023](#bib.bib46 ""))*. (ALL) denotes SWIM-X fine-tuned on all XTREME-UP languages.*

5 Cost Comparison
-----------------

Generating synthetic training data is relatively inexpensive however, not free. The cost associated of generating a synthetic dataset is dependent upon the length of the prompt, input, and output generated from the large language model. The costs also linearly increase with languages, as we need to generate synthetic data for each language pair.
At this writing, PaLM 2 and similar models costs about 0.0005 USD for 1000 characters in the input and output text.131313PaLM 2 pricing: [cloud.google.com/vertex-ai/pricing](https://cloud.google.com/vertex-ai/pricing#generative_ai_models "")Our prompts on average contain about 8-9K characters in the prompt input and generate about 1-2K characters in the output. The relative performance improvement associated with annotation cost in XOR-Retrieve is shown in [Figure 5](#S6.F5 "Figure 5 ‣ 6 Background and Related Work ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"). Generating 200K synthetic training pairs in SWIM-IR will cost roughly $1K USD. SWIM-X (200K) performs comparably to the best supervised baseline (mContriever-X), trained on 15.2K human-annotated pairs, requiring roughly 14 times more, i.e. $14.1K USD to annotate, if we pay an hourly rate of $18.50 USD per hour for the annotator (local minimum wages is $11.50 USD/hr) following *Zhang et al. ([2023c](#bib.bib64 ""))*, assuming an estimated annotation cost of 3.0 minutes per example *Ruder et al. ([2023](#bib.bib46 ""))*.

6 Background and Related Work
-----------------------------

Ad-hoc retrieval tasks require a search system for a given query to find a few relevant passages from within a corpus of many passages. Monolingual and cross-lingual retrieval are well-studied areas within multilingual retrieval. In monolingual retrieval, queries and the corpus are in the same language (e.g., Bengali queries searching passages from a Bengali corpus), as opposed to cross-lingual retrieval, where queries and corpora are in different languages (e.g., Japanese queries searching passages from an English corpus).

The development of pre-trained multilingual LMs has contributed toward recent progress in multilingual retrieval *Asai et al. ([2021a](#bib.bib2 "")); Izacard et al. ([2022](#bib.bib19 "")); Asai et al. ([2021b](#bib.bib3 "")); Li et al. ([2022](#bib.bib27 "")); Ruder et al. ([2023](#bib.bib46 "")); Zhang et al. ([2023c](#bib.bib64 ""), [b](#bib.bib63 ""))*. Notable baselines include mDPR and mContriever. mDPR *Asai et al. ([2021a](#bib.bib2 ""), [b](#bib.bib3 "")); Zhang et al. ([2023b](#bib.bib63 ""))* extends English DPR *Karpukhin et al. ([2020](#bib.bib23 ""))* to the multilingual setting. mContriever *Izacard et al. ([2022](#bib.bib19 ""))* adopts an unsupervised pre-training using the contrastive loss function and data prepared from mC4 *Xue et al. ([2021](#bib.bib60 ""))*, and is further fine-tuned on MS MARCO *Nguyen et al. ([2016](#bib.bib37 ""))*.

| Model | Backbone | Query Gen. | brx | mni | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| 1. Models with Byte-level (UTF-8) tokenizer | | | | | |
| mCon.-X♡ | ByT5 | Human | 1.8 | 1.0 | 2.1 |
| SWIM-X (120k)MT | ByT5 | PaLM 2 | 2.1 | 4.9 | 13.3 |
| SWIM-X (120k) | ByT5 | PaLM 2 | 5.1 | 5.8 | 15.4 |
| 2. Human-generated query replacement in XTREME-UP | | | | | |
| mCon.-X♡ | mT5 | Human | - | - | 13.5 |
| SWIM-X ($\approx$10K) | mT5 | PaLM 2 | - | - | 11.5 |

*Table 6: Ablations in XTREME-UP. First, we replace the mT5 backbone with ByT5. Next, we replace the human-generated queries in the XTREME-UP training dataset with PaLM-2 synthetic queries; MRR@10 scores are macro averaged across all 20 languages; brx denotes Boro and mni denotes Manipuri language.*

Synthetic Data Generation. Traditionally, docT5query *Nogueira and Lin ([2019](#bib.bib39 ""))* for question generation has been prominent for generating synthetic data in English. Thereby, it is used in multiple works involving domain adaptation *Ma et al. ([2021](#bib.bib32 "")); Thakur et al. ([2021](#bib.bib51 "")); Wang et al. ([2022](#bib.bib55 "")); Thakur et al. ([2022](#bib.bib50 ""))*. More recently, using LLMs for synthetic query generation has gained interest. *Bonifacio et al. ([2022](#bib.bib5 ""))* proposed InPars, where they use GPT-3 for few-shot prompting to generate synthetic queries for passages, then fine-tune a T5-based ranker with the synthetic data. *Dai et al. ([2023](#bib.bib10 ""))* proposed Promptagator, which studied task-dependent few-shot prompting on LLMs and further used the synthetic data for both retrieval and ranking models. Another related work, HyDE *Gao et al. ([2023](#bib.bib15 ""))* use LLMs to generate synthetic documents augmented with the input queries.
However, prior work has focused on improving retrieval in English, with the exception of HyDE. In our work, we robustly investigate how LLMs can be used for improving multilingual dense retrieval models.

<img src='x6.png' alt='Refer to caption' title='' width='332' height='238' />

*Figure 5: Recall@5kt improvement (in %) on XOR-Retrieve versus annotation cost in USD ($) to create the training dataset. The amount of training pairs generated are mentioned with each marked datapoint.*

Synthetic Multilingual Datasets. Prior work involved different techniques to build synthetic multilingual datasets for training dense retrieval models. *Bonifacio et al. ([2021](#bib.bib6 ""))* proposed mMARCO, a multilingual version of the MS MARCO dataset*Nguyen et al. ([2016](#bib.bib37 ""))* generated using machine translation.
However, as translated documents are not written by a native speaker, mMARCO and similar translation-based datasets suffer from artifacts such as “Translationese”*Clark et al. ([2020](#bib.bib9 ""))*.
Recently, *Mayfield et al. ([2023](#bib.bib33 ""))* proposes JH-POLO, which uses a pair of positive and negative passages in the target language and prompts GPT-3 *Brown et al. ([2020](#bib.bib7 ""))* to generate an English question.

7 Conclusion
------------

In this work, we present SWIM-IR, a synthetic multilingual retrieval training dataset for synthetic fine-tuning of multilingual dense retrieval models. SWIM-IR is constructed using SAP (summarize-then-ask prompting) containing two stages, summary extraction and query generation. SAP assists the LLM to identify the relevant sections of the input passage, improving the quality of the generated query. SWIM-IR provides 28M query-passage training pairs across 33 diverse languages ranging from high to low-resource languages. By providing SWIM-IR, we focus on fine-tuning multilingual dense retrieval models requiring no supervision, i.e., human-labeled training data, as human annotation is cumbersome and expensive.

Our rigorous evaluation across three multilingual retrieval benchmarks assesses the quality of our SWIM-IR dataset. We find that SWIM-X, fine-tuned on SWIM-IR (while keeping other model and training parameters unchanged) outperform the best supervised baseline, mContriever-X by 7.1 points Recall@5kt on XOR-Retrieve and 11.7 points MRR@10 on XTREME-UP. SWIM-X is competitive on monolingual retrieval on MIRACL, where it underperforms the upper bound of mContriever-X.

8 Limitations of SWIM-IR dataset
---------------------------------

SWIM-IR like any other dataset is not perfect and has limitations. These limitations do not directly affect the downstream multilingual retrieval task, where dense retrieval models learn how to match relevant passages to queries. SWIM-IR dataset has been created for the “sole” purpose of fine-tuning multilingual retrieval models. We describe below few noted limitations:

1. Decontextualization. PaLM 2 captures the salient information from the paragraph, but can generate the query in a reduced context, which cannot be answered without the Wikipedia paragraph.

2. Code-Switching. PaLM 2 can occasionally generate a code-switched query with words combined for English and the target language. Code-switching is more frequently observed for cross-lingual generation in low-resource languages.

3. Passage Quality and Length. A good quality passage contains relevant information about a topic which PaLM 2 uses to generate a synthetic query. However, if the passage is really short with little or zero information, or contains noisy information, this likely can generate a subpar query.

4. Factual inconsistencies in LLM generation. LLMs have been found to generate text lacking sufficient grounding to knowledge sources *Dziri et al. ([2022](#bib.bib12 "")); Ji et al. ([2023](#bib.bib20 ""))*, thereby posing risks
of misinformation and hallucination in their generated outputs *Maynez et al. ([2020](#bib.bib34 "")); Raunak et al. ([2021](#bib.bib41 "")); Muller et al. ([2023](#bib.bib35 ""))*. Queries in SWIM-IR are relevant for the input passage, but are not human-verified, thereby queries may contain factual inconsistencies.

References
----------

* Anil et al. (2023)Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernández Ábrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan A. Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vladimir Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, and et al. 2023.[PaLM 2 Technical Report](https://doi.org/10.48550/ARXIV.2305.10403 "").*CoRR*, abs/2305.10403.
* Asai et al. (2021a)Akari Asai, Jungo Kasai, Jonathan H. Clark, Kenton Lee, Eunsol Choi, and Hannaneh Hajishirzi. 2021a.[XOR QA: Cross-lingual Open-Retrieval Question Answering](https://doi.org/10.18653/V1/2021.NAACL-MAIN.46 "").In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021*, pages 547–564. Association for Computational Linguistics.
* Asai et al. (2021b)Akari Asai, Xinyan Yu, Jungo Kasai, and Hanna Hajishirzi. 2021b.[One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval](https://proceedings.neurips.cc/paper/2021/hash/3df07fdae1ab273a967aaa1d355b8bb6-Abstract.html "").In *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pages 7547–7560.
* Bender et al. (2021)Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021.[On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://doi.org/10.1145/3442188.3445922 "")In *FAccT ’21: 2021 ACM Conference on Fairness, Accountability, and Transparency, Virtual Event / Toronto, Canada, March 3-10, 2021*, pages 610–623. ACM.
* Bonifacio et al. (2022)Luiz Henrique Bonifacio, Hugo Queiroz Abonizio, Marzieh Fadaee, and Rodrigo Frassetto Nogueira. 2022.[InPars: Unsupervised Dataset Generation for Information Retrieval](https://doi.org/10.1145/3477495.3531863 "").In *SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022*, pages 2387–2392. ACM.
* Bonifacio et al. (2021)Luiz Henrique Bonifacio, Israel Campiotti, Roberto de Alencar Lotufo, and Rodrigo Frassetto Nogueira. 2021.[mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset](http://arxiv.org/abs/2108.13897 "").*CoRR*, abs/2108.13897.
* Brown et al. (2020)Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.[Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html "").In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*.
* Chang et al. (2023)Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2023.[A Survey on Evaluation of Large Language Models](https://doi.org/10.48550/ARXIV.2307.03109 "").*CoRR*, abs/2307.03109.
* Clark et al. (2020)Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020.[TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages](https://doi.org/10.1162/tacl_a_00317 "").*Transactions of the Association for Computational Linguistics*, 8:454–470.
* Dai et al. (2023)Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, and Ming-Wei Chang. 2023.[Promptagator: Few-shot Dense Retrieval From 8 Examples](https://openreview.net/pdf?id=gmL46YMpu2J "").In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net.
* Devlin et al. (2019)Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://doi.org/10.18653/v1/N19-1423 "").In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.
* Dziri et al. (2022)Nouha Dziri, Sivan Milton, Mo Yu, Osmar Zaiane, and Siva Reddy. 2022.[On the Origin of Hallucinations in Conversational Models: Is it the Datasets or the Models?](https://doi.org/10.18653/v1/2022.naacl-main.387 "")In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 5271–5285, Seattle, United States. Association for Computational Linguistics.
* Feng et al. (2022)Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022.[Language-agnostic BERT Sentence Embedding](https://doi.org/10.18653/v1/2022.acl-long.62 "").In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, pages 878–891. Association for Computational Linguistics.
* Forcada (2002)Mikel L. Forcada. 2002.[Explaining real MT to translators: between compositional semantics and word-for-word](https://aclanthology.org/2002.eamt-1.16 "").In *Proceedings of the 6th EAMT Workshop: Teaching Machine Translation*, Manchester, England. European Association for Machine Translation.
* Gao et al. (2023)Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023.[Precise Zero-Shot Dense Retrieval without Relevance Labels](https://doi.org/10.18653/V1/2023.ACL-LONG.99 "").In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pages 1762–1777. Association for Computational Linguistics.
* Gehman et al. (2020)Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020.[RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://doi.org/10.18653/v1/2020.findings-emnlp.301 "").In *Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020*, volume EMNLP 2020 of *Findings of ACL*, pages 3356–3369. Association for Computational Linguistics.
* Gospodinov et al. (2023)Mitko Gospodinov, Sean MacAvaney, and Craig Macdonald. 2023.[Doc2Query-: When Less is More](https://doi.org/10.1007/978-3-031-28238-6_31 "").In *Advances in Information Retrieval - 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2-6, 2023, Proceedings, Part II*, volume 13981 of *Lecture Notes in Computer Science*, pages 414–422. Springer.
* Hofstätter et al. (2021)Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021.[Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling](https://doi.org/10.1145/3404835.3462891 "").In *SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021*, pages 113–122. ACM.
* Izacard et al. (2022)Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022.[Unsupervised Dense Information Retrieval with Contrastive Learning](https://openreview.net/forum?id=jKN1pXi7b0 "").*Transactions on Machine Learning Research*.
* Ji et al. (2023)Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023.[Survey of Hallucination in Natural Language Generation](https://doi.org/10.1145/3571730 "").*ACM Comput. Surv.*, 55(12).
* Joshi et al. (2020)Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020.[The State and Fate of Linguistic Diversity and Inclusion in the NLP World](https://doi.org/10.18653/v1/2020.acl-main.560 "").In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 6282–6293, Online. Association for Computational Linguistics.
* Kamalloo et al. (2023)Ehsan Kamalloo, Xinyu Zhang, Odunayo Ogundepo, Nandan Thakur, David Alfonso-Hermelo, Mehdi Rezagholizadeh, and Jimmy Lin. 2023.[Evaluating Embedding APIs for Information Retrieval](https://doi.org/10.18653/V1/2023.ACL-INDUSTRY.50 "").In *Proceedings of the The 61st Annual Meeting of the Association for Computational Linguistics: Industry Track, ACL 2023, Toronto, Canada, July 9-14, 2023*, pages 518–526. Association for Computational Linguistics.
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.[Dense Passage Retrieval for Open-Domain Question Answering](https://doi.org/10.18653/v1/2020.emnlp-main.550 "").In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781, Online. Association for Computational Linguistics.
* Khattab and Zaharia (2020)Omar Khattab and Matei Zaharia. 2020.[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://doi.org/10.1145/3397271.3401075 "").In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR ’20, page 39–48, New York, NY, USA. Association for Computing Machinery.
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.[Natural Questions: a Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/ "").*Transactions of the Association of Computational Linguistics*.
* Lhoest et al. (2021)Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid, Sylvain Gugger, Clément Delangue, Théo Matussière, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar, François Lagunas, Alexander Rush, and Thomas Wolf. 2021.[Datasets: A Community Library for Natural Language Processing](http://arxiv.org/abs/2109.02846 "").In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 175–184, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
* Li et al. (2022)Yulong Li, Martin Franz, Md Arafat Sultan, Bhavani Iyer, Young-Suk Lee, and Avirup Sil. 2022.[Learning Cross-Lingual IR from an English Retriever](https://doi.org/10.18653/v1/2022.naacl-main.329 "").In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4428–4436, Seattle, United States. Association for Computational Linguistics.
* Lin et al. (2021)Jimmy Lin, Rodrigo Frassetto Nogueira, and Andrew Yates. 2021.[*Pretrained Transformers for Text Ranking: BERT and Beyond*](https://doi.org/10.2200/S01123ED1V01Y202108HLT053 "").Synthesis Lectures on Human Language Technologies. Morgan \& Claypool Publishers.
* Lin et al. (2023)Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023.[How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval](https://doi.org/10.48550/arXiv.2302.07452 "").*CoRR*, abs/2302.07452.
* Liu et al. (2023)Nelson F. Liu, Tianyi Zhang, and Percy Liang. 2023.[Evaluating Verifiability in Generative Search Engines](https://doi.org/10.48550/ARXIV.2304.09848 "").*CoRR*, abs/2304.09848.
* Loshchilov and Hutter (2019)Ilya Loshchilov and Frank Hutter. 2019.[Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7 "").In *International Conference on Learning Representations*.
* Ma et al. (2021)Ji Ma, Ivan Korotkov, Yinfei Yang, Keith Hall, and Ryan McDonald. 2021.[Zero-shot Neural Passage Retrieval via Domain-targeted Synthetic Question Generation](https://doi.org/10.18653/v1/2021.eacl-main.92 "").In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 1075–1088, Online. Association for Computational Linguistics.
* Mayfield et al. (2023)James Mayfield, Eugene Yang, Dawn J. Lawrie, Samuel Barham, Orion Weller, Marc Mason, Suraj Nair, and Scott Miller. 2023.[Synthetic Cross-language Information Retrieval Training Data](https://doi.org/10.48550/ARXIV.2305.00331 "").*CoRR*, abs/2305.00331.
* Maynez et al. (2020)Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020.[On Faithfulness and Factuality in Abstractive Summarization](https://doi.org/10.18653/v1/2020.acl-main.173 "").In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 1906–1919, Online. Association for Computational Linguistics.
* Muller et al. (2023)Benjamin Muller, John Wieting, Jonathan H. Clark, Tom Kwiatkowski, Sebastian Ruder, Livio Baldini Soares, Roee Aharoni, Jonathan Herzig, and Xinyi Wang. 2023.[Evaluating and Modeling Attribution for Cross-Lingual Question Answering](https://doi.org/10.48550/ARXIV.2305.14332 "").*CoRR*, abs/2305.14332.
* Neelakantan et al. (2022)Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, and Lilian Weng. 2022.[Text and Code Embeddings by Contrastive Pre-Training](http://arxiv.org/abs/2201.10005 "").*CoRR*, abs/2201.10005.
* Nguyen et al. (2016)Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016.[MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](http://arxiv.org/abs/1611.09268 "").*CoRR*, abs/1611.09268.
* Ni et al. (2022)Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma, Vincent Y. Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. 2022.[Large Dual Encoders Are Generalizable Retrievers](https://doi.org/10.18653/V1/2022.EMNLP-MAIN.669 "").In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pages 9844–9855. Association for Computational Linguistics.
* Nogueira and Lin (2019)Rodrigo Nogueira and Jimmy Lin. 2019.[From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf "").
* Pushkarna et al. (2022)Mahima Pushkarna, Andrew Zaldivar, and Oddur Kjartansson. 2022.[Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI](https://doi.org/10.1145/3531146.3533231 "").In *2022 ACM Conference on Fairness, Accountability, and Transparency*, FAccT ’22, page 1776–1826, New York, NY, USA. Association for Computing Machinery.
* Raunak et al. (2021)Vikas Raunak, Arul Menezes, and Marcin Junczys-Dowmunt. 2021.[The Curious Case of Hallucinations in Neural Machine Translation](https://doi.org/10.18653/v1/2021.naacl-main.92 "").In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1172–1183, Online. Association for Computational Linguistics.
* Reimers and Gurevych (2020)Nils Reimers and Iryna Gurevych. 2020.[Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://doi.org/10.18653/v1/2020.emnlp-main.365 "").In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pages 4512–4525. Association for Computational Linguistics.
* Ren et al. (2021)Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021.[RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://doi.org/10.18653/v1/2021.emnlp-main.224 "").In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021*, pages 2825–2835. Association for Computational Linguistics.
* Roy et al. (2020)Uma Roy, Noah Constant, Rami Al-Rfou, Aditya Barua, Aaron Phillips, and Yinfei Yang. 2020.[LAReQA: Language-Agnostic Answer Retrieval from a Multilingual Pool](https://doi.org/10.18653/V1/2020.EMNLP-MAIN.477 "").In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pages 5919–5930. Association for Computational Linguistics.
* Ruder (2022)Sebastian Ruder. 2022.The State of Multilingual AI.[http://ruder.io/state-of-multilingual-ai/](http://ruder.io/state-of-multilingual-ai/ "").
* Ruder et al. (2023)Sebastian Ruder, Jonathan H. Clark, Alexander Gutkin, Mihir Kale, Min Ma, Massimo Nicosia, Shruti Rijhwani, Parker Riley, Jean Michel A. Sarr, Xinyi Wang, John Wieting, Nitish Gupta, Anna Katanova, Christo Kirov, Dana L. Dickinson, Brian Roark, Bidisha Samanta, Connie Tao, David Ifeoluwa Adelani, Vera Axelrod, Isaac Caswell, Colin Cherry, Dan Garrette, R. Reeve Ingle, Melvin Johnson, Dmitry Panteleev, and Partha Talukdar. 2023.[XTREME-UP: A User-Centric Scarce-Data Benchmark for Under-Represented Languages](https://doi.org/10.48550/ARXIV.2305.11938 "").*CoRR*, abs/2305.11938.
* Santhanam et al. (2022)Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022.[ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://doi.org/10.18653/v1/2022.naacl-main.272 "").In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3715–3734, Seattle, United States. Association for Computational Linguistics.
* Schwenk et al. (2021)Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong, and Francisco Guzmán. 2021.[WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia](https://doi.org/10.18653/v1/2021.eacl-main.115 "").In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 1351–1361, Online. Association for Computational Linguistics.
* Tan et al. (2019)Xu Tan, Jiale Chen, Di He, Yingce Xia, Tao Qin, and Tie-Yan Liu. 2019.[Multilingual Neural Machine Translation with Language Clustering](https://doi.org/10.18653/v1/D19-1089 "").In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 963–973, Hong Kong, China. Association for Computational Linguistics.
* Thakur et al. (2022)Nandan Thakur, Nils Reimers, and Jimmy Lin. 2022.[Injecting Domain Adaptation with Learning-to-hash for Effective and Efficient Zero-shot Dense Retrieval](https://doi.org/10.48550/ARXIV.2205.11498 "").*CoRR*, abs/2205.11498.
* Thakur et al. (2021)Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021.[BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://openreview.net/forum?id=wCu6T5xFjeJ "").In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*.
* Touvron et al. (2023)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov,
and Thomas Scialom. 2023.[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://doi.org/10.48550/arXiv.2307.09288 "").*CoRR*, abs/2307.09288.
* van den Oord et al. (2018)Aäron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.[Representation Learning with Contrastive Predictive Coding](http://arxiv.org/abs/1807.03748 "").*CoRR*, abs/1807.03748.
* Wang et al. (2021)Bingning Wang, Ting Yao, Weipeng Chen, Jingfang Xu, and Xiaochuan Wang. 2021.[Multi-Lingual Question Generation with Language Agnostic Language Model](https://doi.org/10.18653/V1/2021.FINDINGS-ACL.199 "").In *Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021*, volume ACL/IJCNLP 2021 of *Findings of ACL*, pages 2262–2272. Association for Computational Linguistics.
* Wang et al. (2022)Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. 2022.[GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://doi.org/10.18653/v1/2022.naacl-main.168 "").In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2345–2360, Seattle, United States. Association for Computational Linguistics.
* Wenzek et al. (2020)Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. 2020.[CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://aclanthology.org/2020.lrec-1.494/ "").In *Proceedings of The 12th Language Resources and Evaluation Conference, LREC 2020, Marseille, France, May 11-16, 2020*, pages 4003–4012. European Language Resources Association.
* Wieting et al. (2023)John Wieting, Jonathan Clark, William Cohen, Graham Neubig, and Taylor Berg-Kirkpatrick. 2023.[Beyond Contrastive Learning: A Variational Generative Model for Multilingual Retrieval](https://doi.org/10.18653/v1/2023.acl-long.673 "").In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12044–12066, Toronto, Canada. Association for Computational Linguistics.
* Xiong et al. (2021)Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021.[Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://openreview.net/forum?id=zeFrfgyZln "").In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net.
* Xue et al. (2022)Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. 2022.[ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://doi.org/10.1162/tacl_a_00461 "").*Transactions of the Association for Computational Linguistics*, 10:291–306.
* Xue et al. (2021)Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021.[mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://doi.org/10.18653/V1/2021.NAACL-MAIN.41 "").In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021*, pages 483–498. Association for Computational Linguistics.
* Zhang et al. (2023a)Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen R. McKeown, and Tatsunori B. Hashimoto. 2023a.[Benchmarking Large Language Models for News Summarization](https://doi.org/10.48550/ARXIV.2301.13848 "").*CoRR*, abs/2301.13848.
* Zhang et al. (2021)Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin. 2021.[Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval](https://doi.org/10.18653/v1/2021.mrl-1.12 "").In *Proceedings of the 1st Workshop on Multilingual Representation Learning*, pages 127–137, Punta Cana, Dominican Republic. Association for Computational Linguistics.
* Zhang et al. (2023b)Xinyu Zhang, Kelechi Ogueji, Xueguang Ma, and Jimmy Lin. 2023b.[Toward Best Practices for Training Multilingual Dense Retrieval Models](https://doi.org/10.1145/3613447 "").*ACM Trans. Inf. Syst.*, 42(2).
* Zhang et al. (2023c)Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. 2023c.[MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages](https://doi.org/10.1162/tacl_a_00595 "").*Transactions of the Association for Computational Linguistics*, 11:1114–1131.
* Zhu et al. (2023)Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Jiajun Chen, Lei Li, and Shujian Huang. 2023.[Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis](https://doi.org/10.48550/ARXIV.2304.04675 "").*CoRR*, abs/2304.04675.
* Zhuang et al. (2023)Shengyao Zhuang, Linjun Shou, and Guido Zuccon. 2023.[Augmenting Passage Representations with Query Generation for Enhanced Cross-Lingual Dense Retrieval](https://doi.org/10.1145/3539618.3591952 "").In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023*, pages 1827–1832. ACM.

Appendix A Appendix
-------------------

The following supplementary sections in SWIM-IR are arranged as follows:

* •

    [Appendix B](#A2 "Appendix B Details on SWIM-IR Dataset Release ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") provides information on the SWIM-IR dataset release.

* •

    [Appendix C](#A3 "Appendix C SWIM-IR Extra Material ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") provides extra material with SWIM-IR dataset: Datacard, Examples and Prompts. All the prompts for all languages will be provided as text files within our supplementary submission.

* •

    [Appendix D](#A4 "Appendix D Human Validation ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") provides details on the human validation of SWIM-IR question quality.

* •

    [Appendix E](#A5 "Appendix E Additional Technical Details ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") provides detailed information on hyperparameters and training settings for baselines, multilinugal pre-training, synthetic finetuning, and sampling strategies.

* •

    [Appendix F](#A6 "Appendix F Evaluation Dataset Information ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") provides statistics for three multilingual retrieval evaluation datasets: XOR-Retrieve, MIRACL and XTREME-UP.

* •

    [Appendix G](#A7 "Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") contains additional results on the SWIM-IR dataset for XOR-Retrieve and MIRACL evaluation datasets.

Appendix B Details on SWIM-IR Dataset Release
----------------------------------------------

Dataset Release Format. The SWIM-IR dataset will be available as multiple formats. Officially, the dataset will be initially released within the Google Cloud Storage (GCS) cloud storage bucket141414SWIM-IR: [storage.googleapis.com/gresearch/swimir/swim_ir_v1.tar.gz](http://storage.googleapis.com/gresearch/swim-ir/swim_ir_v1.tar.gz "") for the initial review period. Later, for a longer term preservation the dataset will be maintained through TFDS dataset. To enable a wider audience within research, we would also release an official copy of the dataset via HuggingFace datasets *Lhoest et al. ([2021](#bib.bib26 ""))*.

Long Term Preservation. The dataset will be available for a longer time by continually updating the Tensorflow dataset (TFDS) and HuggingFace dataset. The authors will be responsible for maintaining the dataset and in future extension of the work for supporting more languages *Joshi et al. ([2020](#bib.bib21 ""))* and other cross-language retrieval setting: English query retrieving across language specific corpora (En$\rightarrow$L), inclusion of both would improve multilingual neural retrieval models on a wider variety of languages.

Licensing. The SWIM-IR dataset is based on language-specific Wikipedia. We follow the same license as Wikipedia for SWIM-IR: Creative Commons Attribution-ShareAlike 4.0 Unported License (CC BY-SA 4.0).151515 [https://creativecommons.org/licenses/by-sa/4.0](https://creativecommons.org/licenses/by-sa/4.0/ "") Overall, the license allows both researchers and industry alike to access the dataset, and allow them to copy and redistribute the dataset for future work.

Appendix C SWIM-IR Extra Material
----------------------------------

### C.1 SWIM-IR Data Card

We provide the datacard associated with the SWIM-IR dataset along with it’s release. The datacard generated using the template provided by the Data Cards Playbook *Pushkarna et al. ([2022](#bib.bib40 ""))*. The datacard has been generated using the Markdown format.161616The Markdown format and the template of the datacard is available here: [https://github.com/pair-code/datacardsplaybook](https://github.com/pair-code/datacardsplaybook "") The Datacard is provided along with our dataset release in the GitHub repository: [https://github.com/google-research-datasets/swim-ir](https://github.com/google-research-datasets/swim-ir "").

### C.2 SWIM-IR Dataset Statistics

The languages covered and the amount of training pairs available in SWIM-IR are provided in [Table 7](#A7.T7 "Table 7 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"). A majority of the training pairs (sampled a maximum of 1 million per language pair) are provided for 18 languages in MIRACL *Zhang et al. ([2023c](#bib.bib64 ""))*. The rest of 15 Indo-European languages from XTREME-UP contribute for 100K training pairs. We additionally, provide two examples from SWIM-IR dataset for each retrieval task, cross-lingual and monolingual in [Figure 7](#A7.F7 "Figure 7 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"). The cross-lingual example is provided for Chinese (zh) and monolingual for Spanish (es).

There are six fields associated with every SWIM-IR training datapoint. We briefly describe each field available below:
(i) _id: denotes the unique identifier of the training pair. (ii) title: denotes the title of the Wikipedia article.(iii) text: denotes the passage extracted from the Wikipedia article. (iv) query: denotes the synthetic multilingual query generated using PaLM 2 *Anil et al. ([2023](#bib.bib1 ""))*. (v) lang: denotes the language of the synthetic query. (v) code: denotes the ISO code of the synthetic query language.

### C.3 SWIM-IR Prompts

All prompts and their templates (across all 33 languages) used for developing SWIM-IR have been provided in the GitHub repository.171717[https://github.com/google-research-datasets/swim-ir](https://github.com/google-research-datasets/swim-ir "") We show individual prompt examples for a single language for the three datasets in the Appendix: (1) XOR-Retrieve (English passage; Synthetic Bengali query) in [Figure 8](#A7.F8 "Figure 8 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), (2) MIRACL (Chinese passage; Synthetic Chinese query) in [Figure 9](#A7.F9 "Figure 9 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), and (3) XTREME-UP (English Passage: Synthetic Hindi query) in [Figure 10](#A7.F10 "Figure 10 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"). The rest of the prompts will be provided in the SWIM-IR GitHub repository.

Appendix D Human Validation
---------------------------

In this section, we evaluate the quality of the PaLM 2 generated questions available in the SWIM-IR dataset using human annotators who are native speakers of different languages available in the dataset. For our annotation task, we evaluate five languages181818The authors in the paper are native speakers of the five languages chosen for evaluation: Bengali, Spanish, Chinese, Hindi and English. in total: English (en), Bengali (bn), Spanish (es), Chinese (zh) and Hindi (hi). Within the five languages, three are high-resource (en, es, zh), one is medium resource (hi) and low-resource (bn). For each language, we sample a fixed amount of question-passage pairs resulting in overall 500 question-passage pairs human evaluated. For English, Spanish and Chinese, we evaluate monolingual training pairs. For Hindi and Bengali, we mix and evaluate both cross-lingual and monolingual task-specific question-passage pairs.

We compute the question quality on a three-level rating scheme (0/1/2) based on three statistics, fluency, adequacy, and language. (i) Fluency measures the coherence of the generated question, i.e., whether the question can be perfectly understandable and readable by the user containing no spelling or grammatical mistakes. (ii) Adequacy measures the relevancy of the question with the Wikipedia passage (used for generation of the question), whether the question asked contains the answer within the passage. (ii) Language measures whether the generated question is in the correct language, or code-switching occurs in the generated question. We add these details in our annotation guidelines to teach the human annotator and attach it at the end of the Appendix section.

### D.1 Human Validation Results

[Table 2](#S2.T2 "Table 2 ‣ 2.2 SWIM-IR Dataset Construction ‣ 2 SWIM-IR Dataset Overview ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval") shows the results of human validation across five languages. The human annotators get 99-100% for the language metric which denotes the PaLM 2 generated quality is always in the correct language.
For Fluency, the major mistakes are observed in Hindi (12%), where few sampled passages in MIRACL can be too short (2-3 words long), this confuses the PaLM 2 model which duplicates the exact text in the query. For Adequacy, we observe that in Chinese (30%) of the generated synthetic queries are not strongly related to the passage. Similar to fluency, a low adequacy is observed when the LLM-generated query is generated for a short sampled passage or when the query asks a question about a related topic which is not directly mentioned in the passage.

Appendix E Additional Technical Details
---------------------------------------

### E.1 mContriever Pretraining

In the original implementation of mContriever *Izacard et al. ([2022](#bib.bib19 ""))*, the authors initialized the model using the mBERT *Devlin et al. ([2019](#bib.bib11 ""))* pre-trained language model (PLM). Next, the model was jointly pre-trained on 29 languages covering the CCNet dataset *Wenzek et al. ([2020](#bib.bib56 ""))* with a contrastive pre-training objective. In our implementation of mContriever, we initialize the model with the multilingual T5 (mT5) model *Xue et al. ([2021](#bib.bib60 ""))*. Next, we jointly pre-train the model on 101 languages191919The list of all 101 languages in mC4 can be found at: [www.tensorflow.org/datasets/catalog/c4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual "") available in mC4 *Xue et al. ([2021](#bib.bib60 ""))*. We sample two random non-overlapping texts from our document with a maximum size of 256 tokens. Similar to the mT5 pre-training objective *Xue et al. ([2021](#bib.bib60 ""))*, examples are not uniformly sampled over languages, i.e., the probability that a training sample comes from a specific language is directly proportional to the amount of training data available in the language. We randomly sample a maximum of 20k samples per language and keep it as a validation subset.
We optimize our mContriever model with the AdamW optimizer *Loshchilov and Hutter ([2019](#bib.bib31 ""))* with a learning rate of $1e^{-3}$, batch size of 8192, and for 600K training steps. For the first 500K steps, we pre-train with a language-mixed training objective, where a single training batch can contain examples across multiple languages. For the remaining 100k training steps, we pre-train with a language-unmixed training objective, where a single training batch contains all examples from a specific language, i.e., no mixing of different language pairs within a training batch. We internally conducted a quick evaluation of the mContriever pre-trained models with language-mixing (500k) and with both language-mixing and unmixing (600k) checkpoints. On XOR-Retrieve, we observe that the language-unmixed pre-training overall improves the model performance by 7.3 points on XOR-Retrieve.

### E.2 Baseline FT Models

XOR-Retrieve.For the zero-shot baseline model, we fine-tune on the MSMARCO *Nguyen et al. ([2016](#bib.bib37 ""))* dataset. Our base initialization model is mT5 *Xue et al. ([2021](#bib.bib60 ""))*. We use in-batch negatives, AdamW optimizer *Loshchilov and Hutter ([2019](#bib.bib31 ""))* and with a learning rate of $1e^{-3}$. The query sequence length contains a maximum sequence length of 64 tokens, whereas the document contains a maximum sequence length of 256 tokens. On MSMARCO, our models are fine-tuned with a batch size of 4096 and for 50k training steps. For our supervised fine-tuned baselines, we fine-tune on the XOR-Retrieve training dataset. The original dataset authors provide 1 hard negative per each training query in *Asai et al. ([2021a](#bib.bib2 ""))*. We fine-tune our baseline models on XOR-Retrieve on the triplets containing the query, positive passage and a hard negative, AdamW optimizer *Loshchilov and Hutter ([2019](#bib.bib31 ""))*, learning rate of $1e^{-3}$ for a batch size of 4096 for 15K training steps.

MIRACL.For the zero-shot baseline model, we fine-tune on the MSMARCO *Nguyen et al. ([2016](#bib.bib37 ""))* dataset. Details are shown above in XOR-Retrieve. For the monolingual supervised models, we use the MIRACL training data for fine-tuning. The authors of MIRACL provided hard negatives for training samples. We sample up to a maximum of four hard negatives for each query and fine-tune our models on MIRACL for 15K training steps.

XTREME-UP.For the zero-shot baseline model, we fine-tune on the MSMARCO *Nguyen et al. ([2016](#bib.bib37 ""))* dataset. For the supervised baselines, we use the XTREME-UP training data and fine-tune with in-batch negatives for a batch size of 1024 for 5K training steps.

<img src='x7.png' alt='Refer to caption' title='' width='332' height='266' />

*Figure 6: Training batch size ablation of SWIM-X (500K) model on XOR-Retrieve *Asai et al. ([2021a](#bib.bib2 ""))*. The best Recall@5kt (Macro Avg.) is achieved with batch size equal to 4096. To avoid overfitting, we fine-tune SWIM-X models with decreasing training steps of {40K, 40K, 30K, 30K, 20K, 20K, 15K} for increasing batch sizes of {128, 256, 512, 1024, 2048, 4096, 8192} respectively. We fine-tune all SWIM-X models on 500K synthetic SWIM-IR training pairs.*

### E.3 Synthetic FT models

We fine-tune all SWIM-X models using in-batch negatives, AdamW optimizer *Loshchilov and Hutter ([2019](#bib.bib31 ""))* and with a learning rate of $1e^{-3}$. The pre-trained language model for SWIM-X is the mT5 Base model with 580M parameters *Xue et al. ([2021](#bib.bib60 ""))*. The batch size and the training steps varies for each retrieval setting. All training data is always split evenly across all languages present in the training data. For example, given 100K pairs with 5 different languages, each language includes 20K training pairs.

XOR-Retrieve. SWIM-X is fine-tuned with a batch size of 4096 and with a maximum of 50K steps on synthetic SWIM-IR cross-lingual pairs. For the 500K training pairs, we fine-tune for 20K steps, and for the maximum of 7M pairs we fine-tune for 50K training steps. The training pairs within a single batch include language-mixing, i.e., one or more language-specific training pairs are sampled within a single training batch.

MIRACL. SWIM-X is fine-tuned for a batch-size of 4096 and for a maximum of 15K steps. As shown in *Roy et al. ([2020](#bib.bib44 "")); Zhang et al. ([2023b](#bib.bib63 ""))*, language-unmixed training setup is shown to work well for monolingual retrieval. Following prior work, our SWIM-X training pairs include language unmixing, i.e., all pairs are from a single language. The examples are uniformly sampled across all languages, i.e., probability that a training sample comes from a specific language is the same for all languages, unlike the previous experiment in mC4 pre-training.

XTREME-UP. SWIM-X has been fine-tuned for a batch size of 1024 and for a maximum of 15K training steps. Similar to XOR-Retrieve, training pairs include language-mixing with a single batch during fine-tuning.

### E.4 Stratified Sampling Strategy for SWIM-IR

In our work, we use a stratified sampling technique to select a subset of passages from the Wikipedia corpus we use to generate questions for SWIM-IR. We ensure all languages have relatively an equal amount of training samples, wherever possible. Our Wikipedia corpus contains entities which are sorted alphabetically (A-Z). We then compute inclusion threshold $I_{th}$, which is defined as $I_{th}\=D_{sample}/D_{total}$, where $(D_{sample})$ is number of passages required to sample and $(D_{total})$ is the total numbers of passages in corpus.
Next, for each passage ($p_{i}$) in the corpus, we randomly generate an inclusion probability $\hat{p_{i}}\in[0,1]$. We select the passage ($p_{i}$) if $p_{i}\leq I_{th}$. This ensures uniform sampling of passages with Wikipedia entities between all letters (A-Z).202020All Wikipedia entities starting with a non-alphabet are included in the beginning of the Wikipedia corpus.

Appendix F Evaluation Dataset Information
-----------------------------------------

In [Table 8](#A7.T8 "Table 8 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), we provide an overview of the three evaluation datasets and provide statistics for each retrieval dataset. All three multilingual evaluation datasets contain a train split. Only the XTREME-UP dataset has released their test split publicly, as a result it was used for evaluation in the paper. For both XOR-Retrieve and MIRACL, since the test set is hidden from the public, we evaluate on the development split. The list of languages covered by each dataset and samples available for training and evaluation can be found in [Table 8](#A7.T8 "Table 8 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval").

Appendix G Additional Results
-----------------------------

XOR-Retrieve.In [Table 9](#A7.T9 "Table 9 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), we report the Recall@2kt scores across all multilingual retrievers on XOR-Retrieve. We find similar trends for improvement, where SWIM-X (7M) outperforms the best FT model on mContriever-X by 3.9 points on Recall@2kt. The SWIM-X (7M) without pre-training is also a strong baseline outperforming SWIM-X (7M) with pre-training on 4/7 languages in XOR-Retrieve.

MIRACL.In [Table 10](#A7.T10 "Table 10 ‣ Appendix G Additional Results ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"), we report the Recall@100 scores across all multilingual retrievers on MIRACL. We observe that the mContriever-X model overall achieves the highest Recall@100 score of 86.5, SWIM-X models achieve a recall of 78.9 which is competitive on MIRACL outperforming both the zero-shot mDPR and mContriever models. For Yoruba, Our SWIM-X outperforms mContriever which shows the importance of synthetic training data, as the model does not contain supervision for Yoruba (i.e., no human-labeled training pairs).

| Cross-Lingual (33) | | Monolingual (18) | |
| --- | --- | --- | --- |
| Q-P Lang. | # Train Pairs | Q-P Lang. | # Train Pairs |
| Languages available in MIRACL Zhang et al. ([2023c](#bib.bib64 "")) | | | |
| ar-en | 901,363 | ar-ar | 890,389 |
| bn-en | 909,748 | bn-bn | 257,327 |
| de-en | 909,145 | de-de | 943,546 |
| en-en | - | en-en | 936,481 |
| es-en | 905,771 | es-es | 947,340 |
| fa-en | 910,295 | fa-fa | 973,409 |
| fi-en | 906,429 | fi-fi | 967,139 |
| fr-en | 911,694 | fr-fr | 977,900 |
| hi-en | 919,729 | hi-hi | 466,272 |
| id-en | 907,826 | id-id | 837,459 |
| ja-en | 906,862 | ja-ja | 893,520 |
| ko-en | 905,669 | ko-ko | 941,459 |
| ru-en | 904,933 | ru-ru | 915,693 |
| sw-en | 905,242 | sw-sw | 123,099 |
| te-en | 902,190 | te-te | 220,431 |
| th-en | 914,610 | th-th | 451,540 |
| yo-en | 902,467 | yo-yo | 43,211 |
| zh-en | 921,701 | zh-zh | 946,757 |
| Indo-European Languages in XTREME-UP Ruder et al. ([2023](#bib.bib46 "")) | | | |
| as-en | 5,899 | as-as | - |
| bho-en | 5,763 | bho-bho | - |
| gom-en | 5,755 | gom-gom | - |
| gu-en | 5,870 | gu-gu | - |
| kn-en | 5,763 | kn-kn | - |
| mai-en | 5,768 | mai-mai | - |
| ml-en | 5,907 | ml-ml | - |
| mni-en | 5,604 | mni-mni | - |
| mr-en | 5,977 | mr-mr | - |
| or-en | 5,837 | or-or | - |
| pa-en | 5,840 | pa-pa | - |
| ps-en | 5,694 | ps-ps | - |
| sa-en | 5,779 | sa-sa | - |
| ta-en | 5,930 | ta-ta | - |
| ur-en | 5,816 | ur-ur | - |
| Total | 15,532,876 | Total | 12,732,972 |
| Overall Training Pairs \= 28,265,848 | | | |

*Table 7: Dataset Statistics of SWIM-IR for both cross-lingual and monolingual settings; (Q-P Lang.) denotes the language code of the query-passage training pair in SWIM-IR; (# Train Pairs) denotes the count of the relevant training pairs containing the synthetic query and original passage pair.*

| Benchmark | Retrieval | Query | Passage | # L | ISO | Languages | Train Split | | Test Split | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Task |  |  |  |  |  | (#Queries) | (HNeg.) | (#Queries) | (#Passages) | (Metric) |
| XOR-Retrieve Asai et al. ([2021a](#bib.bib2 "")) | Cross-lingual | $L$ | English | 7 | ar, bn, fi, ja, ko, ru, te | Arabic, Bengali, Finnish, Japanese, Korean, Russian, Telugu | 15,250 | Yes (1 each) | 2,110 | 18,003,200 | Recall@5kt |
| MIRACL Zhang et al. ([2023c](#bib.bib64 "")) | Monolingual | $L$ | $L$ | 18 | ar, bn, de, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, yo, zh | Arabic, Bengali, German, English, Spanish, Farsi, Finnish, French, Hindi, Indonesian, Japanese, Korean, Russian, Swahili, Telugu, Thai, Yoruba, Chinese | 88,288 | Yes (max 4) | 13,495 | 106,332,152 | nDCG@10 |
| XTREME-UP Ruder et al. ([2023](#bib.bib46 "")) | Cross-lingual | $L$ | English | 20 | as, bho, brx, gbm, gom, gu, hi, hne, kn, mai, ml, mni, mr, mwr, or, pa, ps, sa, ta, ur | Assamese, Bhojpuri, Boro, Garhwali, Konkani, Gujarati, Hindi, Chhattisgarhi, Kannada, Maithili, Malayalam, Manipuri, Marathi, Marwari, Odia, Punjabi, Pashto, Sanskrit, Tamil, Urdu | 13,270 | No | 5,300 | 112,426 | MRR@10 |

*Table 8: Statistics of multilingual retrieval evaluation benchmarks used in our work: XOR-Retrieve *Asai et al. ([2021a](#bib.bib2 ""))*, MIRACL *Zhang et al. ([2023c](#bib.bib64 ""))* and XTREME-UP *Ruder et al. ([2023](#bib.bib46 ""))*. For each benchmark, we describe the retrieval task, language in which query and passage are available, test and train dataset statistics and the evaluation metric. (HNeg.) denotes whether the dataset contains hard negatives for training multilingual models; (#L) denotes the number of languages covered by the benchmark.*

<img src='x8.png' alt='Refer to caption' title='' width='332' height='141' />

*Figure 7: Dataset examples showing both (a) cross-lingual and (b) monolingual training pairs in the SWIM-IR dataset. The passage is selected from English Wikipedia, and PaLM 2 generates the query. A detailed description of all the dataset column headers are provided in Appendix (§[C.2](#A3.SS2 "C.2 SWIM-IR Dataset Statistics ‣ Appendix C SWIM-IR Extra Material ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")). All translations in the figure above have been provided using Google Translate ([translate.google.com](https://translate.google.com/ "")) for illustration purposes.*

| Model | PLM | PT | Finetune | Recall@2kt | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | (Datasets) | Avg. | Ar | Bn | Fi | Ja | Ko | Ru | Te |
| Existing Supervised Baselines (Prior work) | | | | | | | | | | | |
| Dr. DECR Li et al. ([2022](#bib.bib27 "")) | XLM-R | WikiM | NQ + XOR∗ | 66.0 | – | – | – | – | – | – | – |
| mDPR Asai et al. ([2021a](#bib.bib2 "")) | mBERT | — | XOR | 40.5 | 38.8 | 48.4 | 52.5 | 26.6 | 44.2 | 33.3 | 39.9 |
| mBERT + xQG Zhuang et al. ([2023](#bib.bib66 "")) | mBERT | — | XOR | 46.2 | 42.4 | 54.9 | 54.1 | 33.6 | 52.3 | 33.8 | 52.5 |
| Google MT + DPR Asai et al. ([2021a](#bib.bib2 "")) | BERT | — | NQ | 62.2 | 62.5 | 74.7 | 57.3 | 55.6 | 60.0 | 52.7 | 72.3 |
| OPUS MT + DPR Asai et al. ([2021a](#bib.bib2 "")) | BERT | — | NQ | 42.7 | 43.4 | 53.9 | 55.1 | 40.2 | 50.5 | 30.8 | 20.2 |
| Zero-shot baselines (English-only supervision) | | | | | | | | | | | |
| mContriever | mT5 | mC4 | — | 29.9 | 27.2 | 23.0 | 35.0 | 27.0 | 27.7 | 35.0 | 34.0 |
| mDPR (En) | mT5 | — | MS MARCO | 30.6 | 26.2 | 26.0 | 37.9 | 32.8 | 24.6 | 34.6 | 32.4 |
| mContriever (En) | mT5 | mC4 | MS MARCO | 33.8 | 27.8 | 24.3 | 42.4 | 29.9 | 31.2 | 40.5 | 40.3 |
| Supervised Baselines (Cross-lingual supervision) | | | | | | | | | | | |
| mDPR-X | mT5 | — | XOR | 43.6 | 43.7 | 50.0 | 44.6 | 36.1 | 41.1 | 35.9 | 54.2 |
| mContriever-X | mT5 | mC4 | XOR | 46.6 | 40.1 | 62.5 | 47.1 | 38.2 | 44.2 | 38.4 | 55.5 |
| mDPR-X | mT5 | — | MS MARCO + XOR | 49.5 | 46.0 | 63.8 | 49.0 | 39.0 | 48.4 | 43.9 | 56.3 |
| mContriever-X | mT5 | mC4 | MS MARCO + XOR | 53.0 | 47.6 | 65.1 | 51.6 | 47.3 | 50.2 | 44.3 | 65.1 |
| Synthetic Baselines (Our work) | | | | | | | | | | | |
| SWIM-X (500K) | mT5 | — | SWIM-IR | 49.2 | 46.3 | 57.2 | 49.0 | 42.7 | 45.6 | 44.7 | 58.8 |
| SWIM-X (500K) | mT5 | mC4 | SWIM-IR | 53.3 | 46.6 | 61.8 | 51.9 | 46.5 | 49.1 | 55.3 | 61.8 |
| SWIM-X (7M) | mT5 | — | SWIM-IR | 56.6 | 50.8 | 65.1 | 56.1 | 48.1 | 54.0 | 55.7 | 66.4 |
| SWIM-X (7M) | mT5 | mC4 | SWIM-IR | 56.9 | 53.4 | 67.8 | 55.1 | 49.4 | 52.6 | 55.3 | 64.7 |

*Table 9: Experimental results showing Recall@2kt for cross-lingual retrieval on XOR-Retrieve dev *Asai et al. ([2021a](#bib.bib2 ""))*; (PLM) denotes the pretrained language model; (PT) denotes the pretraining dataset; (∗) Dr.DECR is fine-tuned in a complex training setup across more datasets (§[3.2](#S3.SS2 "3.2 Experimental Methods ‣ 3 Experiments ‣ Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval")); WikiM denotes WikiMatrix *Schwenk et al. ([2021](#bib.bib48 ""))*; XOR denotes XOR-Retrieve; SWIM-X (ours) is fine-tuned on 500K and 7M synthetic data.*

| Model | Avg. | ar | bn | en | es | fa | fi | fr | hi | id | ja | ko | ru | sw | te | th | zh | de | yo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing Supervised Baselines (Prior work) | | | | | | | | | | | | | | | | | | | |
| BM25 | 77.2 | 88.9 | 90.9 | 81.9 | 70.2 | 73.1 | 89.1 | 65.3 | 86.8 | 90.4 | 80.5 | 78.3 | 66.1 | 70.1 | 83.1 | 88.7 | 56.0 | 57.2 | 73.3 |
| mDPR | 79.0 | 84.1 | 81.9 | 76.8 | 86.4 | 89.8 | 78.8 | 91.5 | 77.6 | 57.3 | 82.5 | 73.7 | 79.7 | 61.6 | 76.2 | 67.8 | 94.4 | 89.8 | 79.5 |
| Hybrid | 88.0 | 94.1 | 93.2 | 88.2 | 94.8 | 93.7 | 89.5 | 96.5 | 91.2 | 76.8 | 90.4 | 90.0 | 87.4 | 72.5 | 85.7 | 82.3 | 95.9 | 88.9 | 80.7 |
| Cohere-API | 76.9 | 85.4 | 85.6 | 74.6 | 71.7 | 77.1 | 80.9 | 81.6 | 72.4 | 68.3 | 81.6 | 77.1 | 76.7 | 66.6 | 89.8 | 86.9 | 76.9 | 72.5 | 57.6 |
| Zero-shot baselines (English-only supervision) | | | | | | | | | | | | | | | | | | | |
| mDPR (En) | 76.9 | 85.5 | 85.9 | 72.4 | 66.8 | 79.7 | 86.0 | 71.4 | 74.2 | 67.0 | 80.1 | 77.1 | 77.4 | 80.2 | 91.9 | 84.8 | 68.5 | 70.9 | 58.6 |
| mContriever (En) | 76.6 | 73.5 | 80.8 | 52.1 | 49.5 | 61.7 | 66.0 | 51.8 | 50.3 | 63.5 | 65.6 | 56.3 | 58.9 | 73.5 | 85.9 | 76.6 | 58.2 | 36.3 | 30.2 |
| Supervised Baselines (Monolingual supervision) | | | | | | | | | | | | | | | | | | | |
| mDPR-X | 60.6 | 73.5 | 80.8 | 52.1 | 49.5 | 61.7 | 66.0 | 51.8 | 50.3 | 63.5 | 65.6 | 56.3 | 58.9 | 73.5 | 85.9 | 76.6 | 58.2 | 36.3 | 30.2 |
| mContriever-X | 86.5 | 92.0 | 95.3 | 80.6 | 78.8 | 84.0 | 93.1 | 86.0 | 82.1 | 83.7 | 89.5 | 87.7 | 86.7 | 93.3 | 96.7 | 94.3 | 85.9 | 79.3 | 68.8 |
| Synthetic Baselines (Our work) | | | | | | | | | | | | | | | | | | | |
| SWIM-X (180K) | 78.9 | 89.2 | 87.8 | 72.9 | 70.0 | 76.3 | 91.6 | 75.8 | 72.5 | 74.3 | 77.6 | 76.8 | 77.9 | 87.8 | 84.9 | 92.9 | 69.9 | 72.4 | 69.3 |

*Table 10:  Experimental results for monolingual retrieval on MIRACL dev *Zhang et al. ([2023c](#bib.bib64 ""))*. All scores denote Recall@100; (Hyb.) denotes Hybrid retriever with ranked fusion of three retrievers: mDPR, mColBERT and BM25; BM25, mDPR and Hybrid scores taken from *Zhang et al. ([2023c](#bib.bib64 ""))*; Cohere-API is used as a reranker on top of 100 BM25 results, taken from *Kamalloo et al. ([2023](#bib.bib22 ""))*. SWIM-X is fine-tuned on 180K synthetic data.*

<img src='x9.png' alt='Refer to caption' title='' width='332' height='457' />

*Figure 8: 5-shot SAP (Summarize-then-Ask Prompting) for XOR-Retrieve *Asai et al. ([2021a](#bib.bib2 ""))* is shown for Bengali (bn). There are five exemplars (5-shot) in our cross-lingual question generation task. The passages are randomly selected from XOR-Retrieve. Summaries and questions are manually written in English by the authors. Finally, the questions in exemplars are translated to Bengali using Google Translate ([translate.google.com](https://translate.google.com/ "")).*

<img src='x10.png' alt='Refer to caption' title='' width='332' height='321' />

*Figure 9: 3-shot SAP (Summarize-then-Ask Prompting) for MIRACL *Zhang et al. ([2023c](#bib.bib64 ""))* is shown for Chinese (zh). There are three exemplars (3-shot) in our monolingual question generation task. The query-passage pairs are randomly selected from MIRACL training set. Finally, the summaries in exemplars are automatically generated using Google Bard ([bard.google.com](https://bard.google.com/ "")).*

<img src='x11.png' alt='Refer to caption' title='' width='332' height='457' />

*Figure 10: 5-shot SAP (Summarize-then-Ask Prompting with Machine Translation (MT) for XTREME-UP *Ruder et al. ([2023](#bib.bib46 ""))* is shown for Hindi (hi). There are five exemplars (5-shot) in our cross-lingual question generation. The passages are re-used from the XOR-Retrieve task. Summaries and questions are manually written in English by the authors. Finally, the questions in exemplars are translated to Hindi using Google Translate ([translate.google.com](https://translate.google.com/ "")).*

See pages 1- of [supp/annotation-guidelines.pdf](supp/annotation-guidelines.pdf "")
