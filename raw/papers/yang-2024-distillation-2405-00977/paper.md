Distillation for Multilingual Information Retrieval
===================================================

Eugene YangHLTCOE, Johns Hopkins UniversityBaltimoreMarylandUSA[eugene.yang@jhu.edu](mailto:eugene.yang@jhu.edu),Dawn LawrieHLTCOE, Johns Hopkins UniversityBaltimoreMarylandUSA[lawrie@jhu.edu](mailto:lawrie@jhu.edu)andJames MayfieldHLTCOE, Johns Hopkins UniversityBaltimoreMarylandUSA[mayfield@jhu.edu](mailto:mayfield@jhu.edu)

(2024)

###### Abstract.

Recent work in cross-language information retrieval (CLIR),
where queries and documents are in different languages,
has shown the benefit of the Translate-Distill framework that trains a cross-language neural dual-encoder model
using translation and distillation.
However, Translate-Distill only supports a single document language.
Multilingual information retrieval (MLIR),
which ranks a multilingual document collection,
is harder to train than CLIR
because the model must assign comparable relevance scores to documents in different languages.
This work extends Translate-Distill and propose Multilingual Translate-Distill (MTD) for MLIR.
We show that ColBERT-X models trained with MTD outperform their counterparts trained with Multilingual Translate-Train,
which is the previous state-of-the-art training approach,
by 5% to 25% in nDCG@20 and 15% to 45% in MAP.
We also show that the model is robust to the way languages are mixed in training batches.
Our implementation is available on GitHub.

Dense retrieval, multilingual training, dual encoder architecture

††journalyear: 2024††copyright: rightsretained††conference: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval; July 14–18, 2024; Washington, DC, USA††booktitle: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), July 14–18, 2024, Washington, DC, USA††doi: 10.1145/3626772.3657955††isbn: 979-8-4007-0431-4/24/07††ccs: Information systems Language models††ccs: Information systems Multilingual and cross-lingual retrieval††ccs: Information systems Retrieval effectiveness

![Refer to caption]()

*(a) Mix Passages*

![Refer to caption]()

*(b) Mix Entries*

![Refer to caption]()

*(c) Round Robin Entries*

*Figure 1. Three language mixing strategies for Multilingual Translate-Distill. Each row indicates an entry with a query and a list of sampled passages in the training mini-batch. Circles, diamonds, and squares represent different document languages.*

1. Introduction
----------------

We define Multilingual Information Retrieval (MLIR)
as search over a multilingual collection of monolingual documents to produce a single ranked list*(Peters
et al., [2012]; Si
et al., [2008]; Tsai
et al., [2008]; Rahimi
et al., [2015]; Lawrie
et al., [2023b])*.
The retrieval system must retrieve and rank documents based only on query relevance,
independent of document language.
This is challenging in part because cross-language systems may be unable to exploit surface forms.
Our evaluation uses CLEF data*(Braschler, [2003])* with English queries and French, German, Spanish, and English documents;
CLEF data*(Braschler, [2001a], [b], [2002], [2003])* with English queries and French, German, and Italian documents;
and TREC NeuCLIR data*(Lawrie et al., [2022a], [2023a])* with English queries and Chinese, Persian, and Russian documents.

Dual-encoder retrieval models
such as ColBERT*(Khattab and
Zaharia, [2020])* that matches token embeddings,
and DPR*(Karpukhin et al., [2020])* that matches query and document embeddings,
have shown good results in both monolingual*(Santhanam et al., [2022])* and cross-language*(Nair et al., [2022]; Zhang
et al., [2021]; Yang
et al., [2024a]; Li
et al., [2022])* retrieval.
These approaches use pre-trained language models like multilingual BERT*(Devlin
et al., [2019])* and XLM-RoBERTa*(Conneau
et al., [2020a])* as text encoders to place queries and documents into a joint semantic space;
this allows embedding distances to be calculated across languages.
Multilingual encoders are generally trained monolingually on multiple languages*(Devlin
et al., [2019]; Conneau
et al., [2020b])*,
which leads to limited cross-language ability.
Therefore, careful fine-tuning,
such as Translate-Train*(Nair et al., [2022])*, C3 Pretraining*(Yang et al., [2022])* and Native-Train*(Nair
et al., [2023])*,
are essential to be able to match across languages*(Shi and Lin, [2019]; Yang
et al., [2024a]; Li
et al., [2022])*.

Generalizing from one to multiple document languages is not trivial.
Prior work showed that Multilingual Translate-Train (MTT) *(Lawrie
et al., [2023b])* of ColBERT-X
using
training data translated into all document languages
is more effective than BM25 search over documents translated into the query language.
Searching translated documents with the English ColBERT model is even more effective than MTT,
but incurs a high translation cost
at indexing time
compared to MTT’s amortized cost of translating the training corpus.
This work aims to
develop MLIR training that produces more effective models than its monolingual English counterparts.

Knowledge distillation has shown success monolingually*(Qu et al., [2021]; Santhanam et al., [2022]; Formal
et al., [2021])*,
so we adapt this concept to train MLIR models.
In Translate-Distill*(Yang
et al., [2024a])*, a way to train CLIR ColBERT-X models,
a teacher model scores monolingual training data using text in whichever language produces its best results.
Then when training the student ColBERT-X model,
training data is translated into the languages that match the final CLIR task.
That work showed that the student model is on par with or more effective than a retrieve-and-rerank system
that uses that same teacher model as a reranker.
We propose Multilingual Translate-Distill (MTD), a multilingual generalization of Translate-Distill.
Instead of training with a single document language,
we translate training passages into all document languages.
This opens a design space of how to mix languages in training batches.

This paper contributes
(1) an effective training approach for an MLIR dual-encoder that combines translation and distillation;
(2) models trained with MTD that are more effective than the previously reported state-of-the-art MLIR model,
ColBERT-X trained with MTT; and
(3) a robustness analysis of mini-batch passage mixing strategies.
Models and implementation are available on Huggingface Models111[https://huggingface.co/collections/hltcoe/multilingual-translate-distill-66280df75c34dbbc1708a22f](https://huggingface.co/collections/hltcoe/multilingual-translate-distill-66280df75c34dbbc1708a22f "") and GitHub.222[https://github.com/hltcoe/colbert-x](https://github.com/hltcoe/colbert-x "")

2. Background
--------------

An IR problem can be “multilingual” in several ways.
For example, *Hull and
Grefenstette ([1996])* described a multilingual IR problem of monolingual retrieval in multiple languages, as in*Blloshmi et al. ([2021])*,
or alternatively, multiple
CLIR tasks in several languages*(Lawrie
et al., [2022b]; Braschler, [2001b], [2002], [2003]; Mitamura et al., [2008])*.
We adopt the Cross-Language Evaluation Forum (CLEF)’s notion of MLIR:
using a query to construct one ranked list across documents in several languages*(Peters and
Braschler, [2002])*.
We acknowledge that this definition excludes mixed-language or code-switched queries and documents,
other cases to which “multilingual” has been applied.

Prior to neural retrieval, MLIR systems generally relied on cross-language dictionaries
or machine translation models*(Darwish and Oard, [2003]; Kraaij
et al., [2003]; McNamee and
Mayfield, [2002])*.
Translating documents into the query language casts MLIR as monolingual in that language*(Magdy and Jones, [2011]; Granell, [2014]; Rahimi
et al., [2015])*.
While translating queries into each document language is almost always computationally more economical than translating the documents,
it casts the MLIR problem as multiple monolingual problems
whose results must be merged to form the final MLIR ranked list*(Peters
et al., [2012]; Si
et al., [2008]; Tsai
et al., [2008])*.
Moreover, quality differences between translation models could bias results by systematically
ranking documents in some languages higher*(Lawrie
et al., [2023b]; Huang
et al., [2023])*.

Recent work in representation learning for IR*(Formal
et al., [2021]; Gao and Callan, [2022]; Reimers and
Gurevych, [2019])* and fast dense vector search algorithms*(Malkov and
Yashunin, [2018]; Jegou
et al., [2010]; Johnson
et al., [2019])* spawned a new class of models called dual-encoders.
These models encode queries and documents simultaneously into one or more dense vectors
representing tokens, spans, or entire sequences*(Khattab and
Zaharia, [2020]; Li et al., [2023b]; Li
et al., [2023a]; Karpukhin et al., [2020])*.
While replacing the underlying language model with a multilingual one,
such as multilingual BERT*(Devlin
et al., [2019])* and XLM-RoBERTa*(Conneau
et al., [2020b])*,
produces systems that accept queries and documents in multiple languages,
zero-shot transfer of a model trained only monolingually to a CLIR or MLIR problem is suboptimal;
it leads to systems even less effective than BM25 over document translations*(Nair et al., [2022]; Lawrie
et al., [2023b])*.
Therefore, designing an effective fine-tuning process for transforming multilingual language models into multilingual IR models is critical.

Various retrieval fine-tuning approaches have been explored,
such as contrastive learning*(Karpukhin et al., [2020]; Khattab and
Zaharia, [2020]; Santhanam et al., [2022])*,
hard-negative mining*(Hofstätter et al., [2021]; Formal
et al., [2021])*,
and knowledge distillation*(Formal
et al., [2021]; Santhanam et al., [2022]; Qu et al., [2021])*.
Knowledge distillation has demonstrated more effective results in both monolingual and cross-language IR*(Li
et al., [2022]; Yang
et al., [2024a])* than the others.
The recently proposed Translate-Distill approach decoupled the input languages of the teacher and student models.
This allowed large English rerankers to train ColBERT-X for CLIR,
leading to state-of-the-art CLIR effectiveness measured on the NeuCLIR 22 benchmark*(Lawrie et al., [2022a])*.
Recent work by *Huang
et al. ([2023])* proposes a language-aware decomposition for prompting (or augmenting) the document encoder.
In this work, we explore the simple idea of relying on translations of MS MARCO and distilling the ranking knowledge from a large MonoT5 model with mT5XXL underneath*(Jeronymo
et al., [2023]; Nogueira
et al., [2020]; Xue et al., [2021])*.

3. Multilingual Translate-Distill
-----------------------------------

Our proposed Multilingual Transalte-Distill (MTD) training approach
requires a monolingual training corpus consisting of queries and passages;
no relevance labels are required.

### 3.1. Knowledge Distillation

To train a student dual-encoder model for MLIR,
we first use two teacher models:
a query-passage selector and a query-passage scorer.
Following *Yang
et al. ([2024a])*, the query-passage selector retrieves $k$ passages for each query.
This can be replaced by any hard-negative mining approach*(Qu et al., [2021]; Hofstätter et al., [2021])* or by adapting publicly available mined passages.333For example, [https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives](https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives ""). The query-passage scorer then scores each query-passage pair with high accuracy.
The scorer is essentially a reranker from which we would like to distill ranking knowledge implicit in an expensive model such as MonoT5*(Nogueira
et al., [2020])* that is generally too slow to apply by itself.
The final product from the two teachers is a set of tuples,
each containing a query, a passage, and the associated teacher score.
We use these data
to train the student dual-encoder model.
Specifically, for each training mini-batch of size $n$,
we select $n$ training queries and sample $m$ retrieved passage IDs.
To teach the student model to rank documents across languages,
we translate each passage into all of the target languages.
When constructing the mini-batch, we determine the language for each passage ID, which we discuss in more detail in the next section.
Finally, the loss function is the KL divergence between the teacher and student scores
on the query and the translated passages.

### 3.2. Language Mixing Strategies

To train an effective ColBERT-X model for MLIR,
each training batch must include documents in more than one language*(Lawrie
et al., [2023b])*.
Training with MTD opens a design space for selecting languages for the mini-batch passages.
We experiment with three mixing strategies (see Figure[1]):

Mix Passages. In each training batch entry,
all passages are randomly assigned to one of the document languages.
In this case, each language is equally likely to be present during training.
Each language also has an equal probability of being assigned to any passage in such a way that language representation is balanced, thus a language is just as likely to be assigned to a passage with a high score as a low score.
This mixing method directly trains the student model to rank passages in different languages.

Mix Entries. Alternatively, we can assign the same randomly selected language to all passages associated with a query.
This method ensures the translation quality does not become a possible feature that the student model could rely on if there is a language with which the machine translation model struggles.
While not directly learning MLIR,
this model jointly learns multiple CLIR tasks
with distillation and eventually learns the MLIR task.

Round Robin Entries. To ensure the model equally learns the ranking problem for all languages,
we experiment with training query repetition to present passages from all languages.
In this case, the model learns the CLIR tasks using the same set of queries instead of a random subset
when mixing entries.
However, this reduces the number of queries per mini-batch given some fixed GPU memory size.
Given this memory constraint, round robin may not be feasible if the number of document languages exceeds the number of entries the GPU can hold at once.

4. Experiments
---------------

*Table 1. Collection Statistics*

|  | CLEF | | NeuCLIR | |
| --- | --- | --- | --- | --- |
|  | Subset(Huang et al., [2023]) | 2003 | 2022 | 2023 |
| Languages | de, fr, it | de, fr, es, en | zh, fa, ru | |
| # of Docs | 0.24M | 1.05M | 10.04M | |
| # of Passages | 1.90M | 6.96M | 58.88M | |
| # of Topics | 113 | 60 | 41 | 65 |
| Avg. Rel/Topic | 40.73 | 102.42 | 125.46 | 67.77 |

We evaluate our proposed model on four MLIR evaluation collections:
a subset of CLEF00-03 curated by *Huang
et al. ([2023])*444The collection is reconstructed by using the author-provided document IDs, which excludes a large portion of unjudged documents.
Documents added in subsequent years are also excluded. Thus some judged relevant documents are also excluded.;
CLEF03 with German, French, Spanish, and English*(Braschler, [2003])*;
and NeuCLIR 2022*(Lawrie et al., [2022a])* and 2023*(Lawrie et al., [2023a])*.
Collection statistics are summarized in Table[1].
Queries are English titles concatenated with descriptions.

We use MS MARCO*(Nguyen et al., [2016])* to train the MLIR ColBERT-X models with MTD,
for which we adopt the PLAID-X implementation released by *Yang
et al. ([2024a])*.555[https://github.com/hltcoe/ColBERT-X](https://github.com/hltcoe/ColBERT-X "") We use the English ColBERTv2 model released by *Santhanam et al. ([2022])* that was also trained with knowledge distillation666[https://huggingface.co/colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0 "") and MonoT5 with mT5XXL released by *Jeronymo
et al. ([2023])*777[https://huggingface.co/unicamp-dl/mt5-13b-mmarco-100k](https://huggingface.co/unicamp-dl/mt5-13b-mmarco-100k "") as query-passage selector and scorer, respectively.
Both the selector and the scorer received English MS MARCO queries and passages to generate training teacher scores.

To support MTD training, we translated the MS MARCO passages with Sockeye v2*(Domhan et al., [2020]; Hieber
et al., [2020])* into the document languages.
Student ColBERT-X models are fine-tuned from the XLM-RoBERTa large models*(Conneau
et al., [2020b])* using 8 NVidia V100 GPUs (32GB memory)
for 200,000 gradient steps with a mini-batch size of 8 entries each associated with 6 passages on each GPU.
We use AdamW optimizer with a $5\times 10^{-6}$ learning rate and half-precision floating points.

*Table 2. MLIR system effectiveness.
Numbers in superscripts indicate the system of the row is statistically better than the systems in the superscript with 95% confidence by conducting a one-sided paired t-test.
Numbers in subscripts indicate the system of the row is statistically identical within 0.05 in value to the systems in the subscripts with 95% confidence by conducting paired TOSTs.
Bonferroni corrections are applied to both sets of statistical tests.*

|  |  | CLEF00-03 Subset(Huang et al., [2023]) | | | CLEF 2003 | | | NeuCLIR 2022 MLIR | | | NeuCLIR 2023 MLIR | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Measure | nDCG | MAP | Recall | nDCG | MAP | Recall | nDCG | MAP | Recall | nDCG | MAP | Recall |
|  | Rank Cutoff | 10 | 100 | 100 | 20 | 1000 | 1000 | 20 | 1000 | 1000 | 20 | 1000 | 1000 |
|  | Baselines | | | | | | | | | | | | |
| (0) | KD-SPD(Huang et al., [2023]) | 0.416 | 0.220 | 0.469 | – | – | – | – | – | – | – | – | – |
| (1) | PSQ-HMM | 0.5290 | 0.3390 | 0.6170 | 0.445 | 0.282 | 0.711 | 0.315 | 0.193 | 0.594 | 0.289 | 0.225 | 0.693 |
| (2) | DT ¿¿ BM25 | 0.5680 | 0.38801 | 0.66201 | 0.6361 | 0.4531 | 0.8571 | 0.338 | 0.215 | 0.633 | 0.316 | 0.275 | 0.756 |
| (3) | DT ¿¿ ColBERT | 0.609${}^{01}_{4}$ | 0.422${}^{01}_{4}$ | 0.700${}^{01}_{4}$ | 0.6691 | 0.4971 | 0.88914 | 0.4031 | 0.28512 | 0.708124 | 0.3611 | 0.298${}^{1}_{4}$ | 0.7861 |
| (4) | ColBERT-X MTT | 0.613${}^{01}_{3}$ | 0.411${}^{01}_{3}$ | 0.687${}^{01}_{3}$ | 0.6431 | 0.4511 | 0.8271 | 0.375 | 0.236 | 0.612 | 0.330 | 0.281${}^{1}_{3}$ | 0.760 |
| (5) | ColBERT-X ED | 0.638${}^{012}_{8}$ | 0.457${}^{01234}_{678}$ | 0.732${}^{01234}_{678}$ | 0.699${}^{14}_{8}$ | 0.530${}^{124}_{678}$ | 0.920${}^{124}_{78}$ | 0.393 | 0.263 | 0.68714 | 0.3571 | 0.3171 | 0.827124 |
|  | ColBERT-X MTD with Different Mixing Strategies | | | | | | | | | | | | |
| (6) | Mix Passages | 0.666${}^{01234}_{78}$ | 0.471${}^{01234}_{578}$ | 0.747${}^{01234}_{578}$ | 0.6751 | 0.520${}^{14}_{57}$ | 0.901${}^{14}_{7}$ | 0.44412 | 0.340${}^{1245}_{78}$ | 0.762${}^{1245}_{78}$ | 0.404${}^{1245}_{78}$ | 0.367${}^{12345}_{78}$ | 0.868${}^{12345}_{78}$ |
| (7) | Mix Entries | 0.674${}^{012345}_{68}$ | 0.469${}^{01234}_{568}$ | 0.745${}^{01234}_{568}$ | 0.6861 | 0.522${}^{14}_{56}$ | 0.911${}^{124}_{568}$ | 0.4611245 | 0.347${}^{12345}_{68}$ | 0.768${}^{12345}_{68}$ | 0.397${}^{124}_{68}$ | 0.372${}^{12345}_{68}$ | 0.877${}^{12345}_{678}$ |
| (8) | Round Robin Entries | 0.656${}^{01234}_{567}$ | 0.476${}^{01234}_{567}$ | 0.751${}^{012345}_{567}$ | 0.699${}^{12}_{5}$ | 0.535${}^{1234}_{5}$ | 0.922${}^{1234}_{57}$ | 0.47412345 | 0.341${}^{12345}_{67}$ | 0.761${}^{1245}_{67}$ | 0.388${}^{124}_{67}$ | 0.347${}^{12345}_{67}$ | 0.856${}^{1234}_{67}$ |

Documents are split into 180 token passages with a stride of 90 before indexing.
The number of resulting passages is reported in Table[1].
We index the collection with PLAID-X using one residual bit.
At search time, PLAID-X retrieves passages, and document scores are aggregated using MaxP*(Dai and Callan, [2019])*.
For each query, we return the top 1000 documents for evaluation.

To demonstrate MTD effectiveness,
we report baseline ColBERT models that are trained differently:
English ColBERT*(Santhanam et al., [2022])*,
ColBERT-X with Multilingual Translate-Train (MTT)*(Lawrie
et al., [2023b])*,
and ColBERT-X with English Distillation (ED).
Since English ColBERT does not accept text in other languages,
we index the collection with documents machine-translated into English
(marked “DT” in Table[2]).
ColBERT-X models trained with MTT use the training triples released by MS MARCO
with hyperparameters similar to the MTD ones except for the number of queries per batch per GPU is increased to 32.
Finally, the English Distillation models are only exposed to English queries and passages during fine-tuning instead of the translated text.
It performs a zero-shot language transfer at indexing and search time.

We also compare our models to the recently published KD-SPD*(Huang
et al., [2023])*,
which is a language-aware MLIR model that encodes the entire text sequence as a single vector.
To provide a broader context,
we report sparse retrieval baselines PSQ-HMM*(Darwish and Oard, [2003]; Xu and Weischedel, [2000]; Yang et al., [2024b])* and BM25 with translated documents,
which are two strong MLIR baselines reported in NeuCLIR 2023*(Lawrie et al., [2023a])*.

We report nDCG@20, MAP, and Recall at 1000 for the CLEF03 and NeuCLIR collections.
To enable comparison to *Huang
et al. ([2023])*, we report nDCG@10, MAP@100, and Recall@100 on the CLEF00-03 subset.
To test statistical superiority between two systems,
we use a one-sided paired t-test with 95% confidence on the per-topic metric values.
When testing for statistical “equivalence” where the null hypothesis is that the effectiveness of the two systems differ,
we use a paired Two One-sided T-Tests (TOST)*(Lakens, [2017]; Schuirmann, [1987])* with a threshold of 0.05 and 95% confidence.

5. Results
-----------

*Table 3. nDCG@20 on training with more languages*

|  |  | Training Languages | | |
| --- | --- | --- | --- | --- |
|  | Evaluation Collection | CLEF03 | NeuCLIR | Both |
| Mix Passages | CLEF 2003 | 0.675 | 0.688 | 0.694 |
| | NeuCLIR 2022 MLIR | 0.437 | 0.444 | 0.431 |
| NeuCLIR 2023 MLIR | 0.377 | 0.404 | 0.406 |
| Mix Entries | CLEF 2003 | 0.686 | 0.679 | 0.680 |
| | NeuCLIR 2022 MLIR | 0.424 | 0.461 | 0.445 |
| NeuCLIR 2023 MLIR | 0.359 | 0.397 | 0.379 |

Table[2] summarizes our experiments.
ColBERT-X models trained with MTD are more effective than those with MTT across all four evaluation collections,
demonstrating a 5% (CLEF03 0.643 to 0.675 with mix passages)
to 26% (NeuCLIR22 0.375 to 0.474 with round robin entries)
improvement in nDCG@20 and 15% (CLEF03 0.451 to 0.520 with mix passages) to 47% (NeuCLIR22 0.236 to 0.347 with mix entries) in MAP.
MTD-trained ColBERT-X models over documents in their native form are significantly more effective than
translating all documents into English and searching with English ColBERT.

Since the languages in the two CLEF collections are closer to English than those in NeuCLIR,
the ColBERT-X model trained with English texts (Row 5)
still provides reasonable effectiveness using (partial) zero-shot language transfer during inference.
MTD yields identical effectiveness to ED based on the TOST equivalence test in the two CLEF collections by measuring MAP (Table[2]).
In contrast, NeuCLIR languages do not benefit from this phenomenon.
Instead, training directly with text in document languages
enhances both the general language modeling and retrieval ability of the student models.
In NeuCLIR 2022 and 2023, student ColBERT-X models trained with MTD (Rows 6 to 8) are 9% (NeuCLIR23 0.317 to 0.347 with round robin entries) to 32% (NeuCLIR22 0.263 to 0.347 with mix entries) more effective than ED (Row 5) by measuring MAP.

### 5.1. Ablation on Language Mixing Strategies

Since the TOST equivalence tests show that the three mixing strategies demonstrate statistically similar MAP and Recall for all collections except for a few cases in CLEF 2003
(CLEF 2003 may be an outlier because it has English documents, a known source of bias in MLIR*(Lawrie
et al., [2023b])*).
We conclude that MTD is robust to how languages are mixed during training
as long as multiple languages are present in each training mini-batch*(Lawrie
et al., [2023b])*.
Such robustness provides operational flexibility to practitioners creating MLIR models.
Since passage translation might not be available for all languages,
mixing passages allows selecting passages only from a subset of languages.
Mixing entries also allows training entries to be filtered for specific languages
if relevance is known to drop after translation.

When evaluating with nDCG@20, the differences are larger but less consistent.
For the two CLEF collections and NeuCLIR 2022,
topics were developed for a single language before obtaining relevance judgments across all languages.
These topics may not be well-attested in all document languages,
resulting in some CLIR topics with few relevant documents.
For these three collections, models trained with mixed CLIR tasks
(mix and round-robin entries) are more effective at the top of the ranking.
High variation among topics leads to inconclusive statistical significance results,
suggesting opportunities for result fusion.
NeuCLIR 2023 topics were developed bilingually,
so topics are not socially or culturally tied to a single language;
this leads to statistically equivalent nDCG@20 results.

### 5.2. Training Language Ablation

Finally, we explore training with languages beyond the ones in the document collection.
Table[3] shows MTD-trained models for CLEF 2003, NeuCLIR, and both on each collection.
Due to GPU memory constraints, we exclude the round-robin strategy from this ablation.

We observe that models trained with the mix passages strategy
are more robust than the mix-entries variants when training on CLEF and evaluating on NeuCLIR and vice versa. This shows smaller degradation when facing language mismatch between training and inference.
Surprisingly, training on NeuCLIR languages with the mix passage strategy yields numerically higher nDCG@20 than training on CLEF (0.675 to 0.688).

When training both CLEF and NeuCLIR languages,
effectiveness is generally worse than only training on the evaluation languages.
This trend suggests the models might be facing capability limits in the neural model,
or picking up artifacts from the quality differences in the translation.
This observation demands more experimentation on MLIR dual-encoder models, which we leave for future work.

6. Conclusion
--------------

We propose Multilingual Translate-Distill (MTD) for training MLIR dual-encoder models.
We demonstrated that ColBERT-X models trained with the proposed MTD are more effective than using previously proposed MLIR training techniques on four MLIR collections.
By conducting statistical equivalence tests, we showed that MTD is robust to the mixing strategies of the languages in the training mini-batch.

References
----------

* (1)
* Blloshmi et al. (2021)Rexhina Blloshmi, Tommaso
Pasini, Niccolò Campolungo, Somnath
Banerjee, Roberto Navigli, and
Gabriella Pasi. 2021.IR like a SIR: sense-enhanced information retrieval
for multiple languages. In *Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing*.
1030–1041.
* Braschler (2001a)Martin Braschler.
2001a.CLEF 2000 — Overview of Results. In*Cross-Language Information Retrieval and
Evaluation*, Carol Peters (Ed.).
Springer Berlin Heidelberg, Berlin,
Heidelberg, 89–101.
* Braschler (2001b)Martin Braschler.
2001b.CLEF 2001—Overview of Results. In*Workshop of the Cross-Language Evaluation Forum for
European Languages*. Springer, 9–26.
* Braschler (2002)Martin Braschler.
2002.CLEF 2002—Overview of results. In*Workshop of the Cross-Language Evaluation Forum for
European Languages*. Springer, 9–27.
* Braschler (2003)Martin Braschler.
2003.CLEF 2003–Overview of results. In*Workshop of the Cross-Language Evaluation Forum for
European Languages*. Springer, 44–63.
* Conneau
et al. (2020a)Alexis Conneau, Kartikay
Khandelwal, Naman Goyal, Vishrav
Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott,
Luke Zettlemoyer, and Veselin
Stoyanov. 2020a.Unsupervised Cross-lingual Representation Learning
at Scale. In *Proceedings of the 58th Annual
Meeting of the Association for Computational Linguistics*.
Association for Computational Linguistics,
Online, 8440–8451.[https://aclanthology.org/2020.acl-main.747](https://aclanthology.org/2020.acl-main.747 "")
* Conneau
et al. (2020b)Alexis Conneau, Kartikay
Khandelwal, Naman Goyal, Vishrav
Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott,
Luke Zettlemoyer, and Veselin
Stoyanov. 2020b.Unsupervised Cross-lingual Representation Learning
at Scale. In *Proceedings of the 58th Annual
Meeting of the Association for Computational Linguistics*.
Association for Computational Linguistics,
Online, 8440–8451.
* Dai and Callan (2019)Zhuyun Dai and Jamie
Callan. 2019.Deeper text understanding for IR with contextual
neural language modeling. In *Proceedings of the
42nd International ACM SIGIR Conference on Research and Development in
Information Retrieval*. 985–988.
* Darwish and Oard (2003)Kareem Darwish and
Douglas W Oard. 2003.Probabilistic structured query methods. In*Proceedings of the 26th Annual International ACM
SIGIR Conference on Research and Development in Information Retrieval*.
338–344.
* Devlin
et al. (2019)Jacob Devlin, Ming-Wei
Chang, Kenton Lee, and Kristina
Toutanova. 2019.BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding. In*Proceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long and Short Papers)*.
Association for Computational Linguistics,
Minneapolis, Minnesota, 4171–4186.
* Domhan et al. (2020)Tobias Domhan, Michael
Denkowski, David Vilar, Xing Niu,
Felix Hieber, and Kenneth Heafield.
2020.The Sockeye 2 Neural Machine Translation Toolkit
at AMTA 2020. In *Proceedings of the 14th
Conference of the Association for Machine Translation in the Americas (Volume
1: Research Track)*. Association for Machine Translation
in the Americas, Virtual, 110–115.
* Formal
et al. (2021)Thibault Formal, Benjamin
Piwowarski, and Stéphane Clinchant.
2021.SPLADE: Sparse lexical and expansion model for
first stage ranking. In *Proceedings of the 44th
International ACM SIGIR Conference on Research and Development in Information
Retrieval*. 2288–2292.
* Gao and Callan (2022)Luyu Gao and Jamie
Callan. 2022.Unsupervised Corpus Aware Language Model
Pre-training for Dense Passage Retrieval. In*Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers)*.
Association for Computational Linguistics,
Dublin, Ireland, 2843–2853.
* Granell (2014)Ximo Granell.
2014.*Multilingual information management:
Information, technology and translators*.Chandos Publishing.
* Hieber
et al. (2020)Felix Hieber, Tobias
Domhan, Michael Denkowski, and David
Vilar. 2020.SOCKEYE 2: A Toolkit for Neural Machine
Translation. In *EAMT 2020*.[https://www.amazon.science/publications/sockeye-2-a-toolkit-for-neural-machine-translation](https://www.amazon.science/publications/sockeye-2-a-toolkit-for-neural-machine-translation "")
* Hofstätter et al. (2021)Sebastian Hofstätter,
Sheng-Chieh Lin, Jheng-Hong Yang,
Jimmy Lin, and Allan Hanbury.
2021.Efficiently teaching an effective dense retriever
with balanced topic aware sampling. In *Proceedings
of the 44th International ACM SIGIR Conference on Research and Development in
Information Retrieval*. 113–122.
* Huang
et al. (2023)Zhiqi Huang, Hansi Zeng,
Hamed Zamani, and James Allan.
2023.Soft Prompt Decoding for Multilingual Dense
Retrieval.*arXiv preprint arXiv:2305.09025*(2023).
* Hull and
Grefenstette (1996)David A Hull and Gregory
Grefenstette. 1996.Querying across languages: A dictionary-based
approach to multilingual information retrieval. In*Proceedings of the 19th Annual International ACM
SIGIR Conference on Research and Development in Information Retrieval*.
49–57.
* Jegou
et al. (2010)Herve Jegou, Matthijs
Douze, and Cordelia Schmid.
2010.Product quantization for nearest neighbor search.*IEEE transactions on pattern analysis and
machine intelligence* 33, 1
(2010), 117–128.
* Jeronymo
et al. (2023)Vitor Jeronymo, Roberto
Lotufo, and Rodrigo Nogueira.
2023.NeuralMind-UNICAMP at 2022 TREC NeuCLIR: Large
Boring Rerankers for Cross-lingual Retrieval.*arXiv preprint arXiv:2303.16145*(2023).
* Johnson
et al. (2019)Jeff Johnson, Matthijs
Douze, and Hervé Jégou.
2019.Billion-scale similarity search with GPUs.*IEEE Transactions on Big Data*7, 3 (2019),
535–547.
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas
Oğuz, Sewon Min, Patrick Lewis,
Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih.
2020.Dense passage retrieval for open-domain question
answering.*arXiv preprint arXiv:2004.04906*(2020).
* Khattab and
Zaharia (2020)Omar Khattab and Matei
Zaharia. 2020.Colbert: Efficient and effective passage search via
contextualized late interaction over bert. In*Proceedings of the 43rd International ACM SIGIR
conference on research and development in Information Retrieval*.
39–48.
* Kraaij
et al. (2003)Wessel Kraaij, Jian-Yun
Nie, and Michel Simard.
2003.Embedding web-based statistical translation models
in cross-language information retrieval.*Computational Linguistics*29, 3 (2003),
381–419.
* Lakens (2017)Daniël Lakens.
2017.Equivalence tests: A practical primer for t tests,
correlations, and meta-analyses.*Social psychological and personality
science* 8, 4 (2017),
355–362.
* Lawrie et al. (2022a)Dawn Lawrie, Sean
MacAvaney, James Mayfield, Paul McNamee,
Douglas W. Oard, Luca Soldanini, and
Eugene Yang. 2022a.Overview of the TREC 2022 NeuCLIR Track. In*The Thirty-first Text REtrieval Conference (TREC
2022) Proceedings*.
* Lawrie et al. (2023a)Dawn Lawrie, Sean
MacAvaney, James Mayfield, Paul McNamee,
Douglas W. Oard, Luca Soldanini, and
Eugene Yang. 2023a.Overview of the TREC 2023 NeuCLIR Track. In*The Thirty-second Text REtrieval Conference (TREC
2023) Proceedings*.
* Lawrie
et al. (2022b)Dawn Lawrie, James
Mayfield, Douglas W. Oard, and Eugene
Yang. 2022b.HC4: A New Suite of Test Collections for Ad Hoc
CLIR. In *Proceedings of the 44th European
Conference on Information Retrieval*.
* Lawrie
et al. (2023b)Dawn Lawrie, Eugene Yang,
Douglas W Oard, and James Mayfield.
2023b.Neural Approaches to Multilingual Information
Retrieval. In *European Conference on Information
Retrieval*. Springer, 521–536.
* Li
et al. (2023a)Minghan Li, Sheng-Chieh
Lin, Xueguang Ma, and Jimmy Lin.
2023a.SLIM: Sparsified Late Interaction for Multi-Vector
Retrieval with Inverted Indexes. In *Proceedings of
the 46th International ACM SIGIR Conference on Research and Development in
Information Retrieval* (, Taipei, Taiwan,) *(SIGIR
’23)*. Association for Computing Machinery,
New York, NY, USA, 1954–1959.[https://doi.org/10.1145/3539618.3591977](https://doi.org/10.1145/3539618.3591977 "")
* Li et al. (2023b)Minghan Li, Sheng-Chieh
Lin, Barlas Oguz, Asish Ghoshal,
Jimmy Lin, Yashar Mehdad,
Wen-tau Yih, and Xilun Chen.
2023b.CITADEL: Conditional Token Interaction via
Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval.
In *Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers)*,
Anna Rogers, Jordan
Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics,
Toronto, Canada, 11891–11907.[https://doi.org/10.18653/v1/2023.acl-long.663](https://doi.org/10.18653/v1/2023.acl-long.663 "")
* Li
et al. (2022)Yulong Li, Martin Franz,
Md Arafat Sultan, Bhavani Iyer,
Young-Suk Lee, and Avirup Sil.
2022.Learning Cross-Lingual IR from an English
Retriever. In *Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies*. 4428–4436.
* Magdy and Jones (2011)Walid Magdy and
Gareth J.F. Jones. 2011.Should MT systems be used as black boxes in
CLIR?. In *European Conference on Information
Retrieval*. Springer, 683–686.
* Malkov and
Yashunin (2018)Yu A Malkov and Dmitry A
Yashunin. 2018.Efficient and robust approximate nearest neighbor
search using hierarchical navigable small world graphs.*IEEE transactions on pattern analysis and
machine intelligence* 42, 4
(2018), 824–836.
* McNamee and
Mayfield (2002)Paul McNamee and James
Mayfield. 2002.Comparing cross-language query expansion techniques
by degrading translation resources. In *Proceedings
of the 25th annual international ACM SIGIR conference on Research and
development in information retrieval*. 159–166.
* Mitamura et al. (2008)Teruko Mitamura, Eric
Nyberg, Hideki Shima, Tsuneaki Kato,
Tatsunori Mori, Chin-Yew Lin,
Ruihua Song, Chuan-Jie Lin,
Tetsuya Sakai, Donghong Ji,
et al. 2008.Overview of the NTCIR-7 ACLIA Tasks: Advanced
Cross-Lingual Information Access.. In *NTCIR*.
* Nair et al. (2022)Suraj Nair, Eugene Yang,
Dawn Lawrie, Kevin Duh,
Paul McNamee, Kenton Murray,
James Mayfield, and Douglas W. Oard.
2022.Transfer Learning Approaches for Building
Cross-Language Dense Retrieval Models. In *Advances
in Information Retrieval: 44th European Conference on IR Research, ECIR 2022,
Stavanger, Norway, April 10–14, 2022, Proceedings, Part I* (Stavanger,
Norway). Springer-Verlag, Berlin,
Heidelberg, 382–396.
* Nair
et al. (2023)Suraj Nair, Eugene Yang,
Dawn Lawrie, James Mayfield, and
Douglas W. Oard. 2023.BLADE: Combining Vocabulary Pruning and
Intermediate Pretraining for Scaleable Neural CLIR. In*Proceedings of the 46th International ACM SIGIR
Conference on Research and Development in Information Retrieval* (Taipei,
Taiwan) *(SIGIR ’23)*. Association
for Computing Machinery, New York, NY, USA,
1219–1229.
* Nguyen et al. (2016)Tri Nguyen, Mir
Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and
Li Deng. 2016.MS MARCO: A Human Generated MAchine Reading
COmprehension Dataset.*arXiv preprint arXiv:1611.09268*(2016).arXiv:1611.09268[http://arxiv.org/abs/1611.09268](http://arxiv.org/abs/1611.09268 "")
* Nogueira
et al. (2020)Rodrigo Nogueira, Zhiying
Jiang, Ronak Pradeep, and Jimmy Lin.
2020.Document Ranking with a Pretrained
Sequence-to-Sequence Model. In *Findings of the
Association for Computational Linguistics: EMNLP 2020*.
Association for Computational Linguistics,
Online, 708–718.[https://doi.org/10.18653/v1/2020.findings-emnlp.63](https://doi.org/10.18653/v1/2020.findings-emnlp.63 "")
* Peters and
Braschler (2002)Carol Peters and Martin
Braschler. 2002.The Importance of Evaluation for Cross-Language
System Development: the CLEF Experience.. In*LREC*.
* Peters
et al. (2012)Carol Peters, Martin
Braschler, and Paul Clough.
2012.*Multilingual information retrieval: From
research to practice*.Springer.
* Qu et al. (2021)Yingqi Qu, Yuchen Ding,
Jing Liu, Kai Liu,
Ruiyang Ren, Wayne Xin Zhao,
Daxiang Dong, Hua Wu, and
Haifeng Wang. 2021.RocketQA: An Optimized Training Approach to
Dense Passage Retrieval for Open-Domain Question Answering. In*Proceedings of the 2021 Conference of the North
American Chapter of the Association for Computational Linguistics: Human
Language Technologies*. Association for Computational
Linguistics, Online, 5835–5847.
* Rahimi
et al. (2015)Razieh Rahimi, Azadeh
Shakery, and Irwin King.
2015.Multilingual information retrieval in the language
modeling framework.*Information Retrieval Journal*18, 3 (2015),
246–281.
* Reimers and
Gurevych (2019)Nils Reimers and Iryna
Gurevych. 2019.Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks. In *Proceedings of the 2019
Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP)*. Association for Computational Linguistics.
* Santhanam et al. (2022)Keshav Santhanam, Omar
Khattab, Jon Saad-Falcon, Christopher
Potts, and Matei Zaharia.
2022.ColBERTv2: Effective and Efficient Retrieval
via Lightweight Late Interaction. In *Proceedings
of the 2022 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies*.
Association for Computational Linguistics,
Seattle, United States, 3715–3734.
* Schuirmann (1987)Donald J Schuirmann.
1987.A comparison of the two one-sided tests procedure
and the power approach for assessing the equivalence of average
bioavailability.*Journal of pharmacokinetics and
biopharmaceutics* 15 (1987),
657–680.
* Shi and Lin (2019)P Shi and J Lin.
2019.Cross-lingual relevance transfer for document
retrieval.*arXiv preprint arXiv:1911.02989*(2019).
* Si
et al. (2008)Luo Si, Jamie Callan,
Suleyman Cetintas, and Hao Yuan.
2008.An effective and efficient results merging strategy
for multilingual information retrieval in federated search environments.*Information Retrieval* 11,
1 (2008), 1–24.
* Tsai
et al. (2008)Ming-Feng Tsai, Yu-Ting
Wang, and Hsin-Hsi Chen.
2008.A study of learning a merge model for multilingual
information retrieval. In *Proceedings of the 31st
Annual International ACM SIGIR Conference on Research and Development in
Information Retrieval*. 195–202.
* Xu and Weischedel (2000)Jinxi Xu and Ralph
Weischedel. 2000.Cross-lingual information retrieval using hidden
Markov models. In *2000 Joint SIGDAT Conference
on Empirical Methods in Natural Language Processing and Very Large Corpora*.
95–103.
* Xue et al. (2021)Linting Xue, Noah
Constant, Adam Roberts, Mihir Kale,
Rami Al-Rfou, Aditya Siddhant,
Aditya Barua, and Colin Raffel.
2021.mT5: A Massively Multilingual Pre-trained
Text-to-Text Transformer. In *Proceedings of the
2021 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies*.
Association for Computational Linguistics,
Online, 483–498.[https://doi.org/10.18653/v1/2021.naacl-main.41](https://doi.org/10.18653/v1/2021.naacl-main.41 "")
* Yang
et al. (2024a)Eugene Yang, Dawn Lawrie,
James Mayfield, Douglas W Oard, and
Scott Miller. 2024a.Translate-Distill: Learning Cross-Language Dense
Retrieval by Translation and Distillation. In*Advances in Information Retrieval: 46th European
Conference on IR Research, ECIR 2024*.
* Yang et al. (2022)Eugene Yang, Suraj Nair,
Ramraj Chandradevan, Rebecca
Iglesias-Flores, and Douglas W. Oard.
2022.C3: Continued Pretraining with Contrastive Weak
Supervision for Cross Language Ad-Hoc Retrieval. In*Proceedings of the 45th International ACM SIGIR
Conference on Research and Development in Information Retrieval* (Madrid,
Spain) *(SIGIR ’22)*. Association
for Computing Machinery, New York, NY, USA,
2507–2512.[https://doi.org/10.1145/3477495.3531886](https://doi.org/10.1145/3477495.3531886 "")
* Yang et al. (2024b)Eugene Yang, Suraj Nair,
Dawn Lawrie, James Mayfield,
Douglas W Oard, and Kevin Duh.
2024b.Efficiency-Effectiveness Tradeoff of Probabilistic
Structured Queries for Cross-Language Information Retrieval.*arXiv preprint arXiv:2404.18797*(2024).[https://arxiv.org/abs/2404.18797](https://arxiv.org/abs/2404.18797 "")
* Zhang
et al. (2021)Xinyu Zhang, Xueguang Ma,
Peng Shi, and Jimmy Lin.
2021.Mr. TyDi: A Multi-lingual Benchmark for Dense
Retrieval. In *Proceedings of the 1st Workshop on
Multilingual Representation Learning*. Association for
Computational Linguistics, Punta Cana, Dominican
Republic, 127–137.[https://aclanthology.org/2021.mrl-1.12](https://aclanthology.org/2021.mrl-1.12 "")
