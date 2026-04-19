History-Aware Conversational Dense Retrieval
=============================================

Fengran Mo1, Chen Qu2, Kelong Mao3, Tianyu Zhu1,4, Zhan Su1,5∗, Kaiyu Huang6, Jian-Yun Nie1  
1University of Montreal, Quebec, Canada;2University of Massachusetts Amherst, USA  
3Renmin University of China;4Beihang University, China;  
5University of Copenhagen, Denmark6Beijing Jiaotong University, China;  
fengran.mo@umontreal.ca, nie@iro.umontreal.ca  
∗This work was done when Tianyu Zhu and Zhan Su were working at University of Montreal.

###### Abstract

Conversational search facilitates complex information retrieval by enabling multi-turn interactions between users and the system. Supporting such interactions requires a comprehensive understanding of the conversational inputs to formulate a good search query based on historical information. In particular, the search query should include the relevant information from the previous conversation turns.
However, current approaches for conversational dense retrieval primarily rely on fine-tuning a pre-trained ad-hoc retriever using the whole conversational search session, which can be lengthy and noisy.
Moreover, existing approaches are limited by the amount of manual supervision signals in the existing datasets.
To address the aforementioned issues, we propose a History-Aware Conversational Dense Retrieval (HAConvDR) system, which incorporates two ideas: context-denoised query reformulation and automatic mining of supervision signals based on the actual impact of historical turns.
Experiments on two public conversational search datasets demonstrate the improved history modeling capability of HAConvDR, in particular for long conversations with topic shifts.

1 Introduction
--------------

Conversational search is expected to be the next generation of search engines*Gao et al. ([2022](#bib.bib6 ""))*. It aims to satisfy complex user information needs via multi-turn interactions between a user and the system.
In single-turn ad-hoc search, users typically employ stand-alone queries to convey their information requirements*Bajaj et al. ([2016](#bib.bib3 ""))* in a brief and clearly-expressed manner.
In conversational search, however, queries are usually context-dependent, which highlights the necessity of understanding the search intent within the conversational context.

To uncover the user’s information need, conversational query rewriting (CQR)*(Yu et al., [2020](#bib.bib36 ""); Wu et al., [2021](#bib.bib32 ""); Mo et al., [2023a](#bib.bib20 ""))* employs human-rewritten queries to train a rewriting model that generates de-contextualized queries.
However, obtaining large-scale manual annotations for this purpose is challenging in practice. Besides, CQR models cannot be directly optimized from the downstream retrieval task*Wu et al. ([2021](#bib.bib32 "")); Mo et al. ([2023a](#bib.bib20 ""))*.

In comparison, a more desirable approach is to perform end-to-end conversational dense retrieval (CDR) by training a query encoder that incorporates conversation history*Qu et al. ([2020](#bib.bib24 "")); Yu et al. ([2021](#bib.bib37 ""))*. Since human annotations are usually not available to indicate which conversation turns are relevant to the current query, a common practice is to utilize all historical turns to reformulate the current query as the input to the model.

However, the conversation history can be lengthy and often includes a substantial amount of noise, i.e., historical turns that are irrelevant to the current query.
Despite the observation*Adlakha et al. ([2022](#bib.bib1 ""))* that conversational sessions often center around a specific topic (e.g., sports), it is worth noting that historical turns may focus on different aspects (e.g., match results, or player statistics). Some of them are relevant to the current turn, while others may not. This is especially the case when conversations are long.
This problem can make models suffer from the shortcut history dependency issue*Kim and Kim ([2022](#bib.bib11 ""))*, where models are excessively focused on historical turns while neglecting the current query due to the challenge to comprehend the current information need. We illustrate this issue via an example in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ History-Aware Conversational Dense Retrieval"). Given the current query $q_{7}$, instead of retrieving the passage $p_{7}^{*}$ (addressing the current information need) in top-ranked positions, the retriever ranks $p_{6}^{*}$ and $p_{2}^{*}$ (addressing historical information needs) higher than $p_{7}^{*}$.

To tackle the aforementioned challenge, we put forward HAConvDR, a new History-Aware Conversational Dense Retrieval method. Our approach consists of two prongs of enhancements as detailed in the following sections.

The first prong is to incorporate an explicit denoising mechanism into the model training process so that the model is less affected by the noisy history while being history-aware.
To achieve a similar purpose, recent studies*Mao et al. ([2022a](#bib.bib17 ""), [2023c](#bib.bib19 "")); Mo et al. ([2023b](#bib.bib21 ""))* typically assess whether a historical turn is relevant to the current turn based on the historical query. However, these approaches are inherently lacking because historical queries alone are often not sufficient to fully cover the historical context. To address this shortcoming, we additionally leverage the passages associated with historical queries to better evaluate the intent of a historical turn. Specifically, we use a pseudo-labeling approach to assess the relevance and usefulness of the historical turns – whether they contribute to improving the retrieval effectiveness of the current query. We then retain the relevant historical turns for context-denoised query reformulation.

The second prong is to mine additional supervision signals to further alleviate the pitfall of shortcut history dependency. Despite having context-denoised queries, a single ground-truth passage (given by the dataset) is often indirect and insufficient to guide the training of conversational retrieval due to the remaining noise in the formulated query.
Thus, mining additional supervisions, either positive*Mao et al. ([2022b](#bib.bib18 ""))* or negative*Kim and Kim ([2022](#bib.bib11 ""))*, can enhance the original supervision signal and reduce the negative impact by the distractors in the conversation history.
Different from the aforementioned work that acquires additional supervisions by human annotation or retrieval, we mine pseudo positive and hard negative supervisions from the conversation history based on the same relevance judgment of historical turns used for query reformulation. Intuitively, among the top-ranked historical ground-truth passages in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ History-Aware Conversational Dense Retrieval"), some of them can be highly relevant to the current query, which resembles the pseudo relevant documents in Pseudo Relevance Feedback*Xu and Croft ([1996](#bib.bib34 ""))*, while others are less relevant and can serve as hard negatives for training. These additional supervisions enable the model to be aware of the usefulness or harmfulness of historical ground-truth passages and are used in our history-aware contrastive learning process.

<img src='x1.png' alt='Refer to caption' title='' width='461' height='218' />

*Figure 1: Illustration of shortcut history dependency – passages addressing historical information needs ($p_{6}^{*}$ and $p_{2}^{*}$) can be ranked higher than those addressing the current information need ($p_{7}^{*}$), due to the noise in the reformulated query.*

With our approach HAConvDR incorporated with the two prongs of enhancement described above, we carry out extensive experiments on two conversational search datasets, and the results show our method outperforms most existing strong baselines, demonstrating how relevance judgments of historical turns can benefit conversational retrieval.

Our contributions are summarized as follows: (1) We propose HAConvDR to train a history-aware conversational dense retriever by using the ground-truth passage from historical turns as additional supervision signals.
(2) We conduct pseudo relevance judgment on selecting historical turns to denoise the context for query reformulation, whose results are the foundation of mining additional supervision signals.
(3) We demonstrate the effectiveness of HAConvDR by outperforming different types of strong baselines on two public datasets.
A series of analyses are conducted to understand how historical ground-truth passages work well to solve the conversation with lots of topic shifts.

2 Related Work
--------------

Conversational Query Reformulation. This approach aims to reformulate an explicit query via training a CQR model. Typical methods include query rewriting*Yu et al. ([2020](#bib.bib36 "")); Lin et al. ([2020](#bib.bib14 "")); Vakulenko et al. ([2021](#bib.bib27 "")); Qian and Dou ([2022](#bib.bib23 "")); Mao et al. ([2023a](#bib.bib15 ""), [b](#bib.bib16 ""))* and query expansion*Kumar and Callan ([2020](#bib.bib12 "")); Voskarides et al. ([2020](#bib.bib30 ""))*, which aim to mimic a human query rewriting or selecting useful terms from historical context for expansion. However, the manual annotations needed for training are difficult to obtain in practice and the human-rewritten queries might not necessarily be the optimal search queries*Mo et al. ([2023a](#bib.bib20 ""))*.

Conversational Dense Retrieval. Another research direction is to perform conversational dense retrieval, which leverages the conversational search data to fine-tune a well-trained ad-hoc retriever. Existing studies*Yu et al. ([2021](#bib.bib37 "")); Lin et al. ([2021](#bib.bib13 "")); Mao et al. ([2022b](#bib.bib18 ""))* usually focus on few-shot scenarios or rely on external resources, which ignore the context denoising. To this end, some recent work*Mao et al. ([2022a](#bib.bib17 ""), [2023c](#bib.bib19 "")); Mo et al. ([2023b](#bib.bib21 ""))* designs sophisticated mechanisms to enhance the denoising ability explicitly and implicitly for the models. However, they do not take into account historical feedback.
To further exploit the advantages of context-denoising, our method explicitly selects the useful historical turns, as well as their ground-truth passage via pseudo relevant judgment before model training.

Supervision Signals in Dense Retrieval.*Robinson et al. ([2021](#bib.bib26 ""))* demonstrates that sufficient supervision signals, either positive or negative, are important for the contrastive learning framework, especially the hard negatives. For dense retrieval, hard negatives are usually mined by BM25*Karpukhin et al. ([2020](#bib.bib10 ""))* or a vanilla backbone model*Xiong et al. ([2020](#bib.bib33 ""))*. In the conversational scenario,*Kim and Kim ([2022](#bib.bib11 ""))* uses the CQR model to construct hard negatives and*Mao et al. ([2022a](#bib.bib17 ""))* relies on human annotators to generate augmented positives, but the amount is limited. Differently, our method leverages additional supervision signals from the historical ground-truth passages to enhance the model’s history-awareness (e.g., enjoying the efficiency and avoiding the harmfulness).

<img src='x2.png' alt='Refer to caption' title='' width='461' height='128' />

*Figure 2: Overview of HAConvDR. The first stage (left) is to conduct pseudo relevance judgment (PRJ) between the current query and each historical turn. Based on the PRJ results, the second stage (middle) is to perform context-denoised query reformulation and positive and negative supervision signals mining. The third stage (right) is to conduct conversational dense retrieval training with history-aware contrastive learning.*

3 Methodology
-------------

### 3.1 Task Definition

We are given a conversation session that contains the current query $q_{n}$, and $n-1$ historical turns preceding $q_{n}$. The $i$-th historical turn is denoted as $(q_{i},p_{i}^{*})$, where $q_{i}$ is a historical query and $p_{i}^{*}$ is the historical ground-truth passage that addresses the information need of $q_{i}$. Our task is to retrieve the passage $p_{n}^{*}$ from a passage collection $\mathcal{D}$ to satisfy the information need in $q_{n}$. Our usage of historical ground-truth passages $\mathcal{P}_{h}\={p_{i}^{*}}_{i\=1}^{n-1}$ is consistent with the settings adopted in previous work on conversational search*Choi et al. ([2018](#bib.bib5 "")); Qu et al. ([2019](#bib.bib25 ""))*. In some real-world applications, if $\mathcal{P}_{h}$ is not available, it can be replaced with a set of top-ranked documents for those turns. We discuss and analyze such adaptation in Sec.[4.5](#S4.SS5 "4.5 Impact of Substituting Historical Ground-Truth Passages ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval") for generalizability.

### 3.2 Model Overview

As illustrated in Figure[2](#S2.F2 "Figure 2 ‣ 2 Related Work ‣ History-Aware Conversational Dense Retrieval"), our proposed HAConvDR approach consists of three stages. The first stage is to generate the pseudo relevance judgments (PRJs) for historical turns that evaluate whether a given turn $(q_{i},p_{i}^{*})$ is relevant to the current query $q_{n}$. This is achieved by a pseudo-labeling approach presented in Sec.[3.3](#S3.SS3 "3.3 Relevance Judgement for Historical Turns ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval"). In the second stage, we leverage the generated PRJs of historical turns for two purposes. The first purpose is to use the relevant historical turns to perform a context-denoised query reformulation (Sec.[3.4](#S3.SS4 "3.4 Context-Denoised Query Reformulation ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval")), while the second purpose is to create additional positive and negative training pairs by leveraging historical passages according to their PRJs. Given the reformulated queries and the augmented training pairs from conversation history, we perform history-aware contrastive learning for a dual-encoder model in the third stage (Sec.[3.5](#S3.SS5 "3.5 History-Aware Contrastive Learning ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval")). We highlight that, under our approach, the conversation history becomes an asset that is rich in supervision signals, rather than a liability that we need to handle. We illustrate each stage in the following sections.

### 3.3 Relevance Judgement for Historical Turns

A common practice to obtain a conversational dense retriever is to adapt models for ad-hoc retrieval to a conversational setting by concatenating the entire conversation history to the current query.
In theory, the attention mechanism within the backbone transformer should allow the adapted retriever to implicitly conduct history modeling.
In practice, however, the attention can be easily distracted by the irrelevant information in the conversation history.
Therefore, we argue that it is essential to judge whether a historical turn is relevant to the current turn as part of the history modeling process.

In the literature of information retrieval, relevance is used to denote how well a document meets the information need of a query. Here, we take the liberty of using the same term to describe whether a historical turn is relevant to the current query.

Learning to judge the relevance of historical turns is non-trivial
because conversation datasets rarely contain such labels. *Mo et al. ([2023b](#bib.bib21 ""))* addresses this issue by adopting a simple and effective rule-based approach to derive pseudo labels – a historical query $q_{i}$ is judged relevant if concatenating it to the current query $q_{n}$ leads to an improved retrieval performance for $q_{n}$ (similar to selecting query expansion terms as in*Cao et al. ([2008](#bib.bib4 ""))*). This pseudo-labeling approach is referred to as pseudo relevance judgment for historical turns.
However, this approach is inherently lacking because historical queries alone is often not sufficient to fully articulate the historical context.

To address this issue, we draw inspiration from a fundamental approach in ad-hoc retrieval – Pseudo Relevance Feedback (PRF)*Xu and Croft ([1996](#bib.bib34 ""))*. PRF assumes the top-retrieved documents of the initial query are often relevant and can be used to enrich the original query to better express the information need.
Given this empirical guidance, we propose to perform PRJ for a historical turn $i$ by additionally leveraging its passage $p_{i}^{*}$.

To be specific, we illustrate the enhanced history relevance judgment process in Alg.[1](#alg1 "Algorithm 1 ‣ 3.3 Relevance Judgement for Historical Turns ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval").
As a result, the historical ground-truth passages $\mathcal{P}_{h}\={p_{i}^{*}}_{i\=1}^{n-1}$ for turn $i$ is divided into two disjoint groups
:

|  | $\mathcal{P}^{+}_{h}\={p^{*}_{j}}_{j\=1},\quad\mathcal{P}^{-}_{h}\={p^{*}_{k}}_{k\=1}$ |  | (1) |
| --- | --- | --- | --- |

where $\mathcal{P}^{+}_{h}$ denotes the relevant passage group and $\mathcal{P}^{-}_{h}$ denotes the irrelevant passage group, with $|\mathcal{P}^{+}_{h}|+|\mathcal{P}^{-}_{h}|\=n-1$.
For the use case where historical ground-truth passages are not available, we demonstrate that top-retrieved passages can serve as a substitute in Sec.[4.5](#S4.SS5 "4.5 Impact of Substituting Historical Ground-Truth Passages ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval").

*Algorithm 1  Generating pseudo relevance judgments for historical turns*

0:current query $q_{n}$, historical turn ($q_{i}$, $p_{i}^{*}$), retriever $\phi$, retrieval evaluation metric $\mathcal{M}$

1:RankList-raw $\leftarrow\phi(q_{n})$

2:RankList-reform. $\leftarrow\phi(q_{n}\circ q_{i}\circ p_{i}^{*})$

3:Score-raw $\leftarrow\mathcal{M}(\text{RankList-raw})$

4:Score-reform. $\leftarrow\mathcal{M}(\text{RankList-reform.})$

5: ifScore-reform. > Score-rawthen

6:$\text{PRJ}(q_{n},(q_{i},p_{i}^{*}))\leftarrow\text{relevant}$

7: else

8:$\text{PRJ}(q_{n},(q_{i},p_{i}^{*}))\leftarrow\text{irrelevant}$

9: end if

10: Output $\text{PRJ}(q_{n},(q_{i},p_{i}^{*}))$

### 3.4 Context-Denoised Query Reformulation

Based on the PRJs of historical turns derived in Sec.[3.3](#S3.SS3 "3.3 Relevance Judgement for Historical Turns ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval"), we reformulate the current query $q_{n}$ to obtain the context-denoised query $q^{r}_{n}$:

|  | $\quad\quad\quad q^{r}_{n}\=q_{n}\circ\cdots p_{i}^{*}\circ q_{i}\cdots$ |  | (2) |
| --- | --- | --- | --- |

where $q_{i}$ and $p_{i}^{*}$ are from relevant historical turns, and $\circ$ denotes concatenation.

Since the reformulated query contains historical passages $\mathcal{P}_{h}^{+}$, a potential concern arises regarding the length of the reformulated query – it might exceed the input length limitations of some pre-trained language models.
However, the analysis of the generated PRJ statistics, as presented later in Sec.[4.3](#S4.SS3 "4.3 Deep Dive of PRJs of Historical Turns ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval"), reveals that only a small portion of historical turns are deemed relevant and used for query reformulation.
This proves the practical value of our approach.
Nonetheless, in our future work, we aim to incorporate a more sophisticated mechanism to further select the relevant sections within the historical passages in $\mathcal{P}_{h}^{+}$.

### 3.5 History-Aware Contrastive Learning

Contrastive learning is a prevalent approach to train dense retrievers*Karpukhin et al. ([2020](#bib.bib10 ""))*. This approach first projects queries and passages into an embedding space with dual encoders $\mathcal{F}_{Q}$ and $\mathcal{F}_{P}$. It then evaluates the relevance of any given pair of query and passage $(q,p)$ by taking the dot product similarity $\mathcal{S}(q,p)\=\mathcal{F}_{Q}(q)^{T}\cdot\mathcal{F}_{P}(p)$. Finally, supervision signals are derived from the contrast that the distance between a query and a relevant passage (positive pair) should be closer than that between the same query and an irrelevant passage (negative pair). These supervision signals are back-propagated to train the encoders.

In a research setting, for the current query $q_{n}$, the relevant passage (positive passage) is the ground-truth passage $p_{n}^{*}$ given by the dataset. For the irrelevant passages (negative passages), one option is to simply take the passages other than $p_{n}^{*}$ found in the same training batch. These negative passages are referred to as in-batch negatives, here denoted as $\mathcal{P}^{-}_{b}$. In addition to in-batch negatives, another commonly adopted approach is to leverage retrieved hard negatives $\mathcal{P}^{-}_{r}$*Lin et al. ([2021](#bib.bib13 "")); Kim and Kim ([2022](#bib.bib11 "")); Karpukhin et al. ([2020](#bib.bib10 ""))*. One way to obtain such negatives is to use the top-ranked passages retrieved with $q_{n}$ by an off-the-shelf retriever (e.g., BM25) after removing $p_{n}^{*}$ (if present). Supervision signals generated from these retrieved negatives are believed to be more meaningful than those from in-batch negatives. The power of retrieved negatives suggests that the effectiveness of supervision signals could be heavily impacted by the quality and quantity of the positive and negative pairs.

Given the insight that augmenting positive and negative pairs can boost retrieval performance, we propose to mine additional pairs to further enhance the contrastive learning process. For this very purpose, we found the PRJs of historical turns derived in Sec.[3.3](#S3.SS3 "3.3 Relevance Judgement for Historical Turns ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval") come in handy.

Intuitively, $\mathcal{P}_{h}^{+}$ contains historical passages from the historical turns that are deemed relevant to $q_{n}$. Although $\mathcal{P}_{h}^{+}$ may not directly address the information need of $q_{n}$, $\mathcal{P}_{h}^{+}$ helps enhance or complement $q_{n}$. We believe this relationship can serve as a proxy to claim a certain level of relevance between $\mathcal{P}_{h}^{+}$ and $q_{n}$. Therefore, we use $\mathcal{P}_{h}^{+}$ as pseudo positives.
Similarly, passages in $\mathcal{P}_{h}^{-}$ are less relevant to $q_{n}$ as demonstrated by the irrelevant PRJs. So we use $\mathcal{P}_{h}^{-}$ as additional negatives. More importantly, $\mathcal{P}_{h}^{-}$ resembles retrieved negatives $\mathcal{P}^{-}_{r}$ in the sense that both are hard negatives that can generate more meaningful supervisions. We refer to $\mathcal{P}_{h}^{-}$ as historical hard negatives.

By leveraging these pseudo positives and historical hard negatives mined from the conversation history, we upgrade traditional contrastive learning to history-aware contrastive learning.
Formally, we denote the final positive and negative passages used for training as follows:

|  | $\displaystyle\mathcal{P}^{+}_{n}\={p^{*}_{n}}\cup\mathcal{P}^{+}_{h},$ | $\displaystyle|\mathcal{P}^{+}_{n}|\=N$ |  | (3) |
| --- | --- | --- | --- | --- |
| | $\displaystyle\mathcal{P}^{-}_{n}\=\mathcal{P}^{-}_{b}\cup\mathcal{P}^{-}_{r}\cup\mathcal{P}^{-}_{h},$ | $\displaystyle\ |\mathcal{P}^{-}_{n}\ | |

The final training objective is illustrated in Eq.[4](#S3.E4 "In 3.5 History-Aware Contrastive Learning ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval"), where $p^{+}_{i}\in\mathcal{P}^{+}_{n}$ and $p^{-}_{j}\in\mathcal{P}^{-}_{n}$.

|  | $\mathcal{L}\=\frac{1}{N}\sum^{N}_{i\=1}\frac{e^{\mathcal{S}\left(q^{r}_{n},p^{+}_{i}\right)}}{e^{\mathcal{S}\left(q^{r}_{n},p^{+}_{i}\right)}+\sum^{M}_{j\=1}e^{\mathcal{S}\left(q^{r}_{n},p^{-}_{j}\right)}}$ |  | (4) |
| --- | --- | --- | --- |

| Category | Method | TopiOCQA | | | | QReCC | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | MRR | NDCG@3 | R@10 | R@100 | MRR | NDCG@3 | R@10 | R@100 |
| CQR | GPT2+WS | 12.6 | 12.0 | 22.0 | 33.1 | 33.9 | 30.9 | 53.1 | 72.9 |
| | QuReTeC | 11.2 | 10.5 | 20.2 | 34.3 | 35.0 | 32.6 | 55.0 | 72.9 |
| CQE-sparse | 14.3 | 13.6 | 24.8 | 36.7 | 32.0 | 30.1 | 51.3 | 70.9 |
| T5QR | 23.4 | 22.5 | 39.8 | 56.2 | 34.5 | 31.8 | 53.1 | 72.8 |
| CONQRR | - | - | - | - | 41.8 | - | 65.1 | 84.7 |
| ConvGQR | 25.6 | 24.3 | 41.8 | 58.8 | 42.0 | 39.1 | 63.5 | 81.8 |
| IterCQR | 26.3 | 25.1 | 42.6 | 62.0 | 42.9 | 40.2 | 65.5 | 84.1 |
| InstructorLLM | 25.3 | 23.7 | 45.1 | 69.0 | 43.5 | 40.5 | 66.7 | 85.6 |
| LLM-Aided IQR | - | - | - | - | 45.0 | - | 67.3 | - |
| CDR | Conv-ANCE | 22.9 | 20.5 | 43.0 | 71.0 | 47.1 | 45.6 | 71.5 | 87.2 |
| | SDRConv | 26.1 | 25.4 | 44.4 | 63.2 | 47.3 | 43.6 | 69.8 | 88.4 |
| ConvDR | 27.2 | 26.4 | 43.5 | 61.1 | 38.5 | 35.7 | 58.2 | 77.8 |
| HAConvDR (Ours) | 30.1† | 28.5† | 50.8† | 72.8† | 47.7† | 44.8 | 71.6† | 88.7† |

*Table 1: Performance of different dense retrieval methods on two datasets.
$\dagger$ denotes significant improvements with t-test at $p<0.05$ over the main competitors, all CDR methods. Bold indicate the best results.*

4 Experiments
-------------

Datasets We evaluate our methods on two widely-used conversation datasets. The first is the TopiOCQA*Adlakha et al. ([2022](#bib.bib1 ""))* dataset that contains complex topic-switch phenomena within each conversational session. These sessions have the potential to conceal a wealth of supervision signals in historical turns.
The other dataset we use is QReCC*Anantha et al. ([2021](#bib.bib2 ""))*, where most queries in a conversational session are on the same topic.
The selection of the datasets assures we verify the model performance on conversations with different intrinsic characteristics and enables more informative analyses.
The statistics and more details of the datasets are provided in Appendix[A.1](#A1.SS1 "A.1 Datasets ‣ Appendix A More Detailed Experimental Setup ‣ History-Aware Conversational Dense Retrieval").

Evaluation Metrics For an adequate comparison with previous studies, we use four standard evaluation metrics: MRR, NDCG@3, Recall@10, and Recall@100 to evaluate the retrieval results.

Baselines We compare our method with two lines of conversational search approaches. The first line (CQR) performs conversational query reformulation based on generative rewriter models and off-the-shelf retrievers, including PLM-based GPT2+WS *Yu et al. ([2020](#bib.bib36 ""))*, QuReTeC *Voskarides et al. ([2020](#bib.bib30 ""))*, CQE-Sparse *Lin et al. ([2021](#bib.bib13 ""))*, T5QR *Lin et al. ([2020](#bib.bib14 ""))*, CONQRR *Wu et al. ([2021](#bib.bib32 ""))*, and ConvGQR *Mo et al. ([2023a](#bib.bib20 ""))*, and LLM-based IterCQR *Jang et al. ([2023](#bib.bib7 ""))*, InstructorLLM *Jin et al. ([2023](#bib.bib8 ""))*, and LLM-Aided IQR *Ye et al. ([2023](#bib.bib35 ""))*. The LLM-based methods employ ChatGPT or Llama as backbone models.
The second line (CDR) conducts conversational dense retrieval based on ad-hoc search dense retrievers to learn the latent representation of the reformulated query, including Conv-ANCE *Mao et al. ([2023c](#bib.bib19 ""))* using the original contrastive ranking loss, ConvDR *Yu et al. ([2021](#bib.bib37 ""))* additionally relying on human-rewritten queries as supervision signals and SDRConv *Kim and Kim ([2022](#bib.bib11 ""))* mining additional hard negatives.

Implementation Details The backbone model for conversational dense retriever training is ANCE*Xiong et al. ([2020](#bib.bib33 ""))* and the dense retrieval is performed using Faiss*(Johnson et al., [2019](#bib.bib9 ""))*.
During training, we only update the parameters of the query encoder while keeping the passage encoder frozen.
The number of mined positives and negatives from historical turns can vary across different query turns. Instead of trying to utilize all of them, we randomly select one historical pseudo positive and one historical hard negative (along with the top retrieved hard negative) for each training instance to strike a balance between effectiveness and efficiency.
More details are provided in Appendix[A.2](#A1.SS2 "A.2 Implementation Details ‣ Appendix A More Detailed Experimental Setup ‣ History-Aware Conversational Dense Retrieval") and our code.111<https://github.com/fengranMark/HAConvDR>

### 4.1 Main Results

The main evaluation results on TopiOCQA and QReCC datasets are reported in Table[1](#S3.T1 "Table 1 ‣ 3.5 History-Aware Contrastive Learning ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval").

We find that our method achieves a significantly better performance on both datasets compared with other methods on most metrics. In particular, it improves MRR by 10.7% and NDCG@3 by 8.0% on TopiOCQA over the second-best results ConvDR.
The superior effectiveness can be attributed to the following two aspects. (1) The context-denoised query reformulation and history-aware contrastive learning with mined supervision signals enhance the top-ranking ability of our HAConvDR. (2) Conversational dense retrieval tends to be more effective compared with conversational query rewriting pipelines, including those try to leverage the powerful generation capacity of LLMs.
Besides, the improvements achieved over Conv-ANCE serve as additional validation of the effectiveness of exploiting supplementary supervision signals derived from ground-truth information of past interactions and confirm our underlying assumption.

Moreover, we find that performance improvements are more pronounced on TopiOCQA. This can be attributed to the characteristics of the datasets: the session context is longer in TopiOCQA, and contains more noise. This comparison indicates that our method has a greater potential for longer sessions with topic shifts. In contrast, the turns in QReCC are usually on the same topic, and the ground-truth passages of historical turns can also properly address the information need of the current query. In such a situation, most previous turns can be relevant, making it less critical to select the relevant turns. Notice that TopiOCQA provides a better simulation of real-world scenarios, where a conversation (or search) session is expected to be on related but different topics. Our results demonstrate that our approach is better at addressing this practical situation. More analysis on this is provided in Sec.[4.3](#S4.SS3 "4.3 Deep Dive of PRJs of Historical Turns ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval") and [4.4](#S4.SS4 "4.4 Impact of Historical Supervision Signals ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval").

### 4.2 Ablation Study

Compared to the contrastive learning technique in conversational dense retrieval, our proposed method introduces two extra components, i.e., context-denoised query reformulation and history-aware contrastive signals comprising historical pseudo positives and historical hard negatives. To assess the effectiveness of these individual components, we conduct an ablation study and present the analysis in Table[2](#S4.T2 "Table 2 ‣ 4.2 Ablation Study ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval").

We observe that, on both datasets, removing pseudo positives can cause a more pronounced performance degradation compared with removing hard negatives. This suggests that, although both hard negatives and pseudo positives are useful, the latter serves as a more effective supervision. This insight complements the currently prevalent studies on negative mining. On the other hand, we observe the decrease is more prominent on TopiOCQA, which is true for both removing hard negatives and pseudo positives. This can be attributed to the prevalence of topic-switch phenomena within the sessions in TopiOCQA, where historical supervisions can and should be leveraged to boost performance as illustrated in our approach.

|  | TopiOCQA | | QReCC | |
| --- | --- | --- | --- | --- |
|  | MRR | NDCG@3 | MRR | NDCG@3 |
| Ours | 30.1 | 28.5 | 47.7 | 44.8 |
| - hard neg. | 28.2 | 26.6 | 47.4 | 44.5 |
| - pse. pos. | 26.8 | 25.3 | 46.8 | 44.1 |
| - QR w/ PRJ | 25.0 | 23.0 | 44.5 | 41.4 |

*Table 2: Ablation study of different strategies.*

### 4.3 Deep Dive of PRJs of Historical Turns

The PRJs of historical turns are the foundation of context-denoised query reformulation and history-aware contrastive learning.
The ablation study in Sec.[4.2](#S4.SS2 "4.2 Ablation Study ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval") has shown the effectiveness of the approach. In this section, we take a deeper look to reveal the intuition behind the performance gain. Specifically, for a given turn ID $n$, we pool all historical turns in the dataset and compute the percentage of relevant ones as deemed by PRJs. Intuitively, this number denotes, on average, the portion of relevant historical turns over all historical turns. We plot this number against the turn ID in Figure[3](#S4.F3 "Figure 3 ‣ 4.3 Deep Dive of PRJs of Historical Turns ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval").

We observe that, overall, the relevant historical turns are only a fraction of all historical turns (up to 20%). This verifies the necessity to perform PRJ for historical turns for context-denoised query reformulation.
In addition, we see that the portion of the relevant history of TopiOCQA is generally greater than that of QReCC. This shows that our approach is reacting well to the abundant topic-switch phenomena in TopiOCQA. The PRJs derived from the topic-switches become the source of effectiveness for context-denoised query reformulation and history-aware contrastive learning, which finally results in pronounced gains on TopiOCQA.

Interestingly, the curves of both datasets show an intriguing trend of decrease-then-plateau. In the decreasing region, the amount of relevant history information does not scale as fast as the conversation. This shows the first several rounds of interactions have concentrated dependency on history. In contrast, as the conversation evolves, the amount of relevant history grows proportionally with the conversation (resulting in a plateaued percentage), which indicates a consistent and wide-spread dependency on history. We believe this insight on the change of history dependency over turns can inform future design of history modeling approaches.

<img src='x3.png' alt='Refer to caption' title='' width='461' height='285' />

*Figure 3: Portion of relevant historical turns over all historical turns, as conversations evolve.*

### 4.4 Impact of Historical Supervision Signals

To evaluate how our HAConvDR alleviates the phenomenon that models tend to retrieve the ground-truth passages of historical turns rather than that of the current turn, we conduct an analysis of the impact of historical supervision signals.

Quantitative Analysis The quantitative analysis is presented in Fig.[4](#S4.F4 "Figure 4 ‣ 4.4 Impact of Historical Supervision Signals ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval"), which shows the percentage of the queries that rank the historical ground-truth passages higher than that of the current turn. We observe that our model can decrease the percentage of irrelevant historical gold passages for TopiOCQA, but not much for QReCC.
Such observation indicates the supervision signals for history-aware contrastive learning are stronger in TopiOCQA than in QReCC and it is consistent with the observation in Sec.[4.1](#S4.SS1 "4.1 Main Results ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval") that the improvements in TopiOCQA are more obvious than that of QReCC.

<img src='x4.png' alt='Refer to caption' title='' width='461' height='269' />

*Figure 4: The percentage of the queries whose retrieved list has the ground-truth passage of the historical turns ranked higher than its own.*

<img src='x5.png' alt='Refer to caption' title='' width='461' height='230' />

*Figure 5: T-SNE visualization of query, ground-truth passage, and pseudo positives and history hard negatives embeddings via two ANCE models with and without HAConvDR training.*

Qualitative Analysis To gain more insights into our approach, we did a qualitative study to visualize an example in the embedding space as Figure[5](#S4.F5 "Figure 5 ‣ 4.4 Impact of Historical Supervision Signals ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval"), which shows T-SNE visualization*Van der Maaten and Hinton ([2008](#bib.bib28 ""))* to compare ANCE dense retriever with and without HAConvDR training in the embedding space.
In contrast to the vanilla ANCE, which is unsuccessful in distinguishing the gold passage from the ground-truth of the historical turns, the ANCE trained with our HAConvDR exhibits a stronger ability to differentiate it from the distractors.
Besides, our model can also discriminate the gold passages of relevant and irrelevant turns, showing the effectiveness of these supervision signals toward better search results. The corresponding example is provided in Appendix[B](#A2 "Appendix B Qualitative Example ‣ History-Aware Conversational Dense Retrieval").

### 4.5 Impact of Substituting Historical Ground-Truth Passages

The computation of PRJs for historical turns relies on having access to historical ground-truth passages ${p_{i}^{*}}_{i\=1}^{n-1}$. In many real-world applications, identifying ground-truth passages can be accomplished by analyzing user clicks, engagement, and feedback. However, we acknowledge that there are applications where historical ground-truth passages are difficult to obtain. In such cases, we can use top-retrieved passages as a substitute. This simple substitution allows us to perform the proposed approach described in Sec.[3](#S3 "3 Methodology ‣ History-Aware Conversational Dense Retrieval") with only minor modifications. Specifically, in Alg.[1](#alg1 "Algorithm 1 ‣ 3.3 Relevance Judgement for Historical Turns ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval") and Eq.[2](#S3.E2 "In 3.4 Context-Denoised Query Reformulation ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval"), $p_{i}^{*}$ is approximated by the concatenation of the top-$k$ retrieved passages for $q_{i}$, where $k$ is a hyper-parameter. This retrieval is completed with the same backbone model of the conversational dense retriever. Meanwhile, $\mathcal{P}^{-}_{n}$ in Eq.[3](#S3.E3 "In 3.5 History-Aware Contrastive Learning ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval") degrades to $\mathcal{P}^{-}_{b}\cup\mathcal{P}^{-}_{r}$. The rest of the approach is kept as is.

We conduct an ablation study to verify the effectiveness of our approach under this adaptation, with results presented in Table[3](#S4.T3 "Table 3 ‣ 4.5 Impact of Substituting Historical Ground-Truth Passages ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval").
We see the PRJ information still contributes to the retrieval performance of the reformulated query, further indicating its effectiveness. Besides, we find model performance degrades as $k$ increases, suggesting that longer contexts are more likely to contain noise, which cannot be entirely compensated by our approach. This suggests the potential for more advanced context-denoising approaches.
Finally, we find that using the full model with history-aware contrastive learning under the adapted setting continues to yield better results on top-ranking positions and outperforms most existing systems in Table[1](#S3.T1 "Table 1 ‣ 3.5 History-Aware Contrastive Learning ‣ 3 Methodology ‣ History-Aware Conversational Dense Retrieval").

| Method | $k$ | MRR | NDCG@3 | R@10 | R@100 |
| --- | --- | --- | --- | --- | --- |
| QR w/o PRJ | 1 | 22.66 | 21.14 | 39.57 | 61.21 |
| | 2 | 20.36 | 18.81 | 36.51 | 59.22 |
| 3 | 17.45 | 16.03 | 32.02 | 56.60 |
| QR w/ PRJ | 1 | 24.98 | 23.09 | 43.00 | 65.43 |
| | 2 | 23.54 | 22.00 | 41.28 | 63.92 |
| 3 | 21.96 | 20.43 | 38.51 | 62.13 |
| Full model | 1 | 25.94 | 24.32 | 43.12 | 65.04 |

*Table 3: Performance on TopiOCQA for the adapted use case of historical ground-truth passage substitution.*

5 Conclusion
------------

In this paper, we present a new history-aware contrastive learning strategy for conversational dense retriever training, HAConvDR, which is
based on context-denoised query reformulation and additional supervision signals mining from the historical turns. Extensive experimental results on public datasets overwhelmingly support the effectiveness of our model for conversational search.
Furthermore, we conducte comprehensive analyses to gain insights into the impact of each component of HAConvDR on the enhancements in performance and provide valuable insights on how they can work well for conversations with topic shifts to the research community.

Limitations
-----------

Our work demonstrates the feasibility of using historical ground-truth passages for query reformulation and contrastive supervision signals.
Within our proposed HAConvDR, the context used for query reformulation includes
selected historical passages,
which is usually longer than hundreds of tokens. Thus, an explicit selection mechanism on raw text or an implicit fusion method on the latent representation could be designed to reduce the risk of information loss and the effect of noise.
Besides, the historical supervised signals for model training might not be as important as the original annotation. Thus, a regulatory mechanism can be added to adjust the weight for pseudo positives within the history-aware conversational dense retrieval.

References
----------

* Adlakha et al. (2022)Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Suleman, Harm de Vries, and Siva Reddy. 2022.Topiocqa: Open-domain conversational question answering with topic switching.*Transactions of the Association for Computational Linguistics*, 10:468–483.
* Anantha et al. (2021)Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu, Shayne Longpre, Stephen Pulman, and Srinivas Chappidi. 2021.Open-domain question answering goes conversational via question rewriting.In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 520–534.
* Bajaj et al. (2016)Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. 2016.Ms marco: A human generated machine reading comprehension dataset.*arXiv preprint arXiv:1611.09268*.
* Cao et al. (2008)Guihong Cao, Jian-Yun Nie, Jianfeng Gao, and Stephen Robertson. 2008.Selecting good expansion terms for pseudo-relevance feedback.In *Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval*, pages 243–250.
* Choi et al. (2018)Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. 2018.Quac: Question answering in context.In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2174–2184.
* Gao et al. (2022)Jianfeng Gao, Chenyan Xiong, Paul Bennett, and Nick Craswell. 2022.Neural approaches to conversational information retrieval.*arXiv preprint arXiv:2201.05176*.
* Jang et al. (2023)Yunah Jang, Kang-il Lee, Hyunkyung Bae, Seungpil Won, Hwanhee Lee, and Kyomin Jung. 2023.Itercqr: Iterative conversational query reformulation without human supervision.*arXiv preprint arXiv:2311.09820*.
* Jin et al. (2023)Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. 2023.Instructor: Instructing unsupervised conversational dense retrieval with large language models.In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 6649–6675.
* Johnson et al. (2019)Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.Billion-scale similarity search with gpus.*IEEE Transactions on Big Data*, 7(3):535–547.
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.Dense passage retrieval for open-domain question answering.In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781.
* Kim and Kim (2022)Sungdong Kim and Gangwoo Kim. 2022.Saving dense retriever from shortcut dependency in conversational search.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 10278–10287. Association for Computational Linguistics.
* Kumar and Callan (2020)Vaibhav Kumar and Jamie Callan. 2020.Making information seeking easier: An improved pipeline for conversational search.In *Empirical Methods in Natural Language Processing*.
* Lin et al. (2021)Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021.Contextualized query embeddings for conversational search.In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 1004–1015.
* Lin et al. (2020)Sheng-Chieh Lin, Jheng-Hong Yang, Rodrigo Nogueira, Ming-Feng Tsai, Chuan-Ju Wang, and Jimmy Lin. 2020.Conversational question reformulation via sequence-to-sequence architectures and pretrained language models.*arXiv preprint arXiv:2004.01909*.
* Mao et al. (2023a)Kelong Mao, Zhicheng Dou, Haonan Chen, Fengran Mo, and Hongjin Qian. 2023a.Large language models know your contextual search intent: A prompting framework for conversational search.*arXiv preprint arXiv:2303.06573*.
* Mao et al. (2023b)Kelong Mao, Zhicheng Dou, Bang Liu, Hongjin Qian, Fengran Mo, Xiangli Wu, Xiaohua Cheng, and Zhao Cao. 2023b.Search-oriented conversational query editing.In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 4160–4172.
* Mao et al. (2022a)Kelong Mao, Zhicheng Dou, and Hongjin Qian. 2022a.Curriculum contrastive context denoising for few-shot conversational dense retrieval.In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 176–186.
* Mao et al. (2022b)Kelong Mao, Zhicheng Dou, Hongjin Qian, Fengran Mo, Xiaohua Cheng, and Zhao Cao. 2022b.Convtrans: Transforming web search sessions for conversational dense retrieval.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 2935–2946.
* Mao et al. (2023c)Kelong Mao, Hongjin Qian, Fengran Mo, Zhicheng Dou, Bang Liu, Xiaohua Cheng, and Zhao Cao. 2023c.Learning denoised and interpretable session representation for conversational search.In *Proceedings of the ACM Web Conference 2023*, pages 3193–3202.
* Mo et al. (2023a)Fengran Mo, Kelong Mao, Yutao Zhu, Yihong Wu, Kaiyu Huang, and Jian-Yun Nie. 2023a.Convgqr: Generative query reformulation for conversational search.*arXiv preprint arXiv:2305.15645*.
* Mo et al. (2023b)Fengran Mo, Jian-Yun Nie, Kaiyu Huang, Kelong Mao, Yutao Zhu, Peng Li, and Yang Liu. 2023b.Learning to relate to previous turns in conversational search.*arXiv preprint arXiv:2306.02553*.
* Paszke et al. (2019)Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019.Pytorch: An imperative style, high-performance deep learning library.In *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pages 8024–8035.
* Qian and Dou (2022)Hongjin Qian and Zhicheng Dou. 2022.Explicit query rewriting for conversational dense retrieval.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 4725–4737.
* Qu et al. (2020)Chen Qu, Liu Yang, Cen Chen, Minghui Qiu, W Bruce Croft, and Mohit Iyyer. 2020.Open-retrieval conversational question answering.In *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*, pages 539–548.
* Qu et al. (2019)Chen Qu, Liu Yang, Minghui Qiu, Yongfeng Zhang, Cen Chen, W. Bruce Croft, and Mohit Iyyer. 2019.Attentive history selection for conversational question answering.In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*.
* Robinson et al. (2021)Joshua Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. 2021.Contrastive learning with hard negative samples.In *International Conference on Learning Representations (ICLR)*.
* Vakulenko et al. (2021)Svitlana Vakulenko, Shayne Longpre, Zhucheng Tu, and Raviteja Anantha. 2021.Question rewriting for conversational question answering.In *Proceedings of the 14th ACM International Conference on Web Search and Data Mining*, pages 355–363.
* Van der Maaten and Hinton (2008)Laurens Van der Maaten and Geoffrey Hinton. 2008.Visualizing data using t-sne.*Journal of machine learning research*, 9(11).
* Van Gysel and de Rijke (2018)Christophe Van Gysel and Maarten de Rijke. 2018.Pytrec_eval: An extremely fast python interface to trec_eval.In *SIGIR*. ACM.
* Voskarides et al. (2020)Nikos Voskarides, Dan Li, Pengjie Ren, Evangelos Kanoulas, and Maarten de Rijke. 2020.Query resolution for conversational search with limited supervision.In *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*, pages 921–930.
* Wolf et al. (2019)Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. 2019.[Huggingface’s transformers: State-of-the-art natural language processing](http://arxiv.org/abs/1910.03771 "").*CoRR*, abs/1910.03771.
* Wu et al. (2021)Zeqiu Wu, Yi Luan, Hannah Rashkin, David Reitter, and Gaurav Singh Tomar. 2021.Conqrr: Conversational query rewriting for retrieval with reinforcement learning.*arXiv preprint arXiv:2112.08558*.
* Xiong et al. (2020)Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020.Approximate nearest neighbor negative contrastive learning for dense text retrieval.In *International Conference on Learning Representations*.
* Xu and Croft (1996)Jinxi Xu and W. Bruce Croft. 1996.[Query expansion using local and global document analysis](https://api.semanticscholar.org/CorpusID:53249280 "").In *Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*.
* Ye et al. (2023)Fanghua Ye, Meng Fang, Shenghui Li, and Emine Yilmaz. 2023.Enhancing conversational search: Large language model-aided informative query rewriting.In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 5985–6006.
* Yu et al. (2020)Shi Yu, Jiahua Liu, Jingqin Yang, Chenyan Xiong, Paul Bennett, Jianfeng Gao, and Zhiyuan Liu. 2020.Few-shot generative conversational query rewriting.In *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*, pages 1933–1936.
* Yu et al. (2021)Shi Yu, Zhenghao Liu, Chenyan Xiong, Tao Feng, and Zhiyuan Liu. 2021.Few-shot conversational dense retrieval.In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 829–838.

Appendix A More Detailed Experimental Setup
-------------------------------------------

### A.1 Datasets

The statistics of each dataset are presented in Table[4](#A1.T4 "Table 4 ‣ A.2 Implementation Details ‣ Appendix A More Detailed Experimental Setup ‣ History-Aware Conversational Dense Retrieval") where we eliminate the samples without gold passages in QReCC.
The details of each dataset are in the following:

TopiOCQA addresses the novel issue of topic switching, a common occurrence in realistic scenarios. In typical conversations, there are usually over 10 turns and a minimum of 3 topics. Furthermore, turns related to the same topic tend to have similar gold passages, thus we could leverage them as additional supervision signals.

QReCC primarily addresses the task of query rewriting by attempting to reformulate the query to approach the human-rewritten query. In comparison to TopiOCQA, QReCC involves conversations with a smaller number of turns, and most of these conversations revolve around the same topic. As a result, turns within the same conversation often yield identical gold passage results, making it possible to extract only a limited number of additional supervision signals.

### A.2 Implementation Details

We implement all models by PyTorch*Paszke et al. ([2019](#bib.bib22 ""))* and Huggingface’s Transformers*Wolf et al. ([2019](#bib.bib31 ""))*.
The experiments are conducted on one Nvidia A100 40G GPU. For conversational dense retriever training, we use Adam optimizer with 3e-5 learning rate and set the batch size as 32. The maximum length of the reformulated query and the passage as model input is 512 and 384 for TopiOCQA and both 256 for QReCC, respectively. For the compared baseline systems, we implement the main competitor methods which belong to CDR with the same number of hard negatives and batch size as ours. All the dense retrievers are initiated with ANCE.
For evaluation, We adopt the pytrec_eval tool*(Van Gysel and de Rijke, [2018](#bib.bib29 ""))* for metric computation.

| Dataset | Split | #Conv. | #Turns(Qry.) | #Collection |
| --- | --- | --- | --- | --- |
| TopiOCQA | Train | 3,509 | 45,450 | 25M |
| | Test | 205 | 2,514 | |
| QReCC | Train | 10,823 | 29,596 | 54M |
| | Test | 2,775 | 8,124 | |

*Table 4: Statistics of conversational search datasets.*

Appendix B Qualitative Example
------------------------------

Table[5](#A2.T5 "Table 5 ‣ Appendix B Qualitative Example ‣ History-Aware Conversational Dense Retrieval") presents a qualitative example corresponding to the T-SNE visualization in Figure[5](#S4.F5 "Figure 5 ‣ 4.4 Impact of Historical Supervision Signals ‣ 4 Experiments ‣ History-Aware Conversational Dense Retrieval"), which gives a comprehensive understanding of how historical ground-truth passage can benefit current query retrieval as supervision signals.

| Conversation (id:4-13) |
| --- |
| $q_{1}$: who sang all i want for christmas in 1995? (irrelevant) |
| $p_{1}$: All I Want for Christmas Is You is a Christmas song by American singer-songwriter … (536, -, -) |
| $q_{2}$: who is she? (relevant) |
| $p_{2}$: Mariah Carey (born March 27, 1969 or 1970) is an American singer-songwriter … (5, 20, 17) |
| $q_{3}$: what was her early days like? (irrelevant) |
| $p_{3}$: Mariah Carey was born in Huntington, New York, on March 27, 1969 or 1970 … (614, -, -) |
| $q_{4}$: what are some famous songs she performed during 2010? (relevant) |
| $p_{4}$: It missed out on the top one-hundred in the United Kingdom by one position … (-, -, -) |
| $q_{5}$: who composed the former mentioned one? (irrelevant) |
| $p_{5}$: Cox plated the keyboard and percussion. The background vocals were sung by … (-, -, -) |
| $q_{6}$: how did it perform in the charts? (relevant) |
| $p_{6}$: In the United States, Oh Santa! became a record-breaking entry on … (-, -, -) |
| $q_{7}$: how was it received critically? (relevant) |
| $p_{7}$: Mike Diver of the BBC wrote that Oh Santa! is a “boisterous” song … (-, -, -) |
| $q_{8}$: what was her other song about? (irrelevant) |
| $p_{8}$: Auld Lang Syne (The New Year’s Anthem) is a re-write of Auld Lang Syne … (-, -, -) |
| $q_{9}$: how was it received critically? (relevant) |
| $p_{9}$: Auld Lang Syne (The New Year’s Anthem) garnered a negative response from critics … (937, -, 322) |
| $q_{10}$: what are some philanthropic activities this singer is associated with? (relevant) |
| $p_{10}$: Carey is a philanthropist who has been involved with several … (197, 502, 31) |
| $q_{11}$: what does the latter mentioned foundation do? (relevant) |
| $p_{11}$: The Make-A-Wish Foundation is a 501(c)(3) nonprofit organization founded in … (-, -, -) |
| $q_{12}$: what is her style of music? (relevant) |
| $p_{12}$: Love is the subject of the majority of Carey’s lyrics, although she has written … (6, 68, 29) |
| $q_{13}$: what are some awards she has received? |
| Gold Passage (107, 68, 2) |
| Throughout her career, Carey has earned numerous awards and honors, including the World Music Awards’, Best Selling Female Artist of the Millennium, the Grammy Award for Best New Artist in 1991, and B̈illboards̈ Special Achievement Award for the Artist of the Decade during the 1990s. In a career spanning over 20 years, Carey has sold over 200 million records worldwide, making her one of the best-selling music artists of all time. Carey is ranked as the best-selling female artist of the Nielsen SoundScan era, with over 52 million copies sold. Carey was ranked first in MTV and B̈lenderm̈agazine’s 2003 countdown of the 22 Greatest Voices in Music, and was placed second in C̈ovem̈agazine’s list of T̈he 100 Outstanding Pop Vocalists.Äside from her voice, she has become known for her songwriting. |

*Table 5: A qualitative example of how historical ground-truth passage can benefit current query retrieval as supervision signals within HAConvDR. The brackets following each historical query indicate whether it is relevant or irrelevant to the current turn. The brackets with three numbers after each historical gold passage indicate its rank position by ANCE, Conv-ANCE, and our HAConvDR within top-1000, where “-” means it is ranked outside the top-1000.*
