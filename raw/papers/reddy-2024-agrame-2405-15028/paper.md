[Uncaptioned image] AGRaME: Any-Granularity Ranking with Multi-Vector Embeddings
==================================================================================

Revanth Gangi Reddy1Omar Attia211footnotemark: 1Yunyao Li3 Heng Ji1Saloni Potdar2  
1University of Illinois at Urbana-Champaign 2Apple 3Adobe  
{revanth3,hengji}@illinois.edu  
{oattia,s_potdar}@apple.com yunyaol@adobe.com  
Equal Contribution. Revanth is an external collaborator.Work done during position at Apple.

###### Abstract

Ranking is a fundamental and popular problem in search. However, existing ranking algorithms usually restrict the granularity of ranking to full passages or require a specific dense index for each desired level of granularity. Such lack of flexibility in granularity negatively affects many applications that can benefit from more granular ranking, such as sentence-level ranking for open-domain question-answering, or proposition-level ranking for attribution. In this work, we introduce the idea of any-granularity ranking which leverages multi-vector approaches to rank at varying levels of granularity while maintaining encoding at a single (coarser) level of granularity. We propose a multi-granular contrastive loss for training multi-vector approaches, and validate its utility with both sentences and propositions as ranking units. Finally, we demonstrate the application of proposition-level ranking to post-hoc citation addition in retrieval-augmented generation, surpassing the performance of prompt-driven citation generation.

\NewDocumentCommand\heng

mO Heng[#1]

<img src='extracted/5615713/figures/grammy_logo.png' alt='[Uncaptioned image]' title='' width='14' height='17' />AGRaME: Any-Granularity Ranking with Multi-Vector Embeddings

  
Revanth Gangi Reddy1††thanks: Equal Contribution. Revanth is an external collaborator.Omar Attia211footnotemark: 1Yunyao Li3††thanks: Work done during position at Apple. Heng Ji1Saloni Potdar21University of Illinois at Urbana-Champaign 2Apple 3Adobe{revanth3,hengji}@illinois.edu{oattia,s_potdar}@apple.com yunyaol@adobe.com

1 Introduction
--------------

<img src='extracted/5615713/figures/Any_Granularity_Ranking.png' alt='Refer to caption' title='' width='598' height='427' />

*Figure 1: Ranking at different levels of granularity. X$\rightarrow$Y is used to denote that X represents the query granularity used for ranking, with entire query encoded, and Y indicates the granularity of the retrieval unit being ranked, with entire retrieval unit encoded. In addition to the typical ranking setting (A), our proposed approach enables ranking finer retrieval units (B and D) or using finer query units for ranking (C and D).*

Dense Retrieval approaches leverage dual encoder models to obtain vector representations for both queries and passages. Commonly, single-vector methods*Gautier et al. ([2022]); Karpukhin et al. ([2020])* obtain a single embedding for each query and passage to compute the relevance score using a dot product. In contrast, multi-vector approaches*Khattab and Zaharia ([2020]); Santhanam et al. ([2022])* capture more fine-grained interactions when computing the query-passage relevance score, resulting in better ranking performance*Thakur et al. ([2021])*.
A key advantage of multi-vector approaches is their use of token-level embeddings paired with a MaxSim operation*Khattab and Zaharia ([2020])* for relevance scoring. This enables a more granular scoring mechanism that involes computing dot products between each query token embedding and each passage token embedding, to identify the best matching passage token for each query token. These individual token-level matching scores are subsequently aggregated to obtain the final query-passage relevance score.

We make the important observation that the use of token-level embeddings in multi-vector approaches can facilitate discriminative scoring of different sub-parts within a retrieval unit. For example, even when passages (retrieval units) are input to the encoder, i.e. encoding at passage-level, sub-components such as sentences can be separately scored against the query to identify the most relevant sentence within the passage. We argue that such finer-granularity scoring is inherent to multi-vector approaches, but not possible with single-vector approaches, where only one embedding represents the entire passage, thereby not allowing for scoring of its constituent sentences. Being able to rank at varying levels of granularity is beneficial for a variety of applications. For instance, in open-domain question answering*Lee et al. ([2019]); Karpukhin et al. ([2020])*, ranking sentences within the retrieved passages can better pinpoint the answer. For attribution*Rashkin et al. ([2023]); Chen et al. ([2023a])*, atomic facts (propositions) within sentences need to be used as queries to obtain relevant evidence supporting the facts.

To achieve this, we introduce AGRaME (Any-Granularity Ranking with Multi-vector Embeddings), a method that permits ranking at different levels of granularity while maintaining encoding at a single, coarser level. Our approach enables i) ranking at a finer level than the retrieval unit, and ii) ranking using fragments of the query, as demonstrated in Figure [1]. We hypothesize that encoding at a coarser level–such as the entire retrieval unit or query–can provide additional context for the sub-retrieval units being ranked or sub-parts of the query used for ranking. In contrast, achieving such granularity with single-vector approaches requires the use of specialized encoders, such as a sub-sentence encoder*Chen et al. ([2023b])*, or necessitates a separate encoding at the desired ranking granularity*Chen et al. ([2023c])*.

Firstly, how well do multi-vector approaches perform when used for ranking at a finer granularity? To investigate this, we conduct an exploratory experiment (described in §[2]) using ColBERTv2*Santhanam et al. ([2022])*, a popular multi-vector model, to rank both sentences and passages when encoded at corresponding granularities. From the results summarized in Table [1], we see that performance of sentence ranking is notably inferior when the encoding is at the passage-level, a result that counter-intuitive as passage-level encoding should provide a richer context for scoring sentences.

To improve the model’s ability to rank at finer granularity, we propose a multi-granular contrastive loss during training (outlined in §[3.3]). This introduces an additional sentence-level ranking loss that augments a passagr-level loss, to enable the model to not only accurately select the relevant passage for a query but also discriminatively identify the right sentence within that passage. Our experimental results, presented in Section [4.1], confirm significantly boost in sentence-level ranking while maintaining passage-level performance.

While AGRaME is generally applicable to arbitrary granularity, we explore the effectiveness for proposition-level ranking, crucial for applications requiring fine-grained attribution*Rashkin et al. ([2023])*. Our results (in §[4.2]) indicate that incorporating a sentence-level contrastive loss further improves ranking at proposition-level. Additionally, we demonstrate (in §[4.3]) that proposition-level ranking can effectively integrate citations into generated text post-hoc. Our proposed PropCite method utilizes propositions from generated text as queries to rank input context passages and select relevant citations, showing superior performance over traditional methods that prompt models to include citations in retrieval-augmented generation.

The main contributions are as follows:

* •

    We introduce AGRaME, that leverages multi-vector embeddings for ranking at various granularities while using the same encoding-level.

* •

    We introduce a multi-granular contrastive loss for training multi-vector approaches, which we show improves sentence-level ranking even when encoding at passage-level.

* •

    We demonstrate superior proposition-level ranking using AGRaME, surpassing existing state-of-the-art methods.

* •

    We leverage proposition-level ranking to formulate a post-hoc citation addition approach for retrieval-augmented generation, that outperforms prompt-driven citation generation.

2 Motivating Experiment
-----------------------

Here, we investigate the effectiveness of ColBERTv2*Santhanam et al. ([2022])*, a multi-vector approach, in ranking at a finer granularity than the encoding level. Specifically, when encoding is at passage-level, we measure the performance while ranking at sentence level (using the scoring scheme described in §[3.2]), in addition to ranking at the usual passage-level. A MaxSim operation is applied between query token vectors and token vectors corresponding to the sentence to get a sentence-level score, which is then added with the passage-level score to get the final query-sentence relevance score for ranking. When encoding is at sentence-level, the usual MaxSim score gives query-sentence relevance. On the other hand, the query-passage relevance score for ranking, when encoding is at sentence-level, is obtained as the maximum of the corresponding passage’s query-sentence relevance scores.

We also include Contriever*Gautier et al. ([2022])*, a single-vector approach, for comparison. When encoding is at sentence-level, the same strategy as described before is used to obtain query-passage relevance score for Contriever. On the other hand, when encoding is at passage-level, Contriever does not support sentence-level ranking, which as discussed earlier, is an inherent limitation of single-vector approaches. For the evaluation setting, we consider the development set of the Natural Questions*Kwiatkowski et al. ([2019])* dataset to get the queries, and Wikipedia111We use the 22M passage split of the Wikipedia 2018 dump from *Gautier et al. ([2022])* as the retrieval corpus. To keep the retrieval index size manageable, Contriever is used to index and retrieve 100 passages, which are then ranked by ColBERTv2. When ranking or encoding at sentence-level, only the sentences in these top 100 passages are considered. Evaluation is done for Precision@1 and Recall@5 based on string exact-match of the answer*Rajpurkar et al. ([2016])*.

| Model | Encoding Level | Ranking Level | | | |
| --- | --- | --- | --- | --- | --- |
| | | Sentence | | Passage | |
| P@1 | R@5 | P@1 | R@5 |
| Contriever (Single Vec.) | Sentence | 19.3 | 45.6 | 32.4 | 62.8 |
| | Passage | - | - | 37.8 | 65.1 |
| ColBERTv2 (Multi Vec.) | Sentence | 31.6 | 56.3 | 40.2 | 66.8 |
| | Passage | 27.4 | 48.8 | 43.4 | 69.1 |

*Table 1: Precision@1 (P@1) and Recall@5 (R@5) results on the Natural Questions*Kwiatkowski et al. ([2019])* dev set. We show numbers both at sentence-level and passage-level ranking granularities for when sentences and passages are encoded individually.*

Table [1] shows the results, wherein we see a substantial drop in sentence-level ranking when encoding is at passage-level, and vice versa. As expected, passage-level ranking is better when encoding is at passage-level. However, it is surprising to see sentence-level ranking performance to be lower since passage-level encoding can provide more context when encoding the individual tokens in the sentences. This is more so the case when sentences in the passage that actually ‘answer’ the query do not have any overlapping terms (semantic or lexical) with the query. Table [2] shows an example for this. At both sentence and passage-level encoding, sentences S1 and S2, on account of strong lexical overlap with the query, are scored considerably higher than sentence S3, which actually describes the effects of climate change but has weak semantic overlap with the query. From the token-wise MaxSim score heatmap, we can see that tokens in S3, the most relevant sentence, get lower scores.

We can expect scoring S1 and S2 highly, on account of overlap, to be particularly useful when identifying the passage as relevant amongst a corpus of millions of passages. On the other hand, it can be counter-productive when discriminatively selecting the most relevant sentence within the passage.
This suggests that ranking at different granularities requires the model to have the ability to dynamically switch the notion of relevance when scoring. As we describe later in §[3.2], we introduce a new query marker during encoding to signal the level of granularity, which helps the model distinguish better when scoring at different granularities In §[4.1], we demonstrate that this helps to score appropriately for ranking at sentence-level (a finer granularity), when encoding at passage-level.

<img src='extracted/5615713/figures/motivation_example.png' alt='[Uncaptioned image]' title='' width='598' height='182' />

| Sent. ID | Sentence-level Enc. | Passage-level Enc. |
| --- | --- | --- |
| S1 | Rank:1, Score:23.31 | Rank:1, Score:23.92 |
| S2 | Rank:2, Score:17.47 | Rank:2, Score:20.12 |
| S3 | Rank:3, Score:16.63 | Rank:3, Score:16.96 |

*Table 2: Sentence-level ColBERTv2 scores for different sentences in the same passage, when encoding is at sentence-level and passage-level. We see that the most relevant sentence (S3) is ranked worst (i.e lowest score). Token-wise MaxSim score heatmap is also shown, with tokens in S1 and S2 having higher scores than in S3.*

3 Method
--------

Here, we first provide background (in §[3.1]) on the modeling and training process for ColBERTv2*Santhanam et al. ([2022])*. Then, we describe our proposed multi-granular contrastive training process (in §[3.3]), which provides an additional sentence-level relevance supervision during distillation.

### 3.1 ColBERTv2 Preliminaries

Single-vector retrievers*Karpukhin et al. ([2020]); Gautier et al. ([2022])* typically use a BERT-based*Devlin et al. ([2019])* dual-encoder architecture to obtain a single embedding for a query $q$ and passage $p$ separately. This is usually the CLS output or a pooled representation of the individual token outputs from the encoder $E(.)$. The query-passage relevance score is computed as the dot product of their corresponding representation:

|  | $\vec{Q_{q}}\=Pool(E(q));\vec{P_{p}}\=Pool(E(p))$ |  |
| --- | --- | --- |

|  | $Score(q,p)\=\vec{Q_{q}}^{T}\vec{P_{p}}$ |  |
| --- | --- | --- |

In constrast, ColBERTv2*Santhanam et al. ([2022])* is a multi-vector retrieval model, that uses token-level dense embeddings of the query and passage. Given a query $q$ containing $n$ tokens $t^{q}_{i}$ and passage $p$ containing $m$ tokens $t^{p}_{i}$, additional query and passage marker tokens $m_{q}$ and $m_{p}$ are prepended to the query and passage respectively before encoding, to provide additional signal to the encoder. The query-passage relevance score $S_{CB}(q,p)$ is obtained as below using the MaxSim operator introduced in*Khattab and Zaharia ([2020])*:

|  | $[\vec{Q_{t^{q}_{1}}},\vec{Q_{t^{q}_{1}}},...,\vec{Q_{t^{q}_{n}}}]\=E(q)\=E(cat(m% _{q},t^{q}_{1},...,t^{q}_{n}))$ |  |
| --- | --- | --- |

|  | $[\vec{P_{t^{p}_{1}}},\vec{P_{t^{p}_{2}}},...,\vec{P_{t^{p}_{m}}}]\=E(p)\=E(cat(m% _{p},t^{p}_{1},...,t^{p}_{m}))$ |  |
| --- | --- | --- |

|  | $S_{CB}(q,p)\=MaxSim(q,p)\=\sum_{i\=1}^{n}\max_{1\leq j\leq m}\vec{Q_{t^{q}_{i}}}^% {T}\vec{P_{t^{p}_{j}}}$ |  |
| --- | --- | --- |

The training process for neural retrievers typically involves a contrastive loss over the <query $q$, postitive $p^{+}$, negative $p^{-}$> triples. ColBERTv2 instead incorporates a distillation-based training strategy wherein $k$ negative passages are sampled from the retrieval corpus, to form a $(k+1)$-way passage set $[p]\={p^{+},p^{-}_{1},...,p^{-}_{k}}$ for each query. The relevance supervision is in the form of soft scores $S_{CE}(.)$ from a cross-encoder reranker. A KL-Divergence loss $\mathcal{L}_{psg}$ between the cross-encoder and ColBERT passage scoring distributions, $D_{CE}(q,[p])$ and $D_{CB}(q,[p])$ respectively, is used for training:

|  | $D_{CB}(q,[p])\=\left[Softmax(S_{CB}(q,p_{i}))\right]_{i\=1}^{k+1}$ |  |
| --- | --- | --- |

|  | $D_{CE}(q,[p])\=\left[Softmax(S_{CE}(q,p_{i}))\right]_{i\=1}^{k+1}$ |  |
| --- | --- | --- |

|  | $\mathcal{L}_{psg}(q,[p])\=KL(D_{CE}(q,[p])||D_{CB}(q,[p]))$ |  |
| --- | --- | --- |

### 3.2 AGRaME: Any-Granularity Ranking with Multi-Vector Embeddings

<img src='extracted/5615713/figures/sentence_scoring_v2.png' alt='Refer to caption' title='' width='598' height='237' />

*Figure 2: Figure demonstrating our sentence-level scoring methodology using multi-vector representations with encoding at passage-level. Query marker $m_{q}$ is used while getting passage-level score $P$, while marker $m^{\prime}_{q}$ is used for getting sentence-level scores $S1$, $S2$, $S3$.*

Here, we introduce our approach for scoring sub-units within the retrieval unit. This is made possible by the access to token-level embeddings in multi-vector approaches. While AGRaME can rank at any granularity, in this section, we will consider sentences as the sub-units for simplicity. With the entire passage input to the encoder, only the output embeddings corresponding to tokens within a given sentence are used during the MaxSim operation for scoring that sentence.

Let $t^{p_{i}}_{jr}$ correspond to the $j^{th}$ token of sentence $s_{j}^{p_{i}}$ from passage $p_{i}$ that is passed as input to encoder $E$. To signal the model to score discriminatively within the passage for sentence-level ranking, we prepend a new query marker token $m^{\prime}_{q}$, different from $m_{q}$ used when ranking at passage-level. The in-passage query-sentence relevance score $S_{CB}(q,s_{j}^{p_{i}})$ is computed as follows:

|  | $[\vec{Q^{\prime}_{t^{q}_{1}}},\vec{Q^{\prime}_{t^{q}_{1}}},...,\vec{Q^{\prime}% _{t^{q}_{n}}}]\=E(cat(m^{\prime}_{q},t^{q}_{1},...,t^{q}_{n}))$ |  |
| --- | --- | --- |

|  | $S_{CB}(q,s_{j}^{p_{i}})\=\sum_{i\=1}^{n}\max_{1\leq r\leq|s_{j}^{p_{i}}|}\vec{Q^% {\prime}_{t^{q}_{i}}}^{T}\vec{P_{t^{p_{i}}_{jr}}}$ |  |
| --- | --- | --- |

Note that the passage encoding is the same as before, meaning the same multi-vector index can be used for both passage-level and sentence-level ranking. As we demonstrate in §[4.1], encoding at passage-level provides more context to the token embeddings to benefit sentence-level ranking.

We note that our proposed sentence-level loss (described in §[3.3]) teaches the model to rank sentences discriminatively within a passage, and not across passages. Hence, at inference to get a final sentence-level relevance score $Score(q,s_{j}^{p_{i}})$ to rank sentences across passages, we combine the in-passage sentence relevance score $S_{CB}(q,s_{j}^{p_{i}})$ with the usual passage-level relevance score $S_{CB}(q,p_{i})$:

|  | $Score(q,s_{j}^{p_{i}})\=S_{CB}(q,s_{j}^{p_{i}})+\alpha S_{CB}(q,p_{i})$ |  |
| --- | --- | --- |

### 3.3 Multi-Granular Contrastive Training

As discussed in §[3.1], given a query $q$ and a passage set $[p]$, the ColBERTv2 training process aims to teach the model to identify the most relevant passage within $[p]$. To enable the model to discriminatively select sub-units within the passage, we propose to incorporate a more finer-level of training supervision, by teaching to further identify the most relevant sentence within each passage.

Since ColBERTv2 uses passage-level cross-encoder scores as teacher supervision, we train a different cross-encoder model $CE^{\prime}$ to provide in-passage sentence-level relevance supervision. Specifically, $CE^{\prime}$ takes a passage $p_{i}$ as input, with a given sentence $s_{j}^{p_{i}}$ marked with delimiters $, to give a relevance score $S_{CE^{\prime}}(q,s_{j}^{p_{i}})$ for the sentence.

|  | $S_{CE^{\prime}}(q,s_{j}^{p_{i}})\=CE^{\prime}(q,cat(s_{1}^{p_{i}},...,\$s_{j}^{% p_{i}}\$,...,s_{l}^{p_{i}}))$ |  |
| --- | --- | --- |

$CE^{\prime}$ is trained using question answering data in the form <query, passage, answer> triples. A binary cross-entropy loss is used while training $CE^{\prime}$, wherein any sentence within the passage that contains the answer is marked as a positive, with the other sentences marked as negatives.

The cross encoder $CE^{\prime}$ provides soft scores for sentence-level relevance superivision when training our model. For each passage $p_{i}$, we compute a KL-divergence loss $\mathcal{L}_{s}(q,p_{i})$ between the $CE^{\prime}$ and ColBERTv2 sentence-level scoring distributions, $D_{CE^{\prime}}(q,[s^{p_{i}}])$ and $D_{CB}(q,[s^{p_{i}}])$ respectively.

|  | $D_{CB}(q,[s^{p_{i}}])\=\left[Softmax(S_{CB}(q,s_{j}^{p_{i}}))\right]_{j\=1}^{l}$ |  |
| --- | --- | --- |

|  | $D_{CE^{\prime}}(q,[s^{p_{i}}])\=\left[Softmax(S_{CE^{\prime}}(q,s_{j}^{p_{i}}))% \right]_{j\=1}^{l}$ |  |
| --- | --- | --- |

|  | $\mathcal{L}_{s}(q,p_{i})\=KL(D_{CE^{\prime}}(q,[s^{p_{i}}])||D_{CB}(q,[s^{p_{i}% }])))$ |  |
| --- | --- | --- |

We then aggregate each passage’s sentence-level scoring loss $\mathcal{L}_{s}(q,p_{i})$, by weighting with the corresponding passage’s relevance supervision score $S_{CE}(q,p_{i})$, to get a single loss ${L}_{sent.}(q,[p])$. The passage score weight ensures that the model is penalized higher on sentence-level losses for passages that are more relevant. The sentence-level loss ${L}_{sent.}(q,[p])$ is finally added to original passage-level loss ${L}_{psg}(q,[p])$ to get the training loss $\mathcal{L}$.

|  | $\mathcal{L}_{sent.}(q,[p])\=\sum_{i\=1}^{k+1}Softmax(S_{CE}(q,p_{i}))\mathcal{L}% _{s}(q,p_{i})$ |  |
| --- | --- | --- |

|  | $\mathcal{L}(q,[p])\=\mathcal{L}_{psg}(q,[p])+\mathcal{L}_{sent.}(q,[p])$ |  |
| --- | --- | --- |

4 Experiments
-------------

AGRaME can rank at different granularities, as shown in Figure [1], which involves ranking sub-parts of the retrieval unit or ranking using sub-parts of the query. In our experiments, we aim to investigate two research questions: RQ1: Can the training approach proposed in §[3.3] improve ranking at a finer granularity than the level of encoding, i.e. Query$\rightarrow$Sub-Retrieval Unit? In §[4.1], we show the improvements at sentence-level ranking from our proposed multi-granular contrastive loss, while maintaining performance at passage-level, i.e.Query$\rightarrow$Retrieval Unit; RQ2: Can multi-vector embeddings be used to rank with sub-parts of the query? In §[4.2], we demonstrate the application of multi-vector approaches in Sub-Query$\rightarrow$Sub-Retrieval Unit ranking for proposition-level attribution. Here, a given proposition within a sentence is used as the query to rank and identify relevant propositions in a corpus of sentences. Further, in §[4.3], we introduce PropCite, a post-hoc citation addition approach based on Sub-Query$\rightarrow$Retrieval Unit ranking. PropCite scores input context passages based on propositions in the generated text to add citations in retrieval-augmented generation.

### 4.1 Query$\rightarrow$Sub-Retrieval Unit Ranking for Open-Domain QA

In §[2], we saw that with a multi-vector approach (ColBERTv2), sentence ranking performance drops when changing the encoding from sentence-level to passage-level.
We addressed this in two ways: (a) our proposed multi-granular contrastive loss (in §[3.3]) provides sentence-level relevance supervision at training; b) AGRaME introduces a new query marker (in §[3.2]) to signal scoring at sentence-level.
In this section, we empirically demonstrate the benefits of our proposed approach by evaluating sentence-level (sub-retrieval unit) ranking performance when encoding is at passage-level.

| Model | Encoding Level | Natural Questions | | | | TriviaQA | | | | Web Questions | | | | Entity Questions | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | Sentence | | Passage | | Sentence | | Passage | | Sentence | | Passage | | Sentence | | Passage | |
| P@1 | R@5 | P@1 | R@5 | P@1 | R@5 | P@1 | R@5 | P@1 | R@5 | P@1 | R@5 | P@1 | R@5 | P@1 | R@5 |
| Contriever | Sentence | 20.6 | 48.9 | 35.0 | 65.4 | 31.0 | 58.8 | 48.5 | 72.1 | 14.5 | 39.1 | 28.8 | 57.9 | 14.7 | 42.7 | 39.8 | 64.9 |
| | Passage | - | - | 40.3 | 66.0 | - | - | 50.1 | 71.5 | - | - | 36.9 | 63.6 | - | - | 36.9 | 63.6 |
| ColBERTv2 | Sentence | 32.7 | 58.8 | 42.0 | 68.8 | 43.2 | 66.1 | 55.6 | 74.7 | 29.0 | 51.9 | 38.8 | 63.7 | 38.1 | 59.4 | 50.9 | 68.1 |
| | Passage | 27.9 | 51.1 | 43.2 | 70.0 | 43.5 | 65.6 | 57.5 | 75.6 | 27.6 | 50.7 | 41.0 | 65.1 | 39.2 | 55.3 | 53.9 | 69.2 |
| Ours | Passage | 36.8 | 60.5 | 44.0 | 69.9 | 48.9 | 68.1 | 57.9 | 75.6 | 33.2 | 55.6 | 41.2 | 65.4 | 43.8 | 61.5 | 54.2 | 69.5 |

*Table 3: Precision@1 (P@1) and Recall@5 (R@5) results on various open-domain QA datasets. We show numbers both at sentence-level and passage-level ranking for when sentences and passages are encoded individually.*

| Model | Encoding Level | Finance | | Recreation | | Lifestyle | | Science | | Technology | | Writing | | Biomedical | | Average | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | Sent. | Psg. | Sent. | Psg. | Sent. | Psg. | Sent. | Psg. | Sent. | Psg. | Sent. | Psg. | Sent. | Psg. | Sent. | Psg. |
| Contriever | Sentence | 13.8 | 22.2 | 17.9 | 29.4 | 19.7 | 32.7 | 10.9 | 18.8 | 11.3 | 18.3 | 23.0 | 36.1 | 10.7 | 16.6 | 15.3 | 24.9 |
| | Passage | - | 27.2 | - | 34.7 | - | 40.4 | - | 17.5 | - | 21.4 | - | 39.6 | - | 4.6 | - | 26.5 |
| ColBERTv2 | Sentence | 15.8 | 23.7 | 24.0 | 33.6 | 22.6 | 34.2 | 17.6 | 25.0 | 15.5 | 23.4 | 33.4 | 46.6 | 12.8 | 17.3 | 20.2 | 29.1 |
| | Passage | 17.1 | 29.8 | 25.5 | 40.7 | 23.9 | 41.9 | 18.4 | 28.7 | 16.7 | 27.1 | 34.7 | 51.3 | 13.1 | 16.9 | 21.4 | 33.8 |
| Ours | Passage | 19.5 | 29.8 | 29.2 | 40.4 | 30.0 | 42.6 | 20.5 | 28.1 | 18.4 | 26.4 | 36.7 | 50.2 | 15.4 | 17.5 | 24.2 | 33.6 |

*Table 4: Precision@1 results on various domains from the RobustQA dataset*Han et al. ([2023])*. We show numbers at sentence-level and passage-level ranking for when sentences and passages are encoded individually.*

#### 4.1.1 Setup

##### Datasets

We first evaluate on different popular open-domain QA datasets: Natural Questions (NQ)*Kwiatkowski et al. ([2019])*, TriviaQA*Joshi et al. ([2017])*, Web Questions*Berant et al. ([2013])* and Entity Questions*Sciavolino et al. ([2021])*. For the retrieval corpus, we use the 2018 Wikipedia dump released by*Lee et al. ([2019])*.

For cross-domain evaluation, we consider the RobustQA*Han et al. ([2023])* dataset, a large-scale OpenQA benchmark specifically designed for evaluating cross-domain generalization capabilities. The QA pairs and documents correspond to various domains like finance (adopted from FiQA*Maia et al. ([2018])*), biomedical (adopted from BioASQ*Tsatsaronis et al. ([2015])*), along with recreation, lifestyle, science, technology and writing, which are all adopted from LOTTE*Santhanam et al. ([2022])*.

##### Baselines

We use Contriever*Gautier et al. ([2022])* as the single-vector baseline, and ColBERTv2*Santhanam et al. ([2022])* as the multi-vector baseline. All models use MS MARCO*Nguyen et al. ([2016])* as the training dataset. Due to storage constraints, we create a single-vector index with Contriever and rank the top-100 retrieval results from Contriever using the multi-vector approaches to report numbers.

#### 4.1.2 Results

Table [3] shows ranking results on various open-domain QA datasets. Firstly, as expected, for both Contriever and ColBERTv2, passage-level ranking performance is best when encoding is at passage-level. We observe that our proposed approach significantly improves sentence-level ranking performance with passage-level encoding, even outperforming sentence-level ranking at sentence-level encoding. This result confirms our intuition that passage-level encoding benefits sentence-level ranking, since it can provide more context to the individual sentences during encoding. Moreover, we ensure that passage-level ranking performance is not compromised, with our approach matching that of ColBERTv2 at passage-level encoding.

Table [4] shows sentence-level and passage-level ranking results for cross-domain evaluation on the RobustQA benchmark. We observe that our approach is robust and extends to cross-domain settings, with consistent improvements in sentence-level ranking across the board, while passage-level ranking performance almost the same.

#### 4.1.3 Analysis

We do an ablation study to analyze the effect of using the new query marker $m^{\prime}_{q}$, instead of the default query marker $m_{q}$, when scoring at sentence-level. Note that the markers $m_{q}$ and $m^{\prime}_{q}$ at inference only affect the query token embeddings. We consider three different settings: A1) Using $m^{\prime}_{q}$ for sentence-level ranking at training and inference, which corresponds to our proposed approach, A2) Using $m^{\prime}_{q}$ for sentence-level ranking at training but using $m_{q}$ at inference, and finally A3) Using the default $m_{q}$ for sentence-level ranking at training and inference. We also include the baseline ColBERTv2 for comparison, which does not have sentence-level supervision at training and uses $m_{q}$ at inference.
From the results in Table [5], we can see the benefit of using a different query marker, with A1 outperforming A3 in the majority of the cases. Moreover, using $m_{q}$ at inference even while being trained with $m^{\prime}_{q}$ (A2) shows some gains over baseline ColBERTv2, implying that the model also learns to encode passage tokens to be better at discriminatively scoring sentences in the passage.

In addition, we show the training loss curves in Figure [3] when the same query marker ($m_{q}$) vs different query markers ($m^{\prime}_{q}$ and $m_{q}$) are used for sentence-level and passage-level loss respectively. We can see that the model converges faster at sentence-level loss when new marker $m^{\prime}_{q}$ is used. Further, Table [6] shows the sentence-wise scores for the example in Table [2] from using $m^{\prime}_{q}$ vs $m_{q}$ for sentence-level scoring. We observe that sentence-level ranking changes when $m^{\prime}_{q}$ is used, with the most relevant sentence (S3) ranked best.

| Setting | NQ | TQA | WebQ | EntQ |
| --- | --- | --- | --- | --- |
| ColBERTv2 | 27.9 | 43.5 | 27.6 | 39.2 |
| A1) Train$\rightarrow$$m^{\prime}_{q}$, Rank$\rightarrow$$m^{\prime}_{q}$ | 36.8 | 48.9 | 33.2 | 43.8 |
| A2) Train$\rightarrow$$m^{\prime}_{q}$, Rank$\rightarrow$$m_{q}$ | 29.1 | 44.8 | 29.4 | 40.8 |
| A3) Train$\rightarrow$$m_{q}$, Rank$\rightarrow$$m_{q}$ | 35.9 | 47.6 | 32.9 | 44.1 |

*Table 5: Precision@1 of sentence-level ranking performance for various variants of using a different query marker. ColBERTv2 is trained only with a passage-level loss and uses the $m_{q}$ query marker. The latter three variants are represented with the query marker used while training with sentence-level contrastive loss and that used for sentence-level ranking at inference.*

<img src='extracted/5615713/figures/q2s_loss_v2.png' alt='Refer to caption' title='' width='107' height='75' />

<img src='extracted/5615713/figures/q2p_loss_v2.png' alt='Refer to caption' title='' width='107' height='75' />

*Figure 3: Comparison of training curves for sentence-level and passage-level loss, when a different query marker is used. The model converges faster at sentence-level with a different query marker, while passage-level loss is mostly similar for the two.*

| Sent. ID | Query Marker $m^{\prime}_{q}$ | Query Marker $m_{q}$ |
| --- | --- | --- |
| S1 | Rank:2, Score:14.32 | Rank:1, Score:24.04 |
| S2 | Rank:3, Score:14.16 | Rank:2, Score:21.07 |
| S3 | Rank:1, Score:15.92 | Rank:3, Score:16.81 |

*Table 6: Sentence-level scores from our model at passage-level encoding for the example in Table [2], when different query markers are used. The most relevant sentence (S3) is ranked best when new marker $m^{\prime}_{q}$ is used.*

### 4.2 Sub-Query$\rightarrow$Sub-Retrieval Unit Ranking for Fine-Grained Attribution

Attributing model-generated text with supporting information from known sources is an emerging research topic*Gao et al. ([2023a]); Liu et al. ([2023])*. Each sentence in the generation can have multiple atomic facts or propositions*Min et al. ([2023])* for which evidence needs to be obtained. In this context, we explore ranking at the sub-sentence granularity, wherein given a sentence as a query, fine-grained attributions*Rashkin et al. ([2023])* need to be obtained for a sub-part of the sentence. Specifically, we consider the Atomic Fact Retrieval task, wherein given an atomic proposition (sub-query) in the sentence, the system is expected to identify and retrieve evidence from atomic propositions as facts, each of which can be a sub-part of sentences within a corpus.

We consider this task to demonstrate that multi-vector embeddings can be leveraged to natively rank at the sub-sentence level, and compare them against specialized models*Chen et al. ([2023b])* explicitly trained to encode propositions. We note that the encoding here is at sentence-level, unlike in §[4.1] where encoding is at passage-level. Since the marker $m^{\prime}_{q}$ in our multi-granular training loss was for sentence-level ranking with passage-level encoding, we use the default marker $m_{q}$ when ranking at proposition-level with sentence-level encoding.

<img src='extracted/5615713/figures/attribution.png' alt='Refer to caption' title='' width='598' height='127' />

*Figure 4: Figure illustrating PropCite, our proposed approach for post-hoc addition of citations to long-form answers. PropCite encodes sentences and uses the propositions within them as queries for attribution. The figure shows the propositions highlighted within the current sentence (in yellow), and the corresponding supporting evidence highlighted in the input context passages. PropCite correctly attributes proposition P2 to context C1, while directly encoding and querying using P2 incorrectly attributes to C2.*

#### 4.2.1 Setup

##### Datasets

For the proposition-level ranking evaluation, we use the PropSegmEnt *Chen et al. ([2023a])* dataset, which involves 8.8k propositions as sub-queries for which evidence needs to be obtained from a corpus of 45k human-labeled atomic propositions from 1.5k News or Wikipedia documents in total.

##### Baselines

For this task, we consider SubEncoder *Chen et al. ([2023b])* as the primary baseline, a state-of-the-art sub-sentence encoder for proposition-level ranking. SubEncoder has been specifically trained to produce contextual embeddings for atomic propositions in a sentence. Being a single-vector model, SubEncoder produces a single sub-sentence embedding for each atomic proposition in the sentence. We also include other sentence-level embedding approaches, such as GTR*Ni et al. ([2022b])*, Sentence-T5*Ni et al. ([2022a])*, as baselines that*Chen et al. ([2023a])* adapt for this task by specifically pooling over the tokens of the proposition to get a single-vector embedding.

#### 4.2.2 Results

| Model | Proposition | | Sentence | |
| --- | --- | --- | --- | --- |
| | P@1 | R@5 | P@1 | R@5 |
| GTR | 21.9 | 52.5 | 49.4 | 77.0 |
| ST5 | 26.2 | 57.7 | 50.6 | 79.4 |
| SubEncoder (GTR) | 40.8 | 72.9 | 42.9 | 82.3 |
| SubEncoder (ST5) | 41.0 | 72.2 | 43.5 | 81.4 |
| ColBERTv2 | 46.9 | 74.2 | 54.7 | 87.8 |
| Ours | 47.7 | 74.7 | 55.0 | 87.4 |

*Table 7: Evaluation results on the Atomic Fact Retrieval task in PropSegmEnt *Chen et al. ([2023a])*. The encoding level is individual sentences, with each sentence consisting of multiple propositions. All models are based on encoders with 110M parameters. Numbers for GTR, ST5, SubEncoder are from*Chen et al. ([2023b])*.*

Table [7] shows results from the Atomic Fact Retrieval task. First, we observe that the baseline ColBERTv2 already outperforms the state-of-the-art SubEncoder at proposition-level (sub-sentence) ranking. Although our proposed approach adds a sentence-level constrastive loss at passage-level encoding, we do see some improvements when ranking at proposition-level. However, we hypothesize that better proposition-level ranking can be expected by further training with a proposition-level loss in §[3.3], which we leave for future work to explore. Nevertheless, given the state-of-the-art performance of multi-vector approaches at proposition-level ranking, we introduce next (in §[4.3]) a practical application leveraging this for the task of adding citations to machine-generated text.

### 4.3 Sub-Query$\rightarrow$Retrieval Unit Ranking for Citation Addition

Retrieval-augmented generation (RAG)*Lewis et al. ([2020]); Izacard et al. ([2023])* produces a long-form answer to a query, given a set of relevant passages as input context. Here, we explore the ability of multi-vector approaches to act as a citation addition approach for attribution in RAG. Specifically, given a set of $K$ passages, and the generated long-form answer, the task involves adding citations to one or more of the input passages, for each sentence in the generated answer.

We introduce PropCite, a post-hoc citation approach that adds citations to the input context supporting propositions (atomic facts) in the generated text. Specifically, PropCite makes use of propositions tagged222We employ the approach from*Chen et al. ([2023b])*, which uses a T5 model*Raffel et al. ([2020])* to segment sentences into propositions, that are then converted into token masks by aligning the tokens in each proposition to the sentence. within the generated sentences, so that the corresponding sub-parts can be used as the query to score the input passages and identify the ones that need to be cited. Figure [4] illustrates PropCite. Our approach is ‘post-hoc’ since citations are added after the text is generated, as opposed to the typical approach of generating text with citations by directly prompting the generation model*Gao et al. ([2023c])* to add citations.

#### 4.3.1 Setup

##### Datasets and Metrics

We consider various long-form question answering datasets, specifically ASQA*Stelmakh et al. ([2022])* and ELI5*Fan et al. ([2019])*. The RAG setting involves both $K$\=5 and $K$\=10 passages as input to the language model to generate the answer. The evaluation of attribution quality is based on the citation precision and recall metrics introduced in*Gao et al. ([2023c])*. Citation recall determines if the output is entirely supported by cited passages and citation precision identifies any irrelevant citations. The metrics are computed using TRUE*Honovich et al. ([2022])*, a 11B-parameter model trained on a collection of natural language inference datasets, commonly used*Bohnet et al. ([2022]); Gao et al. ([2023b])* to evaluate attribution by checking whether the cited passages entail the claims in the sentence.

##### Baselines

We compare PropCite against the commonly used instruction-driven citation generation*Gao et al. ([2023c])*, which we call Generate, where the generation model is prompted to output text with citations. We use the same few-shot prompt (provided in appendix) as*Gao et al. ([2023c])* to instruct the model to add citations while generating the answer. We consider variants of the generation model, a smaller333We also experimented with Google’s Gemma 2B and Microsoft Phi-2 models. Refusal rate was too high for Gemma 2B while Phi-2 had an input context length of only 2048. 4B Qwen1.5*Bai et al. ([2023])* and a larger 7B Mistral-Instruct*Jiang et al. ([2023])*. We also include comparison with Self-RAG*Asai et al. ([2023])*, which uses a self-reflective generation framework to adaptively pick input passages to generate from and thereby cite.

#### 4.3.2 Results

Table [8] shows the citation precision (P) and recall (R) numbers comparing the citation quality in the generated text vs our post-hoc PropCite approach. Firstly, the ability to generate text with citations depends heavily on the instruction-following capability of the generation model, with weaker models such as Qwen1.5 4B considerably worse-off compared to Mistral-Instruct 7B. Moreover, even the citation quality of post-hoc approaches depends on the quality of generated text, i.e. when weaker models hallucinate or generate text that cannot be supported by the input context, citation quality is expected to be lower.

We observe that PropCite has significantly better citation quality on the 4B and 7B model generations. Even for the Self-RAG models, which were explicitly finetuned for RAG by adding special reflection tokens to cite passsages, we see improvements with PropCite. It is important to note PropCite is post-hoc, and hence can be used with any RAG framework, without needing to tweak the generation model. Moreover, PropCite is light-weight444While we use a T5 model to explicitly segment sentences into propositions, faster approaches relying on syntactic dependency parsing*Goyal and Durrett ([2020]); Wanner et al. ([2024])* can be a cheaper alternative to get the sub-structures with a sentence that represent the propositions or atomic claims. and can post-hoc add citations as sentences are generated one-by-one in a streaming setting.

| Generation Model | Psg. | Citation Method | ASQA | | ELI5 | |
| --- | --- | --- | --- | --- | --- | --- |
| | | | P | R | P | R |
| Qwen1.5 4B | 5 | Generate | 26.9 | 21.3 | 11.0 | 8.6 |
| | | PropCite | 48.9 | 54.5 | 19.5 | 23.4 |
| 10 | Generate | 14.8 | 11.7 | 5.7 | 4.7 |
| | PropCite | 45.3 | 52.0 | 18.3 | 22.9 |
| Mistral 7B | 5 | Generate | 64.9 | 69.5 | 40.5 | 49.0 |
| | | PropCite | 65.7 | 74.2 | 43.0 | 51.9 |
| 10 | Generate | 60.2 | 66.7 | 38.0 | 48.8 |
| | PropCite | 61.6 | 71.9 | 41.9 | 53.0 |
| Self-RAG 7B | 5 | Generate | 67.9 | 67.1 | - | - |
| | | PropCite | 68.5 | 68.4 | - | - |
| Self-RAG 13B | Generate | 71.4 | 70.5 | - | - |
| | PropCite | 71.6 | 71.5 | - | - |

*Table 8: Table showing precision (P) and recall (R) for different citation addition approaches on the long-form ASQA*Stelmakh et al. ([2022])* and ELI5*Fan et al. ([2019])* question answering datasets. For Self-RAG, we directly use generation outputs from*Asai et al. ([2023])*.*

| Setting | Precision | Recall |
| --- | --- | --- |
| Generate | 64.9 | 69.5 |
| PropCite | 65.7 | 74.2 |
| + Thresholding | 69.2 | 71.1 |
| (i) Propositions as query | 63.5 | 73.9 |
| (ii) Sentence as query (top 1) | 69.0 | 67.5 |
| (iii) Sentence as query (top 2) | 51.2 | 72.6 |

*Table 9: Analysis of citation precision and recall performance on ASQA for Mistral 7B when using top-5 passages as input. We consider different settings, wherein the generated propositions or the sentence itself are used as the query when searching for relevant citations.*

#### 4.3.3 Analysis

Table [9] shows results for an ablation study with different variants of post-hoc citation addition to demonstrate the benefits of using propositions within generated sentences as the query. Firstly, we show numbers for a higher-precision version of PropCite that incorporates thresholding555To mitigate false positives, we only add a citation if the top-scored passage has a relevance score margin of atleast 1.0. to decide whether to add a citation for a given proposition in the sentence. Next, we compare against different variants that directly encode the proposition (i) or query using the entire sentence (ii, iii). We can see that directly encoding the proposition, instead of encoding the sentence and using tokens corresponding to the proposition, leads to a drop in precision. This supports our primary hypothesis that encoding at lower-granularity (sentence-level in this case) gives additional context to the token vectors when used for querying at higher granularity (proposition-level in this case). Figure [4] illustrates this with an example from the ASQA dataset. PropCite correctly attributes proposition P2 to input passage C1 which mentions U.S. Open as the tournament in September that was won by Ouiment. However, directly encoding P2 misses the context that the tournament occured in September and incorrectly attributes to input passage C2, that mentions a different tournament, the Massachusetts Amateur, that Ouiment won.

Moreover, we compare against an alternate approach that just uses the entire sentence as one single query, instead of separately using the individual propositions within the sentences as queries. Tagging the top-1 scored passage as the citation (ii) for that sentence gives a high precision but considerably low recall, while tagging top-2 scored passages as the citations does improve recall but precision suffers a lot. Overall, with PropCite, 66% of sentences had 1 citation, 30% had 2 citations and remaining 4% had more than 2 citations.

5 Related Work
--------------

The phrase ‘multi-granularity’ can have different meaning depending on the domain in which it is being used. In space of image retrieval, it corresponds to representing different regions of the image separately*[Wang et al.] ; Zhang et al. ([2022])*. For representation learning, it refers to encoding information at different granularities, i.e. output embedding dimensions, to adapt to the computational constraints of downstream tasks*Kusupati et al. ([2022]); Li et al. ([2024])*. Our definition of granularity in text ranking corresponds to ranking relevant sub-units within a given retrieval unit.

Multi-vector approaches*Luan et al. ([2021]); Khattab and Zaharia ([2020]); Santhanam et al. ([2022])* have primarily been used for ranking at the same granularity as the level of encoding, which is typically passage-level. Single vector approaches, on the other hand, inherently do not support ranking at a finer granularity than the level of encoding, thereby needing a separate dense index for each granularity*Chen et al. ([2023c])*. Hence, specialized models for single-vector embeddings have be introduced for embedding phrases*Lee et al. ([2021])*, propositions*Chen et al. ([2023b])*, sentences*Reimers and Gurevych ([2019])* or passages*Karpukhin et al. ([2020])*. Our approach leverages multi-vector approaches for ranking at different granularities, while still encoding at a single coarser level of granularity.

Prior approaches that score at different granularities have leveraged custom scoring functions or incorporate separate embeddings.*Chang et al. ([2023])* proposes a multi-granularity matching model that uses a convolutional filter for scoring, instead of dot similarity, meaning it cannot be scaled to a retrieval-scale corpus due to the matching function. Hierarchical ranking approaches*Liu et al. ([2019]); Chu et al. ([2022]); Ma et al. ([2024])* also consider multi-granular ranking but require use separate embeddings for each granularity to rank at. In contrast, our approach directly uses multi-vector embeddings from a single-level of encoding to rank at any granularity. Further, our approach use a dot product for scoring at all levels of granularity, meaning the same pre-computed dense corpus index can be used for any granularity.

6 Conclusion
------------

In this work, we introduce AGRaME, which leverages multi-vector embeddings to rank at finer granularities, while encoding is still at a single, coarser level. Our proposed multi-granular contrastive loss for training multi-vector approaches improves sentence ranking performance even with passage-level encoding. We demonstrate that AGRaME can rank at any-granularity, even by using sub-parts of the query for ranking. Leveraging multi-vector approaches’ superior performance at proposition-level ranking, our post-hoc attribution approach uses propositions in the generated text to rank input context passages to identify the relevant one to cite. We show superior performance with PropCite over the conventional approach of prompt-driven citation in retrieval-augmented generation.

Acknowledgement
---------------

We would like to thank Omar Khattab and members of the Blender NLP group for helpful comments and feedback. We are also grateful to members of the Apple Knowledge Platform team, especially Mostafa Arefiyan, Ihab Ilyas, Theodoros Rekatsinas and Benjamin Han for early discussions. This research is based on work supported by U.S. DARPA KAIROS Program No. FA8750-19-2-1004. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

References
----------

* Asai et al. (2023)Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.Self-rag: Learning to retrieve, generate, and critique through self-reflection.In *The Twelfth International Conference on Learning Representations*.
* Bai et al. (2023)Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023.Qwen technical report.*arXiv preprint arXiv:2309.16609*.
* Berant et al. (2013)Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013.Semantic parsing on freebase from question-answer pairs.In *Proceedings of the 2013 conference on empirical methods in natural language processing*, pages 1533–1544.
* Bohnet et al. (2022)Bernd Bohnet, Vinh Q Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Massimiliano Ciaramita, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, et al. 2022.Attributed question answering: Evaluation and modeling for attributed large language models.*arXiv preprint arXiv:2212.08037*.
* Chang et al. (2023)Guanghui Chang, Weihan Wang, and Shiyang Hu. 2023.Matchacnn: A multi-granularity deep matching model.*Neural Processing Letters*, 55(4):4419–4438.
* Chen et al. (2023a)Sihao Chen, Senaka Buthpitiya, Alex Fabrikant, Dan Roth, and Tal Schuster. 2023a.Propsegment: A large-scale corpus for proposition-level segmentation and entailment recognition.In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 8874–8893.
* Chen et al. (2023b)Sihao Chen, Hongming Zhang, Tong Chen, Ben Zhou, Wenhao Yu, Dian Yu, Baolin Peng, Hongwei Wang, Dan Roth, and Dong Yu. 2023b.Sub-sentence encoder: Contrastive learning of propositional semantic representations.*arXiv preprint arXiv:2311.04335*.
* Chen et al. (2023c)Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Dong Yu, and Hongming Zhang. 2023c.Dense x retrieval: What retrieval granularity should we use?*arXiv preprint arXiv:2312.06648*.
* Chu et al. (2022)Xiaokai Chu, Jiashu Zhao, Lixin Zou, and Dawei Yin. 2022.H-ernie: A multi-granularity pre-trained language model for web search.In *Proceedings of the 45th International ACM SIGIR conference on research and development in information retrieval*, pages 1478–1489.
* Devlin et al. (2019)Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.Bert: Pre-training of deep bidirectional transformers for language understanding.In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186.
* Fan et al. (2019)Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019.Eli5: Long form question answering.In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 3558–3567.
* Gao et al. (2023a)Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu. 2023a.[RARR: Researching and revising what language models say, using language models](https://doi.org/10.18653/v1/2023.acl-long.910 "").In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 16477–16508, Toronto, Canada. Association for Computational Linguistics.
* Gao et al. (2023b)Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. 2023b.Rarr: Researching and revising what language models say, using language models.In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 16477–16508.
* Gao et al. (2023c)Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023c.Enabling large language models to generate text with citations.In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 6465–6488.
* Gautier et al. (2022)Izacard Gautier, Caron Mathilde, Hosseini Lucas, Riedel Sebastian, Bojanowski Piotr, Joulin Armand, and Grave Edouard. 2022.Unsupervised dense information retrieval with contrastive learning.*Transactions on Machine Learning Research*.
* Goyal and Durrett (2020)Tanya Goyal and Greg Durrett. 2020.Evaluating factuality in generation with dependency-level entailment.In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 3592–3603.
* Han et al. (2023)Rujun Han, Peng Qi, Yuhao Zhang, Lan Liu, Juliette Burger, William Yang Wang, Zhiheng Huang, Bing Xiang, and Dan Roth. 2023.Robustqa: Benchmarking the robustness of domain adaptation for open-domain question answering.In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 4294–4311.
* Honovich et al. (2022)Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. 2022.[TRUE: Re-evaluating factual consistency evaluation](https://doi.org/10.18653/v1/2022.naacl-main.287 "").In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3905–3920, Seattle, United States. Association for Computational Linguistics.
* Izacard et al. (2023)Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023.Atlas: Few-shot learning with retrieval augmented language models.*Journal of Machine Learning Research*, 24(251):1–43.
* Jiang et al. (2023)Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023.Mistral 7b.*arXiv preprint arXiv:2310.06825*.
* Joshi et al. (2017)Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017.Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611.
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.Dense passage retrieval for open-domain question answering.In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781.
* Khattab and Zaharia (2020)Omar Khattab and Matei Zaharia. 2020.Colbert: Efficient and effective passage search via contextualized late interaction over bert.In *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*, pages 39–48.
* Kusupati et al. (2022)Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. 2022.Matryoshka representation learning.*Advances in Neural Information Processing Systems*, 35:30233–30249.
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019.Natural questions: a benchmark for question answering research.*Transactions of the Association for Computational Linguistics*, 7:453–466.
* Lee et al. (2021)Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2021.Learning dense representations of phrases at scale.In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 6634–6647.
* Lee et al. (2019)Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019.Latent retrieval for weakly supervised open domain question answering.In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 6086–6096.
* Lewis et al. (2020)Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020.Retrieval-augmented generation for knowledge-intensive nlp tasks.*Advances in Neural Information Processing Systems*, 33:9459–9474.
* Li et al. (2024)Xianming Li, Zongxi Li, Jing Li, Haoran Xie, and Qing Li. 2024.2d matryoshka sentence embeddings.*arXiv preprint arXiv:2402.14776*.
* Liu et al. (2023)Nelson F Liu, Tianyi Zhang, and Percy Liang. 2023.Evaluating verifiability in generative search engines.In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 7001–7025.
* Liu et al. (2019)Wei Liu, Lei Zhang, Longxuan Ma, Pengfei Wang, and Feng Zhang. 2019.Hierarchical multi-dimensional attention model for answer selection.In *2019 International Joint Conference on Neural Networks (IJCNN)*, pages 1–8. IEEE.
* Luan et al. (2021)Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021.Sparse, dense, and attentional representations for text retrieval.*Transactions of the Association for Computational Linguistics*, 9:329–345.
* Ma et al. (2024)Kai Ma, Junyuan Deng, Miao Tian, Liufeng Tao, Junjie Liu, Zhong Xie, Hua Huang, and Qinjun Qiu. 2024.Multi-granularity retrieval of mineral resource geological reports based on multi-feature association.*Ore Geology Reviews*, page 105889.
* Maia et al. (2018)Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018.Www’18 open challenge: financial opinion mining and question answering.In *Companion proceedings of the the web conference 2018*, pages 1941–1942.
* Min et al. (2023)Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023.Factscore: Fine-grained atomic evaluation of factual precision in long form text generation.In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 12076–12100.
* Nguyen et al. (2016)Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016.Ms marco: A human generated machine reading comprehension dataset.In *CoCo@ NIPs*.
* Ni et al. (2022a)Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. 2022a.Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models.In *Findings of the Association for Computational Linguistics: ACL 2022*, pages 1864–1874.
* Ni et al. (2022b)Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, et al. 2022b.Large dual encoders are generalizable retrievers.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 9844–9855.
* Raffel et al. (2020)Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020.Exploring the limits of transfer learning with a unified text-to-text transformer.*Journal of Machine Learning Research*, 21:1–67.
* Rajpurkar et al. (2016)Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.Squad: 100,000+ questions for machine comprehension of text.In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2383–2392.
* Rashkin et al. (2023)Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. 2023.Measuring attribution in natural language generation models.*Computational Linguistics*, 49(4):777–840.
* Reimers and Gurevych (2019)Nils Reimers and Iryna Gurevych. 2019.[Sentence-bert: Sentence embeddings using siamese bert-networks](https://arxiv.org/abs/1908.10084 "").In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics.
* Santhanam et al. (2022)Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022.Colbertv2: Effective and efficient retrieval via lightweight late interaction.In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3715–3734.
* Sciavolino et al. (2021)Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. 2021.Simple entity-centric questions challenge dense retrievers.In *2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021*, pages 6138–6148. Association for Computational Linguistics (ACL).
* Stelmakh et al. (2022)Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. 2022.Asqa: Factoid questions meet long-form answers.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 8273–8288.
* Thakur et al. (2021)Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021.Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models.In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*.
* Tsatsaronis et al. (2015)George Tsatsaronis, Georgios Balikas, Prodromos Malakasiotis, Ioannis Partalas, Matthias Zschunke, Michael R Alvers, Dirk Weissenborn, Anastasia Krithara, Sergios Petridis, Dimitris Polychronopoulos, et al. 2015.An overview of the bioasq large-scale biomedical semantic indexing and question answering competition.*BMC bioinformatics*, 16:1–28.
* (48)Chengji Wang, Zhiming Luo, Yaojin Lin, and Shaozi Li.Text-based person search via multi-granularity embedding learning.
* Wanner et al. (2024)Miriam Wanner, Seth Ebner, Zhengping Jiang, Mark Dredze, and Benjamin Van Durme. 2024.A closer look at claim decomposition.*arXiv preprint arXiv:2403.11903*.
* Zhang et al. (2022)Jiacheng Zhang, Xiangru Lin, Minyue Jiang, Yue Yu, Chenting Gong, Wei Zhang, Xiao Tan, Yingying Li, Errui Ding, and Guanbin Li. 2022.A multi-granularity retrieval system for natural language-based vehicle retrieval.In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3216–3225.
