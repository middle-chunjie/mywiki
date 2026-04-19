# Phrase Retrieval Learns Passage Retrieval, Too

Jinhyuk Lee $^{1,2*}$  Alexander Wettig $^{1}$  Danqi Chen $^{1}$

Department of Computer Science, Princeton University

Department of Computer Science and Engineering, Korea University

{jinhyuklee,awettig,danqic}@cs.princeton.edu

# Abstract

Dense retrieval methods have shown great promise over sparse retrieval methods in a range of NLP problems. Among them, dense phrase retrieval—the most fine-grained retrieval unit—is appealing because phrases can be directly used as the output for question answering and slot filling tasks. In this work, we follow the intuition that retrieving phrases naturally entails retrieving larger text blocks and study whether phrase retrieval can serve as the basis for coarse-level retrieval including passages and documents. We first observe that a dense phrase-retrieval system, without any retraining, already achieves better passage retrieval accuracy ( $+3 - 5\%$  in top-5 accuracy) compared to passage retrievers, which also helps achieve superior end-to-end QA performance with fewer passages. Then, we provide an interpretation for why phrase-level supervision helps learn better fine-grained entailment compared to passage-level supervision, and also show that phrase retrieval can be improved to achieve competitive performance in document-retrieval tasks such as entity linking and knowledge-grounded dialogue. Finally, we demonstrate how phrase filtering and vector quantization can reduce the size of our index by  $4 - 10x$ , making dense phrase retrieval a practical and versatile solution in multi-granularity retrieval.

# 1 Introduction

Dense retrieval aims to retrieve relevant contexts from a large corpus, by learning dense representations of queries and text segments. Recently, dense retrieval of passages (Lee et al., 2019; Karpukhin et al., 2020; Xiong et al., 2021) has been shown


Figure 1: Comparison of passage representations from DPR (Karpukhin et al., 2020) and DensePhrases (Lee et al., 2021). Unlike using a single vector for each passage, DensePhrases represents each passage with multiple phrase vectors and the score of a passage can be computed by the maximum score of phrases within it.

to outperform traditional sparse retrieval methods such as TF-IDF and BM25 in a range of knowledge-intensive NLP tasks (Petroni et al., 2021), including open-domain question answering (QA) (Chen et al., 2017), entity linking (Wu et al., 2020), and knowledge-grounded dialogue (Dinan et al., 2019).

One natural design choice of these dense retrieval methods is the retrieval unit. For instance, the dense passage retriever (DPR) (Karpukhin et al., 2020) encodes a fixed-size text block of 100 words as the basic retrieval unit. On the other extreme, recent work (Seo et al., 2019; Lee et al., 2021) demonstrates that phrases can be used as a retrieval unit. In particular, Lee et al. (2021) show that learning dense representations of phrases alone can achieve competitive performance in a number of open-domain QA and slot filling tasks. This is particularly appealing since the phrases can directly serve as the output, without relying on an additional reader model to process text passages.

In this work, we draw on an intuitive motivation that every single phrase is embedded within a larger text context and ask the following question: If a retriever is able to locate phrases, can

we directly make use of it for passage and even document retrieval as well? We formulate phrase-based passage retrieval, in which the score of a passage is determined by the maximum score of phrases within it (see Figure 1 for an illustration). By evaluating DensePhrases (Lee et al., 2021) on popular QA datasets, we observe that it achieves competitive or even better passage retrieval accuracy compared to DPR, without any re-training or modification to the original model (Table 1). The gains are especially pronounced for top- $k$  accuracy when  $k$  is smaller (e.g., 5), which also helps achieve strong open-domain QA accuracy with a much smaller number of passages as input to a generative reader model (Izacard and Grave, 2021b).

To better understand the nature of dense retrieval methods, we carefully analyze the training objectives of phrase and passage retrieval methods. While the in-batch negative losses in both models encourage them to retrieve topically relevant passages, we find that phrase-level supervision in DensePhrases provides a stronger training signal than using hard negatives from BM25, and helps DensePhrases retrieve correct phrases, and hence passages. Following this positive finding, we further explore whether phrase retrieval can be extended to retrieval of coarser granularities, or other NLP tasks. Through fine-tuning of the query encoder with document-level supervision, we are able to obtain competitive performance on entity linking (Hoffart et al., 2011) and knowledge-grounded dialogue retrieval (Dinan et al., 2019) in the KILT benchmark (Petroni et al., 2021).

Finally, we draw connections to multi-vector passage encoding models (Khattab and Zaharia, 2020; Luan et al., 2021), where phrase retrieval models can be viewed as learning a dynamic set of vectors for each passage. We show that a simple phrase filtering strategy learned from QA datasets gives us a control over the trade-off between the number of vectors per passage and the retrieval accuracy. Since phrase retrievers encode a larger number of vectors, we also propose a quantization-aware fine-tuning method based on Optimized Product Quantization (Ge et al., 2013), reducing the size of the phrase index from 307GB to 69GB (or under 30GB with more aggressive phrase filtering) for full English Wikipedia, without any performance degradation. This matches the index size of passage retrievers and makes dense phrase retrieval a practical and versatile solution for multi-granularity retrieval.

# 2 Background

Passage retrieval Given a set of documents  $\mathcal{D}$ , passage retrieval aims to provide a set of relevant passages for a question  $q$ . Typically, each document in  $\mathcal{D}$  is segmented into a set of disjoint passages and we denote the entire set of passages in  $\mathcal{D}$  as  $\mathcal{P} = \{p_1,\dots ,p_M\}$ , where each passage can be a natural paragraph or a fixed-length text block. A passage retriever is designed to return top- $k$  passages  $\mathcal{P}_k\subset \mathcal{P}$  with the goal of retrieving passages that are relevant to the question. In open-domain QA, passages are considered relevant if they contain answers to the question. However, many other knowledge-intensive NLP tasks (e.g., knowledge-grounded dialogue) provide human-annotated evidence passages or documents.

While traditional passage retrieval models rely on sparse representations such as BM25 (Robertson and Zaragoza, 2009), recent methods show promising results with dense representations of passages and questions, and enable retrieving passages that may have low lexical overlap with questions. Specifically, Karpukhin et al. (2020) introduce DPR that has a passage encoder  $E_{p}(\cdot)$  and a question encoder  $E_{q}(\cdot)$  trained on QA datasets and retrieves passages by using the inner product as a similarity function between a passage and a question:

$$
f (p, q) = E _ {p} (p) ^ {\top} E _ {q} (q). \tag {1}
$$

For open-domain QA where a system is required to provide an exact answer string  $a$ , the retrieved top  $k$  passages  $\mathcal{P}_k$  are subsequently fed into a reading comprehension model such as a BERT model (Devlin et al., 2019), and this is called the retriever-reader approach (Chen et al., 2017).

Phrase retrieval While passage retrievers require another reader model to find an answer, Seo et al. (2019) introduce the phrase retrieval approach that encodes phrases in each document and performs similarity search over all phrase vectors to directly locate the answer. Following previous work (Seo et al., 2018, 2019), we use the term 'phrase' to denote any contiguous text segment up to  $L$  words (including single words), which is not necessarily a linguistic phrase and we take phrases up to length  $L = 20$ . Given a phrase  $s^{(p)}$  from a passage  $p$ , their similarity function  $f$  is computed as:

$$
f (s ^ {(p)}, q) = E _ {s} \left(s ^ {(p)}\right) ^ {\top} E _ {q} (q), \tag {2}
$$

<table><tr><td rowspan="2">Retriever</td><td colspan="5">Natural Questions</td><td colspan="5">TriviaQA</td></tr><tr><td>Top-1</td><td>Top-5</td><td>Top-20</td><td>MRR@20</td><td>P@20</td><td>Top-1</td><td>Top-5</td><td>Top-20</td><td>MRR@20</td><td>P@20</td></tr><tr><td>DPR◇</td><td>46.0</td><td>68.1</td><td>79.8</td><td>55.7</td><td>16.5</td><td>54.4†</td><td>-</td><td>79.4‡</td><td>-</td><td>-</td></tr><tr><td>DPR♠</td><td>44.2</td><td>66.8</td><td>79.2</td><td>54.2</td><td>17.7</td><td>54.6</td><td>70.8</td><td>79.5</td><td>61.7</td><td>30.3</td></tr><tr><td>DensePhrases◇</td><td>50.1</td><td>69.5</td><td>79.8</td><td>58.7</td><td>20.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DensePhrases♠</td><td>51.1</td><td>69.9</td><td>78.7</td><td>59.3</td><td>22.7</td><td>62.7</td><td>75.0</td><td>80.9</td><td>68.2</td><td>38.4</td></tr></table>

Table 1: Open-domain QA passage retrieval results. We retrieve top  $k$  passages from DensePhrases using Eq. (3). We report top-  $k$  passage retrieval accuracy (Top-  $k$  ),mean reciprocal rank at  $k\left( {\mathrm{{MRR}}@k}\right)$  ,and precision at  $k\left( {\mathrm{P}@k}\right)$  .  ${}^{ \diamond  }$  : trained on each dataset independently.  ${}^{ \spadesuit  }$  : trained on multiple open-domain QA datasets. See §3.1 for more details.  ${}^{ \dagger  }$  : (Yang and Seo,2020).  ${}^{ \ddagger  }$  : (Karpukhin et al.,2020).

where  $E_{s}(\cdot)$  and  $E_{q}(\cdot)$  denote the phrase encoder and the question encoder, respectively. Since this formulates open-domain QA purely as a maximum inner product search (MIPS), it can drastically improve end-to-end efficiency. While previous work (Seo et al., 2019; Lee et al., 2020) relied on a combination of dense and sparse vectors, Lee et al. (2021) demonstrate that dense representations of phrases alone are sufficient to close the performance gap with retriever-reader systems. For more details on how phrase representations are learned, we refer interested readers to Lee et al. (2021).

# 3 Phrase Retrieval for Passage Retrieval

Phrases naturally have their source texts from which they are extracted. Based on this fact, we define a simple phrase-based passage retrieval strategy, where we retrieve passages based on the phrase-retrieval score:

$$
\tilde {f} (p, q) := \max  _ {s ^ {(p)} \in \mathcal {S} (p)} E _ {s} \left(s ^ {(p)}\right) ^ {\top} E _ {q} (q), \tag {3}
$$

where  $S(p)$  denotes the set of phrases in the passage  $p$ . In practice, we first retrieve a slightly larger number of phrases, compute the score for each passage, and return top  $k$  unique passages. Based on our definition, phrases can act as a basic retrieval unit of any other granularity such as sentences or documents by simply changing  $S(p)$  (e.g.,  $s^{(d)} \in S(d)$  for a document  $d$ ). Note that, since the cost of score aggregation is negligible, the inference speed of phrase-based passage retrieval is the same as for phrase retrieval, which is shown to be efficient in Lee et al. (2021). In this section, we evaluate the passage retrieval performance (Eq. (3)) and also how phrase-based passage retrieval can contribute to end-to-end open-domain QA.

# 3.1 Experiment: Passage Retrieval

Datasets We use two open-domain QA datasets: Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017), following the standard train/dev/test splits for the open-domain QA evaluation. For both models, we use the 2018-12-20 Wikipedia snapshot. To provide a fair comparison, we use Wikipedia articles pre-processed for DPR, which are split into 21-million text blocks and each text block has exactly 100 words. Note that while DPR is trained in this setting, DensePhrases is trained with natural paragraphs. $^4$

Models For DPR, we use publicly available checkpoints trained on each dataset  $(\mathrm{DPR}^{\diamond})$  or multiple QA datasets  $(\mathrm{DPR}^{\spadesuit})$ , which we find to perform slightly better than the ones reported in Karpukhin et al. (2020). For DensePhrases, we train it on Natural Questions  $(\mathrm{DensePhrases}^{\diamond})$  or multiple QA datasets  $(\mathrm{DensePhrases}^{\spadesuit})$  with the code provided by the authors. Note that we do not make any modification to the architecture or training methods of DensePhrases and achieve similar open-domain QA accuracy as reported. For phrase-based passage retrieval, we compute Eq. (3) with DensePhrases and return top  $k$  passages.

Metrics Following previous work on passage retrieval for open-domain QA, we measure the top- $k$  passage retrieval accuracy (Top- $k$ ), which denotes the proportion of questions whose top  $k$  retrieved passages contain at least one of the gold answers.

To further characterize the behavior of each system, we also include the following evaluation metrics: mean reciprocal rank at  $k$  (MRR@k) and precision at  $k$  (P@k). MRR@k is the average reciprocal rank of the first relevant passage (that contains an answer) in the top  $k$  passages. Higher MRR@k means relevant passages appear at higher ranks. Meanwhile, P@k is the average proportion of relevant passages in the top  $k$  passages. Higher P@k denotes that a larger proportion of top  $k$  passages contains the answers.

Results As shown in Table 1, DensePhrases achieves competitive passage retrieval accuracy with DPR, while having a clear advantage on top-1 or top-5 accuracy for both Natural Questions (+6.9% Top-1) and TriviaQA (+8.1% Top-1). Although the top-20 (and top-100, which is not shown) accuracy is similar across different models, MRR@20 and P@20 reveal interesting aspects of DensePhrases—it ranks relevant passages higher and provides a larger number of correct passages. Our results suggest that DensePhrases can also retrieve passages very accurately, even though it was not explicitly trained for that purpose. For the rest of the paper, we mainly compare the DPR $\clubsuit$  and DensePhrases $\clubsuit$  models, which were both trained on multiple QA datasets.

# 3.2 Experiment: Open-domain QA

Recently, Izacard and Grave (2021b) proposed the Fusion-in-Decoder (FiD) approach where they feed top 100 passages from DPR into a generative model T5 (Raffel et al., 2020) and achieve the state-of-the-art on open-domain QA benchmarks. Since their generative model computes the hidden states of all tokens in 100 passages, it requires large GPU memory and Izacard and Grave (2021b) used 64 Tesla V100 32GB for training.

In this section, we use our phrase-based passage retrieval with DensePhrases to replace DPR in FiD and see if we can use a much smaller number of passages to achieve comparable performance, which can greatly reduce the computational requirements. We train our model with 4 24GB RTX GPUs for training T5-base, which are more affordable with academic budgets. Note that training T5-base with 5 or 10 passages can also be done with 11GB GPUs. We keep all the hyperparameters the same as in Izacard and Grave (2021b).

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="2">NaturalQ</td><td>TriviaQA</td></tr><tr><td>Dev</td><td>Test</td><td>Test</td></tr><tr><td colspan="2">ORQA (Lee et al., 2019)</td><td>-</td><td>33.3</td><td>45.0</td></tr><tr><td colspan="2">REALM (Guu et al., 2020)</td><td>-</td><td>40.4</td><td>-</td></tr><tr><td colspan="2">DPR (reader: BERT-base)</td><td>-</td><td>41.5</td><td>56.8</td></tr><tr><td colspan="2">DensePhrases</td><td>-</td><td>41.3</td><td>53.5</td></tr><tr><td colspan="5">FiD with DPR (Izacard and Grave, 2021b)</td></tr><tr><td rowspan="5">Reader: T5-base</td><td>k = 5</td><td>37.8</td><td>-</td><td>-</td></tr><tr><td>k = 10</td><td>42.3</td><td>-</td><td>-</td></tr><tr><td>k = 25</td><td>45.3</td><td>-</td><td>-</td></tr><tr><td>k = 50</td><td>45.7</td><td>-</td><td>-</td></tr><tr><td>k = 100</td><td>46.5</td><td>48.2</td><td>65.0</td></tr><tr><td colspan="5">FiD with DensePhrases (ours)</td></tr><tr><td rowspan="4">Reader: T5-base</td><td>k = 5</td><td>44.2</td><td>45.9</td><td>59.5</td></tr><tr><td>k = 10</td><td>45.5</td><td>45.9</td><td>61.0</td></tr><tr><td>k = 25</td><td>46.4</td><td>47.2</td><td>63.4</td></tr><tr><td>k = 50</td><td>47.2</td><td>47.9</td><td>64.5</td></tr></table>

Table 2: Open-domain QA results. We report exact match (EM) of each model by feeding top  $k$  passages into a T5-base model. DensePhrases can greatly reduce the computational cost of running generative reader models while having competitive performance.

Results As shown in Table 2, using DensePhrases as a passage retriever achieves competitive performance to DPR-based FiD and significantly improves upon the performance of original DensePhrases (NQ = 41.3 EM without a reader). Its better retrieval quality at top-k for smaller  $k$  indeed translates to better open-domain QA accuracy, achieving +6.4% gain compared to DPR-based FiD when  $k = 5$ . To obtain similar performance with using 100 passages in FiD, DensePhrases needs fewer passages ( $k = 25$  or 50), which can fit in GPUs with smaller RAM.

# 4 A Unified View of Dense Retrieval

As shown in the previous section, phrase-based passage retrieval is able to achieve competitive passage retrieval accuracy, despite that the models were not explicitly trained for that. In this section, we compare the training objectives of DPR and DensePhrases in detail and explain how DensePhrases learn passage retrieval.

# 4.1 Training Objectives

Both DPR and DensePhrases set out to learn a similarity function  $f$  between a passage or phrase and a question. Passages and phrases differ primarily in characteristic length, so we refer to either as

Figure 2: Comparison of training objectives of DPR and DensePhrases. While both models use in-batch negatives, DensePhrases use in-passage negatives (phrases) compared to BM25 hard-negative passages in DPR. Note that each phrase in DensePhrases can directly serve as an answer to open-domain questions.


a retrieval unit  $x$ .<sup>8</sup> DPR and DensePhrases both adopt a dual-encoder approach with inner product similarity as shown in Eq. (1) and (2), and they are initialized with BERT (Devlin et al., 2019) and SpanBERT (Joshi et al., 2020), respectively.

These dual-encoder models are then trained with a negative log-likelihood loss for discriminating positive retrieval units from negative ones:

$$
\mathcal {L} = - \log \frac {e ^ {f \left(x ^ {+} , q\right)}}{e ^ {f \left(x ^ {+} , q\right)} + \sum_ {x ^ {-} \in \mathcal {X} ^ {-}} e ^ {f \left(x ^ {-} , q\right)}}, \tag {4}
$$

where  $x^{+}$  is the positive phrase or passage corresponding to question  $q$ , and  $\mathcal{X}^{-}$  is a set of negative examples. The choice of negatives is critical in this setting and both DPR and DensePhrases make important adjustments.

In-batch negatives In-batch negatives are a common way to define  $\mathcal{X}^{-}$ , since they are available at no extra cost when encoding a mini-batch of examples. Specifically, in a mini-batch of  $B$  examples, we can add  $B - 1$  in-batch negatives for each positive example. Since each mini-batch is randomly sampled from the set of all training passages, in-batch negative passages are usually topically negative, i.e., models can discriminate between  $x^{+}$  and  $\mathcal{X}^{-}$  based on their topic only.

Hard negatives Although topic-related features are useful in identifying broadly relevant passages, they often lack the precision to locate the exact passage containing the answer in a large corpus.

Karpukhin et al. (2020) propose to use additional hard negatives which have a high BM25 lexical overlap with a given question but do not contain the answer. These hard negatives are likely to share a similar topic and encourage DPR to learn more fine-grained features to rank  $x^{+}$  over the hard negatives. Figure 2 (left) shows an illustrating example.

In-passage negatives While DPR is limited to use positive passages  $x^{+}$  which contain the answer, DensePhrases is trained to predict that the positive phrase  $x^{+}$  is the answer. Thus, the fine-grained structure of phrases allows for another source of negatives, in-passage negatives. In particular, DensePhrases augments the set of negatives  $\mathcal{X}^{-}$  to encompass all phrases within the same passage that do not express the answer.<sup>9</sup> See Figure 2 (right) for an example. We hypothesize that these in-passage negatives achieve a similar effect as DPR's hard negatives: They require the model to go beyond simple topic modeling since they share not only the same topic but also the same context. Our phrase-based passage retriever might benefit from this phrase-level supervision, which has already been shown to be useful in the context of distilling knowledge from reader to retriever (Izacard and Grave, 2021a; Yang and Seo, 2020).

# 4.2 Topical vs. Hard Negatives

To address our hypothesis, we would like to study how these different types of negatives used by DPR and DensePhrases affect their reliance on topical

Figure 3: Comparison of DPR and DensePhrases on NQ (dev) with  $\mathcal{L}_{\mathrm{topic}}$  and  $\mathcal{L}_{\mathrm{hard}}$ . Starting from each model trained with in-batch negatives (in-batch), we show the effect of using hard negatives (+BM25), in-passage negatives (+in-passage), as well as training on multiple QA datasets (+multi. dataset). The  $x$ -axis is in log-scale for better visualization. For both metrics, lower numbers are better.

and fine-grained entailment cues. We characterize their passage retrieval based on two metrics (losses):  $\mathcal{L}_{\mathrm{topic}}$  and  $\mathcal{L}_{\mathrm{hard}}$ . We use Eq. (4) to define both  $\mathcal{L}_{\mathrm{topic}}$  and  $\mathcal{L}_{\mathrm{hard}}$ , but use different sets of negatives  $\mathcal{X}^{-}$ . For  $\mathcal{L}_{\mathrm{topic}}$ ,  $\mathcal{X}^{-}$  contains passages that are topically different from the gold passage—In practice, we randomly sample passages from English Wikipedia. For  $\mathcal{L}_{\mathrm{hard}}$ ,  $\mathcal{X}^{-}$  uses negatives containing topically similar passages, such that  $\mathcal{L}_{\mathrm{hard}}$  estimates how accurately models locate a passage that contains the exact answer among topically similar passages. From a positive passage paired with a question, we create a single hard negative by removing the sentence that contains the answer. $^{10}$  In our analysis, both metrics are estimated on the Natural Questions development set, which provides a set of questions and (gold) positive passages.

Results Figure 3 shows the comparison of DPR and DensePhrases trained on NQ with the two losses. For DensePhrases, we compute the passage score using  $\tilde{f}(p,q)$  as described in Eq. (3). First, we observe that in-batch negatives are highly effective at reducing  $\mathcal{L}_{\mathrm{topic}}$  as DensePhrases trained with only in-passage negatives has a relatively high  $\mathcal{L}_{\mathrm{topic}}$ . Furthermore, we observe that using in-passage negatives in DensePhrases (+in-passage) significantly lowers  $\mathcal{L}_{\mathrm{hard}}$ , even lower than DPR

<table><tr><td>Type</td><td>D={p}</td><td>D= Dsmall</td></tr><tr><td>DensePhrases</td><td>71.8</td><td>61.3</td></tr><tr><td>+ BM25 neg.</td><td>71.8</td><td>60.6</td></tr><tr><td>+ Same-phrase neg.</td><td>72.1</td><td>60.9</td></tr></table>

Table 3: Effect of using hard negatives in DensePhrases on the NQ development set. We report EM when a single gold passage is given  $(\mathcal{D} = \{p\})$  or 6K passages are given by gathering all the gold passages from NQ development set  $(\mathcal{D} = \mathcal{D}_{\mathrm{small}})$ . The two hard negatives do not give any noticeable improvement in DensePhrases.

that uses BM25 hard negatives (+BM25). Using multiple datasets (+multi. dataset) further improves  $\mathcal{L}_{\mathrm{hard}}$  for both models. DPR has generally better (lower)  $\mathcal{L}_{\mathrm{topic}}$  than DensePhrases, which might be due to the smaller training batch size of DensePhrases (hence a smaller number of in-batch negatives) compared to DPR. The results suggest that DensePhrases relies less on topical features and is better at retrieving passages based on fine-grained entailment cues. This might contribute to the better ranking of the retrieved passages in Table 1, where DensePhrases shows better MRR@20 and P@20 while top-20 accuracy is similar.

Hard negatives for DensePhrases? We test two different kinds of hard negatives in DensePhrases to see whether its performance can further improve in the presence of in-passage negatives. For each training question, we mine for a hard negative passage, either by BM25 similarity or by finding another passage that contains the gold-answer phrase, but possibly with a wrong context. Then we use all phrases from the hard negative passage as additional hard negatives in  $\mathcal{X}^{-}$  along with the existing in-passage negatives. As shown in Table 3, DensePhrases obtain no substantial improvements from additional hard negatives, indicating that in-passage negatives are already highly effective at producing good phrase (or passage) representations.

# 5 Improving Coarse-grained Retrieval

While we showed that DensePhrases implicitly learns passage retrieval, Figure 3 indicates that DensePhrases might not be very good for retrieval tasks where topic matters more than fine-grained entailment, for instance, the retrieval of a single evidence document for entity linking. In this section, we propose a simple method that can adapt DensePhrases to larger retrieval units, especially when the topical relevance is more important.

Method We modify the query-side fine-tuning proposed by Lee et al. (2021), which drastically improves the performance of DensePhrases by reducing the discrepancy between training and inference time. Since it is prohibitive to update the large number of phrase representations after indexing, only the query encoder is fine-tuned over the entire set of phrases in Wikipedia. Given a question  $q$  and an annotated document set  $\mathcal{D}^*$ , we minimize:

$$
\mathcal {L} _ {\mathrm {d o c}} = - \log \frac {\sum_ {s \in \tilde {\mathcal {S}} (q) , d (s) \in \mathcal {D} ^ {*}} e ^ {f (s , q)}}{\sum_ {s \in \tilde {\mathcal {S}} (q)} e ^ {f (s , q)}}, \tag {5}
$$

where  $\tilde{S}(q)$  denotes top  $k$  phrases for the question  $q$ , out of the entire set of phrase vectors. To retrieve coarse-grained text better, we simply check the condition  $d(s) \in \mathcal{D}^*$ , which means  $d(s)$ , the source document of  $s$ , is included in the set of annotated gold documents  $\mathcal{D}^*$  for the question. With  $\mathcal{L}_{\mathrm{doc}}$ , the model is trained to retrieve any phrases that are contained in a relevant document. Note that  $d(s)$  can be changed to reflect any desired level of granularity such as passages.

Datasets We test DensePhrases trained with  $\mathcal{L}_{\mathrm{doc}}$  on entity linking (Hoffart et al., 2011; Guo and Barbosa, 2018) and knowledge-grounded dialogue (Dinan et al., 2019) tasks in KILT (Petroni et al., 2021). Entity linking contains three datasets: AIDA CoNLL-YAGO (AY2) (Hoffart et al., 2011), WNED-WIKI (WnWi) (Guo and Barbosa, 2018), and WNED-CWEB (WnCw) (Guo and Barbosa, 2018). Each query in entity linking datasets contains a named entity marked with special tokens (i.e., [STARTEnt], [ENDENT]), which need to be linked to one of the Wikipedia articles. For knowledge-grounded dialogue, we use Wizard of Wikipedia (WoW) (Dinan et al., 2019) where each query consists of conversation history, and the generated utterances should be grounded in one of the Wikipedia articles. We follow the KILT guidelines and evaluate the document (i.e., Wikipedia article) retrieval performance of our models given each query. We use R-precision, the proportion of successfully retrieved pages in the top R results, where R is the number of distinct pages in the provenance set. However, in the tasks considered, R-precision is equivalent to precision@1, since each question is annotated with only one document.

Models DensePhrases is trained with the original query-side fine-tuning loss (denoted as  $\mathcal{L}_{\mathrm{phrase}}$ ) or with  $\mathcal{L}_{\mathrm{doc}}$  as described in Eq. (5). When

<table><tr><td rowspan="2">Model</td><td colspan="3">Entity Linking</td><td>Dialogue</td></tr><tr><td>AY2</td><td>WnWi</td><td>WnCw</td><td>WoW</td></tr><tr><td colspan="5">Retriever Only</td></tr><tr><td>TF-IDF</td><td>3.7</td><td>0.2</td><td>2.1</td><td>49.0</td></tr><tr><td>DPR</td><td>1.8</td><td>0.3</td><td>0.5</td><td>25.5</td></tr><tr><td>DensePhrases-Phrase</td><td>7.7</td><td>12.5</td><td>6.4</td><td>-</td></tr><tr><td>DensePhrases-Ldoc</td><td>61.6</td><td>32.1</td><td>37.4</td><td>47.0</td></tr><tr><td>DPR♣</td><td>26.5</td><td>4.9</td><td>1.9</td><td>41.1</td></tr><tr><td>DensePhrases-Ldoc♣</td><td>68.4</td><td>47.5</td><td>47.5</td><td>55.7</td></tr><tr><td colspan="5">Retriever + Additional Components</td></tr><tr><td>RAG</td><td>72.6</td><td>48.1</td><td>47.6</td><td>57.8</td></tr><tr><td>BLINK + flair</td><td>81.5</td><td>80.2</td><td>68.8</td><td>-</td></tr></table>

Table 4: Results on the KILT test set. We report page-level R-precision on each task, which is equivalent to precision@1 on these datasets.  $\clubsuit$ : Multi-task models.

DensePhrases is trained with  $\mathcal{L}_{\mathrm{phrase}}$ , it labels any phrase that matches the title of gold document as positive. After training, DensePhrases returns the document that contains the top passage. For baseline retrieval methods, we report the performance of TF-IDF and DPR from Petroni et al. (2021). We also include a multi-task version of DPR and DensePhrases, which uses the entire KILT training datasets. While not our main focus of comparison, we also report the performance of other baselines from Petroni et al. (2021), which uses generative models (e.g., RAG (Lewis et al., 2020)) or task-specific models (e.g., BLINK (Wu et al., 2020), which has additional entity linking pre-training). Note that these methods use additional components such as a generative model or a cross-encoder model on top of retrieval models.

Results Table 4 shows the results on three entity linking tasks and a knowledge-grounded dialogue task. On all tasks, we find that DensePhrases with  $\mathcal{L}_{\mathrm{doc}}$  performs much better than DensePhrases with  $\mathcal{L}_{\mathrm{phrase}}$  and also matches the performance of RAG that uses an additional large generative model to generate the document titles. Using  $\mathcal{L}_{\mathrm{phrase}}$  does very poorly since it focuses on phrase-level entailment, rather than document-level relevance. Compared to the multi-task version of DPR (i.e., DPR), DensePhrases- $\mathcal{L}_{\mathrm{doc}}$  can be easily adapted to non-QA tasks like entity linking and generalizes better on tasks without training sets (WnWi, WnCw).

# 6 DensePhrases as a Multi-Vector Passage Encoder

In this section, we demonstrate that DensePhrases can be interpreted as a multi-vector passage encoder, which has recently been shown to be very effective for passage retrieval (Luan et al., 2021; Khattab and Zaharia, 2020). Since this type of multi-vector encoding models requires a large disk footprint, we show that we can control the number of vectors per passage (and hence the index size) through filtering. We also introduce quantization techniques to build more efficient phrase retrieval models without a significant performance drop.

# 6.1 Multi-Vector Encodings

Since we represent passages not by a single vector, but by a set of phrase vectors (decomposed as token-level start and end vectors, see Lee et al. (2021)), we notice similarities to previous work, which addresses the capacity limitations of dense, fixed-length passage encodings. While these approaches store a fixed number of vectors per passage (Luan et al., 2021; Humeau et al., 2020) or all token-level vectors (Khattab and Zaharia, 2020), phrase retrieval models store a dynamic number of phrase vectors per passage, where many phrases are filtered by a model trained on QA datasets.

Specifically, Lee et al. (2021) trains a binary classifier (or a phrase filter) to filter phrases based on their phrase representations. This phrase filter is supervised by the answer annotations in QA datasets, hence denotes candidate answer phrases. In our experiment, we tune the filter threshold to control the number of vectors per passage for passage retrieval.

# 6.2 Efficient Phrase Retrieval

The multi-vector encoding models as well as ours are prohibitively large since they contain multiple vector representations for every passage in the entire corpus. We introduce a vector quantization-based method that can safely reduce the size of our phrase index, without performance degradation.

Optimized product quantization Since the multi-vector encoding models are prohibitively large due to their multiple representations, we further introduce a vector quantization-based method that can safely reduce the size of our phrase index, without performance degradation. We use Product Quantization (PQ) (Jegou et al., 2010) where the original vector space is decomposed into the Cartesian

Figure 4: Top-5 passage retrieval accuracy on Natural Questions (dev) for different index sizes of DensePhrases. The index size (GB) and the average number of saved vectors per passage (# vec / p) are controlled by the filtering threshold  $\tau$ . For instance, # vec / p reduces from 28.0 to 5.1 with higher  $\tau$ , which also reduces the index size from 69GB to 23GB. OPQ: Optimized Product Quantization (Ge et al., 2013).

product of subspaces. Using PQ, the memory usage of using  $N$  number of  $d$ -dimensional centroid vectors reduces from  $Nd$  to  $N^{1 / M}d$  with  $M$  subspaces while each database vector requires  $\log_2N$  bits. Among different variants of PQ, we use Optimized Product Quantization (OPQ) (Ge et al., 2013), which learns an orthogonal matrix  $R$  to better decompose the original vector space. See Ge et al. (2013) for more details on OPQ.

Quantization-aware training While this type of aggressive vector quantization can significantly reduce memory usage, it often comes at the cost of performance degradation due to the quantization loss. To mitigate this problem, we use quantization-aware query-side fine-tuning motivated by the recent successes on quantization-aware training (Jacob et al., 2018). Specifically, during query-side fine-tuning, we reconstruct the phrase vectors using the trained (optimized) product quantizer, which are then used to minimize Eq. (5).

# 6.3 Experimental Results

In Figure 4, we present the top-5 passage retrieval accuracy with respect to the size of the phrase index in DensePhrases. First, applying OPQ can reduce the index size of DensePhrases from 307GB to 69GB, while the top-5 retrieval accuracy is poor without quantization-aware query-side fine-tuning. Furthermore, by tuning the threshold  $\tau$  for the phrase filter, the number of vectors per each passage (# vec / p) can be reduced without hurting the

performance significantly. The performance improves with a larger number of vectors per passage, which aligns with the findings of multi-vector encoding models (Khattab and Zaharia, 2020; Luan et al., 2021). Our results show that having 8.8 vectors per passage in DensePhrases has similar retrieval accuracy with DPR.

# 7 Related Work

Text retrieval has a long history in information retrieval, either for serving relevant information to users directly or for feeding them to computationally expensive downstream systems. While traditional research has focused on designing heuristics, such as sparse vector models like TF-IDF and BM25, it has recently become an active area of interest for machine learning researchers. This was precipitated by the emergence of open-domain QA as a standard problem setting (Chen et al., 2017) and the spread of the retriever-reader paradigm (Yang et al., 2019; Nie et al., 2019). The interest has spread to include a more diverse set of downstream tasks, such as fact checking (Thorne et al., 2018), entity-linking (Wu et al., 2020) or dialogue generation (Dinan et al., 2019), where the problems require access to large corpora or knowledge sources. Recently, REALM (Guu et al., 2020) and RAG (retrieval-augmented generation) (Lewis et al., 2020) have been proposed as general-purpose pre-trained models with explicit access to world knowledge through the retriever. There has also been a line of work to integrate text retrieval with structured knowledge graphs (Sun et al., 2018, 2019; Min et al., 2020). We refer to Lin et al. (2020) for a comprehensive overview of neural text retrieval methods.

# 8 Conclusion

In this paper, we show that phrase retrieval models also learn passage retrieval without any modification. By drawing connections between the objectives of DPR and DensePhrases, we provide a better understanding of how phrase retrieval learns passage retrieval, which is also supported by several empirical evaluations on multiple benchmarks. Specifically, phrase-based passage retrieval has better retrieval quality on top  $k$  passages when  $k$  is small, and this translates to an efficient use of passages for open-domain QA. We also show that DensePhrases can be fine-tuned for more coarse-grained retrieval units, serving as a basis for any

retrieval unit. We plan to further evaluate phrase-based passage retrieval on standard information retrieval tasks such as MS MARCO.

# Acknowledgements

We thank Chris Sciavolino, Xingcheng Yao, the members of the Princeton NLP group, and the anonymous reviewers for helpful discussion and valuable feedback. This research is supported by the James Mi *91 Research Innovation Fund for Data Science and gifts from Apple and Amazon. It was also supported in part by the ICT Creative Consilience program (IITP-2021-0-01819) supervised by the IITP (Institute for Information & communications Technology Planning & Evaluation) and National Research Foundation of Korea (NRF-2020R1A2C3010638).

# Ethical Considerations

Models introduced in our work often use question answering datasets such as Natural Questions to build phrase or passage representations. Some of the datasets, like SQuAD, are created from a small number of popular Wikipedia articles, hence could make our model biased towards a small number of topics. We hope that inventing an alternative training method that properly regularizes our model could mitigate this problem. Although our efforts have been made to reduce the computational cost of retrieval models, using passage retrieval models as external knowledge bases will inevitably increase the resource requirements for future experiments. Further efforts should be made to make retrieval more affordable for independent researchers.

# References

Petr Baudis and Jan Šedivý. 2015. Modeling of the question answering task in the YodaQA system. In International Conference of the Cross-Language Evaluation Forum for European Languages.  
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on Freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1533-1544, Seattle, Washington, USA. Association for Computational Linguistics.  
Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In Proceedings of the 55th Annual Meeting of the Association for Computational

Linguistics (Volume 1: Long Papers), pages 1870-1879, Vancouver, Canada. Association for Computational Linguistics.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.  
Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019. Wizard of wikipedia: Knowledge-powered conversational agents. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.  
Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence, 36(4):744-755.  
Zhaochen Guo and Denilson Barbosa. 2018. Robust named entity disambiguation with random walks. Semantic Web, 9(4):459-479.  
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: Retrieval-augmented language model pre-training. In International Conference on Machine Learning.  
Johannes Hoffart, Mohamed Amir Yosef, Ilaria Bordino, Hagen Fürstenau, Manfred Pinkal, Marc Spanirol, Bilyana Taneva, Stefan Thater, and Gerhard Weikum. 2011. Robust disambiguation of named entities in text. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing, pages 782-792, Edinburgh, Scotland, UK. Association for Computational Linguistics.  
Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.  
Gautier Izacard and Edouard Grave. 2021a. Distilling knowledge from reader to retriever for question answering. In International Conference on Learning Representations.  
Gautier Izacard and Edouard Grave. 2021b. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 874-880, Online. Association for Computational Linguistics.

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew G. Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 2704-2713. IEEE Computer Society.  
Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128.  
Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and Omer Levy. 2020. SpanBERT: Improving pre-training by representing and predicting spans. Transactions of the Association for Computational Linguistics, 8:64-77.  
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601-1611, Vancouver, Canada. Association for Computational Linguistics.  
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769-6781, Online. Association for Computational Linguistics.  
Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 39-48. ACM.  
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452-466.  
Jinhyuk Lee, Minjoon Seo, Hannaneh Hajishirzi, and Jaewoo Kang. 2020. Contextualized sparse representations for real-time open-domain question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 912-919, Online. Association for Computational Linguistics.

Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2021. Learning dense representations of phrases at scale. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6634-6647, Online. Association for Computational Linguistics.  
Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6086-6096, Florence, Italy. Association for Computational Linguistics.  
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-tus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.  
Jimmy Lin, Rodrigo Nogueira, and Andrew Yates. 2020. Pretrained transformers for text ranking: BERT and beyond. arXiv preprint arXiv:2010.06467.  
Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329-345.  
Sewon Min, Danqi Chen, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2020. Knowledge guided text retrieval and reading for open domain question answering. ArXiv preprint, abs/1911.03868.  
Yixin Nie, Songhe Wang, and Mohit Bansal. 2019. Revealing the importance of semantic retrieval for machine reading at scale. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2553-2566, Hong Kong, China. Association for Computational Linguistics.  
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Roktaschel, and Sebastian Riedel. 2021. KILT: a benchmark for knowledge intensive language tasks. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2523-2544, Online. Association for Computational Linguistics.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou,

Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21:1-67.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392, Austin, Texas. Association for Computational Linguistics.  
Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333-389.  
Minjoon Seo, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Hajishirzi. 2018. Phrase-indexed question answering: A new challenge for scalable document comprehension. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 559-564, Brussels, Belgium. Association for Computational Linguistics.  
Minjoon Seo, Jinhyuk Lee, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Hajishirzi. 2019. Real-time open-domain question answering with dense-sparse phrase index. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4430-4441, Florence, Italy. Association for Computational Linguistics.  
Haitian Sun, Tania Bedrax-Weiss, and William Cohen. 2019. PullNet: Open domain question answering with iterative retrieval on knowledge bases and text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2380-2390, Hong Kong, China. Association for Computational Linguistics.  
Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and William Cohen. 2018. Open domain question answering using early fusion of knowledge bases and text. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4231-4242, Brussels, Belgium. Association for Computational Linguistics.  
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERIFICATION. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809-819, New Orleans, Louisiana. Association for Computational Linguistics.  
Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. 2020. Scalable zero-shot entity linking with dense entity retrieval. In

Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6397-6407, Online. Association for Computational Linguistics.  
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations.  
Sohee Yang and Minjoon Seo. 2020. Is retriever merely an approximator of reader? ArXiv preprint, abs/2010.10999.  
Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen Tan, Kun Xiong, Ming Li, and Jimmy Lin. 2019. End-to-end open-domain question answering with BERTserini. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 72-77, Minneapolis, Minnesota. Association for Computational Linguistics.

# Footnotes:

Page 0: *This work was done when JL worked as a visiting research scholar at Princeton University. $^{1}$ Following previous work (Seo et al., 2018, 2019), the term phrase denotes any contiguous text segment up to  $L$  words, which is not necessarily a linguistic phrase (see Section 2). 2Our code and models are available at https:// github.com/princeton-nlp/DensePhrases. 
Page 2: <sup>4</sup>We expect DensePhrases to achieve even higher performance if it is re-trained with 100-word text blocks. We leave it for future investigation. $^{5}$ https://github.com/facebookresearch/DPR. $^{6}\mathrm{DPR}^{\spadesuit}$  is trained on NaturalQuestions, TriviaQA, CuratedTREC (Baudis and Sedivy, 2015), and WebQuestions (Berant et al., 2013). DensePhrases additionally includes SQuAD (Rajpurkar et al., 2016), although it does not contribute to Natural Questions and TriviaQA much. 3In most cases, retrieving  $2k$  phrases is sufficient for obtaining  $k$  unique passages. If not, we try  $4k$  and so on. 
Page 3: <sup>7</sup>We also accumulate gradients for 16 steps to match the effective batch size of the original work. 
Page 4: Technically, DensePhrases treats start and end representations of phrases independently and use start (or end) representations other than the positive one as negatives. Note that phrases may overlap, whereas passages are usually disjoint segments with each other. 
Page 5: 10While  $\mathcal{L}_{\mathrm{hard}}$  with this type of hard negatives might favor DensePhrases, using BM25 hard negatives for  $\mathcal{L}_{\mathrm{hard}}$  would favor DPR since DPR was directly trained on BM25 hard negatives. Nonetheless, we observed similar trends in  $\mathcal{L}_{\mathrm{hard}}$  regardless of the choice of hard negatives. 
Page 6: <sup>11</sup>We follow the same steps described in Petroni et al. (2021) for training the multi-task version of DensePhrases. 
