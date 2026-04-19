Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines
=================================================================================

Peitian Zhang1, 
Zheng Liu2, 
Yujia Zhou1, 
Zhicheng Dou1, 
Zhao Cao2  
1: Renmin University, Beijing, China  
2: Huawei Tech Co. Ltd., Shenzhen, China  
{namespace.pt,zhengliu1026}@gmail.comCorresponding author

###### Abstract

Auto-regressive search engines emerge as a promising paradigm for next-gen information retrieval systems. These methods work with Seq2Seq models, where each query can be directly mapped to the identifier of its relevant document. As such, they are praised for merits like being end-to-end differentiable. However, auto-regressive search engines also confront challenges in retrieval quality, given the requirement for the exact generation of the document identifier. That’s to say, the targeted document will be missed from the retrieval result if a false prediction about its identifier is made in any step of the generation process.
In this work, we propose a novel framework, namely AutoTSG (Auto-regressive Search Engine with Term-Set Generation), which is featured by 1) the unordered term-based document identifier and 2) the set-oriented generation pipeline. With AutoTSG, any permutation of the term-set identifier will lead to the retrieval of the corresponding document, thus largely relaxing the requirement of exact generation.
Besides, the Seq2Seq model is enabled to flexibly explore the optimal permutation of the document identifier for the presented query, which may further contribute to the retrieval quality.
AutoTSG is empirically evaluated with Natural Questions and MS MARCO, where notable improvements can be achieved against the existing auto-regressive search engines.

1 Introduction
--------------

Search engines, standing as the most representative form of information retrieval, are fundamentally important to real-world applications like web search, question answering, advertising, and recommendation*(Karpukhin et al., [2020](#bib.bib9 ""); Lewis et al., [2021](#bib.bib12 ""))*. Nowadays, they are also regarded as a critical tool for the augmentation of large language models (LLMs), where external information can be introduced to facilitate faithful and knowledge-grounded generation*(Komeili et al., [2021](#bib.bib10 ""); Nakano et al., [2022](#bib.bib19 ""); Wang et al., [2023](#bib.bib24 ""))*. A typical search engine calls for the utilization of two basic modules: representation and indexing. For example, a sparse retrieval system uses lexicon-based representations and an inverted index, while a dense retrieval system is based on latent embeddings and an ANN index*(Robertson and Zaragoza, [2009](#bib.bib22 ""); Malkov and Yashunin, [2018](#bib.bib14 ""))*.

Recently, a new type of method, the auto-regressive search engines, e.g., GENRE *(Cao et al., [2021](#bib.bib2 ""))*, DSI *(Tay et al., [2022](#bib.bib23 ""))*, emerge as a promising direction for next-gen information retrieval*(Metzler et al., [2021b](#bib.bib18 ""))*.
Briefly speaking, the auto-regressive search engine allocates each document with a sequential ID, called document identifier111In this work, we focus on document identifiers based on explicit features, e.g., text. Unlike those based on implicit features, such identifiers are directly compatible with the pre-trained language models and do not depend on an addition embedding model and hierarchical clustering.; e.g., n-grams within the document*(Bevilacqua et al., [2022](#bib.bib1 ""))*, or semantic IDs acquired by hierarchical clustering*(Tay et al., [2022](#bib.bib23 ""))*.
Next, it learns to predict the document identifier for an input query with a Seq2Seq model.
Compared with traditional retrieval methods, the autoregressive search engine is praised for being end-to-end differentiable: instead of optimizing each module individually, the entire retrieval pipeline can be optimized by the Seq2Seq learning and does not need a separate index*(Metzler et al., [2021b](#bib.bib18 ""); Tay et al., [2022](#bib.bib23 ""))*.

Despite the preliminary progresses achieved by recent works*(Tay et al., [2022](#bib.bib23 ""); Cao et al., [2021](#bib.bib2 ""); Bevilacqua et al., [2022](#bib.bib1 ""); Wang et al., [2022](#bib.bib25 ""))*, we argue that the auto-regressive search is much more challenging than typical Seq2Seq problems. Particularly, auto-regressive search engines require the exact generation of identifier for the targeted document. If incorrect predictions are made in any steps of the generation process, it will falsely produce the identifier of a different document, which causes the missing of targeted document in the final retrieval result (a.k.a. false pruning).
Furthermore, considering that the sequence length of the identifier must be large enough to guarantee the discrimination of all documents, the generation process has to go through a large number of decoding steps.
If we regard the generation process as sequential decision making, the probability of false pruning will gradually accumulate step-by-step and finally result in a bad retrieval quality.
A derived problem from false-pruning is that the permutation of document identifier becomes critical.
While retrieving, the targeted document will be falsely pruned if the prefix of its predefined identifier is bad, i.e. relatively hard to generate conditioned on the query. However, it can be successfully retrieved as long as its prefix is sufficiently good. We introduce the following concrete example to better illustrate the above points.

<img src='img/autotsg.png' alt='Refer to caption' title='' width='574' height='327' />

*Figure 1:  (A) The targeted document (D3) is falsely pruned given the predefined sequential identifier. (B) The targeted document (D3) is retrieved via the highlighted permutation on top of AutoTSG.*

###### Example 1

We use a sample query from Natural Questions, "Who cooks for the president of the United States", for discussion. We have three candidate documents from Wikipedia: D1, D2, and D3. D3 is the target as it contains the correct answer. Each document is identified by keywords from its title and first paragraph. All document identifiers are organized by a prefix tree (trie) as Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines") (A).

We apply the Seq2Seq model from GENRE*(Cao et al., [2021](#bib.bib2 ""))* for our example. As we may observe, D3 is falsely pruned in the first step, as the generation likelihood $P(\textit{cristeta}|\textbf{Q})$ is lower than the other candidates, “white” and “executive”.
We may also derive two interesting observations from this example. Firstly, if the identifier of D3 can be re-ordered as “executive, chef, cristeta, comerford”, it will achieve a much higher generation likelihood -12.8, making D3 successfully retrieved (greater than -16.5 from D1, and -31.0 from D2). This reflects the importance of identifier’s permutation. Secondly, although the document identifier is problematic regarding the presented query, it can be favorable to other queries, like “who is cristeta comerford?”. In other words, there is probably no universally favorable permutation of identifier for the document.

Our Method. We propose a novel framework, namely AutoTSG, to overcome the above challenges in auto-regressive search engines. The proposed framework is highlighted by two featured designs. First of all, the document identifier is no longer one (or a few) predefined sequence, but a set of unordered terms from the document, known as the unordered term-based identifier. Any permutation of the term-set will be a valid identification for the corresponding document; that is, the targeted document can be retrieved if any permutation of its identifier is generated by the Seq2Seq model.
Thus, it will be more tolerable and largely relaxed for the requirement of exact generation. Secondly, given the change of document identifier, the Seq2Seq model is switched to perform the set-oriented generation: it aims to generate the included terms of the document identifier, rather than exactly predict any required sequences. With such flexibility, the Seq2Seq model may explore the “favorable permutation” of the document identifier given different queries. This model is easier to train and therefore contributes to a better retrieval quality for the generation process.

Back to our example (Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines") B.), the terms white, house, executive, chef, etc., are selected as the document identifier of D3.
Therefore, all permutations, like “white, house, …, executive, chef”, “white, house, …, cristeta, comerford”, etc., will be valid identification of D3.
Given the query “Q: who cooks for the president of the United States”, the Seq2Seq explores the entire term space ($\bigcup_{\text{Terms}}$), where it figures out “executive” to be the most probable (with the highest generation likelihood) and valid (belongs to a valid document) term to decode. In the second step, it further explores the term space. This time, it selects “chef” given its high likelihood and validity. Note that although combinations like “executive, director”, “executive, manager” may also give large enough likelihood, they will be reckoned invalid since they do not belong to any existing document identifiers. The Seq2Seq model will keep on exploring; with the permutation “executive, chef, white, house, …” generated (other terms are omitted due to limited space), document D3 is successfully retrieved for the query.

While the framework is upgraded in terms of document identifier and generation pipeline, it still needs to conquer several challenges in order to achieve competitive retrieval performance, including how to select appropriate terms for a document identifier, how to explore the optimal permutation of document identifier while ensuring its validity, how to learn the Seq2Seq model effectively to perform the exploration task.
In our work, we develop the following techniques to address these challenges.
(1) The matching-oriented term selection for constructing document identifiers, which determines a concise and discriminative set of terms for each document based on the importance to query-document matching. (2) The constrained greedy search, which explores the optimal identifier permutation while ensuring its validity. (3) Likelihood-adapted Seq2Seq learning: as there is no predefined permutation of document identifier, the Seq2Seq learning is performed with iteratively updated objectives determined by concrete query and model snapshot.

In summary, the main technical contributions of this paper are highlighted by the following points.

* •

    We propose a novel framework AutoTSG for auto-regressive search engines. The proposed method is featured by its unordered term-based document identifier and the set-oriented generation pipeline. With both designs, the requirement for exact generation of the identifier is relaxed, and the Seq2Seq model is enabled to explore its favorable identifier permutation.

* •

    We devise three technical components which jointly contribute to AutoTSG’s retrieval performance: 1) the matching-oriented term selection, 2) the constrained greedy search for document identifier’s generation, and 3) the likelihood-adapted Seq2Seq learning.

* •

    We conduct comprehensive empirical analyses on top of popular evaluation benchmarks: Natural Questions and MSMARCO. Experimental results verify the effectiveness of AutoTSG, as notable improvements in retrieval quality can be achieved over the existing auto-regressive search engines under a variety of experimental settings.

2 Related Work
--------------

Document retrieval has been extensively studied for a long time. Conventional methods resort to lexical representations and inverted indexes, where query-document relationships can be estimated by relevance functions, like BM25*(Robertson and Zaragoza, [2009](#bib.bib22 ""))*. With the development of pre-trained language models*(Devlin et al., [2019](#bib.bib6 ""))*, dense retrieval becomes another popular option*(Karpukhin et al., [2020](#bib.bib9 ""); Xiong et al., [2021](#bib.bib26 ""); Izacard et al., [2021](#bib.bib8 ""))*, where the relevance is measured by embedding similarity. Apart from these well-established methods, the auto-regressive search engines emerge as a promising direction*(Metzler et al., [2021a](#bib.bib17 ""); Tay et al., [2022](#bib.bib23 ""); Cao et al., [2021](#bib.bib2 ""))*. These methods treat document retrieval as a Seq2Seq problem, where the document identifier can be directly generated for the query.
The document identifier is one of the most decisive factors for the corresponding methods*(Tay et al., [2022](#bib.bib23 ""); Bevilacqua et al., [2022](#bib.bib1 ""))*: the Seq2Seq model must generate the exact same identifier for the targeted document, and the ranking of the document is determined by the generation likelihood of its identifier. Based on different formations, the current works can be roughly partitioned into three groups: 1) the semantic ID based methods*(Tay et al., [2022](#bib.bib23 ""); Mehta et al., [2022](#bib.bib16 ""))*, 2) the atomic ID based methods*(Tay et al., [2022](#bib.bib23 ""); Zhou et al., [2022](#bib.bib27 ""))*, 3) the explicit term based methods*(Cao et al., [2021](#bib.bib2 ""); De Cao et al., [2022](#bib.bib5 ""); Bevilacqua et al., [2022](#bib.bib1 ""))*. By comparison, the last category is more compatible with pre-trained language models, as the explicit terms are directly perceptible. Thus, our proposed framework also adopts such features. As discussed, the existing works call for the exact generation of the document identifier, which is a too challenging requirement. It is a major cause for the false pruning of the relevant document, which severely restricts the retrieval quality. In light of such a deficiency, our work reformulates the document identifier based on unordered terms; together with the set-oriented generation pipeline, it achieves substantial improvements in retrieval quality.

3 Methodology
-------------

An auto-regressive search engine usually constitutes two basic components*(Tay et al., [2022](#bib.bib23 ""); Bevilacqua et al., [2022](#bib.bib1 ""))*. One is a document identifier schema - a unique identifier set $\mathcal{I}(D)$ (e.g., a family of sequences) needs to be assigned to each document $D$. The other one is a Seq2Seq model $\boldsymbol{\Theta}(\cdot)$. For an input query $Q$, the Seq2Seq model estimates the relevance between $Q$ and $D$ based on the following generation likelihood:

|  | $\mathrm{Rel}(Q,D)\=\mathrm{Agg}\left(\left{\prod\nolimits_{i\=1}^{|I|}\Pr(I_{i}\mid I_{<i},Q;\boldsymbol{\Theta}):~{}I\in\mathcal{I}(D)\right}\right),$ |  | (1) |
| --- | --- | --- | --- |

where $I$ is an element of $\mathcal{I}(D)$; $\Pr(I_{i}\mid I_{<i},Q;\boldsymbol{\Theta})$ indicates the generation probability of $i$-th element $I_{i}$ given the prefix $I_{<i}$, the query $Q$, and the Seq2Seq model $\boldsymbol{\Theta}$. The function $\mathrm{Agg}(\cdot)$ stands for aggregation of the likelihood for sequences within $\mathcal{I}(D)$.
Many of the existing works*(Tay et al., [2022](#bib.bib23 ""); Cao et al., [2021](#bib.bib2 ""); Wang et al., [2022](#bib.bib25 ""))* make use of one single sequence for document identification. In those cases, $\mathrm{Agg}(\cdot)$ will simply be the identity function $\mathbbm{1}(\cdot)$. In SEAL*(Bevilacqua et al., [2022](#bib.bib1 ""))*, the whole collection of n-grams from the document are used as the identifier, where an intersective scoring function is introduced to aggregate the generation likelihood of different n-grams. With the above formulation, the document retrieval can be made through a sequence generation workflow: the Seq2Seq model generates the most likely identifiers for the given query via a beam search, then the corresponding documents, ranked by their generation likelihoods, are returned as the retrieval result.

Although AutoTSG also relies on a Seq2Seq model for document retrieval as existing methods, it is fundamentally different in terms of document identification.
Particularly, it uses a set of $N$ unordered terms to form the document identifier: $\mathcal{T}(D)\={t_{1},\dots,t_{N}}$.
With the assumption that $\mathcal{T}(D)$ is unique within the corpus, any permutation of $\mathcal{T}(D)$ is unique as well. Then, we define that $D$ is retrieved if one permutation of $\mathcal{T}(D)$ is generated by the Seq2Seq model; and if multiple permutations are generated for a single document, we take their maximum likelihood: $\mathrm{Agg}(\cdot)\leftarrow\mathrm{max}(\cdot)$.

In the remaining part of this section, we will introduce corresponding components of AutoTSG: (1) the document identifier schema: how to decide the terms in the document identifier in the pre-processing stage (Section[3.1](#S3.SS1 "3.1 Matching-oriented Term Selection For Document Identifier ‣ 3 Methodology ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines")) and how to generate it in the prediction stage (Section[3.2](#S3.SS2 "3.2 Constrained Greedy Search ‣ 3 Methodology ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines")). (2) the Seq2Seq generation model: how to train the document identifier generation model (Section[3.3](#S3.SS3 "3.3 Likelihood-Adapted Sequence-to-Sequence Learning ‣ 3 Methodology ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines")).

### 3.1 Matching-oriented Term Selection For Document Identifier

The selection of terms in a document identifier is performed based on the following principles. Firstly, the number of terms $N$ should be sufficiently large that all documents within the corpus can be uniquely identified, i.e., no collision of identifiers between two different documents. Secondly, the term selection needs to be concise as well. As mentioned, longer sequences are more prone to false prediction. Thirdly, the selected terms must sufficiently capture the semantic information within the document; by doing so, the query-document relevance can be precisely reflected by the generation likelihood. With the above principles, we introduce the following mechanism for term selection, where representative terms are selected
depending on their importance to the query-document matching.

Each document $D$ is partitioned into a list of terms in the first place: $[t^{D}_{1},\dots,t^{D}_{L}]$. Then, the term importance is acquired through the estimation pipeline in Eq. ([2](#S3.E2 "In 3.1 Matching-oriented Term Selection For Document Identifier ‣ 3 Methodology ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines")).

|  | $\mathcal{M}([t^{D}_{1},\dots,t^{D}_{L}])\xRightarrow{1.}[\boldsymbol{e}^{D}_{1},\dots,\boldsymbol{e}^{D}_{L}]\xRightarrow{2.}[\sigma(W^{T}\boldsymbol{e}^{D}_{1}),\dots,\sigma(W^{T}\boldsymbol{e}^{D}_{L})]\xRightarrow{3.}[w^{D}_{1},\dots,w^{D}_{L}]$ |  | (2) |
| --- | --- | --- | --- |

The encoding model $\mathcal{M}(\cdot)$ is applied to transform each term $t_{i}$ into its latent representation $\boldsymbol{e}^{D}_{i}\in\mathbb{R}^{d\times 1}$. Following the common practice, we leverage BERT*(Devlin et al., [2019](#bib.bib6 ""))* for this operation. The latent representation is further mapped into real-valued importance $w^{D}_{i}$ via linear transformation $W\in\mathbb{R}^{d\times 1}$ and ReLU activation $\sigma(\cdot)$. Following the existing practice on semantic matching*(Mallia et al., [2021](#bib.bib15 ""); Gao et al., [2021](#bib.bib7 ""); Lin and Ma, [2021](#bib.bib13 ""))*, the selection modules, i.e., $\mathcal{M}$ and $W$, are learned to optimize the semantic matching between query and document. Particularly, given the annotations $\mathcal{A}\={\langle Q,D^{+},{D_{i}^{-}}_{i\=1}^{M}\rangle}$ where $D^{+}$ is the relevant document to $Q$, and ${D_{i}^{-}}_{i\=1}^{M}$ are $M$ irrelevant documents to $Q$, the following InfoNCE loss is optimized for estimating term importance:

|  | $\min\left(-\log\frac{\exp(\sum_{t^{Q}_{i}\=t^{D^{+}}_{j}}w^{Q}_{i}w^{D^{+}}_{j}/\tau)}{\exp(\sum_{t^{Q}_{i}\=t^{D^{+}}_{j}}w^{Q}_{i}w^{D^{+}}_{j}/\tau)+\sum_{m\=1}^{M}\exp(\sum_{t^{Q}_{i}\=t^{D_{m}^{+}}_{j}}w^{Q}_{i}w^{D_{m}^{+}}_{j}/\tau)}\right).$ |  | (3) |
| --- | --- | --- | --- |

In this place, $\tau$ is the temperature; “$t^{Q}_{i}\=t^{D}_{j}$” indicates the constraint which regularizes $t^{Q}_{i}$ and $t^{D}_{j}$ to be the same term.
By minimizing the above loss, large importance scores can be learned for the terms which bridge the matching between query and its relevant document. We select the top-$N$ terms as the identifier: $\mathcal{T}(D)\leftarrow{t_{i}^{D}:w^{D}_{i}\in\text{top-}N\left({w^{D}_{i}}_{i\=1}^{L}\right)}$. The same number of selection is applied to all documents. We choose the smallest value of $N$ while ensuring the discrimination; e.g., for a moderate-scale corpus like NQ320K, $N\=12$ is already enough to have all documents discriminated.

### 3.2 Constrained Greedy Search

Given the unordered term-based identifier, the relevance between query and document can be measured in the following naive way: firstly, the generation likelihood is enumerated for all possible permutations of the document identifier ($N!$); then, the highest value is used as the measurement of relevance. Since the naive method is intractable, we need to design a mechanism where the language model may generate plausible document identifiers and their near optimal permutations for the given query, to ensure acceptable generation efficiency. The search mechanism needs to satisfy the following two properties: optimality and validity. First of all, it is expected to produce the document identifier of the highest generation likelihood. Knowing that the optimal solution is intractable, we resort to the greedy algorithm for its approximation. Particularly, we set the following local optimality while making stepwise term selection.
At the $i$-th decoding step, given the collection of previously generated terms ${I_{<i}^{*}}_{K}$ ($K$: the beam size), the decoding result of the current step (${I_{\leq i}^{*}}_{K}$) is made w.r.t. the following condition:

|  | ${I_{\leq i}^{*}}_{K}\leftarrow\underset{I_{\leq i}}{\text{argtop-}K}\left(\left{\prod\nolimits_{j\=1,\dots,i}\Pr(I_{j}\mid I_{<j};Q;\boldsymbol{\Theta})\right}\right).$ |  | (4) |
| --- | --- | --- | --- |

In other words, we greedily select the terms which give rise to the top-$K$ generation likelihood until the current step. Apart from the optimality, the generated term set must also correspond to valid document identifiers. To guarantee the validity, for each prefix $I_{<i}\in{I_{<i}^{*}}_{K}$, we regularize the selection of $I_{i}$ with the following set-difference based constraint:

|  | $1.~{}I_{i}\notin{I_{1},\dots,I_{i-1}}\wedge 2.~{}\exists D:I_{i}\in\mathcal{T}(D)/{I_{1},\dots,I_{i-1}}.$ |  | (5) |
| --- | --- | --- | --- |

The first condition prevents the selection of a repetitive term given the current prefix $I_{<i}$; while the second condition ensures that the newly selected term and its prefix, i.e., ${I_{1},\dots,I_{i-1}}\cup I_{i}$, will always constitute a subset of a valid document identifier.

Since it’s time consuming to verify the constraint case-by-case, we implement the following data structure for efficient generation. We maintain an inverted index during generation, pointing from each prefix $I_{<i}$ to the documents whose identifiers constitute the super sets of ${I_{1},\dots,I_{i-1}}$. The union is computed for all such identifiers: $\boldsymbol{X}\=\bigcup{\mathcal{T}(D^{\prime}):~{}{I_{1},\dots,I_{i-1}}\subseteq\mathcal{T}(D^{\prime})}$, and let the difference set $\boldsymbol{X}/{I_{1},\dots,I_{i-1}}$ be the feasible scope for next-step decoding. Note that at the begining of decoding, all terms in all document identifiers are valid. With the selection of $I_{i}$, the inverted index is updated accordingly, with the invalid documents pruned from the entry of $I_{<\=i}$. As most of the documents will be pruned for one specific prefix within very few steps, the above data structure helps to achieve a high running efficiency for the constrained greedy search.

### 3.3 Likelihood-Adapted Sequence-to-Sequence Learning

Unlike the existing works where ground-truth sequences are predefined, the document identifier becomes a term-set in AutoTSG. Since one document is retrieved if any permutation of its identifier is generated, it is straightforward to make random sampling from the $N!$ permutations, so that the Seq2Seq learning can be conducted. However, the sampled sequence will probably be inconsistent with the decoding order of constrained greedy search (unfavorable to recall), nor will it likely be the one with the highest generation likelihood of document identifier (unfavorable to the final ranking).

To facilitate the recall of relevant documents from the generation process and have them better ranked in the final retrieval result, we expect the Seq2Seq model to learn from the permutations of document identifiers. Therefore, we propose a new training workflow named likelihood-adapted Seq2Seq learning.
The proposed method adopts an iterative pipeline. In each iteration, it samples the favorable permutation of document identifier as the learning objective. Specifically, given the current Seq2Seq model $\boldsymbol{\Theta}^{t-1}$, the query, and the previously generated terms $I_{<i}$, the top-K sampling is performed to the difference set of $\mathcal{T}(D)$ and $I_{<i}$ according to the following distribution:

|  | $P(I_{i})\propto\Pr(I_{i}\mid I_{<i};Q;\boldsymbol{\Theta}^{t-1}),~{}I_{i}\in\mathcal{T}(D)/{I_{0},\dots,I_{i-1}}.$ |  | (6) |
| --- | --- | --- | --- |

With the sampling of multiple candidate sequences $\boldsymbol{I}$, the one with the highest overall likelihood is selected as the learning objective for the current iteration ($I^{t}$):

|  | $I^{t}\leftarrow\mathrm{argmax}\left(\left{\prod\nolimits_{i\=1,\dots,N}\Pr(I_{i}\mid I_{<i};Q;\boldsymbol{\Theta}^{t-1}):~{}I\in\boldsymbol{I}\right}\right).$ |  | (7) |
| --- | --- | --- | --- |

With this new objective, the Seq2Seq model is updated as $\boldsymbol{\Theta}^{t}$ via another round of learning. The above process, i.e., the likelihood-dependent permutation sampling and the Seq2Seq learning, is iteratively conducted until a desirable model is produced.

There are still two remaining issues. One is the initial order of permutation. Although there are different options, e.g., purely randomized permutation, or sampling from a pre-trained LM (T5, GPT), we find that ordering the terms by their estimated importance in term selection brings forth the best empirical performance. The other one is about the convergence. Although the sampled permutation is always changing, we keep track of the Seq2Seq model’s retrieval accuracy on a hold-out validation set. In our experiment, it merely takes two iterations to reach the convergence of accuracy growth.

4 Experiments
-------------

The experimental studies are performed to explore the following research questions. RQ 1. AutoTSG’s impact on retrieval quality against the existing auto-regressive search engines. RQ 2. The impact from each of the technical designs in AutoTSG. RQ 3. The impact on running efficiency.

### 4.1 Settings

Datasets. We leverage two popular datasets which are widely used by previous evaluations for auto-regressive search engines. One is the NQ320k dataset*(Tay et al., [2022](#bib.bib23 ""); Bevilacqua et al., [2022](#bib.bib1 ""))* curated from Natural Questions*(Kwiatkowski et al., [2019](#bib.bib11 ""))*, including 320k training queries and 7830 testing queries. Each query is corresponding to a Wikipedia article containing its answer. The other one is the MS300k dataset*(Chen et al., [2023](#bib.bib3 ""); Zhou et al., [2022](#bib.bib27 ""))* curated from MSMARCO*(Nguyen et al., [2016](#bib.bib20 ""))*, which contains 320k documents, 360k training queries, and 772 testing queries.

Metrics. Two evaluation metrics are introduced to measure the retrieval quality at the top-$K$ cut-off: MRR@K and Recall@K, which focus on the perspective of ranking and recall, respectively.

Implementations. Some critical facts about the implementations are presented as follows. Backbone LM. We leverage T5*(Raffel et al., [2020](#bib.bib21 ""))* as our backbone, which is consistent with the majority of previous works*(Chen et al., [2023](#bib.bib3 ""); Mehta et al., [2022](#bib.bib16 ""); Wang et al., [2022](#bib.bib25 ""))*. T5 (base) is the default option, yet T5 (large) is also explored. Term Granularity. We treat each single word, separated by space, as one term. Since a term may contain multiple tokens, we append a “,” to the last token, which indicates the termination of term. (The same treatment can be applied to handle other granularities, e.g., n-grams.) Data Augmentation. Following the previous works*(Wang et al., [2022](#bib.bib25 ""); Mehta et al., [2022](#bib.bib16 ""); Zhou et al., [2022](#bib.bib27 ""))*, we leverage DocT5*(Cheriton, [2019](#bib.bib4 ""))* to generate pseudo training queries. Beam Size. The beam size is 100 throughout the experiments, which is also same as previous works. We’ve uploaded our implementations to an anonymous repo 222[https://github.com/namespace-Pt/Adon/tree/AutoTSG](https://github.com/namespace-Pt/Adon/tree/AutoTSG "") for the reference of more details.

*Table 1: Overall evaluations on NQ320k. $\dagger$ denotes the results copied from*(Wang et al., [2022](#bib.bib25 ""))*.*

| Method | MRR@10 | MRR@100 | Recall@1 | Recall@10 | Recall@100 |
| --- | --- | --- | --- | --- | --- |
| BM25$\dagger$ | – | 0.211 | 0.151 | 0.325 | 0.505 |
| DPR$\dagger$ | – | 0.366 | 0.287 | 0.534 | 0.732 |
| GENRE | 0.653 | 0.656 | 0.591 | 0.756 | 0.814 |
| DSI | 0.594 | 0.598 | 0.533 | 0.715 | 0.816 |
| SEAL$\dagger$ | – | 0.655 | 0.570 | 0.800 | 0.914 |
| Ultron | 0.726 | 0.729 | 0.654 | 0.854 | 0.911 |
| NCI$\dagger$ | – | 0.731 | 0.659 | 0.852 | 0.924 |
| AutoTSG | 0.757 | 0.760 | 0.690 | 0.875 | 0.932 |

*Table 2: Overall evaluations on MS300k.*

| Method | MRR@10 | MRR@100 | Recall@1 | Recall@10 | Recall@100 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.313 | 0.325 | 0.196 | 0.591 | 0.861 |
| DPR | 0.424 | 0.433 | 0.271 | 0.764 | 0.948 |
| GENRE | 0.361 | 0.368 | 0.266 | 0.579 | 0.751 |
| DSI | 0.339 | 0.346 | 0.257 | 0.538 | 0.692 |
| SEAL | 0.393 | 0.402 | 0.259 | 0.686 | 0.879 |
| Ultron | 0.432 | 0.437 | 0.304 | 0.676 | 0.794 |
| NCI | 0.408 | 0.417 | 0.301 | 0.643 | 0.851 |
| AutoTSG | 0.484 | 0.491 | 0.359 | 0.766 | 0.907 |

Baselines. To analyze the effectiveness of AutoTSG, especially the proposed formulation of document identifier and its generation workflow, we introduce a diverse collection of auto-regressive search engines with different forms of document identifier. GENRE*(Cao et al., [2021](#bib.bib2 ""))*: using titles; DSI*(Tay et al., [2022](#bib.bib23 ""))*: using semantic IDs; SEAL*(Bevilacqua et al., [2022](#bib.bib1 ""))*: using n-grams and FM index; Ultron*(Zhou et al., [2022](#bib.bib27 ""))*: using titles and urls; NCI*(Wang et al., [2022](#bib.bib25 ""))*: enhancing DSI with data augmentation. Given the limitation of space, we omit many of the repetitive comparisons with other conventional retrieval methods, as they have been extensively analyzed in the above works.

### 4.2 Main Analysis

The overall evaluations on NQ320k and MS300k are shown in Table[1](#S4.T1 "Table 1 ‣ 4.1 Settings ‣ 4 Experiments ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines") and[2](#S4.T2 "Table 2 ‣ 4.1 Settings ‣ 4 Experiments ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines"), respectively. According to the experiment results, AutoTSG notably improves the retrieval quality over the existing auto-regressive search engines. For example, on NQ320k, it outperforms the strongest baseline by $+3.9\%$ and $+2.4\%$ on MRR@100 and Recall@10; on MS300k, it also achieves the relative improvements of $+12.3\%$ on MRR@100 and $+11.6\%$ on Recall@10 over the baseline methods. In our detailed analysis, we’ll demonstrate that the new formulation of document identifier and the corresponding generation workflow are the main contributors to such advantages. Despite the overall advantages, we may observe that the conventional approach DPR*(Karpukhin et al., [2020](#bib.bib9 ""))* leads to the highest recall@100 on MS300k. In fact, this observation reveals a general challenge for the current auto-regressive search engines: it is easier to achieve high ranking performances (reflected by MRR) thanks to the expressiveness of Seq2Seq models, but comparatively difficult to achieve equally competitive recall. Much of the reason is due to the aforementioned false-pruning problem: once the document identifier is false predicted in any step of the generation process, it is impossible for back-tracking (thus unfavorable for recall); however, if the document can be returned by generation, it will probably be ranked with a favorable position. Fortunately with AutoTSG, we make a critical step-forward to mitigate the above problem: it relaxes the requirement of exact generation, and enables the Seq2Seq model to explore the optimal permutation of identifier w.r.t. the given query. Both designs substantially improve the recall, and further expand the advantage on ranking.

Besides the above overall evaluations, we present more detailed analysis between the auto-regressive search engines in terms of their memorization and generation capability. Particularly, the existing auto-regressive search engines highly rely on the presence of training queries*(Wang et al., [2022](#bib.bib25 ""); Zhou et al., [2022](#bib.bib27 ""); Tay et al., [2022](#bib.bib23 ""); Mehta et al., [2022](#bib.bib16 ""))*: it is desirable to provide each document identifier with sufficient training queries. By learning to generate a document’s identifier with training queries, it will be much easier to make exact generation of the document’s identifier for its testing queries, given that the queries associated with the same document are somewhat similar. In other words, the existing auto-regressive models are more of memorization rather than generalization, which is unfavorable to handling a massive or constantly changing corpus. To evaluate AutoTSG’s impact for the corresponding capabilities, we design the experiment where the corpus is partitioned into two halves: with training queries preserved for 50% of the documents (Seen), and with training queries removed for the other 50% of the documents (Unseen).
Given the above setting, the Seq2Seq model is prevented from memorizing the document identifiers on the unseen half during the training stage. According to the experiment results in Table[3](#S4.T3 "Table 3 ‣ 4.2 Main Analysis ‣ 4 Experiments ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines") and[4](#S4.T4 "Table 4 ‣ 4.2 Main Analysis ‣ 4 Experiments ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines"): AutoTSG marginally outperforms the baselines on the “seen” half; nevertheless, its advantage is significantly magnified on the “unseen” half. As discussed, AutoTSG is largely relaxed from the requirement of exact generation, making it is less restricted by memorization; and together with the flexibility to explore optimal identifier permutation, it becomes more generalizable when dealing with unseen documents.

*Table 3: Analysis of retrieval quality w.r.t. seen and unseen documents on NQ320k.*

|  | Seen (50%) | | Unseen (50%) | | Seen+Unseen (100%) | |
| --- | --- | --- | --- | --- | --- | --- |
| Method | MRR@10 | Recall@10 | MRR@10 | Recall@10 | MRR@10 | Recall@10 |
| GENRE | 0.763 | 0.869 | 0.138 | 0.187 | 0.448 | 0.558 |
| DSI | 0.713 | 0.802 | 0.011 | 0.040 | 0.360 | 0.428 |
| Ultron | 0.782 | 0.891 | 0.300 | 0.383 | 0.471 | 0.570 |
| NCI | 0.751 | 0.842 | 0.050 | 0.159 | 0.393 | 0.459 |
| AutoTSG | 0.809 | 0.900 | 0.466 | 0.654 | 0.552 | 0.700 |

*Table 4: Analysis of retrieval quality w.r.t. seen and unseen documents on MS300k.*

|  | Seen (50%) | | Unseen (50%) | | Seen+Unseen (100%) | |
| --- | --- | --- | --- | --- | --- | --- |
| Method | MRR@10 | Recall@10 | MRR@10 | Recall@10 | MRR@10 | Recall@10 |
| GENRE | 0.361 | 0.579 | 0.150 | 0.312 | 0.196 | 0.411 |
| DSI | 0.339 | 0.538 | 0.030 | 0.075 | 0.171 | 0.298 |
| Ultron | 0.432 | 0.676 | 0.197 | 0.246 | 0.313 | 0.492 |
| NCI | 0.408 | 0.643 | 0.034 | 0.082 | 0.260 | 0.412 |
| AutoTSG | 0.484 | 0.766 | 0.390 | 0.588 | 0.391 | 0.642 |

### 4.3 Ablation Studies

The ablation studies are performed for each influential factor based on NQ320k dataset as Table[5](#S4.T5 "Table 5 ‣ 4.3 Ablation Studies ‣ 4 Experiments ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines").

Identifiers. We compare the proposed formulation of document identifier, the one based on unordered terms (term-set), with the conventional sequence-based formulation; that is, the terms are ordered as a sequence by their estimated importance (empirically more competitive than other sequence orders). It can be observed that the retrieval quality can be notably improved for both recall and ranking metrics on top of the proposed formulation. As discussed, the generation task is largely relaxed with term-set: there are no longer requirements to follow the exact sequence order; in contrast, any permutations of the identifier may lead to the retrieval of the corresponding document, and the Seq2Seq model may flexibly explore the favorable permutation depending on the presented query.

Term Selection. We compare three alternative term selection methods. Random: purely randomized selection (from the document); Title: terms within the title; Matching Oriented: the default option used by AutoTSG. We may derive the following observations from the experiment. Firstly, there are huge differences between different selection methods, which verifies the importance of term selection. Secondly, although directly making use of title is a strong baseline (also a common practice in many works*(Cao et al., [2021](#bib.bib2 ""); De Cao et al., [2022](#bib.bib5 ""); Zhou et al., [2022](#bib.bib27 ""))*), the matching-oriented approach is more effective:
by estimating the term’s importance based on its utility to query-document matching, the selected terms will not only facilitate the identifier’s generation (considering the higher relevance to the potential queries), but also better reflect the relationship between query and document.

*Table 5: Ablation studies on NQ320k. The default settings of AutoTSG are marked with *.*

| Factor | Setting | MRR@10 | MRR@100 | Recall@1 | Recall@10 | Recall@100 |
| --- | --- | --- | --- | --- | --- | --- |
| Identify | Sequence | 0.733 | 0.736 | 0.668 | 0.848 | 0.904 |
| | Term Set∗ | 0.757 | 0.760 | 0.690 | 0.875 | 0.932 |
| Select | Random | 0.628 | 0.631 | 0.568 | 0.739 | 0.811 |
| | Title | 0.743 | 0.745 | 0.677 | 0.856 | 0.915 |
| Matching Oriented∗ | 0.757 | 0.760 | 0.690 | 0.875 | 0.932 |
| Learning | Non-adaptive | 0.743 | 0.745 | 0.671 | 0.865 | 0.927 |
| | Likelihood Adapted∗ | 0.757 | 0.760 | 0.690 | 0.875 | 0.932 |
| Initialize | Random | 0.723 | 0.727 | 0.652 | 0.854 | 0.925 |
| | Likelihood | 0.715 | 0.718 | 0.643 | 0.844 | 0.916 |
| Importance∗ | 0.757 | 0.760 | 0.690 | 0.875 | 0.932 |
| Q-Gen | Ultron w.o. QG | 0.670 | 0.672 | 0.605 | 0.779 | 0.845 |
| | NCI w.o. QG | – | 0.679 | 0.602 | 0.802 | 0.909 |
| AutoTSG w.o. QG | 0.707 | 0.710 | 0.635 | 0.836 | 0.916 |
| AutoTSG∗ | 0.757 | 0.760 | 0.690 | 0.875 | 0.932 |
| Scale | DSI large | 0.613 | 0.620 | 0.553 | 0.733 | 0.835 |
| | SEAL large | – | 0.677 | 0.599 | 0.812 | 0.909 |
| NCI large | – | 0.734 | 0.662 | 0.853 | 0.925 |
| AutoTSG large | 0.766 | 0.768 | 0.697 | 0.882 | 0.938 |

Learning. We compare our proposed likelihood-adapted sequence-to-sequence learning with its non-adaptive variation: the document identifier’s permutation is fixed as its initialization. Note that the constrained greedy search is still maintained in the testing stage for the non-adaptive baseline, despite that it relies on a fixed permutation in the training stage. It can be observed that our proposed learning method indeed contributes to the retrieval quality. Such an advantage is easy to comprehend, considering that the training objective (the permutation of document identifier) can be iteratively adapted to keep consistent with the plausible permutations in the testing stage.

Initialization. We make evaluations for the three alternative initialization approaches for the Seq2Seq learning. 1) Random: the selected terms are randomly permuted; 2) Likelihood: the selected terms are permuted based on the generation likelihood of a pre-trained T5; 3) Importance: the selected terms are permuted by their estimated importance (default option). We can observe that the initialization turns out to be another critical factor for the Seq2Seq learning: the importance-based method is notably stronger than the other two baselines. This is probably because the importance-based initialization presents “a more plausible permutation” of document identifier, that is, easier to be generated and better reflect the query-document relationship. Besides, considering the iterative workflow of the learning process, the initialization will not only determine the current training objective, but also largely influence the final permutation where the Seq2Seq model will converge.

Query Generation. Query generation is a widely used data augmentation strategy to enhance auto-regressive search engines*(Wang et al., [2022](#bib.bib25 ""); Zhou et al., [2022](#bib.bib27 ""); Mehta et al., [2022](#bib.bib16 ""))*. It is also found helpful in AutoTSG (by using the off-the-shelf DocT5*(Cheriton, [2019](#bib.bib4 ""))*), whose impact is explored in the ablation studies. As expected, the retrieval quality is substantially improved on top of query generation. Note that the relative improvement of AutotSG is mainly from the proposed formulation of document identifier and its generation workflow, rather than the extra data augmentation. When query generation is disabled, AutoTSG maintains its advantage over the baselines.

Model Scaling. The scaling-up of backbone Seq2Seq model is another common approach for the enhancement of auto-regressive search engines. In our experiment, empirical improvements may also be observed when we switch to a T5-large backbone. Meanwhile, it maintains the advantage when other baselines are scaled up as well.

*Table 6: Efficiency analysis on NQ320k.*

| Method | Memory | Query Latency (s) | |
| --- | --- | --- | --- |
|  | (MB) | bs \= 10 | bs \= 100 |
| GENRE | 27 | 0.05 | 0.57 |
| DSI | 12 | 0.03 | 0.21 |
| SEAL | 210 | 0.32 | 3.14 |
| Ultron | 27 | 0.05 | 0.57 |
| NCI | 12 | 0.03 | 0.21 |
| AutoTSG | 35 | 0.06 | 0.69 |

Efficiency. The running efficiency is evaluated in Table[6](#S4.T6 "Table 6 ‣ 4.3 Ablation Studies ‣ 4 Experiments ‣ Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines"). Particularly, we measure the memory consumption for hosting the entire corpus; we also measure the time cost (query latency) with different beam sizes. We may observe that most of the approaches incur very close memory and time costs given their similar workflow. However, one exception is SEAL*(Bevilacqua et al., [2022](#bib.bib1 ""))*, where much more memory and running time are resulted from the usage of FM index.

5 Conclusion
------------

In this work, we propose a novel framework for auto-regressive search engines. The new framework is featured by two designs: 1) the unordered term-based document identifier, 2) the set-oriented generation pipeline. With both features, the challenge of generating document identifier becomes significantly relaxed, where the Seq2Seq model may flexibly explore the favorable permutation of document identifier. To support high-quality document retrieval, we devise three key techniques for the proposed framework: the matching-oriented term selection, the constrained greedy search for the document identifier and its optimal permutation, the likelihood-adapted Seq2Seq learning. With comprehensive experiments, we empirically verify the following technical contributions: 1) the proposed framework achieves substantial improvements over the existing auto-regressive search engines, especially in terms of generalizability, where superior retrieval quality can be achieved; 2) all of the proposed technical designs bring forth notable positive impacts to the retrieval quality; and 3) the improvements are achieved with very little extra cost on running efficiency.

References
----------

* Bevilacqua et al. [2022]Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Scott Yih, Sebastian
Riedel, and Fabio Petroni.Autoregressive search engines: Generating substrings as document
identifiers.In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho,
editors, *Advances in Neural Information Processing Systems*, 2022.URL [https://openreview.net/forum?id\=Z4kZxAjg8Y](https://openreview.net/forum?id=Z4kZxAjg8Y "").
* Cao et al. [2021]Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni.Autoregressive entity retrieval.In *9th International Conference on Learning Representations,
ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021.URL [https://openreview.net/forum?id\=5k8F6UU39V](https://openreview.net/forum?id=5k8F6UU39V "").
* Chen et al. [2023]Xiaoyang Chen, Yanjiang Liu, Ben He, Le Sun, and Yingfei Sun.Understanding differential search index for text retrieval.*CoRR*, abs/2305.02073, 2023.doi: 10.48550/arXiv.2305.02073.URL [https://doi.org/10.48550/arXiv.2305.02073](https://doi.org/10.48550/arXiv.2305.02073 "").
* Cheriton [2019]David R. Cheriton.From doc2query to doctttttquery.2019.
* De Cao et al. [2022]Nicola De Cao, Ledell Wu, Kashyap Popat, Mikel Artetxe, Naman Goyal, Mikhail
Plekhanov, Luke Zettlemoyer, Nicola Cancedda, Sebastian Riedel, and Fabio
Petroni.Multilingual autoregressive entity linking.*Transactions of the Association for Computational Linguistics*,
10:274–290, 2022.doi: 10.1162/tacl_a_00460.URL [https://aclanthology.org/2022.tacl-1.16](https://aclanthology.org/2022.tacl-1.16 "").
* Devlin et al. [2019]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.BERT: pre-training of deep bidirectional transformers for language
understanding.In Jill Burstein, Christy Doran, and Thamar Solorio, editors,*Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies,
NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and
Short Papers)*, pages 4171–4186. Association for Computational Linguistics,
2019.doi: 10.18653/v1/n19-1423.URL [https://doi.org/10.18653/v1/n19-1423](https://doi.org/10.18653/v1/n19-1423 "").
* Gao et al. [2021]Luyu Gao, Zhuyun Dai, and Jamie Callan.COIL: revisit exact lexical match in information retrieval with
contextualized inverted list.In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek
Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy
Chakraborty, and Yichao Zhou, editors, *Proceedings of the 2021
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June
6-11, 2021*, pages 3030–3042. Association for Computational Linguistics,
2021.doi: 10.18653/v1/2021.naacl-main.241.URL [https://doi.org/10.18653/v1/2021.naacl-main.241](https://doi.org/10.18653/v1/2021.naacl-main.241 "").
* Izacard et al. [2021]Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr
Bojanowski, Armand Joulin, and Edouard Grave.Unsupervised dense information retrieval with contrastive learning,
2021.URL [https://arxiv.org/abs/2112.09118](https://arxiv.org/abs/2112.09118 "").
* Karpukhin et al. [2020]Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih.Dense passage retrieval for open-domain question answering.In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors,*Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pages
6769–6781. Association for Computational Linguistics, 2020.doi: 10.18653/v1/2020.emnlp-main.550.URL [https://doi.org/10.18653/v1/2020.emnlp-main.550](https://doi.org/10.18653/v1/2020.emnlp-main.550 "").
* Komeili et al. [2021]Mojtaba Komeili, Kurt Shuster, and Jason Weston.Internet-augmented dialogue generation.In *Annual Meeting of the Association for Computational
Linguistics*, 2021.
* Kwiatkowski et al. [2019]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins,
Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob
Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey,
Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.Natural questions: a benchmark for question answering research.*Trans. Assoc. Comput. Linguistics*, 7:452–466, 2019.doi: 10.1162/tacl\_a\_00276.URL [https://doi.org/10.1162/tacl_a_00276](https://doi.org/10.1162/tacl_a_00276 "").
* Lewis et al. [2021]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela.Retrieval-augmented generation for knowledge-intensive nlp tasks,
2021.
* Lin and Ma [2021]Jimmy Lin and Xueguang Ma.A few brief notes on deepimpact, coil, and a conceptual framework for
information retrieval techniques.*CoRR*, abs/2106.14807, 2021.URL [https://arxiv.org/abs/2106.14807](https://arxiv.org/abs/2106.14807 "").
* Malkov and Yashunin [2018]Yu A Malkov and Dmitry A Yashunin.Efficient and robust approximate nearest neighbor search using
hierarchical navigable small world graphs.*IEEE transactions on pattern analysis and machine
intelligence*, 42(4):824–836, 2018.
* Mallia et al. [2021]Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto.Learning passage impacts for inverted indexes.In Fernando Diaz, Chirag Shah, Torsten Suel, Pablo Castells, Rosie
Jones, and Tetsuya Sakai, editors, *SIGIR ’21: The 44th International
ACM SIGIR Conference on Research and Development in Information
Retrieval, Virtual Event, Canada, July 11-15, 2021*, pages 1723–1727. ACM,
2021.doi: 10.1145/3404835.3463030.URL [https://doi.org/10.1145/3404835.3463030](https://doi.org/10.1145/3404835.3463030 "").
* Mehta et al. [2022]Sanket Vaibhav Mehta, Jai Prakash Gupta, Yi Tay, Mostafa Dehghani, Vinh Q.
Tran, Jinfeng Rao, Marc Najork, Emma Strubell, and Donald Metzler.DSI++: updating transformer memory with new documents.*CoRR*, abs/2212.09744, 2022.doi: 10.48550/arXiv.2212.09744.URL [https://doi.org/10.48550/arXiv.2212.09744](https://doi.org/10.48550/arXiv.2212.09744 "").
* Metzler et al. [2021a]Donald Metzler, Yi Tay, Dara Bahri, and Marc Najork.Rethinking search.*ACM SIGIR Forum*, 55(1):1–27,
2021a.
* Metzler et al. [2021b]Donald Metzler, Yi Tay, Dara Bahri, and Marc Najork.Rethinking search: making domain experts out of dilettantes.*SIGIR Forum*, 55(1):13:1–13:27,
2021b.doi: 10.1145/3476415.3476428.URL [https://doi.org/10.1145/3476415.3476428](https://doi.org/10.1145/3476415.3476428 "").
* Nakano et al. [2022]Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina
Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders,
Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew
Knight, Benjamin Chess, and John Schulman.Webgpt: Browser-assisted question-answering with human feedback,
2022.
* Nguyen et al. [2016]Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan
Majumder, and Li Deng.MS MARCO: A human generated machine reading comprehension
dataset.In Tarek Richard Besold, Antoine Bordes, Artur S. d’Avila Garcez, and
Greg Wayne, editors, *Proceedings of the Workshop on Cognitive
Computation: Integrating neural and symbolic approaches 2016 co-located with
the 30th Annual Conference on Neural Information Processing Systems (NIPS
2016), Barcelona, Spain, December 9, 2016*, volume 1773 of *CEUR
Workshop Proceedings*. CEUR-WS.org, 2016.URL [https://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf](https://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf "").
* Raffel et al. [2020]Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.Exploring the limits of transfer learning with a unified text-to-text
transformer.*J. Mach. Learn. Res.*, 21:140:1–140:67, 2020.URL [http://jmlr.org/papers/v21/20-074.html](http://jmlr.org/papers/v21/20-074.html "").
* Robertson and Zaragoza [2009]Stephen E. Robertson and Hugo Zaragoza.The probabilistic relevance framework: BM25 and beyond.*Found. Trends Inf. Retr.*, 3(4):333–389,
2009.doi: 10.1561/1500000019.URL [https://doi.org/10.1561/1500000019](https://doi.org/10.1561/1500000019 "").
* Tay et al. [2022]Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta,
Zhen Qin, Kai Hui, Zhe Zhao, Jai Prakash Gupta, Tal Schuster, William W.
Cohen, and Donald Metzler.Transformer memory as a differentiable search index.*CoRR*, abs/2202.06991, 2022.URL [https://arxiv.org/abs/2202.06991](https://arxiv.org/abs/2202.06991 "").
* Wang et al. [2023]Boxin Wang, Wei Ping, Peng Xu, Lawrence McAfee, Zihan Liu, Mohammad Shoeybi,
Yi Dong, Oleksii Kuchaiev, Bo Li, Chaowei Xiao, Anima Anandkumar, and Bryan
Catanzaro.Shall we pretrain autoregressive language models with retrieval? a
comprehensive study, 2023.
* Wang et al. [2022]Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun,
Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie,
Hao Allen Sun, Weiwei Deng, Qi Zhang, and Mao Yang.A neural corpus indexer for document retrieval.*CoRR*, abs/2206.02743, 2022.doi: 10.48550/arXiv.2206.02743.URL [https://doi.org/10.48550/arXiv.2206.02743](https://doi.org/10.48550/arXiv.2206.02743 "").
* Xiong et al. [2021]Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett,
Junaid Ahmed, and Arnold Overwijk.Approximate nearest neighbor negative contrastive learning for dense
text retrieval.In *9th International Conference on Learning Representations,
ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021.URL [https://openreview.net/forum?id\=zeFrfgyZln](https://openreview.net/forum?id=zeFrfgyZln "").
* Zhou et al. [2022]Yujia Zhou, Jing Yao, Zhicheng Dou, Ledell Wu, Peitian Zhang, and Ji-Rong Wen.Ultron: An ultimate retriever on corpus with a model-based indexer,
2022.
