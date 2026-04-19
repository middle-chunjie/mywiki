Scaling Laws For Dense Retrieval
================================

Yan Fang[fangy21@mails.tsinghua.edu.cn](mailto:fangy21@mails.tsinghua.edu.cn%20) ,Jingtao Zhan[jingtaozhan@gmail.com](mailto:jingtaozhan@gmail.com)Department of Computer Science and Technology, Tsinghua UniversityQuan Cheng LaboratoryBeijing 100084China,Qingyao Ai[aiqy@tsinghua.edu.cn](mailto:aiqy@tsinghua.edu.cn)Quan Cheng LaboratoryDepartment of Computer Science and Technology, Tsinghua UniversityBeijing 100084China,Jiaxin Mao[maojiaxin@gmail.com](mailto:maojiaxin@gmail.com)Gaoling School of Artificial Intelligence, Renmin University of ChinaBeijing 100872China,Weihang Su[swh22@mails.tsinghua.edu.cn](mailto:swh22@mails.tsinghua.edu.cn)Department of Computer Science and Technology, Tsinghua UniversityZhongguancun LaboratoryBeijing 100084China,Jia Chen[chenjia2@xiaohongshu.com](mailto:chenjia2@xiaohongshu.com)Xiaohongshu IncBeijingChinaandYiqun Liu[yiqunliu@tsinghua.edu.cn](mailto:yiqunliu@tsinghua.edu.cn)Department of Computer Science and Technology, Tsinghua UniversityZhongguancun LaboratoryBeijing 100084China

(2018)

###### Abstract.

Scaling up neural models has yielded significant advancements in a wide array of tasks, particularly in language generation.
Previous studies have found that the performance of neural models frequently adheres to predictable scaling laws, correlated with factors such as training set size and model size.
This insight is invaluable, especially as large-scale experiments grow increasingly resource-intensive.
Yet, such scaling law has not been fully explored in dense retrieval due to the discrete nature of retrieval metrics and complex relationships between training data and model sizes in retrieval tasks.
In this study, we investigate whether the performance of dense retrieval models follows the scaling law as other neural models.
We propose to use contrastive log-likelihood as the evaluation metric and conduct extensive experiments with dense retrieval models implemented with different numbers of parameters and trained with different amounts of annotated data.
Results indicate that, under our settings, the performance of dense retrieval models follows a precise power-law scaling related to the model size and the number of annotations.
Additionally, we examine scaling with prevalent data augmentation methods to assess the impact of annotation quality, and apply the scaling law to find the best resource allocation strategy under a budget constraint.
We believe that these insights will significantly contribute to understanding the scaling effect of dense retrieval models and offer meaningful guidance for future research endeavors.

Dense retrieval, Neural scaling law, Large language models

††copyright: acmcopyright††journalyear: 2018††doi: XXXXXXX.XXXXXXX††conference: The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval; July 14 - 18, 2024; Washington D.C., USA††price: 15.00††isbn: 978-1-4503-XXXX-X/18/06

1. Introduction
----------------

The studies of scaling law in language data can be traced back to a century ago.
In the 1920s, a couple of linguisticians discovered that the frequency of a word is proportional to the inverse of its rank when sorting vocabulary based on each word’s frequency in the corpus, which is widely known as the Zipf’s law *(Adamic and Huberman, [2002](#bib.bib2 ""); Newman, [2005](#bib.bib28 ""))*.
Later in the 1960s, Gustav Herdan found that the number of distinct words in a corpus approximately follows a function of the corpus size, which can be approximated with a power function.
This is often referred to as the Heaps’s law*(Lü et al., [2010](#bib.bib24 ""))*.
The discovery of these scaling laws in language has significantly influenced not only the research of linguistic, but also many studies in natural language processing (NLP) and information retrieval (IR).
For example, Zipf’s law has provided inspiration for the development of several statistical retrieval models, and Heap’s law has served as the key principle for the estimation of inverted index, which serves as the foundation of many retrieval systems.

Recently, as the focus of language modeling has switched from statistic analysis to semantic representation learning, the studies of scaling law is also shifting from analyzing the statistical characteristics of text data to the modeling of empirical relationships between representation/model structures and downstream language tasks.
For instance, pre-trained neural language models have received considerable attention due to their impressive language comprehension and generation performance.
Particularly, pre-trained language models with an extreme large number of model parameters, i.e., large language models (LLM), have achieved a human-level performance in many language-related tasks.
In addition to the design of model structures and training objectives, researchers have spent considerable effort investigating the relationships between parameter size and language generation quality of LLMs.
Previous studies have shown that variations in the model size, data size and computation reliably of LLMs often significantly affect the converged loss of LLM pre-training and downstream task performance.
They have identified a precise power law scaling relationship between model performance and such scaling factors.
These scaling laws enable researchers and developers to empirically predict model performance under certain conditions without actually constructing the model.
Since the training of modern LLMs demands substantial time and financial resources, such scaling laws are of great importance to both the research and applications of LLMs.

<img src='x1.png' alt='Refer to caption' title='' width='368' height='174' />

*Figure 1. Contrastive perplexity of different models on MSMARCO Passage Ranking (Left) and T2Ranking (Right) datasets.*

Besides LLMs, pre-trained neural language models have also served as the foundation of dense retrieval models.
Due to their superior ability to capture the semantic similarity between term-based queries and documents, dense retrieval models*(Guo et al., [2021](#bib.bib13 ""); Lin et al., [2021](#bib.bib22 ""))* have become the most popular neural retrieval models in recent years.
They have achieved the state-of-the-art retrieval performance in many IR tasks*(Qu et al., [2020](#bib.bib33 ""); Shao et al., [2023](#bib.bib38 ""); Ma et al., [2023](#bib.bib27 ""))*.
Typical dense retrieval models use a transformer-based encoder, usually initialized with a pre-trained language model and trained with relevant query-document pairs annotated by humans, to encode both queries and documents into a shared, fixed-dimensional latent semantic space.
In this space, relevant documents can be directly retrieved based on their similarity with the input queries.

Dense retrieval models and LLMs share many similarities in terms of structures and applications. First, both dense retrieval models and LLMs employ the Transformer-based architecture. Second, LLMs are optimized to maximize the probability of generating the target next token from the vocabulary based on a given context, while dense retrieval models are trained to maximize the probability of retrieving the target positive passage from the candidates based on a given query.
Therefore, an obvious and important question is: whether the performance of dense retrieval models follows similar scaling laws as those observed on LLMs?

Interestingly, while some studies have indicated that larger models exhibit improved generalization capabilities in zero-shot dense retrieval tasks*(Ni et al., [2021](#bib.bib30 ""); Rosa et al., [2022](#bib.bib36 ""))*, to the best of our knowledge, there isn’t any published literature explicitly investigating the scaling laws in dense retrieval models.
We believe that there are two key challenges that prevent researchers from successfully uncovering such phenomenon in dense retrieval.
First, the training of dense retrieval models often requires large-scale relevant annotations created by human annotators or users, which are usually expensive to collect.
Unlike LLMs, whose training data can be unsupervisedly and effectively extracted from raw text data in a large scale (usually in petabytes), the annotated training data of dense retrieval models are usually limited (usually in gigabytes).
Thus, the observed performance of dense retrieval models is often a twisted result affected jointly by model size and data size, making it difficult to be interpreted with a concrete function.
Second, existing evaluation paradigms for dense retrieval models mostly estimate model quality with discrete functions such as NDCG and MAP.
The discontinuity of these evaluation metrics makes them insensitive to model size and data size changes within some ranges, but extremely sensitive to changes that pass certain thresholds.
This poses significant difficulties to the construction of a smooth and reliable function to fit the behaviors of dense retrieval models.

In this paper, we focus on the investigation of scaling laws for dense retrieval models.
We first propose to evaluate the quality of dense retrieval models with a contrastive perplexity metric.
The idea of contrastive perplexity is inspired by the popular contrastive ranking loss and the analysis of token generation perplexity in LLMs.
It measures the likelihood of retrieving a relevant document from a randomly sampled candidate set, and shares a similar structure with the training loss of dense retrieval models.
The smooth nature of this metric considerably facilitates our subsequent analysis.
Second, to disentangle the effects of model size and data size in dense retrieval, we conducted experiments with models implemented with different pre-trained language models with non-embedding parameter sizes ranging from 0.5 to 87 million, on two of the largest web search datasets, i.e., MSMARCO and T2Ranking.
Experimental results show that, under proper experimental conditions, the performance of dense retrieval models follows a precise power-law scaling with respect to training factors. Figure [1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Scaling Laws For Dense Retrieval") illustrates such power-law scaling with model size.
To investigate the effect of annotation quality, we adopted several LLMs and weak supervision methods to generate training data for dense retrieval models.
Our results indicate that the observed scaling laws of dense retrieval are uniformly valid across models trained with different types of annotation data.
Additionally, we show that the joint effect of model and data sizes can be nicely fitted and predicted with a single function within a certain range.
Such functions can be used to find the best resource allocation strategy given a restricted budget, and could potentially provide important insights for the practical implementation of dense retrieval models and green IR*(Strubell et al., [2019](#bib.bib39 ""))*

The contributions of this paper can be summarized as the following:

* •

    We investigate and identify several scaling laws for dense retrieval models.

* •

    We conduct extensive experiments with different model sizes, data sizes, and types of annotation. Experiment results show that the scaling law of dense retrieval models is valid across a wide range of experimental settings.

* •

    We derive empirical functions to fit the relationship between model performance and multiple factors, and showcase their potential applications in several tasks.

2. Background and Related Work
-------------------------------

### 2.1. Scaling Laws in Linguistic Language Data

Zipf’s law*(Adamic and Huberman, [2002](#bib.bib2 ""); Newman, [2005](#bib.bib28 ""))* is one of the best pieces of evidence about the existence of universal power laws in cognitive science and the social sciences. This law finds significant application in linguistics, asserting an inverse correlation between the frequency of a word’s occurrence in natural language and its rank in the frequency distribution.

Furthermore, Zipf’s law is intrinsically connected to other statistical scaling laws in linguistics, notably Heaps’ law*(Lü et al., [2010](#bib.bib24 ""); Gelbukh and Sidorov, [2001](#bib.bib12 ""); Lansey and Bukiet, [2009](#bib.bib21 ""))*. Heaps’ law delineates a sublinear growth trajectory between a text’s vocabulary size and its total word count. As the total word count increases, the rate of introducing new words diminishes, leading to a plateau in vocabulary expansion. This phenomenon is particularly significant in information retrieval, which serves as the key principle for the estimation of inverted index.

### 2.2. Neural Scaling Law

Neural scaling laws articulate the relationship between model size, dataset size, computational budget, and performance in neural network training. This concept, first introduced by Hestness et al.*(Hestness et al., [2017](#bib.bib14 ""))*, identified a power-law relationship, which was subsequently expanded upon for larger models by Kaplan et al.*(Kaplan et al., [2020](#bib.bib19 ""))*. Researchers further refined this concept by developing a unified formula for scaling laws, incorporating data-dependent scaling terms for compute-optimal training.*(Hoffmann et al., [2022](#bib.bib15 ""))*

These empirical scaling laws offer crucial insights for training large Transformer-based models, particularly in accurately predicting losses under specific settings. Notably, results from experiments with smaller models can be extrapolated to larger ones and may even be applicable to other downstream tasks. Recent studies have diversified the application of these laws, examining different Transformer parameterizations. For instance, Clark et al.*(Clark et al., [2022](#bib.bib5 ""))* explored their application in Mixture of Experts (MoE) models. Gao’s research*(Gao et al., [2023](#bib.bib11 ""))* delved into the scaling effects in model optimization with Reinforcement Learning, pivotal in AI alignment.

Beyond language-centric tasks, these scaling principles have been adapted for domain-specific applications, such as speech recognition*(Radford et al., [2023](#bib.bib35 ""))*, computer vision*(Zhai et al., [2022](#bib.bib42 ""); Dehghani et al., [2023](#bib.bib7 ""))*, and multi-modal language-vision settings*(Jia et al., [2021](#bib.bib18 ""); Pham et al., [2023](#bib.bib32 ""); Radford et al., [2021](#bib.bib34 ""))*.
In Information Retrieval (IR), Ardalani et al.*(Ardalani et al., [2022](#bib.bib4 ""))* investigated the application of scaling laws in Click-Through Rate (CTR) recommendation tasks, and Zhang et al.*(Zhang et al., [2023](#bib.bib45 ""))* addressed their relevance in conventional ID-based sequential recommendation models. Nonetheless, there has been limited research into whether scaling laws remain applicable in dense retrieval.

### 2.3. Dense Retrieval

We now briefly revisit prior studies in the field of dense retrieval.
The training data for dense retrieval tasks typically comprises annotated pairs, each consisting of a query and a human-labeled relevant passage.
Early research primarily concentrated on effective negative sampling strategies used for dense retrieval training, such as employing random passages or the top irrelevant passages retrieved by BM25 as negative samples. ANCE*(Xiong et al., [2020](#bib.bib41 ""))* introduced the concept of self-mined negatives, wherein negatives are refreshed every epoch, which greatly improved the retrieval performance, especially in the top ranking results. Furthermore, Zhan et al.*(Zhan et al., [2021a](#bib.bib43 ""), [b](#bib.bib44 ""))* proposed dynamic hard negatives to further enhance both training efficiency and retrieval effectiveness. RocketQA*(Qu et al., [2020](#bib.bib33 ""))* and TAS-B*(Hofstätter et al., [2021](#bib.bib16 ""))* introduced knowledge distillation, utilizing a well-trained cross-encoder model to generate soft labels for training pairs.

Subsequently, researchers’ attention shifted to retrieval-oriented second-stage pre-training for the language model. Previous pre-training tasks, such as Masked Language Modeling (MLM) or Causal Language Modeling, were not specifically tailored for Information Retrieval (IR) scenarios. Therefore, dense retrieval models might be not able to fully utilize the language comprehension capacity developed through large-scale pre-training. Condenser*(Gao and Callan, [2021a](#bib.bib8 ""))* and coCondenser*(Gao and Callan, [2021b](#bib.bib9 ""))* employed the Sequence Contrastive Learning (SCL) task to improve the representational capability of the [CLS] token, enabling it to encapsulate a richer information content. Furthermore, RetroMAE*(Liu and Shao, [2022](#bib.bib23 ""))* introduces an encoder-decoder architecture, wherein a shallow decoder compels the encoder to produce higher-quality representations. Contriever*(Izacard et al., [2021](#bib.bib17 ""))* pre-trains dense retrieval models on crawled large-scale web pages using the Inverse Cloze Task (ICT) and the Independent Cropping Task, not only enhancing downstream retrieval performance with supervised fine-tuning but also demonstrating significant potential in zero-shot dense retrieval.

Additionally, researchers have observed that the single, fixed-dimension vector representation in dense retrieval could become a limitation for further advancements. Various studies have explored more complex scoring techniques. ME-BERT*(Luan et al., [2021](#bib.bib25 ""))*, for instance, introduces multi-vector representations to enable more precise retrieval of long documents, while ColBERT*(Khattab and Zaharia, [2020](#bib.bib20 ""); Santhanam et al., [2021](#bib.bib37 ""))*, on the other hand, investigates token-level vector representations and aggregates scores using a late-interaction mechanism. Other research endeavors have attempted to expand the vector dimension to vocabulary size. This expansion allows dense retrieval models to directly generate term weights, facilitating retrieval similar to sparse models.

From the perspective of training data, dense retrieval often relies on query generation techniques for data augmentation. Query generation involves generating multiple relevant queries for a given passage. The most basic approach employs unsupervised heuristic methods, such as the previously mentioned Sequence Contrastive Learning (SCL) or Inverse Cloze Task (ICT). However, the quality of the weak supervision data generated by these methods is relatively low. Therefore, they are primarily used in the unsupervised pre-training phase due to their accessibility. More advanced methods leverage pre-trained language models like T5 to generate more precise relevant queries for data augmentation*(Nogueira et al., [2019](#bib.bib31 ""))*. Nevertheless, these generated queries are often used for document expansion to enhance the retrieval performance in lexical matching models. As training data, these queries are usually exploited in scenarios where human annotations are scarce, such as in out-of-domain situations. Recently, approaches like HyDE*(Gao et al., [2022](#bib.bib10 ""))* have leveraged the exceptional language generation capabilities of Large Language Models (LLMs) for data augmentation. This generated data is of superior quality and can be used to replace the original queries or passages in retrieval tasks.

To summarize the prior explorations of dense retrieval models, we conclude that they mainly focus on techniques of enhancing retrieval performance within certain resource constraints. This includes the development of sophisticated training strategies and model architectures with fixed model sizes, as well as the incorporation of various augmented data into the dense retrieval (DR) lifecycle when high-quality human annotations are limited.
However, our study aims to explore the scaling laws of dense retrieval, a direction orthogonal to previous research.
Specifically, given the same training strategies, we examine how model performance varies under different scales of available resources (such as model size and the number of annotated training pairs).

Specifically, in this study, we aim to thoroughly investigate the following three research questions:

* •

    How does the variation of model sizes impact dense retrieval performance?

* •

    How does the change of annotated training data sizes influence dense retrieval performance?

* •

    Do different types of data annotations result in distinct scaling effects on dense retrieval models?

3. Methedology
---------------

In this section, we first introduce the model architecture and datasets used for exploring the scaling effect of dense retrieval. We further discuss the training strategy used in the experiments and the performance evaluation metrics adopted.

<img src='x2.png' alt='Refer to caption' title='' width='415' height='149' />

*Figure 2. IR metrics with contrastive perplexity for different models on MSMARCO Passage Ranking*

### 3.1. Problem Formulation

We first give a formal description of the dense retrieval task. In a fixed corpus ${C}$, the goal of retrieval is to identify the top $K$ relevant passages for a given query $q$. Dense retrieval models accomplish this by employing an encoder that maps both queries and candidate passages into a shared dense embedding space. Subsequently, a scoring function, such as inner product or cosine similarity, is applied to the encoded dense vectors to model relevance:

| (1) |  | $s(q,p)\=\left\langle f(q;\theta),f(p;\theta)\right\rangle$ |  |
| --- | --- | --- | --- |

where $q$ and $p$ denote the query and the passage, respectively. $f(\cdot;\theta)$ is the mapping function of the dense retrieval model parameterized by $\theta$.
Note that in this paper, we only focus on the dense retrieval models with the above structure as it is one of the most popular and representative structures in practice. We leave the studies of other types of dense retrieval models to future studies.

The training data for dense retrieval typically comprises a set of training queries and associated human annotations. Each query is annotated with one or more relevant passages, and the remaining unannotated passages are generally presumed irrelevant. In this paper, we adhere to this annotation standard and consider each query-positive-passage pair as an individual data point. Formally, the training set ${S}_{1}^{n}$ consists of $n$ data points:

| (2) |  | ${S}_{1}^{n}\={(q_{i},p_{i}^{+})}_{i\=1}^{n}$ |  |
| --- | --- | --- | --- |

where $q_{i}$ and $p_{i}^{+}$ denote the $i$-th query in the training set and its corresponding annotated positive passage.

### 3.2. Model Architechture

With the development of large-scale pre-trained language models, advanced dense retrieval models in recent years have followed the Transformer’s structure.
While some studies have explored using decoder-only architectures to generate dense vector representations of texts, mainstream dense retrieval models still employ encoder-only models such as BERT. Formally, a pre-trained Transformer, augmented with a projection layer, serves as the text encoder:

| (3) |  | $v\=W\cdot\left({\rm Transformer}(x;\phi)\right)+b$ |  |
| --- | --- | --- | --- |

where $x$ represents the text input, $\phi$ denotes the parameters of the Transformer encoder, and $W$ and $b$ are the parameters of the projection layer.

Typically, the generated vector representation is derived from the [CLS] token representation (in BERT series models) or the mean pooling of the outputs from the last Transformer layer. The main function of the projection layer is to map these vectors into the target semantic space.

In our study, we experimented with Transformer models of various model sizes. With limited annotated query-passage pairs, it is usually difficult to train a large dense retrieval model from scratch. As a result, most dense retrieval models are initialized with pre-trained language models and then perform fine-tuning on the annotated data. Therefore, to align with prevailing research practices, we focus our analysis on fine-tuning dense retrieval models constructed with pre-trained language models in different sizes.

Previous studies have shown that different pre-training tasks could significantly affect the performance of dense retrieval models*(Gao and Callan, [2021a](#bib.bib8 ""), [b](#bib.bib9 ""); Izacard et al., [2021](#bib.bib17 ""); Liu and Shao, [2022](#bib.bib23 ""); Ma et al., [2022](#bib.bib26 ""))*.
To minimize such influence, we selected a series of models with identical pre-training configurations and only differ in parameter sizes. Specifically, for experiments on the English corpus, we chose 24 BERT checkpoints from the original Google release, with model sizes ranging from 0.6 million (BERT-Tiny) to 87 million parameters (BERT-Base)111https://github.com/google-research/bert. The model sizes are defined by non-embedding parameters in this paper.. For experiments on Chinese retrieval benchmarks, we selected the ERNIE series, which were pre-trained on Chinese corpora using tasks similar to BERT. To each model, we attached a projection layer to standardize the output vector (e.g., query and document vectors) dimensionality to 768 for consistent comparisons.

### 3.3. Training Data

We utilize publicly available retrieval datasets for exploring the scaling effect for dense retrieval models. To ensure the generalizability and completeness of our study, we follow recent DR research and use MS MARCO Passage Ranking dataset*(Nguyen et al., [2016](#bib.bib29 ""))* (English) and T2Ranking*(Xie et al., [2023](#bib.bib40 ""))* (Chinese) for the experiments. MS MARCO Passage Ranking is a large-scale annotated dataset with a corpus of 8.8M passages from English web pages and 0.5M training queries. Each training query is coupled with a manually labeled positive passage, which together constitute the annotated pairs. MS MARCO also provides around 7,000 validation queries for performance evaluation. T2Ranking is a recently released large-scale Chinese benchmark for passage ranking, which comprises more than 300k queries and over 2M unique passages collected from real-world search engines.

### 3.4. Training Setting

As discussed previously, in this paper, we construct dense retrieval models from the pre-trained language model checkpoints and perform fine-tuning with the annotated query-document pairs in each dataset.
One of the most important parts of dense retrieval model training is the negative sampling strategy.
Previous work has shown that mining hard negative samples in the training process can significantly improve the retrieval performance.
However, the primary objective of this work is to investigate the scaling effects of dense retrieval models. As a result, we do not focus on sophisticated training strategies. For simplicity, we adopt the most straightforward approaches, namely random negative sampling and in-batch negative techniques, for the training of all dense retrieval models in this paper. These methods are employed to minimize the influence of sampling strategies.

Formally, for each query-passage pair ($q_{i},p_{i}^{+}$), we randomly select a set of unlabeled passages from the corpus as the negative. Then we can optimize the following contrastive ranking loss:

| (4) |  | $\displaystyle\mathcal{L}(\theta)$ | $\displaystyle\=-{1\over B}\sum_{i\=1}^{B}\log{\exp\left(s(q_{i},p_{i}^{+};\theta)\right)\over{\exp\left(s(q_{i},p_{i}^{+};\theta)\right)+\sum_{j}\exp\left(s(q_{i},p_{j}^{-};\theta)\right)}}$ |  |
| --- | --- | --- | --- | --- |

where $B$ denotes the training batch size, ${p_{j}^{-}}$ is the set of negative passages and $s(q,p;\theta)$ is the scoring function of query and passage:

| (5) |  | $\displaystyle s(q,d;\theta)$ | $\displaystyle\=\left\langle f(q;\theta),f(d;\theta)\right\rangle$ |  |
| --- | --- | --- | --- | --- |

Here, $\left\langle\cdot\right\rangle$ denotes inner product and $\theta$ denotes the parameters of the text encoder.

We fine-tune the models for a fixed 10,000 steps and random sample 256 negatives at each step.

### 3.5. Evaluation Protocol

We now discuss how we evaluate the retrieval performance. The most widely adopted retrieval paradigm is to rank passages in the corpus based on the relevance scores predicted by the retrieval model and retrieve the Top-K candidates to form a ranked list. The performance of the retrieval model is then assessed based on the ranked list using well-defined ranking metrics such as NDCG@K and MAP@K. However, such metrics are not continuous due to their discrete nature and reliance on a cutoff parameter, K. Because the ranking metrics of a ranked list would not change unless the sequence of the passages changes, these ranking metrics are not sensitive to the changes of model outputs in many cases.
Also, with the cutoff in ranking metric, a positive passage only contributes to the metric when ranked within the top K results. If it falls beyond K, whether at K+1 or further, it has no impact on the metric score. The characteristics of these existing ranking metrics make them unsuitable for the investigation of scaling laws in dense retrieval. To solve these problems, we propose to construct a more fluid and continuous metric that better reflects the overall retrieval capability of the models under various settings. Inspired by the analysis of scaling laws in large language models, which utilize the perplexity of token generations as evaluation metrics, we propose to use the contrastive perplexity as our evaluation metric. Formally, for each query-passage pair in the test set, we randomly select $W$(256 in this paper) unlabeled passages and define the contrastive perplexity as:

| (6) |  | $\displaystyle\mathcal{L}(W;\theta)$ | $\displaystyle\=-\log{\exp\left(s(q_{i},p_{i}^{+};\theta)\right)\over{\exp\left(s(q_{i},p_{i}^{+};\theta)\right)+\sum_{j\=1}^{W}\exp\left(s(q_{i},p_{j}^{-};\theta)\right)}}$ |  |
| --- | --- | --- | --- | --- |

We plot the IR metrics with the proposed contrastive perplexity for different models in Figure [2](#S3.F2 "Figure 2 ‣ 3. Methedology ‣ Scaling Laws For Dense Retrieval").
As we can see from the figure, the correlation between the contrastive perplexity and existing ranking metrics is strong and positive.
The correlation between contrastive perplexity and recall@1000, a popular metric used in most dense retrieval benchmarks, is particularly close to a linear correlation.
Besides, the structure of contrastive perplexity is closely related to the training loss of dense retrieval models in our experiments.
Therefore, we believe that using contrastive perplexity is an effective measure to assess the overall retrieval ability of models in our study.

<img src='x3.png' alt='Refer to caption' title='' width='368' height='170' />

*Figure 3. Contrastive perplexity of different models on MSMARCO Passage Ranking (Left) and T2Ranking (Right) datasets. The curve demonstrates precise linearity in the logarithmic scale.*

4. Scaling Laws For Dense Retrieval
------------------------------------

In this section, we show the results of our experiments and summarize our initial investigation of the scaling laws for dense retrieval.

### 4.1. Model Size Scaling

We first fine-tune models of various sizes using the human-annotated training pairs, while keeping all other settings identical. We utilized the entire training sets to simulate conditions of infinite data, in order to minimize the influence of performance constraints due to limited training data. We continuously monitored the contrastive perplexity in the test set and reported the best result of each model. It is important to note that this approach was adopted to mitigate the influence of suboptimal early stopping, which could lead to models being underfitted or overfitted.

In alignment with neural scaling law analysis*(Kaplan et al., [2020](#bib.bib19 ""))*, we define model size by the number of non-embedding parameters. Figure [3](#S3.F3 "Figure 3 ‣ 3.5. Evaluation Protocol ‣ 3. Methedology ‣ Scaling Laws For Dense Retrieval") illustrates the evaluation metric, namely the contrastive perplexity on the test set, with respect to model sizes. As shown in the figure, the retrieval performance improves (indicated by a lower test contrastive perplexity) as the model size increases. On the left side of the diagram, red stars represent the official checkpoints of variously sized BERT models, while blue points denote other official variants released concurrently. These variants differ in aspects such as the number of attention heads or feed-forward dimensions. The right diagram, in contrast, only features red stars, as the different shape variants of ERNIE are not publicly available.

Based on observation, we propose to fit the scaling law of dense retrieval models in terms of model sizes as follows:

| (7) |  | $L(N)\=\left({A\over N}\right)^{\alpha}+\delta_{N}$ |  |
| --- | --- | --- | --- |

where $N$ represents the number of non-embedding parameters of the model, and $L(N)$ denotes the model’s contrastive perplexity on the test set. Parameters $A$, $\alpha$ and $\delta_{N}$ are the constants estimated based on the observations of different models’ performance.

Note that we introduce a parameter $\delta_{N}$. This means that, when we have a sufficiently large model that has been adequately trained (setting $N$ to infinity), the contrastive perplexity approaches $\delta_{N}$ rather than zero. This is reasonable because, under the current settings of training and testing process, even an ideal model cannot reduce the contrastive perplexity to zero. Specifically, an important reason for this phenomenon is related to the annotation data in our experimental datasets.
The contrastive perplexity used in our experiments directly reflects the distances between the model’s predictions and the actual human-annotated labels in the datesets. On one hand, human annotations are far from complete in MSMARCO and T2Ranking, leading to issues with false negatives. On the other hand, most existing retrieval datasets employ binary annotations, while an ideal model should fit a continuous relevance distribution. Therefore, even the best dense retrieval models cannot fit the human annotations perfectly, which explains why we need a bias term $\delta_{N}$ in Eq. (8)

We employed the commonly used least squares method to fit the linear curve, and the coefficient of determination (R²) suggests a good fit. Based on these results, we observe that the contrastive perplexity follows a power-law scaling in relation to the size of non-embedding parameters. The parameters of the fitted curve are detailed in Table [1](#S4.T1 "Table 1 ‣ 4.1. Model Size Scaling ‣ 4. Scaling Laws For Dense Retrieval ‣ Scaling Laws For Dense Retrieval").

<img src='x4.png' alt='Refer to caption' title='' width='368' height='168' />

*Figure 4. Contrastive perplexity of different numbers of training data on MSMARCO Passage Ranking (Left) and T2Ranking (Right) datasets. The curve demonstrates precise linearity on the logarithmic scale.*

*Table 1. Fitting parameters for model size scaling*

| Dataset | $A$ | $\alpha$ | $\delta_{N}$ | $R^{2}$ |
| --- | --- | --- | --- | --- |
| MSMARCO | $4.54\times 10^{4}$ | 0.46 | 0.02 | 0.990 |
| T2Ranking | $9.88\times 10^{6}$ | 0.53 | 0.14 | 0.999 |

Such discoveries offer new perspectives for future research experiments. For example, given this scaling law, once a training dataset is available, we can initially train smaller models, fit the corresponding scaling curves, and then extrapolate them to predict the performance of larger models. This approach could significantly reduce the cost and effort of conducting experiments directly on larger models. More importantly, experimenting with different training strategies on smaller models can quickly validate the effectiveness of new approaches.

### 4.2. Data Size Scaling

We then fixed the model size and varied the size of the training data, defined by the number of annotated query-passage pairs. To minimize potential underfit problem caused by small models, we specifically fine-tuned the largest models in our experiment, i.e., the BERT-Base model. Here we only present the experiment results up to using all available annotatation data.

The results are plotted in Figure [4](#S4.F4 "Figure 4 ‣ 4.1. Model Size Scaling ‣ 4. Scaling Laws For Dense Retrieval ‣ Scaling Laws For Dense Retrieval"). Similarly, we fit our experiment observations with a log-linear curve. The coefficient of determination (R²) indicates a good fit. Based on these results, we infer that the contrastive perplexity follows a power-law scaling relative to the number of annotated query-passage pairs, with specific parameters detailed in Table [2](#S4.T2 "Table 2 ‣ 4.2. Data Size Scaling ‣ 4. Scaling Laws For Dense Retrieval ‣ Scaling Laws For Dense Retrieval")

| (8) |  | $L(D)\=\left({B\over D}\right)^{\beta}+\delta_{D}$ |  |
| --- | --- | --- | --- |

where $D$ represents the number of annotated query-passage pairs, and $L(D)$ denotes the contrastive perplexity. Parameters $B$, $\beta$ and $\delta_{D}$ are the constants to be determined.

*Table 2. Fitting parameters for data size scaling*

| Dataset | $B$ | $\beta$ | $\delta_{D}$ | $R^{2}$ |
| --- | --- | --- | --- | --- |
| MSMARCO | $2.37\times 10^{-2}$ | 0.19 | 0.01 | 0.971 |
| T2Ranking | $6.04\times 10^{4}$ | 0.50 | 0.15 | 0.991 |

Orthogonal to our previous findings, this conclusion offers an alternative perspective for future dense retrieval experiments and the data annotation process. For instance, when dealing with a completely new corpus and determining the requisite amount of annotations, the traditional approach relies on past experience without a clear understanding of the sufficiency of data annotation. Now, by taking advantage of the data size scaling law, a potential approach involves initiating with a minimal amount of annotations, training a model, and fitting the corresponding scaling curve. Accordingly, we can approximate the necessary size of data annotation based on the target performance of the dense retrieval model. This method is more economical since, as observed, the loss diminishes almost linearly on a logarithmic scale with increasing data size, indicating diminishing returns on performance improvements given more data annotations. Therefore, while more data invariably enhances performance, excessive annotation may not be cost-effective. Predetermining the expected goals and corresponding data size through our observed scaling laws would be a more efficient and economical strategy.

5. Annotation Quality
----------------------

So far, we have observed strong scaling phenomena of dense retrieval model performance with respect to model sizes and data sizes. Yet, in the IR scenario, another aspect that remained unexplored is the quality of data annotations, which can significantly impact the effectiveness of the data. A pertinent question arises: Does the scaling effect hold true for data of different quality? To investigate this, we conducted experiments using annotations of different quality. Due to constraints in time and resources, our experiments were exclusively conducted on the MSMARCO Passage Ranking dataset. Specifically, we employed query generation techniques to create three distinct types of annotations:

Inverse Cloze Task (ICT): The unsupervised ICT method extracts sentences from passages and pairs them with their context to form query-positive pairs. Given its reliance on topic and semantic similarity within a passage, this method often includes noisy data, positioning it as the lowest-quality annotation.

Supervised Generation Models: We utilize the well-trained docT5query to produce multiple queries for each passage, thereby forming query-positive pairs. The model is trained by human annotations, rendering it a higher-quality annotation method.

Large Language Models (LLMs): We instruct LLMs to generate relevant queries for given passages. Leveraging the substantial language comprehension and generation capabilities of LLMs, this method is considered to be of the highest quality. Practically, we used the recently open-sourced ChatGLM3, noted for its impressive performance in various downstream language tasks.

For ICT and ChatGLM3, we generate a query for each positive document annotated by humans in the original datasets. For docT5query, we randomly sampled 500,000 passages from the corpus for query generation, since it is originally trained on the human annotated passages. In this way, we can align the training passages with human annotations and other annotations.
Also, it’s important to note that, despite employing different data generation methods, our evaluations consistently utilized the human-annotated development set. The results are reported in Figure [5](#S5.F5 "Figure 5 ‣ 5. Annotation Quality ‣ Scaling Laws For Dense Retrieval").

<img src='x5.png' alt='Refer to caption' title='' width='438' height='335' />

*Figure 5. Contrastive perplexity of different annotations on MS MARCO.*

From the figure, we observe that the retrieval performance scales with respect to different annotation qualities. Comparing the three methods of query generation, it is evident that, on the log-linear curve, ICT exhibits the smallest slope.
This observation aligns with our expectations.
ICT, as a weak supervision method, involves considerable noise, thus limiting the enhancements in retrieval tasks when we increase the data size.
Therefore, previous studies usually use such weak supervision only in the retrieval-oriented pre-training phase, not the fine-tuning phase.
The data quality from docT5query is better, but not as good as human annotations, which could be attributed to the limited generation ability for a relatively small T5-base model.
The data from ChatGLM3 is of the highest quality, which results in a more steep slope in Figure [5](#S5.F5 "Figure 5 ‣ 5. Annotation Quality ‣ Scaling Laws For Dense Retrieval").

Additionally, we include human-annotated data for reference and notice an interesting phenomenon in Figure [5](#S5.F5 "Figure 5 ‣ 5. Annotation Quality ‣ Scaling Laws For Dense Retrieval"): the scaling slope of models trained with ChatGLM3 generated data is larger than the slope of human annotations.
This observation indicates that, with larger annotation sizes, the performance of dense retrieval models trained purely based on the data generated by ChatGLM3 might potentially outperform the models trained with human annotations, which shows the potential of LLMs in terms of data augmentation in retrieval tasks.
Yet, it’s crucial to note that it’s not entirely appropriate to directly compare the curve for human-annotated data with the other curves.
On the one hand, besides docT5query, we select the human-annotated positive passages in MSMARCO as candidates when generating queries, which may benefit the query generation methods in terms of data collection because low-quality documents are naturally excluded from the annotation process.
On the other hand, the nature of human annotation – labeling relevant passages for a given query – differs inherently from query generation, which involves generating a relevant query for a given passage.
Still, the results in Figure [5](#S5.F5 "Figure 5 ‣ 5. Annotation Quality ‣ Scaling Laws For Dense Retrieval") reveal the potential of LLM-based data annotation for dense retrieval, which is worth more investigations in future studies.

6. Application in Budget Allocation
------------------------------------

In this section, we showcase a potential application of the scaling laws for dense retrieval observed in our experiments. To get started, we first describe how to combine the above observations into a single function between model performance and model/data sizes. Inspired by the scaling laws of LLMs*(Kaplan et al., [2020](#bib.bib19 ""))*, on MSMARCO, we employ the following equation to describe the scaling effect as:

| (9) |  | $\displaystyle L(N,D)\=\left[\left({A\over N}\right)^{\alpha\over\beta}+{B\over D}\right]^{\beta}+\delta$ |  |
| --- | --- | --- | --- |
| (10) |  | $\displaystyle A\approx 8.2\times 10^{5},~{}~{}B\approx 5.2\times 10^{3}$ |  |
| --- | --- | --- | --- |
| (11) |  | $\displaystyle\alpha\approx 0.57,~{}~{}\beta\approx 1.39,~{}~{}\delta\approx 0.03$ |  |
| --- | --- | --- | --- |

Similarly, $N,D$ represents the model size and data size, respectively. $A,B,\alpha,\beta,\delta$ are the fitting parameters.
We employed results with different model sizes and data sizes to fit the parameters, which are used to predict the remaining data points. Figure [6](#S6.F6 "Figure 6 ‣ 6. Application in Budget Allocation ‣ Scaling Laws For Dense Retrieval") illustrates the contrastive perplexity across different resource settings. In this figure, solid dots represent the data used for curve fitting, while the dashed line indicates the resulting fitted curve. The red stars denote data points utilized to evaluate the accuracy of our predictions.

<img src='x6.png' alt='Refer to caption' title='' width='438' height='335' />

*Figure 6. Contrastive perplexity of different model size (different lines) and different data size (x-axis). The dots are used for fitting the curve and the red stars are used to test the predictions.*

We subsequently attempt to estimate the comprehensive cost associated with the lifecycle of dense retrieval models, including data annotation, model training, and model inference. The total cost of training a model with $N$ parameters using $D$ data points is given by:

| (12) |  | $Z(N,D)\=Z_{\rm data}\cdot D+Z_{\rm train}\cdot N+Z_{\rm infer}\cdot N$ |  |
| --- | --- | --- | --- |

Here, $Z_{\rm data},Z_{\rm train},Z_{\rm infer}$ represent cost factors corresponding to annotations, training, and inference, respectively.

To better illustrate the relationship between the predicted contrastive perplexity and the total cost, we provide approximate values for the cost factors as follows.
The cost of human annotations is approximated at $0.6 per query-passage pair*(Althammer et al., [2023](#bib.bib3 ""))*.
For computational costs, according to previous studies*(Kaplan et al., [2020](#bib.bib19 ""); Clark et al., [2020](#bib.bib6 ""))*,
the training and inference computation for Transformer can be assumed by $6N$ and $2N$ FLOPs, respectively. We refer to common cloud computing and the price for using an A100 80G GPU is assumed to be $3.93 per hour222From https://cloud.google.com/compute/gpus-pricing, the cost for A100 80G GPU is $2867.5 per month., with the peak computational power around 312 TFLOPs.
For the training phase, we assume that the model is trained for 10,000 steps on a single A100 GPU. At each step, the model encodes a query, a positive passage and a negative passage with a batch size of 256. Each query is around 30 tokens and each passage is around 60 tokens.
For the inference phase, we assume that the model is employed in a web search engine.
Based on public statistics, we assume that there are around 30 trillion web pages in Google’s index333From https://en.wikipedia.org/wiki/Google_Search, the estimated size of Google’s index is around 30 trillion in 2012.. The inference cost for a dense retrieval model predominantly involves encoding the entire corpus. We estimate that each web page contains approximately 512 tokens.
We assume the GPU utilization efficiency is 25%, then we have

| (13) |  | $\displaystyle Z_{\rm data}$ | $\displaystyle\approx 0.6$ |  |
| --- | --- | --- | --- | --- |
| (14) |  | $\displaystyle Z_{\rm train}$ | $\displaystyle\approx{10000\times(30+2\times 60)\times 256\times 6\times 3.93\over 312T\times 3600\times 25\%}\=3.22\times 10^{-8}$ |  |
| --- | --- | --- | --- | --- |
| (15) |  | $\displaystyle Z_{\rm infer}$ | $\displaystyle\approx{30\times 10^{12}\times 512\times 2\times 3.93\over 312T*3600\times 25\%}\=0.43$ |  |
| --- | --- | --- | --- | --- |

To get started, our first analysis excludes the cost of inference and focuses on annotation and training.
Figure [7](#S6.F7 "Figure 7 ‣ 6. Application in Budget Allocation ‣ Scaling Laws For Dense Retrieval") shows the predicted contrastive perplexity against model size under different cost budget.
It is clear that for a fixed cost budget, as the training model size increases, the predicted retrieval performance initially improves and then deteriorates.
With relatively small models, despite that we have a sufficient budget for data annotation, the capacity of the dense retrieval model would limit its overall performance.
Conversely, when training models with large parameter sizes, the budget constraints for data annotation limit the amount of data that can be used for training, leading to suboptimal retrieval performance.
Also, Figure [7](#S6.F7 "Figure 7 ‣ 6. Application in Budget Allocation ‣ Scaling Laws For Dense Retrieval") indicates that, without the consideration of inference cost, the optimal model size grows with an increased budget and can be over 13 billion parameters, which is not always practical in real-world scenarios.
This is primarily due to that, compared to the cost of model training, human annotation is significantly more expensive. Therefore, under a limited budget, maximizing model size can yield better results.

<img src='x7.png' alt='Refer to caption' title='' width='415' height='324' />

*Figure 7. Predicted contrastive perplexity of different model sizes under different cost budgets (w/o inference cost)*

Now, after including the inference costs, we have the analysis results in Figure [8](#S6.F8 "Figure 8 ‣ 6. Application in Budget Allocation ‣ Scaling Laws For Dense Retrieval").
It is clear that the optimal model size significantly diminishes, resulting in models with million-scale parameters, even under a larger budget. This observation suggests a notable distinction between dense retrieval and language generation tasks: the corpus size for retrieval is often large, making the encoding cost prohibitively high when opting for billion-scale models.

<img src='x8.png' alt='Refer to caption' title='' width='415' height='324' />

*Figure 8. Predicted contrastive perplexity of different model sizes under different cost budgets (w/ inference cost)*

7. Conclusions and Future Work
-------------------------------

In this paper, we have focused on investigating the scaling laws for dense retrieval models. We conducted extensive experiments with varying model sizes, data sizes, and annotation data. Our key finding is that in dense retrieval tasks, when employing the proposed contrastive perplexity as the evaluation metric, retrieval performance exhibits a precise power-law scaling in relation to both model size and data size.
Moreover, we further investigate the influence of different data augmentation methods and showcase potential applications of the fitted scaling law.
However, there are limitations to this work. Firstly, due to limited time and computation resources, our experiments only involve BERT-series models, and the variations in model size and data size are conducted within a relatively small range. Therefore, whether the corresponding scaling laws hold true for models on the scale of billion-scale models requires further investigation. Secondly, different training strategies, training steps, and the corpus size of datasets are factors that could affect scaling. These aspects were not explored in this study and are left for future work.

###### Acknowledgements.

This work is supported by Quan Cheng Laboratory (Grant No. QCLZD202301).

References
----------

* (1)
* Adamic and Huberman (2002)Lada A Adamic and
Bernardo A Huberman. 2002.Zipf’s law and the Internet.*Glottometrics* 3,
1 (2002), 143–150.
* Althammer et al. (2023)Sophia Althammer, Guido
Zuccon, Sebastian Hofstätter, Suzan
Verberne, and Allan Hanbury.
2023.Annotating Data for Fine-Tuning a Neural Ranker?
Current Active Learning Strategies are not Better than Random Selection. In*Proceedings of the Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval in the Asia
Pacific Region*. 139–149.
* Ardalani et al. (2022)Newsha Ardalani,
Carole-Jean Wu, Zeliang Chen,
Bhargav Bhushanam, and Adnan Aziz.
2022.Understanding Scaling Laws for Recommendation
Models.*arXiv preprint arXiv:2208.08489*(2022).
* Clark et al. (2022)Aidan Clark, Diego
De Las Casas, Aurelia Guy, Arthur
Mensch, Michela Paganini, Jordan
Hoffmann, Bogdan Damoc, Blake Hechtman,
Trevor Cai, Sebastian Borgeaud,
et al. 2022.Unified scaling laws for routed language models.
In *International Conference on Machine Learning*.
PMLR, 4057–4086.
* Clark et al. (2020)Kevin Clark, Minh-Thang
Luong, Quoc V Le, and Christopher D
Manning. 2020.Electra: Pre-training text encoders as
discriminators rather than generators.*arXiv preprint arXiv:2003.10555*(2020).
* Dehghani et al. (2023)Mostafa Dehghani, Josip
Djolonga, Basil Mustafa, Piotr
Padlewski, Jonathan Heek, Justin Gilmer,
Andreas Peter Steiner, Mathilde Caron,
Robert Geirhos, Ibrahim Alabdulmohsin,
et al. 2023.Scaling vision transformers to 22 billion
parameters. In *International Conference on Machine
Learning*. PMLR, 7480–7512.
* Gao and Callan (2021a)Luyu Gao and Jamie
Callan. 2021a.Condenser: a pre-training architecture for dense
retrieval.*arXiv preprint arXiv:2104.08253*(2021).
* Gao and Callan (2021b)Luyu Gao and Jamie
Callan. 2021b.Unsupervised corpus aware language model
pre-training for dense passage retrieval.*arXiv preprint arXiv:2108.05540*(2021).
* Gao et al. (2022)Luyu Gao, Xueguang Ma,
Jimmy Lin, and Jamie Callan.
2022.Precise zero-shot dense retrieval without relevance
labels.*arXiv preprint arXiv:2212.10496*(2022).
* Gao et al. (2023)Leo Gao, John Schulman,
and Jacob Hilton. 2023.Scaling laws for reward model overoptimization. In*International Conference on Machine Learning*.
PMLR, 10835–10866.
* Gelbukh and Sidorov (2001)Alexander Gelbukh and
Grigori Sidorov. 2001.Zipf and Heaps laws’ coefficients depend on
language. In *Computational Linguistics and
Intelligent Text Processing: Second International Conference, CICLing 2001
Mexico City, Mexico, February 18–24, 2001 Proceedings 2*. Springer,
332–335.
* Guo et al. (2021)Jiafeng Guo, Yinqiong
Cai, Yixing Fan, Fei Sun,
Ruqing Zhang, and Xueqi Cheng.
2021.Semantic models for the first-stage retrieval: A
comprehensive review.*arXiv preprint arXiv:2103.04831*(2021).
* Hestness et al. (2017)Joel Hestness, Sharan
Narang, Newsha Ardalani, Gregory Diamos,
Heewoo Jun, Hassan Kianinejad,
Md Mostofa Ali Patwary, Yang Yang, and
Yanqi Zhou. 2017.Deep learning scaling is predictable, empirically.*arXiv preprint arXiv:1712.00409*(2017).
* Hoffmann et al. (2022)Jordan Hoffmann, Sebastian
Borgeaud, Arthur Mensch, Elena
Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne
Hendricks, Johannes Welbl, Aidan Clark,
Tom Hennigan, Eric Noland,
Katie Millican, George van den Driessche,
Bogdan Damoc, Aurelia Guy,
Simon Osindero, Karen Simonyan,
Erich Elsen, Jack W. Rae,
Oriol Vinyals, and Laurent Sifre.
2022.Training Compute-Optimal Large Language Models.arXiv:2203.15556 [cs.CL]
* Hofstätter et al. (2021)Sebastian Hofstätter,
Sheng-Chieh Lin, Jheng-Hong Yang,
Jimmy Lin, and Allan Hanbury.
2021.Efficiently teaching an effective dense retriever
with balanced topic aware sampling. In *Proceedings
of the 44th International ACM SIGIR Conference on Research and Development in
Information Retrieval*. 113–122.
* Izacard et al. (2021)Gautier Izacard, Mathilde
Caron, Lucas Hosseini, Sebastian Riedel,
Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2021.Towards unsupervised dense information retrieval
with contrastive learning.*arXiv preprint arXiv:2112.09118*(2021).
* Jia et al. (2021)Chao Jia, Yinfei Yang,
Ye Xia, Yi-Ting Chen,
Zarana Parekh, Hieu Pham,
Quoc Le, Yun-Hsuan Sung,
Zhen Li, and Tom Duerig.
2021.Scaling up visual and vision-language
representation learning with noisy text supervision. In*International conference on machine learning*.
PMLR, 4904–4916.
* Kaplan et al. (2020)Jared Kaplan, Sam
McCandlish, Tom Henighan, Tom B Brown,
Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford,
Jeffrey Wu, and Dario Amodei.
2020.Scaling laws for neural language models.*arXiv preprint arXiv:2001.08361*(2020).
* Khattab and Zaharia (2020)Omar Khattab and Matei
Zaharia. 2020.Colbert: Efficient and effective passage search via
contextualized late interaction over bert. In*Proceedings of the 43rd International ACM SIGIR
conference on research and development in Information Retrieval*.
39–48.
* Lansey and Bukiet (2009)Jonathan C Lansey and
Bruce Bukiet. 2009.Internet Search Result Probabilities: Heaps’ Law
and Word Associativity.*Journal of Quantitative Linguistics*16, 1 (2009),
40–66.
* Lin et al. (2021)Jimmy Lin, Rodrigo
Nogueira, and Andrew Yates.
2021.Pretrained transformers for text ranking: Bert and
beyond.*Synthesis Lectures on Human Language
Technologies* 14, 4
(2021), 1–325.
* Liu and Shao (2022)Zheng Liu and Yingxia
Shao. 2022.RetroMAE: Pre-training Retrieval-oriented
Transformers via Masked Auto-Encoder.*arXiv preprint arXiv:2205.12035*(2022).
* Lü et al. (2010)Linyuan Lü, Zi-Ke
Zhang, and Tao Zhou. 2010.Zipf’s law leads to Heaps’ law: Analyzing their
relation in finite-size systems.*PloS one* 5,
12 (2010), e14139.
* Luan et al. (2021)Yi Luan, Jacob
Eisenstein, Kristina Toutanova, and
Michael Collins. 2021.Sparse, dense, and attentional representations for
text retrieval.*Transactions of the Association for
Computational Linguistics* 9 (2021),
329–345.
* Ma et al. (2022)Xinyu Ma, Jiafeng Guo,
Ruqing Zhang, Yixing Fan, and
Xueqi Cheng. 2022.Pre-Train a Discriminative Text Encoder for Dense
Retrieval via Contrastive Span Prediction. In*Proceedings of the 45th International ACM SIGIR
Conference on Research and Development in Information Retrieval* (Madrid,
Spain) *(SIGIR ’22)*. Association
for Computing Machinery, New York, NY, USA,
848–858.[https://doi.org/10.1145/3477495.3531772](https://doi.org/10.1145/3477495.3531772 "")
* Ma et al. (2023)Yixiao Ma, Yueyue Wu,
Qingyao Ai, Yiqun Liu,
Yunqiu Shao, Min Zhang, and
Shaoping Ma. 2023.Incorporating Structural Information into Legal
Case Retrieval.*ACM Transactions on Information Systems*42, 2 (2023),
1–28.
* Newman (2005)Mark EJ Newman.
2005.Power laws, Pareto distributions and Zipf’s law.*Contemporary physics* 46,
5 (2005), 323–351.
* Nguyen et al. (2016)Tri Nguyen, Mir
Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and
Li Deng. 2016.MS MARCO: A human generated machine reading
comprehension dataset. In *CoCo@ NIPS*.
* Ni et al. (2021)Jianmo Ni, Chen Qu,
Jing Lu, Zhuyun Dai,
Gustavo Hernández Ábrego, Ji Ma,
Vincent Y Zhao, Yi Luan,
Keith B Hall, Ming-Wei Chang,
et al. 2021.Large dual encoders are generalizable retrievers.*arXiv preprint arXiv:2112.07899*(2021).
* Nogueira et al. (2019)Rodrigo Nogueira, Jimmy
Lin, and AI Epistemic. 2019.From doc2query to docTTTTTquery.*Online preprint* 6
(2019), 2.
* Pham et al. (2023)Hieu Pham, Zihang Dai,
Golnaz Ghiasi, Kenji Kawaguchi,
Hanxiao Liu, Adams Wei Yu,
Jiahui Yu, Yi-Ting Chen,
Minh-Thang Luong, Yonghui Wu,
et al. 2023.Combined scaling for zero-shot transfer learning.*Neurocomputing* 555
(2023), 126658.
* Qu et al. (2020)Yingqi Qu, Yuchen Ding,
Jing Liu, Kai Liu,
Ruiyang Ren, Wayne Xin Zhao,
Daxiang Dong, Hua Wu, and
Haifeng Wang. 2020.RocketQA: An optimized training approach to dense
passage retrieval for open-domain question answering.*arXiv preprint arXiv:2010.08191*(2020).
* Radford et al. (2021)Alec Radford, Jong Wook
Kim, Chris Hallacy, Aditya Ramesh,
Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell,
Pamela Mishkin, Jack Clark,
et al. 2021.Learning transferable visual models from natural
language supervision. In *International conference
on machine learning*. PMLR, 8748–8763.
* Radford et al. (2023)Alec Radford, Jong Wook
Kim, Tao Xu, Greg Brockman,
Christine McLeavey, and Ilya
Sutskever. 2023.Robust speech recognition via large-scale weak
supervision. In *International Conference on
Machine Learning*. PMLR, 28492–28518.
* Rosa et al. (2022)Guilherme Moraes Rosa,
Luiz Bonifacio, Vitor Jeronymo,
Hugo Abonizio, Marzieh Fadaee,
Roberto Lotufo, and Rodrigo Nogueira.
2022.No parameter left behind: How distillation and
model size affect zero-shot retrieval.*arXiv preprint arXiv:2206.02873*(2022).
* Santhanam et al. (2021)Keshav Santhanam, Omar
Khattab, Jon Saad-Falcon, Christopher
Potts, and Matei Zaharia.
2021.ColBERTv2: Effective and Efficient Retrieval via
Lightweight Late Interaction.arXiv:2112.01488 [cs.IR]
* Shao et al. (2023)Yunqiu Shao, Yueyue Wu,
Yiqun Liu, Jiaxin Mao, and
Shaoping Ma. 2023.Understanding Relevance Judgments in Legal Case
Retrieval.*ACM Transactions on Information Systems*41, 3 (2023),
1–32.
* Strubell et al. (2019)Emma Strubell, Ananya
Ganesh, and Andrew McCallum.
2019.Energy and Policy Considerations for Deep Learning
in NLP. In *Proceedings of the 57th Annual
Meeting of the Association for Computational Linguistics*,
Anna Korhonen, David
Traum, and Lluís Màrquez (Eds.).
Association for Computational Linguistics,
Florence, Italy, 3645–3650.[https://doi.org/10.18653/v1/P19-1355](https://doi.org/10.18653/v1/P19-1355 "")
* Xie et al. (2023)Xiaohui Xie, Qian Dong,
Bingning Wang, Feiyang Lv,
Ting Yao, Weinan Gan,
Zhijing Wu, Xiangsheng Li,
Haitao Li, Yiqun Liu, et al.2023.T2Ranking: A large-scale Chinese Benchmark for
Passage Ranking.*arXiv preprint arXiv:2304.03679*(2023).
* Xiong et al. (2020)Lee Xiong, Chenyan Xiong,
Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul Bennett,
Junaid Ahmed, and Arnold Overwijk.
2020.Approximate nearest neighbor negative contrastive
learning for dense text retrieval.*arXiv preprint arXiv:2007.00808*(2020).
* Zhai et al. (2022)Xiaohua Zhai, Alexander
Kolesnikov, Neil Houlsby, and Lucas
Beyer. 2022.Scaling vision transformers. In*Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition*. 12104–12113.
* Zhan et al. (2021a)Jingtao Zhan, Jiaxin Mao,
Yiqun Liu, Jiafeng Guo,
Min Zhang, and Shaoping Ma.
2021a.Optimizing dense retrieval model training with hard
negatives. In *Proceedings of the 44th
International ACM SIGIR Conference on Research and Development in Information
Retrieval*. 1503–1512.
* Zhan et al. (2021b)Jingtao Zhan, Jiaxin Mao,
Yiqun Liu, Jiafeng Guo,
Min Zhang, and Shaoping Ma.
2021b.Optimizing dense retrieval model training with hard
negatives. In *Proceedings of the 44th
International ACM SIGIR Conference on Research and Development in Information
Retrieval*. 1503–1512.
* Zhang et al. (2023)Gaowei Zhang, Yupeng Hou,
Hongyu Lu, Yu Chen,
Wayne Xin Zhao, and Ji-Rong Wen.
2023.Scaling Law of Large Sequential Recommendation
Models.*arXiv preprint arXiv:2311.11351*(2023).
