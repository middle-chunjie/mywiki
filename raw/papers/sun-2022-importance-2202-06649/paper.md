On the Importance of Building High-quality Training Datasets for Neural Code Search
====================================================================================

Zhensu Sun[zhensuuu@gmail.com](mailto:zhensuuu@gmail.com)Monash UniversityMelbourneVictoriaAustralia,Li Li[1853549@tongji.edu.cn](mailto:1853549@tongji.edu.cn)Tongji UniversityShanghaiChina,Yan Liu[yanliu.sse@tongji.edu.cn](mailto:yanliu.sse@tongji.edu.cn)Tongji UniversityShanghaiChina,Xiaoning Du[xiaoning.du@monash.edu](mailto:xiaoning.du@monash.edu)Monash UniversityMelbourneVictoriaAustraliaandLi Li[li.li@monash.edu](mailto:li.li@monash.edu)Monash UniversityMelbourneVictoriaAustralia

(2022)

###### Abstract.

The performance of neural code search is significantly influenced by the quality of the training data from which the neural models are derived.
A large corpus of high-quality query and code pairs is demanded to establish a precise mapping from the natural language to the programming language.
Due to the limited availability, most widely-used code search datasets are established with compromise, such as using code comments as a replacement of queries.
Our empirical study on a famous code search dataset reveals that over one-third of its queries contain noises that make them deviate from natural user queries.
Models trained through noisy data are faced with severe performance degradation when applied in real-world scenarios.
To improve the dataset quality and make the queries of its samples semantically identical to real user queries is critical for the practical usability of neural code search.
In this paper, we propose a data cleaning framework consisting of two subsequent filters: a rule-based syntactic filter and a model-based semantic filter.
This is the first framework that applies semantic query cleaning to code search datasets.
Experimentally, we evaluated the effectiveness of our framework on two widely-used code search models and three manually-annotated code retrieval benchmarks.
Training the popular DeepCS model with the filtered dataset from our framework improves its performance by 19.2% MRR and 21.3% Answer@1, on average with the three validation benchmarks.

Code search, dataset, data cleaning, deep learning

††journalyear: 2022††copyright: acmcopyright††conference: 44th International Conference on Software Engineering; May 21–29, 2022; Pittsburgh, PA, USA††booktitle: 44th International Conference on Software Engineering (ICSE ’22), May 21–29, 2022, Pittsburgh, PA, USA††price: 15.00††doi: 10.1145/3510003.3510160††isbn: 978-1-4503-9221-1/22/05††ccs: Software and its engineering Reusability

1. Introduction
----------------

A semantic code search engine is a vital software development assistant, which significantly improves the development efficiency and quality.
With a description of the intended code functionality in natural language, a search engine can retrieve a list of semantically best-matched code snippets from its codebase.
Recently, deep learning (DL) has been widely applied in this area in view of its advantages in semantic modeling and understanding of languages. In the task of code search, DL models learn and represent the semantic mappings between the natural language and the programming language from query-code pairs.

<img src='x1.png' alt='Refer to caption' title='' width='442' height='195' />

*Figure 1. Statistics of 394,471 code comments used in CodeSearchNet (Java). The feature definitions are presented in [Section 3.1](#S3.SS1 "3.1. The Rule-based Syntactic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").*

Like many other DL tasks, code search models are data-hungry and require large-scale and high-quality training datasets.
Nevertheless, collecting a large set of query-code pairs is challenging, where the queries are supposed to be natural expressions from developers and the code to be a valid semantic match.
Instead, considering the scale and availability, code comments are popularly used as an alternative to the queries, many of which describe the core functionalities and with the corresponding code implementation rightly available.
To better understand the quality of datasets hence constructed, we investigated a Github dataset, CodeSearchNet (Java)*[20](#bib.bib20 "")*, which is popularly used in current code search research.
Surprisingly, we found a considerable amount of noise and unnaturalness in the queries of its data samples, which can hinder the training of high-quality models for practical usage.
As shown in[Fig. 1](#S1.F1 "In 1. Introduction ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search"), one-third of its queries contain text features (see Table[1](#S1.T1 "Table 1 ‣ 1. Introduction ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search") for examples of different features) that hardly exist in actual user queries.
The features are summarized based on our observations of the dataset, and may not be sufficient.
Comments may also be used for other purposes, such as copyright and to-do, instead of describing the core functionalities, thus shall not be seen as queries.
The proportion of noise data can be higher than one-third.

Code search models trained with noisy queries will face severe performance degradation when dealing with actual user queries.
The gap between the collected comment-code pairs and the natural user queries violates the basic assumption of learning algorithms that the training data and the evaluation data share a similar distribution.
It is also noteworthy that evaluating the model with a noisy comment-code benchmark can hardly reflect how useful the model would be in practice, and, even worse, may bring non-negligible bias to the model design, evaluation and application.
Many other researchers*[8](#bib.bib8 ""), [31](#bib.bib31 ""), [45](#bib.bib45 "")* also point out the misalignment between code comments and natural user queries, and report it as a threat to the validity of their approaches.
As mentioned in*[32](#bib.bib32 ""), [58](#bib.bib58 "")*, improving the quality of the training data is still a research opportunity for machine learning, including DL-based code search models.
Considering that there are still plenty of comments close to actual user queries and naturally paired with high-quality code snippets, a promising solution is to filter out the noisy ones.
Manual filtering can produce the most accurate results but is hardly practical for large-scale datasets.
Automated data cleaning methods are of demanding needs.

Queries for code retrieval possess specific *syntactic* and *semantic* characteristics, which can be utilized as key features to distinguish genuine user queries from noise.
Typical syntactic features include text attributes such as keywords, sentence length, and language type. Semantic features are related to the intention underneath the text expression, which usually describes the computational functionality of code snippets and might be influenced by the design convention of common program APIs.
Compared with syntactic features, semantic features are more abstract, implicit, and hard to be matched by simple rules.
Recently, some initial efforts have emerged on query quality enhancement, but primarily focusing on the regularization of syntactic features.
Simple filtering heuristics are proposed, based on the appearance of verb and noun phrases*[31](#bib.bib31 "")*, keywords uncommonly used in queries*[8](#bib.bib8 "")*, and constraints on the query length*[20](#bib.bib20 ""), [31](#bib.bib31 "")*.

However, the improvement in data quality is limited.
As declared in*[20](#bib.bib20 "")*, the collected dataset is still noisy despite their data cleaning efforts.
The proposed rules are not sufficient to cover the various syntactic violations, let alone the semantic misalignment.
For example, warning messages such as “Use of this property requires Java 6” widely exist in the code comments, but few code queries would request this way.
Hence, a remaining challenge is recognizing the code comments that are syntactically valid but encode semantics rarely seen in natural user queries.

To tackle this challenge, we propose an automated and effective data cleaning framework that distills high-quality queries from generally collected code comments on a large scale.
The framework is orthogonal to the design of code search algorithms and could be integrated with any of them to improve the quality of the training dataset.
Basically, it encompasses two subsequent filters: *a rule-based syntactic filter* and *a model-based semantic filter*.
The rule-based filter includes a set of systematically designed heuristic rules and weeds out data with anomalous syntactic features, e.g., HTML tags and Javadoc tags.
It is developed to cover a diverse range of syntactic violations, and each member inside is validated to reduce the noises effectively.
It is also extensible to fulfill the specific requirements for the dataset based on the applications.
The model-based semantic filter further refines the dataset produced by the rule-based filter and retains the comments that are semantically close to the natural queries.
The filter relies on a bootstrap query corpus, a set of high-quality queries, which represents how semantically the queries should look.
It learns the semantic features of the corpus, such as the expression style and topic, with a DL model, and leverages it to identify samples with similar semantics.
The bootstrap query corpus could be constructed with any trusted sources of natural user queries, and we formulate it with question titles from StackOverflow in this work.
These titles are an ideal approximation of natural queries and could be re-used by related studies.
Then, a Variational Auto-Encoder*[23](#bib.bib23 "")* is trained with the bootstrap query corpus, which maps the inputs into a latent space and attempts to reconstruct the original inputs solely based on the latent features.
The reconstruction loss reflects “how far away” an input is from the training data distribution, i.e., the distribution of queries in the bootstrap query corpus.
The lower the reconstruction loss, the more qualified an input is as a natural query.
We compute the reconstruction loss for each code comment in the raw dataset and cluster them into two groups.
The group of qualified queries is retained for training, and the group of noises is discarded.

*Table 1. Examples of syntactic rules.*

| Syntax Feature | Rule Action | Example |
| --- | --- | --- |
| HTML tags | Partly Remove | <p>parse line</p> |
| Parentheses | Partly Remove | (TODO) Send requests |
| Javadoc tags | Fully Remove | Returns a {@link Support} |
| URLs | Fully Remove | See https://github.com/ |
| | Non-English | | --- | | Languages | | Fully Remove | 创建临时文件 |
| Punctuation | Fully Remove | \=\=\=\=\=\=\=\=\=\=\=\=\=\= |
| Interrogation | Fully Remove | Is this a name declaration? |
| Short Sentence | Fully Remove | DEPRECATED |

To evaluate the effectiveness of our data cleaning framework, we compare the performance of code search models trained with datasets before and after the filtering.
One training dataset, two neural models, and three manually annotated validation datasets are used in the experiments, and our framework brings a significant performance improvement under all settings.
In particular, the performance of the popular DeepCS*[16](#bib.bib16 "")* model is improved by 19.2% MRR and 21.3% Answer@1, on average with the three validation datasets.
More importantly, with less training data used after the filtering, we also save the training time and computation resources.
Further, we carry out a comprehensive ablation study to validate the usefulness of each filter component and each rule and manually inspect the quality of the rejected and retained data.
Finally, we release the implementation of our framework, *NLQF*, and a cleaned code search dataset, *COFIC*, to facilitate future research.
The source code and datasets are available at <https://github.com/v587su/NLQF>.

To the best of our knowledge, this is the first systematic data cleaning framework for comment-based code search datasets. Our main contributions include:

* •

    A two-step data cleaning framework for code search datasets, which bridges the gap between code comments and natural user queries, both syntactically and semantically.

* •

    Implementation of the framework as a Python library for the code search task in academia and industry.

* •

    A comprehensive evaluation of our framework’s effectiveness, which demonstrates significant model performance improvement on three manually-annotated validation benchmarks.

* •

    The first systematically distilled Github dataset for neural code search, containing over one million comment-code pairs.

2. Preliminaries
-----------------

We prepare readers with the primary sources for collecting query-code pairs and the Variational Auto-Encoder, a major building block of our framework.

### 2.1. Data Source

A query for neural code search describes, in natural language, the functionality of the code snippets desired by users, e.g., “convert string to JSON object”.
The ideal data source for genuine code queries is the production data from existing neural code search engines.
However, these queries are not publicly accessible due to privacy and business sensitivity.
In academia, researchers use texts with similar intentions (e.g., code comments) as a replacement.
The primary alternative data sources for semantic code search research include GitHub and StackOverflow.

#### 2.1.1. Github

Github*[2](#bib.bib2 "")* is an open-source community, hosting more than 100 million repositories. It is the most popular platform for developers to share and manage their projects.
The large-scale well-maintained repositories on Github are a treasury for code reuse during development, thus naturally becoming the main retrieval source for code search tasks.
Moreover, mature projects are usually accompanied by canonical development documents.
According to Javadoc*[3](#bib.bib3 "")*, a code comment style guide, the first sentence of doc comments should be a summary sentence. Therefore, it is convenient to construct a code search dataset by collecting the code snippets paired with the first sentence of comments, forming the comment-code pairs. Javadoc-generated comments have hence been widely used in practice for various software engineering purposes*[28](#bib.bib28 ""), [34](#bib.bib34 ""), [27](#bib.bib27 ""), [26](#bib.bib26 "")* due to their large scale, ease of obtaining, and being close to
actual use scenarios.

However, developers write comments for their software projects without considering the retrieval purposes. Not all the comments properly map to queries.
As mentioned in[Section 1](#S1 "1. Introduction ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search"), the CodeSearchNet collected from Github contains plenty of anomalies that rarely exist in natural user queries.
It is not appropriate to include these comments in the dataset, and we call for more attention to be drawn to this problem.

#### 2.1.2. StackOverflow

StackOverflow*[5](#bib.bib5 "")* serves as a Q$\\&$A community specialized for software developers. It is a rich resource of software-related questions and answers.
When asking about codes or APIs for implementing a specific functionality, users would propose a question title to express their intention.
These are natural user queries with valid syntax and semantics.
Hence, researchers*[55](#bib.bib55 ""), [53](#bib.bib53 "")* also collect the titles of StackOverflow questions paired with proper answers containing sample code snippets, which also form the query-code pairs.
Others also evaluate their code search models with queries manually selected from StackOverflow*[8](#bib.bib8 ""), [16](#bib.bib16 ""), [33](#bib.bib33 "")*, and additional public evaluation benchmarks could be found in*[25](#bib.bib25 ""), [51](#bib.bib51 "")*.

Compared with the Github data source, queries from StackOverflow have a significant advantage of being closer to natural user queries, but the quality of code samples is hard to guarantee.
Hence, the dataset collected from StackOverflow is still not as desired.
Nevertheless, the query corpus is valuable.
It is worth investigating whether and how it could be leveraged to improve the other query-code datasets.

### 2.2. Variational Auto-Encoder

Variational Auto-Encoder (VAE)*[6](#bib.bib6 "")* is a neural model that learns the distribution of a set of data. A VAE model consists of an encoder and a decoder.
The encoder learns to map an input data $x$ into a prior distribution $p_{\theta}(z)$, from which a latent variable $z$ is sampled, and the decoder maps $z$ back to $\hat{x}$, a reconstruction of $x$. It is expensive to calculate $p_{\theta}(z)$ directly, so VAE introduces an approximate posterior $q_{\phi}(z|x)$.
$\theta$ and $\phi$ are parameters of the prior and the approximate posterior.

The loss function, Evidence Lower Bound (ELBO), which seeks to maximize the likelihood of reconstructing the original data and minimize the Kullback-Leibler (KL) divergence between the actual and estimated posterior distributions, is represented as:

| (1) |  | $\mathcal{L}\=\mathbb{E}_{q_{\phi}(z|x)}[-logp_{\theta}(x|z)]+KL(q_{\phi}(z|x)||p_{\theta}(z)),$ |  |
| --- | --- | --- | --- |

where $KL$ represents the KL divergence.
Theoretically, the distributions of $q_{\phi}(z|x)$ and $p_{\theta}(z)$ can be arbitrary. In practice, the Gaussian distribution is mostly adopted.

3. The Data Cleaning Framework
-------------------------------

<img src='x2.png' alt='Refer to caption' title='' width='212' height='152' />

*Figure 2. An overview of our data cleaning framework.*

This section introduces our automated and effective data cleaning framework for code search datasets, mainly to filter out query-code pairs with inappropriate queries.
The framework consists of two subsequent filters, the *rule-based syntactic filter* and *model-based semantic filter*.
An overview of the framework, when applied to clean the comment-code pairs collected from Github, is shown in[Fig. 2](#S3.F2 "In 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
The raw comment-code pairs are firstly cleaned by the rule-based filter, where a ruleset is applied to detect the existence of invalid query syntax.
Next, for the model-based filter, leveraging a small bootstrap query corpus as the semantics reference, a VAE model is trained to model its characteristics and further used to reject comments violating the natural query semantics.
Here we collect the bootstrap query corpus (no need of the paired code snippets) from StackOverflow, and more details can be found in[Section 4.1](#S4.SS1 "4.1. Datasets ‣ 4. Experiment setup ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
Such, we take advantage of both the Github and StackOverflow data sources and produce a large set of high-quality query-code pairs.
In the following, we elaborate on the design of the two filters.

### 3.1. The Rule-based Syntactic Filter

Code comments contain richer information than just descriptions on code functionalities and manifest various syntactic features rarely existing in actual code search quires.
For example, URLs are used for external references, and HTML tags are used in comments for documentation autogeneration.
To reduce such deviations from natural queries, we sampled 1% code comments from CodeSearchNet, manually inspected and summarized noises in these 3,949 instances.
We establish a black list of invalid syntax features to reject unqualified code comments.
If a comment matches any of these features, we remove the invalid parts if they are detachable; otherwise, we abandon this comment-code pair.

Based on a comparative observation of code comments and user queries, we develop a set of rules to precisely identify synthetically inappropriate queries and leave the fine-grained semantic check to the model-based filter.
To facilitate the management, we define three criteria that the ruleset must comply with: 1) any rule should define a unique and specific construction pattern, 2) the rules should be conservative and limit the preclusion of valid queries within an acceptable range, and 3) any rule is not a subrule of other rules in the set.
As a plug-in framework, the ruleset is extensible, and any rules that meet these criteria can be appended to the set.

We introduce the syntax features covered by our ruleset in the following, and their examples can be found in[Table 1](#S1.T1 "In 1. Introduction ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
We empirically decide whether to keep the content enclosed by a feature structure or not and validate the decisions with experiments (see our website*[4](#bib.bib4 "")* for more details) and manual inspection (see[Section 5.2](#S5.SS2 "5.2. RQ2: The impact of each filter component and each rule ‣ 5. Results ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search")).
From the results, our decisions help improve the naturalness and bring greater improvement to the model performance.

HTML tags HTML tags are used for documentation autogeneration in comments and should not appear in user queries. However, the content wrapped by the tags can still be informative.
Therefore, we remove the HTML tags from the comments but keep the wrapped content.

Parentheses Parentheses in comments are for adding supplementary information and do not appear in user queries. Due to such purpose, the removal of the content inside the parentheses does not have much influence on the completeness of the comments. We only retain the content outside of the parentheses.

Javadoc tags Javadoc tags starting with an “@” sign are special tags indicating a Javadoc comment. Such comments are only consumed by the Javadoc project for autogenerating well-formatted documentation. Considering that the special syntax of the tags may mislead code search models on natural language understanding, we reject all comments containing Javadoc tags.

URLs URLs in comments provide external references to relevant code snippets, but natural language queries do not contain any URLs. We reject all comments containing URLs.

Non-English Languages Non-English expressions exist as developers from different countries may write comments in their first languages. However, current code search models are not designed to handle multi-languages. We reject all non-English comments.

Punctuation Sometimes, punctuation symbols are used for section partitioning in the comments.
For example, developers use a row of equal signs ($\=$) or asterisks signs ($*$) (see examples in[Table 1](#S1.T1 "In 1. Introduction ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search")) to indicate a new section.
For effectiveness, we reject comments containing no English letters in our implementation.

Interrogation Based on our observation, some of the comments in the dataset are interrogative. Developers seem to use comments to communicate with their collaborators during the code review process.
There may be some sparse information about the code functionality, but the quality is hard to control.
We reject comments ending with a question mark.

Short Sentence The sentence length is a commonly used criterion for comment filtering.
Extremely short comments are not informative enough for code search models to establish their mapping to the corresponding code snippets.
We reject comments containing no more than two words.

### 3.2. The Model-based Semantic Filter

This section introduces the model-based semantic filter, which takes the initially cleaned comment-code pairs from the rule-based filter as input and further selects the pairs with comments semantically close to the queries in a pre-collected bootstrap query corpus.
We present the detailed design of the VAE model and discuss how it is used for filtering.

#### 3.2.1. The VAE Model

The two main components of a VAE model are the encoder and decoder, which are generally composed of deep neural networks.
Here, we use Gated Recurrent Unit (GRU)*[12](#bib.bib12 "")* for both the encoder and decoder in our VAE model.
GRU is a variant of Recurrent Neural Network (RNN), which enables the model to capture information from sequential texts. [Fig. 3](#S3.F3 "In 3.2.1. The VAE Model ‣ 3.2. The Model-based Semantic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search") illustrates an overview of the design of the model structure. Details about each layer are as follows.

<img src='x3.png' alt='Refer to caption' title='' width='272' height='128' />

  

*Figure 3. The structure of the Variational Auto-Encoder in the model-based filter. The dashed lines denote the propagation of hidden states in neural cells.*

Embedding Given a query $w_{0}w_{1}\dots w_{n}$ of length $n$, the $i$-th token is $w_{i}$.
The embedding layer is responsible for mapping each token into an embedding vector.
It consists of an embedding matrix $\textbf{E}\in\mathbb{R}^{o_{w}\times d}$, where $o_{w}$ is the vocabulary size of the query language and $d$ is the dimension of embedding vectors.
The matrix is initialized with random values and updated during training.

GRU Encoder We design the encoder of VAE as a bi-directional GRU.
Sequentially, it deals with the input tokens, and propagates the upstream and downstream context through the hidden states, respectively in the forward and backward directions, as shown in[Eq. 2](#S3.Ex1 "In 3.2.1. The VAE Model ‣ 3.2. The Model-based Semantic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search") and[Eq. 3](#S3.Ex2 "In 3.2.1. The VAE Model ‣ 3.2. The Model-based Semantic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
$emb$ maps a token to its embedding vector.
Finally, we sum up the last hidden states of both directions to get the final hidden state, as in[Eq. 4](#S3.Ex3 "In 3.2.1. The VAE Model ‣ 3.2. The Model-based Semantic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search"), and pass it to the next layer.

| (2) |  | $\displaystyle\overrightarrow{\textbf{h}_{i}}$ | $\displaystyle\=\overrightarrow{GRU}(emb(w_{i}),\overrightarrow{\textbf{h}_{i-1}})$ |  |
| --- | --- | --- | --- | --- |
| (3) |  | $\displaystyle\overleftarrow{\textbf{h}_{i}}$ | $\displaystyle\=\overleftarrow{GRU}(emb(w_{i}),\overleftarrow{\textbf{h}_{i+1}})$ |  |
| --- | --- | --- | --- | --- |
| (4) |  | h | $\displaystyle\=\overrightarrow{\textbf{h}_{n}}+\overleftarrow{\textbf{h}_{n}}$ |  |
| --- | --- | --- | --- | --- |

Latent Variable Based on the hidden state h from the encoder, we estimate the parameters of a Gaussian distribution with a fully-connected layer, which are the mean vector $\bm{\mu}$ and variance vector $\bm{\sigma}^{2}$.
The latent variable z is randomly sampled from this distribution.
The equations are as follows:

|  | $\displaystyle\bm{\mu};\bm{\sigma^{2}}$ | $\displaystyle\=FC(\textbf{h})$ |  |
| --- | --- | --- | --- |
|  | z | $\displaystyle\=\bm{\mu}+\textbf{r}\cdot\textbf{e}^{\bm{\sigma^{2}}/2}$ |  |
| --- | --- | --- | --- |

where $FC$ is a fully-connected layer and r is a random vector from the standard normal distribution.

GRU Decoder The latent variable represents the key features of the original input in a highly abstract and compact way.
The decoder works to reconstruct the input solely based on the latent variable. Iteratively, the decoder computes the hidden state $s_{i}$ at each step $i$ and reconstructs token $w_{i}^{\prime}$, based on the previous state $s_{i-1}$ (or z at step 0) and $w_{i-1}^{\prime}$ generated in the previous step.
The equations are as follows:

|  | $\displaystyle\textbf{s}_{i}$ | $\displaystyle\=\begin{cases}GRU(emb(bos),\textbf{z}))\&\text{i\=0}\\ GRU(emb(w_{i-1}^{\prime}),\textbf{s}_{i-1}))\&\text{i\textgreater{}0}\end{cases}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle p_{i}$ | $\displaystyle\=FC(\textbf{s}_{i})$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle w_{i}^{\prime}$ | $\displaystyle\=argmax(p_{i})$ |  |
| --- | --- | --- | --- |

where $bos$ is a special token indicating the start of a sentence, and $p_{i}\in\mathbb{R}^{o_{w}}$ represents the probability of $i$-th token to be generated.

Loss We measure the likelihood of reconstructing the original input with the Cross-Entropy (CE) loss. Hence, the ELBO loss introduced in [Section 2.2](#S2.SS2 "2.2. Variational Auto-Encoder ‣ 2. Preliminaries ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search") can be computed as:

|  | $\mathcal{L}\=-\frac{1}{n}\sum_{i\=1}^{n}CELoss(w_{i},w_{i}^{\prime})+KLDivergence(\bm{\mu},\bm{\sigma}^{2})$ |  |
| --- | --- | --- |

where $CELoss$ and $KLDivergence$ represents the calculation of the CE loss and KL divergence.

#### 3.2.2. The Filtering Algorithm

We train the VAE model with a set of high-quality code search queries collected from near-real scenarios, which we call the bootstrap query corpus.
After the training, the VAE model is able to recognize whether a query semantically resembles those in the corpus.
We measure the reconstruction loss, i.e., the CE loss, of an input when fed to the VAE model, which just reflects how well it is within the training set distribution.
Intuitively, the loss value is the anomaly score gauging how far an input stays away from the queries in bootstrap query corpus.
Comments with smaller losses are more likely to be query-appropriate.

To select comments resembling queries in bootstrap query corpus, we sort the comments based on their reconstruction losses, in ascending order, and retain the top-ranked ones.
It is tricky to decide an appropriate dividing point for retaining the portion with better quality and discarding the remaining.
The less data we keep from the top, the higher the dataset quality.
However, a sharp reduction in the data size hinders the performance of the trained code search model. [Fig. 4](#S3.F4 "In 3.2.2. The Filtering Algorithm ‣ 3.2. The Model-based Semantic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search") shows a theoretical model illustrating the relation between the dividing point and the model performance.
As the amount of retained data increases, the model performance firstly increases and then decreases after reaching the peak.
There is a trade-off between the quality and quantity of the dataset.

<img src='x4.png' alt='Refer to caption' title='' width='151' height='118' />

*Figure 4. An illustration of the relation between the portion of retained data and the performance of the code search model trained with it.*

We leverage an unsupervised clustering algorithm, EM-GMM (Expectation-Maximization for Gaussian Mixture Model)*[15](#bib.bib15 "")*, to decide the partition automatically.
It is widely used to model the mixed distributions of a dataset.
For our task, EM-GMM divides a set of comments into the qualified and the unqualified groups based on the reconstruction loss. For each group, GMM fits a Gaussian probability density function and mixes them together as the distribution of the whole dataset, which can be represented as:

|  | $\displaystyle P(x)\=\pi N(x|\mu_{q},\sigma_{q})+(1-\pi)N(x|\mu_{uq},\sigma_{uq})$ |  |
| --- | --- | --- |

where $\pi$ is the mixture coefficient for the qualified group, ($\mu_{q},\sigma_{q}$) and ($\mu_{uq},\sigma_{uq}$) are the parameters for the Gaussian probability density functions of the qualified and unqualified groups, respectively.
Finally, the EM algorithm*[15](#bib.bib15 "")* is applied to estimate a set of optimal values for all the parameters.

Note that, to establish a high-quality code search dataset, all comments are processed together with their paired code snippets.
Hence, we obtain a set of comment-code pairs after applying this semantic filter, where the comments are syntactically and semantically close to natural user queries.

4. Experiment setup
--------------------

*Table 2. The Answered@k and MRR scores of the DeepCS and CARLCS models trained over different datasets.*

| Model | Test Set | #Query | Train Set | #Pairs | Train Hours | A@1 | | A@5 | | A@10 | | MRR | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepCS | CSN-v | 434 | CSN-t (all) | 394,471 | 8 h | 123 |  | 248 |  | 306 |  | 0.407 |  |
| | | | CSN-t (controlled) | 192,031 | 4 h | 107 | | 235 | | 275 | | 0.376 | |
| CSN-t (filtered) | 192,031 | 4 h | 168 | 36.6$\%\uparrow$ | 299 | 20.6$\%\uparrow$ | 348 | 13.7$\%\uparrow$ | 0.512 | 26.0$\%\uparrow$ |
| CB | 52 | CSN-t (all) | 394,471 | 8 h | 25 |  | 31 |  | 38 |  | 0.522 |  |
| | | CSN-t (controlled) | 192,031 | 4 h | 20 | | 26 | | 28 | | 0.438 | |
| CSN-t (filtered) | 192,031 | 4 h | 29 | 16.0$\%\uparrow$ | 38 | 22.6$\%\uparrow$ | 40 | 5.3$\%\uparrow$ | 0.644 | 23.3$\%\uparrow$ |
| NCSED | 287 | CSN-t (all) | 394,471 | 8 h | 44 |  | 101 |  | 136 |  | 0.250 |  |
| | | CSN-t (controlled) | 192,031 | 4 h | 31 | | 90 | | 135 | | 0.210 | |
| CSN-t (filtered) | 192,031 | 4 h | 49 | 11.4$\%\uparrow$ | 110 | 8.9$\%\uparrow$ | 142 | 4.4$\%\uparrow$ | 0.271 | 8.4$\%\uparrow$ |
| CARLCS | CSN-v | 434 | CSN-t (all) | 394,471 | 6 h | 54 |  | 210 |  | 292 |  | 0.283 |  |
| | | | CSN-t (controlled) | 192,031 | 3 h | 54 | | 202 | | 284 | | 0.281 | |
| CSN-t (filtered) | 192,031 | 3 h | 62 | 14.8$\%\uparrow$ | 221 | 5.2$\%\uparrow$ | 296 | 1.4$\%\uparrow$ | 0.302 | 6.7$\%\uparrow$ |
| CB | 52 | CSN-t (all) | 394,471 | 6 h | 1 |  | 2 |  | 6 |  | 0.038 |  |
| | | CSN-t (controlled) | 192,031 | 3 h | 0 | | 1 | | 3 | | 0.012 | |
| CSN-t (filtered) | 192,031 | 3 h | 2 | 100.0$\%\uparrow$ | 4 | 100.0$\%\uparrow$ | 7 | 16.7$\%\uparrow$ | 0.056 | 49.4$\%\uparrow$ |
| NCSED | 287 | CSN-t (all) | 394,471 | 6 h | 20 |  | 34 |  | 57 |  | 0.105 |  |
| | | CSN-t (controlled) | 192,031 | 3 h | 15 | | 33 | | 49 | | 0.097 | |
| CSN-t (filtered) | 192,031 | 3 h | 35 | 75.0$\%\uparrow$ | 58 | 70.6$\%\uparrow$ | 74 | 29.8$\%\uparrow$ | 0.168 | 59.9$\%\uparrow$ |

We introduce the research questions, the basic experimental setup about the datasets and models, and the evaluation metrics used throughout the evaluation. The research questions we aim to answer include:

RQ1: How effective is our data cleaning framework?

RQ2: What is the impact of each filter component and each rule on the effectiveness of our framework?

RQ3: How is the dividing point determined by the clustering algorithm during the model-based filtering?

### 4.1. Datasets

Three types of datasets are involved in our evaluation, including the training and validation datasets used to train and assess the performance of code search models and the bootstrap query corpus used to develop our model-based filter.
To make the best use of existing resources, we focus on the Java programming language in this work, for which there have been the most public datasets and models in the field of neural code search.
Theoretically, our framework is language-independent and applicable to other programming languages with a proper adaptation of the filtering rules.

#### 4.1.1. Training Datasets

We use the popular CodeSearchNet (CSN)*[20](#bib.bib20 "")* dataset to train all the code search models.
CSN is a collection of datasets and benchmarks for semantic code retrieval.
It extracts functions and their paired comments from Github repositories.
It covers six programming languages, and we take the training dataset for Java, which contains 394,471 data points.
We took the first sentence of each comment.
In what follows, we denote it as CSN-t.
Another widely used dataset in DeepCS*[16](#bib.bib16 "")* is not included because the authors only released the processed data, but our framework cannot work without accessing the raw data.

#### 4.1.2. Validation Datasets

We utilize human-annotated validation datasets to evaluate how well a code search model performs in a real-world scenario, and three widely used datasets are adopted.
It is noteworthy that the validation datasets are *never* filtered by our framework in order to ensure the fairness of our experiments.
They are listed as follows:

* •

    CSN-v CSN also offers a validation benchmark for Java, containing query-code pairs collected from Bing and StackOverflow.
    Human annotators are hired to rate the relevance between the query and the code.
    Pairs with a score greater than <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S4.I1.i1.p1.1.m1.1"><semantics id="S4.I1.i1.p1.1.m1.1a"><mn id="S4.I1.i1.p1.1.m1.1.1" xref="S4.I1.i1.p1.1.m1.1.1.cmml">0</mn><annotation-xml encoding="MathML-Content" id="S4.I1.i1.p1.1.m1.1b"><cn id="S4.I1.i1.p1.1.m1.1.1.cmml" type="integer" xref="S4.I1.i1.p1.1.m1.1.1">0</cn></annotation-xml></semantics></math> -->00 are deemed as relevant, and there are 434 relevant pairs in total.
    In the dataset, each pair is accompanied by 999 distractor code snippets.
    It means, given a query, the code search model needs to retrieve the ground truth among 1000 candidates.

* •

    CB CosBench (CB)*[51](#bib.bib51 "")* is a validation dataset consisting of 52 selective queries from StackOverflow.
    For each query, the authors prepared around ten paring code snippets as its ground truths, including its best answer on StackOverflow and several other matched code snippets selected from GitHub.
    Additionally, there is a pool of 4,199,769 distractor code snippets.
    The model needs to search the ground truths from a mixture with the complete code pool given a query.

* •

    NCSED Proposed in*[25](#bib.bib25 "")*, the NCSED dataset contains 287 question queries manually collected from StackOverflow.
    For each query, there are around three pairing code snippets selected from GitHub.
    The ground truths are mixed with other 4,716,814 distractor code snippets collected from GitHub.
    The search model is required to retrieve the ground truths from the large corpus for a query.

The extremely
large search space in NCSED and CB makes it extremely hard for code search models to achieve a good performance, and the performances variations brought by data cleaning can also be too marginal to compare. Without loss of generality, for each query in NCSED and CB,
we construct 999 distractor snippets, following a similar fashion as CSN-v.

#### 4.1.3. Bootstrap Query Corpus

StackOverflow is an ideal source for collecting resemblers of actual user queries, though the quality of the pairing code snippets is hard to guarantee.
It becomes an optimal choice to establish the bootstrap query corpus.
We surveyed existing StackOverflow datasets in the code search field, and found that they were of severely limited size.
With their aim to collect high-quality question-code pairs, numerous questions were discarded due to the lack of qualified code answers.
Hence, to better facilitate the training of our VAE model, we determined to construct a question-only corpus from StackOverflow instead of using existing ones.

According to a study*[37](#bib.bib37 "")*, the StackOverflow questions can be divided into four types: “Debug/Corrective”, “Need-To-Know”, “How-To-Do-It” and “Seeking-Different-Solution”. Among them, questions of the “How-To-Do-It” type are most relevant to queries for the code search task.
Aiming to select the most qualified resemblers, we require the question titles to 1) start with “how to”, 2) be tagged with “Java”, and 3) pass the rule-based syntactic filter proposed in[Section 3.1](#S3.SS1 "3.1. The Rule-based Syntactic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search") (except for the Interrogation rule).
In the end, 168,779 out of 1,709,703 Java-related question titles were retained.
Afterwards, we transformed them into declarative sentences by removing the starting “how to” and the question marks if any, thus forming the bootstrap query corpus, which is used to train the VAE model later.

### 4.2. Code Search Model

Two code search models, DeepCS*[16](#bib.bib16 "")* and CARLCS*[42](#bib.bib42 "")*, are used in our experiments.
They are designed with representative architectures among most neural code search models.
DeepCS is based on the Siamese architecture, and CARLCS is an Interaction-based network*[36](#bib.bib36 "")*.
The Siamese architecture consists of two DL models to represent the query and code, respectively, with independent embedding vectors, and the similarity between these vectors is used to measure the relevance between query and code.
The Interaction-based network compares the query and code directly by generating an interaction matrix to reflect their relevance.

When training the models with our training dataset, we adopted the recommended settings for all the hyper-parameters, except for the training epoch of the DeepCS model.
In order to save some time and computation resources, we set the maximum training epoch of DeepCS to 100 instead of the recommended 500.
Without loss of fairness, the same setting has been used for training with the dataset either before or after the data cleaning.
This change should not affect the evaluation conclusion on the effectiveness of our framework, which focuses more on whether the model performance improves after removing the noises instead of its absolute level.

### 4.3. Evaluation Metrics

Two widely used metrics are adopted in our experiments to evaluate the code retrieval performance.

* •

    Answered@k: Answered@k (abbrev. A@k) is the number of queries answered by snippets in the top-k results.

* •

    Mean Reciprocal Rank (MRR): MRR is the average of the reciprocal ranks of the ground truth in the result list.

5. Results
-----------

In this section, we show the experimental results and answer the research questions.
Measures for both evaluation metrics are reported as the medium over five independent runs.

### 5.1. RQ1: Effectiveness

This experiment evaluates the effectiveness of our data cleaning solution as a pre-processing step when training neural code search models.
Specifically, one training dataset (CSN-t), two code search models (DeepCS and CARLCS), and three validation datasets (CSN-v, CB, and NCSED) are used in the evaluation.
Thus, we have six $(1\times 2\times 3)$ experimental settings in total.
During experiments, a relatively smaller filtered training set will be derived from CSN-t after our framework is applied for the data cleaning.
To also benchmark the performance variation brought by the size shrinking, we further derive a controlled training set by randomly selecting from CSN-t an equivalent number of data as the filtered set.
We observe the model performance resulted from training with these three datasets respectively.

The model performance is measured with four evaluation metrics, namely, A@1, A@5, A@10, and MRR, and the results are shown in[Table 2](#S4.T2 "In 4. Experiment setup ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
Under all the six experimental settings, our data cleaning framework demonstrates a positive influence on the model’s searching ability and helps it hit the best score.
On average of the three validation datasets, DeepCS trained over the filtered data outperforms the one trained over original data by 21.3% A@1, 17.4% A@5, 7.8% A@10, and 19.2% MRR.
Correspondingly, the improvements of CARLCS are 63.3% A@1, 58.6% A@5, 16.0% A@10, and 38.6% MRR.
Regarding the MRR on the three validation datasets, CSN-v,CB, and NCSED, DeepCS achieves 0.512, 0.644 and 0.271, and CARLCS achieves 0.302, 0.056 and 0.168, respectively.
Basically, DeepCS and CARLCS are boosted to their new best records, and CARLCS sees a greater improvement.
Note that the A@1 score of CARLCS over NCSED is increased by 75.0% (from 20 to 35), which is an extraordinary improvement.

Overall, with around half of the data quantity and half of the training time, models trained over the filtered data achieve a significant improvement on the number of answered queries and the rank of ground truth in search results.


*Table 3. Results of the ablation experiments on the filter components.*

| Model | Test Set | #Query | Train Set | #Pairs | A@1 | | A@5 | | A@10 | | MRR | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepCS | CSN-v | 434 | CSN-t (all) | 394,471 | 123 |  | 248 |  | 306 |  | 0.407 |  |
| | | | CSN-t (filtered) | 192,031 | 168 | | 299 | | 348 | | 0.512 | |
| Rule Filter only | 285,372 | 157 | 6.5$\%\downarrow$ | 283 | 5.4$\%\downarrow$ | 335 | 3.7$\%\downarrow$ | 0.490 | 4.3$\%\downarrow$ |
| Model filter only | 286,306 | 158 | 6.0$\%\downarrow$ | 276 | 7.7$\%\downarrow$ | 323 | 7.2$\%\downarrow$ | 0.491 | 4.2$\%\downarrow$ |
| CB | 52 | CSN-t (all) | 394,471 | 25 |  | 31 |  | 38 |  | 0.522 |  |
| | | CSN-t (filtered) | 192,031 | 29 | | 38 | | 40 | | 0.644 | |
| Rule Filter only | 285,372 | 28 | 3.4$\%\downarrow$ | 35 | 7.9$\%\downarrow$ | 38 | 5.0$\%\downarrow$ | 0.598 | 7.2$\%\downarrow$ |
| Model filter only | 286,306 | 24 | 17.2$\%\downarrow$ | 35 | 7.9$\%\downarrow$ | 38 | 5.0$\%\downarrow$ | 0.539 | 16.3$\%\downarrow$ |
| NCSED | 287 | CSN-t (all) | 394,471 | 44 |  | 101 |  | 136 |  | 0.250 |  |
| | | CSN-t (filtered) | 192,031 | 49 | | 110 | | 142 | | 0.271 | |
| Rule Filter only | 285,372 | 46 | 6.1$\%\downarrow$ | 106 | 3.6$\%\downarrow$ | 139 | 2.1$\%\downarrow$ | 0.265 | 2.3$\%\downarrow$ |
| Model filter only | 286,306 | 48 | 2.0$\%\downarrow$ | 106 | 3.6$\%\downarrow$ | 137 | 3.5$\%\downarrow$ | 0.264 | 2.4$\%\downarrow$ |
| CARLCS | CSN-v | 434 | CSN-t (all) | 394,471 | 54 |  | 210 |  | 292 |  | 0.283 |  |
| | | | CSN-t (filtered) | 192,031 | 62 | | 221 | | 296 | | 0.302 | |
| Rule Filter only | 285,372 | 57 | 8.1$\%\downarrow$ | 211 | 4.5$\%\downarrow$ | 288 | 2.7$\%\downarrow$ | 0.293 | 3.0$\%\downarrow$ |
| Model filter only | 286,306 | 57 | 8.1$\%\downarrow$ | 219 | 0.9$\%\downarrow$ | 294 | 0.7$\%\downarrow$ | 0.300 | 0.6$\%\downarrow$ |
| CB | 52 | CSN-t (all) | 394,471 | 1 |  | 2 |  | 6 |  | 0.038 |  |
| | | CSN-t (filtered) | 192,031 | 2 | | 4 | | 7 | | 0.056 | |
| Rule Filter only | 285,372 | 1 | 50.0$\%\downarrow$ | 2 | 50.0$\%\downarrow$ | 5 | 28.6$\%\downarrow$ | 0.039 | 31.2$\%\downarrow$ |
| Model filter only | 286,306 | 1 | 50.0$\%\downarrow$ | 3 | 25.0$\%\downarrow$ | 6 | 14.3$\%\downarrow$ | 0.049 | 14.0$\%\downarrow$ |
| NCSED | 287 | CSN-t (all) | 394,471 | 20 |  | 34 |  | 57 |  | 0.105 |  |
| | | CSN-t (filtered) | 192,031 | 35 | | 58 | | 74 | | 0.168 | |
| Rule Filter only | 285,372 | 18 | 48.6$\%\downarrow$ | 57 | 1.7$\%\downarrow$ | 74 | 0.0$\%\downarrow$ | 0.122 | 27.6$\%\downarrow$ |
| Model filter only | 286,306 | 21 | 40.0$\%\downarrow$ | 46 | 20.7$\%\downarrow$ | 63 | 14.9$\%\downarrow$ | 0.120 | 28.4$\%\downarrow$ |

### 5.2. RQ2: The impact of each filter component and each rule

We evaluate the effectiveness of each filter component with ablation experiments and conduct manual inspection on the queries accepted/rejected by each syntactic rule and the model-based filter to study their precision in identifying noises.

Each time, one of the two filter components is muted for the ablation experiments.
We observe the model performance after training with such derived filtered dataset and compare it with their previous performance (in [Section 5.1](#S5.SS1 "5.1. RQ1: Effectiveness ‣ 5. Results ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search")).
If the performance declines compared with when both filters are enabled, we can infer a positive impact of the muted component on the framework effectiveness.
We evaluate the performance of DeepCS and CARLCS trained under ablation and report the results in[Table 3](#S5.T3 "In 5.1. RQ1: Effectiveness ‣ 5. Results ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
The removal of any filter leads to worse performance scores.
Without the model-based filter, the A@1, A@5, A@10, and MRR scores of DeepCS on the three validation sets reduce by 5.4%, 5.6%, 3.6%, and 4.6% on average. CARLCS sees a much more severe deduction, and on average, A@1, A@5, A@10, and MRR decrease by 35.5%, 18.7%, 10.4%, and 20.6%.
After removing the rule-based filter, the performance of DeepCS averagely drops by 8.4% A@1, 6.4% A@5, 5.2% A@10, and 7.6% MRR. Meanwhile, the average reduction percentages of CARLCS on all the validation sets are 32.7% A@1, 15.5% A@5, 9.9% A@10, and 14.3% MRR.
It is noteworthy that the A@1 score of CARLCS on NCSED drops from 35 to 18 when the rule filter is muted, indicating that the ruleset plays a very influential part during the data cleaning.

For the manual inspection, two annotators, with over two years’ development experience, are hired to rate how likely a sentence is to be used as a code search query.
The rating score ranges from 0 to 2, where 0 means worst and 2 best.
There are 11 groups of data to annotate, including eight groups of comments rejected by each rule, the group of comments discarded by the model filer, the original CSN-t dataset, and the filtered dataset after the two-filter cleaning.
The last two groups are for comparison purposes.
For rules focusing on detachable features, i.e., the Parentheses and HTML tags, we let the annotators judge how well the removed part can help with a query expression.
We sample a subset of data from its full set for each group.
The sample size $ss$ of each group is computed by a statistical formula which is extracted from *[13](#bib.bib13 "")*,
$ss\=\frac{z^{2}*p*(1-p)/c^{2}}{1+\frac{z^{2}*p*(1-p)/(c^{2}-1)}{population}},$
where $population$ is the size of the entire dataset, $p$ is the standard deviation of the population, $c$ is the confidence interval (margin of error), $z$ is the Z-Score determined by the confidence level. In this experiment, we choose to work with a 95% confidence level (i.e., 1.96 Z-Score according to *[21](#bib.bib21 "")*), a standard deviation of 0.5 (0.5 is the maximum standard deviation, and a safe choice given the exact figure unknown), and a confidence interval of 5%.
We also measure the agreement between the two annotators with Cohen’s Kappa*[14](#bib.bib14 "")*, which is 0.69 and within the range of fair to good.

For each data, we finalize its score as the average of scores from the two annotators and display the statistics in [Table 4](#S5.T4 "In 5.2. RQ2: The impact of each filter component and each rule ‣ 5. Results ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
We report the number of data examined in each group, the respective portion of data scored as 0 or no less than 1, and the group’s average score, in the last four columns.
In general, the comments rejected by either the rule-based filter or the model-based filter poorly resemble real user queries, with 96.9% and 85.9% of them receiving a score of 0 and the average scores being as low as 0.04 and 0.20, respectively.
Still, it comes at an acceptable cost of losing a small set of good quality data, where 3.1% and 14.1% of the discarded data by the two filters score at least 1.
Each of the eight rules rejects code comments in an effective way, with four of them rejecting non-query-like data at 100% precision.
The precision of the Parentheses rule is relatively low, where 11% of the discarded data is of high quality.
In the future, when deciding whether the content inside the parentheses should be removed, a more refined rule can be derived.
Also, the model-based semantic filter is accompanied by a larger sacrifice, indicating it as a more challenging task.

*Table 4. Results of the manual inspection.*

| Type | Rule | # | Likeness score | | |
| --- | --- | --- | --- | --- | --- |
| | | | $\=$0 | $\geq$1 | Avg. |
| Origin | - | 394,471 | 79.0% | 21.0% | 0.27 |
| Discarded byRule Filer | HTML Tag | 32,989 | 100.0% | 0.0% | 0.00 |
| | Parentheses | 19,305 | 89.0% | 11.0% | 0.15 |
| Javadoc Tags | 47,106 | 94.9% | 5.1% | 0.08 |
| URLs | 640 | 97.3% | 2.7% | 0.05 |
| Non-Eng. Lan. | 6,503 | 100.0% | 0.0% | 0.00 |
| Punctuation | 39,032 | 100.0% | 0.0% | 0.00 |
| Interrogation | 516 | 100.0% | 0.0% | 0.00 |
| Short Sentence | 15,186 | 97.3% | 2.7% | 0.04 |
| In total | 161,277 | 96.9% | 3.1% | 0.04 |
| | Discarded by | | --- | | Model Filer | | - | 93,457 | 85.9% | 14.1% | 0.20 |
| Retained | - | 192,031 | 59.2% | 40.8% | 0.61 |

Overall, through the two-phase filtering, the average likeness score increases from 0.27 to 0.61.
In particular, the portion of non-query-like data drops from 79.0% to 59.2%, and the portion of highly query-like data scoring at least 1 improves from 21% to 40.8%.
There are still many comments inappropriate to be seen as code search queries, but our data cleaning framework makes a substantial contribution to alleviating the situation.
We call for more attention to be drawn to overcoming related challenges.


*Table 5. Results of changing the EM-GMM to other methods.*

| Model | Dividing Point | # | CSN-v (434) | | | | CB (52) | | | | NCSED (287) | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | A@1 | A@5 | A@10 | MRR | A@1 | A@5 | A@10 | MRR | A@1 | A@5 | A@10 | MRR |
| DeepCS | Percentile (25%) | 71,343 | 134 | 240 | 285 | 0.420 | 22 | 31 | 34 | 0.500 | 40 | 80 | 115 | 0.215 |
| | Percentile (50%) | 142,686 | 138 | 277 | 324 | 0.459 | 23 | 33 | 36 | 0.528 | 34 | 91 | 129 | 0.212 |
| KMeans (57.5%) | 164,194 | 146 | 274 | 321 | 0.471 | 23 | 34 | 39 | 0.530 | 39 | 99 | 127 | 0.229 |
| EM-GMM (67.3%) | 192,031 | 168 | 299 | 348 | 0.512 | 29 | 38 | 40 | 0.644 | 49 | 110 | 142 | 0.271 |
| Percentile (75%) | 214,029 | 160 | 284 | 332 | 0.505 | 28 | 36 | 38 | 0.604 | 47 | 108 | 138 | 0.266 |
| Percentile (100%) | 285,372 | 157 | 283 | 335 | 0.490 | 28 | 35 | 38 | 0.598 | 46 | 106 | 139 | 0.265 |
| CARLCS | Percentile (25%) | 71,343 | 51 | 190 | 270 | 0.264 | 0 | 2 | 6 | 0.031 | 6 | 23 | 35 | 0.053 |
| | Percentile (50%) | 142,686 | 57 | 217 | 290 | 0.295 | 1 | 2 | 5 | 0.031 | 8 | 32 | 48 | 0.070 |
| KMeans (57.5%) | 164,194 | 58 | 220 | 292 | 0.299 | 1 | 2 | 5 | 0.032 | 24 | 44 | 59 | 0.119 |
| EM-GMM (67.3%) | 192,031 | 62 | 221 | 296 | 0.302 | 2 | 4 | 7 | 0.056 | 35 | 58 | 74 | 0.168 |
| Percentile (75%) | 214,029 | 61 | 216 | 288 | 0.298 | 1 | 3 | 5 | 0.035 | 22 | 53 | 72 | 0.130 |
| Percentile (100%) | 285,372 | 57 | 211 | 288 | 0.293 | 1 | 2 | 5 | 0.039 | 18 | 57 | 74 | 0.122 |

### 5.3. RQ3: Quality of dividing point determined in the model-based filtering

In the model-based filter, we use EM-GMM to decide the dividing point between the qualified and the unqualified groups.
To assess the quality of the dividing point, we observe the model performance resulting from alternative dividing points, including fix proportions and the one decided by K-means, another widely used clustering algorithm.
For the fixed proportions, we set a 25% step and select 25%, 50%, 75%, and 100% top-ranked comments, respectively.

The results on DeepCS and CARLCS are reported in Table[5](#S5.T5 "Table 5 ‣ 5.2. RQ2: The impact of each filter component and each rule ‣ 5. Results ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").
For DeepCS, EM-GMM outperforms K-means and the fixed proportions on all the validation sets. Compared with the second-best partition, 75%, EM-GMM still achieves higher average performances by 4.3% A@1, 4.2% A@5, 4.3% A@10, and 3.3% MRR. The superiority is also observed on CARLCS at every metric, and EM-GMM outperforms K-means on average of CSN-v and NCSED by 30.4% A@1, 5.9% A@5, 2.8% A@10, and 15.0% MRR.

EM-GMM ultimately retains 192,031 data points, accounting for 67.3% of the original dataset,
which locates between 57.5%, the dividing point set by K-means and 75%.
As discussed in[Section 3.2.2](#S3.SS2.SSS2 "3.2.2. The Filtering Algorithm ‣ 3.2. The Model-based Semantic Filter ‣ 3. The Data Cleaning Framework ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search"), the relation between the data quantity and the model performance should be a convex function.
According to the property of the convex function, if there exists another optimal dividing point, it would locate between 57.5% and 75.0%.
Therefore, EM-GMM successfully identifies an optimal solution of the dividing point with an error less than 9.8% (calculated by $67.3\%-57.5\%$).


6. Application
---------------

This section presents the applications of our filtering framework, including a proof-of-concept data cleaning toolbox and a high-quality code search dataset.

### 6.1. *NLQF*: Natural Language Query Filter

We release the implementation of our filtering framework as a third-party Python library, Natural Language Query Filter (*NLQF*), which is designed to systemically filter queries for neural code search models.
As a lightweight library with convenient APIs, *NLQF* can be easily integrated into the development pipeline of any code search model. Besides, *NLQF* is extensible at several features to ensure its applicability in a wide range of contexts:

Extensible Ruleset The ruleset in *NLQF* is configurable, which enables users to specify the rules based on the characteristics of their own data. Besides, *NLQF* accepts user-defined functions as a part of rule-based filtering. One can easily extend the filter implementation by creating the filtering function for any new rule.

Open-source Filtering Model *NLQF* requires a trained VAE model in the model-based filter. We release the source code for training the VAE model used in this paper. Following the instructions, users can easily train a new model with their own bootstrap query corpus, which may boost the filtering performance further.

Tunable Dividing Proportion Besides the recommended clustering method, EM-GMM, *NLQF* also provides an interface accepting user-defined dividing points. Users can create their own method for finding the dividing point and configure *NLQF* to adopt it easily.

### 6.2. *COFIC*: Codebase Paired with Filtered Comments

We build and release a Codebase paired with Filtered Comments (*COFIC*) for Java programming language.

#### 6.2.1. Dataset Building

We collect the source code of Java repositories from Github according to the list maintained by Libraries.io*[38](#bib.bib38 "")*,
From these files, we extract the methods and corresponding comments using the scripts provided by CodeSearchNet*[20](#bib.bib20 "")*.
In the end, 2,475,692 raw comment-code pairs are obtained.
Through the processing with *NLQF*, there are 1,048,519 data points left in the cleaned query-code dataset.
Detailed statistics of the dataset during filtering are reported in[Table 6](#S6.T6 "In 6.2.1. Dataset Building ‣ 6.2. COFIC: Codebase Paired with Filtered Comments ‣ 6. Application ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").

*Table 6. The statistics during the data filtering.*

| Step | Rule | #Discarded | #Retained |
| --- | --- | --- | --- |
| Rule-based | HTML tags | 189,250 | 2,475,692 |
| | Parentheses | 129,130 | 2,475,692 |
| Javadoc tags | 423,313 | 2,052,379 |
| URLs | 3,119 | 2,049,260 |
| Non-English Languages | 67,943 | 1,981,317 |
| Punctuation | 201,881 | 1,779,436 |
| Interrogation | 3,300 | 1,776,136 |
| Short Sentence | 112,133 | 1,664,003 |
| Model-based | - | 615,484 | 1,048,519 |

*Table 7. A comparison between the training datasets for code search tasks.*

| Dataset | Source | Language | Likeness | # |
| --- | --- | --- | --- | --- |
| COFIC | Github | Java | 0.52 | 1 M |
| CSN (Java)[20](#bib.bib20 "") | Github | Java | 0.27 | 543 K |
| Hu et al.[19](#bib.bib19 "") | Github | Java | 0.48 | 69 K |
| StaQC[53](#bib.bib53 "") | StackOverflow | Python | 0.80 | 148 K |
| | | SQL | 0.80 | 120 K |
| Barone et al.[7](#bib.bib7 "") | Github | Python | 0.43 | 150 K |

#### 6.2.2. Dataset Comparison

We compare *COFIC*, on the query quality, with several other datasets currently used in neural code search research.
Following the same manual inspection convention as in[Section 5.2](#S5.SS2 "5.2. RQ2: The impact of each filter component and each rule ‣ 5. Results ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search"), the annotators rate the queries sampled from each dataset, reported in[Table 7](#S6.T7 "In 6.2.1. Dataset Building ‣ 6.2. COFIC: Codebase Paired with Filtered Comments ‣ 6. Application ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search"). Again, we measure the agreement level between the two annotators with Cohen’s Kappa, which is 0.73 and within the range of fair to good.
Among all the datasets collected from Github, *COFIC* receives the highest score on data quality, but there is still a gap compared with the StackOverflow dataset, StaQC.
Indeed, the datasets collected from StackOverflow have high-quality queries, but they suffer from the unstable code quality in answers*[46](#bib.bib46 ""), [56](#bib.bib56 "")*.
With our filtering framework, a Github dataset with better quality is established.

Besides the user study, we also experimentally compare *COFIC* with CSN-t. We train the DeepCS and CARLCS models with three datasets: CSN-t, *COFIC*, and a controlled *COFIC* (same size as CSN-t).
The model trained with *COFIC* outperforms other experimental settings on the three validation datasets (CSV-v, CB, and NCSED).
The detailed results are reported in [Table 8](#S6.T8 "In 6.2.2. Dataset Comparison ‣ 6.2. COFIC: Codebase Paired with Filtered Comments ‣ 6. Application ‣ On the Importance of Building High-quality Training Datasets for Neural Code Search").

*Table 8. The results of the experimental comparison between COFIC and CSN-t.*

| Model | Test Set | #Query | Train Set | #Pairs | A@1 | | A@5 | | A@10 | | MRR | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepCS | CSN-v | 434 | CSN-t | 394,471 | 123 |  | 248 |  | 306 |  | 0.407 |  |
| | | | COFIC (controlled) | 394,471 | 188 | 52.8$\%\uparrow$ | 297 | 19.8$\%\uparrow$ | 344 | 12.4$\%\uparrow$ | 0.555 | 36.3$\%\uparrow$ |
| COFIC | 1,048,519 | 191 | 55.3$\%\uparrow$ | 314 | 26.6$\%\uparrow$ | 354 | 15.7$\%\uparrow$ | 0.577 | 41.9$\%\uparrow$ |
| NCSED | 287 | CSN-t | 394,471 | 44 |  | 101 |  | 136 |  | 0.250 |  |
| | | COFIC (controlled) | 394,471 | 58 | 31.8$\%\uparrow$ | 109 | 7.9$\%\uparrow$ | 137 | 0.7$\%\uparrow$ | 0.281 | 12.4$\%\uparrow$ |
| COFIC | 1,048,519 | 72 | 63.6$\%\uparrow$ | 118 | 16.8$\%\uparrow$ | 148 | 8.8$\%\uparrow$ | 0.327 | 31.0$\%\uparrow$ |
| CB | 52 | CSN-t | 394,471 | 25 |  | 31 |  | 38 |  | 0.522 |  |
| | | COFIC (controlled) | 394,471 | 34 | 36.0$\%\uparrow$ | 38 | 22.6$\%\uparrow$ | 39 | 2.6$\%\uparrow$ | 0.693 | 32.6$\%\uparrow$ |
| COFIC | 1,048,519 | 38 | 52.0$\%\uparrow$ | 39 | 25.8$\%\uparrow$ | 41 | 7.9$\%\uparrow$ | 0.744 | 42.3$\%\uparrow$ |
| CARLCS | CSN-v | 434 | CSN-t | 394,471 | 54 |  | 210 |  | 292 |  | 0.283 |  |
| | | | COFIC (controlled) | 394,471 | 66 | 22.2$\%\uparrow$ | 237 | 12.9$\%\uparrow$ | 319 | 9.2$\%\uparrow$ | 0.328 | 15.8$\%\uparrow$ |
| COFIC | 1,048,519 | 69 | 27.8$\%\uparrow$ | 247 | 17.6$\%\uparrow$ | 322 | 10.3$\%\uparrow$ | 0.339 | 19.8$\%\uparrow$ |
| NCSED | 287 | CSN-t | 394,471 | 20 |  | 34 |  | 57 |  | 0.105 |  |
| | | COFIC (controlled) | 394,471 | 34 | 70.0$\%\uparrow$ | 68 | 100.0$\%\uparrow$ | 87 | 52.6$\%\uparrow$ | 0.172 | 64.1$\%\uparrow$ |
| COFIC | 1,048,519 | 47 | 135.0$\%\uparrow$ | 75 | 120.6$\%\uparrow$ | 92 | 61.4$\%\uparrow$ | 0.211 | 101.1$\%\uparrow$ |
| CB | 52 | CSN-t | 394,471 | 1 |  | 2 |  | 6 |  | 0.038 |  |
| | | COFIC (controlled) | 394,471 | 1 | - | 3 | 50.0$\%\uparrow$ | 7 | 16.7$\%\uparrow$ | 0.043 | 12.9$\%\uparrow$ |
| COFIC | 1,048,519 | 1 | - | 4 | 100.0$\%\uparrow$ | 7 | 16.7$\%\uparrow$ | 0.059 | 56.3$\%\uparrow$ |

7. Threats to validity
-----------------------

Rule Design Though our experiments have evaluated the usefulness of each rule in the ruleset; the rule-based filter may still introduce a few false positives or false negatives due to its design and implementation.
For example, the widely used query “quick sort” can be filtered out by the rule Short Sentences.
Besides, some rules are tricky to be implemented exactly in line with our aim.
For example, non-English letters in the comments are identified based on ASCII encoding. It may leave out several other languages also using English letters. But no English sentences will be falsely filtered out.
Overall, it requires further exploration on balancing the trade-off between precision and recall better.

Bootstrap Query Corpus The bootstrap query corpus in this work is built based on the questions on StackOverflow. Only titles starting with “how to” are collected into the corpus, which limits the sentence pattern. The VAE model trained over this corpus may not have a good tolerance to other patterns.
Besides, StackOverflow titles are also not fully query-appropriate.
Although we filter the titles by rules, there are still semantically irrelevant texts left.

Generalization Limited by the accessibility of models and evaluation benchmarks for code search tasks, we evaluate our solution only on Java datasets. In theory, our approach is capable of any comment-based code search dataset. Yet, the generalization of our filtering framework in different programming languages has not been experimentally verified. Besides, we only evaluate our filtering framework on two code search models, DeepCS and CARLCS, which is also a threat to the generalizability of our approach.

8. Related Work
----------------

Code Search Dataset Recent years have witnessed a growing interest in the semantic search for code snippets*[22](#bib.bib22 "")*.
DL models are applied to establish links between natural language and programming language. To train these models*[16](#bib.bib16 ""), [10](#bib.bib10 ""), [8](#bib.bib8 ""), [47](#bib.bib47 ""), [52](#bib.bib52 ""), [42](#bib.bib42 ""), [31](#bib.bib31 ""), [54](#bib.bib54 ""), [48](#bib.bib48 ""), [17](#bib.bib17 ""), [29](#bib.bib29 ""), [41](#bib.bib41 ""), [18](#bib.bib18 "")*, code snippets paired with comments are collected from Github*[7](#bib.bib7 ""), [16](#bib.bib16 ""), [19](#bib.bib19 ""), [47](#bib.bib47 ""), [20](#bib.bib20 "")*. According to a manual investigation*[40](#bib.bib40 "")*, there are 16 categories of comments in source code, most of which, e.g., TODO, License, and Exception, are not appropriate to serve as queries. However, to the best of our knowledge, the comments in code search datasets have never been fully cleaned. For example, Barone et al.*[7](#bib.bib7 "")* remove empty or non-alphanumeric lines from the docstrings. CodeSearchNet*[20](#bib.bib20 "")* filters each comment-code pair with its comment length. Ling et al.*[31](#bib.bib31 "")* use heuristic rules (e.g., the existence of verb and noun phrases) to filter comments. Cambronero et al.*[8](#bib.bib8 "")* filter out queries that contain specific keywords. These simple and scattered efforts are not enough to filter out the various noises, especially the texts that are semantically unrelated to real queries. Liu et al.*[32](#bib.bib32 "")* also mention that improving the data quality is still a research opportunity for deep-learning-based code search models, which well motivates our work.

There are two evaluation methods for neural code search research: train-test split and actual user query evaluation. A lot of works*[8](#bib.bib8 ""), [42](#bib.bib42 ""), [48](#bib.bib48 ""), [54](#bib.bib54 "")* split their datasets into train and test sets. The queries of their test set contain the same defects as the train set so that the results fail to reflect the model performance in an actual environment.
Manually reviewed queries*[25](#bib.bib25 ""), [20](#bib.bib20 ""), [51](#bib.bib51 "")* can overcome this problem but they are usually on a small scale and cannot serve as the training dataset.

Unsupervised Anomaly Detection Comments cleaning is an application of the unsupervised anomaly detection algorithm as labeled comments are non-trivial to obtain. Unsupervised anomaly detection algorithms identify the outliers solely based on the intrinsic properties of the data instances. Various techniques can be applied, such as Principal Component Analysis*[49](#bib.bib49 "")*, Generative Adversarial Network*[24](#bib.bib24 "")*, Spatio Temporal Networks*[11](#bib.bib11 "")* and LSTM*[43](#bib.bib43 "")*. Among them, Auto-Encoder (AE) is the fundamental architecture for unsupervised anomaly detection*[39](#bib.bib39 "")*. It has been applied in many tasks. For example, Zhang et al.*[57](#bib.bib57 "")* detect the rumors in social media using multi-layer AE. Castellini et al.*[9](#bib.bib9 "")* apply AE to detect false followers on Twitter. Luo and Nagarajan*[35](#bib.bib35 "")* use AE to identify the error events of interest such as equipment faults and undiscovered phenomena in wireless sensor networks.

The encoder of AE maps an input to a point in the latent space, while VAE maps an input to a region. In this way, VAE can extract more abstract semantic features. It has been applied to unsupervised anomaly detection with promising evaluation scores*[6](#bib.bib6 ""), [44](#bib.bib44 ""), [50](#bib.bib50 ""), [30](#bib.bib30 "")*.

9. Conclusion
--------------

We propose the first data cleaning framework for code search tasks, which improves the quality and naturalness of the queries.
The framework leverages two subsequent filters, the rule-based syntactic filter, and the model-based semantic filter.
The rule-based filter uses configurable heuristics rules to filter out comments with syntactic anomalies.
The model-based filter aims to refine the dataset semantically.
It trains a VAE model over a pre-collected bootstrap query corpus, and exploits it to select comments with smaller reconstruction losses. Experiments show that our filtering framework can significantly save computing resources and improve the model accuracy.
Finally, we release our framework as a Python library *NLQF* and make public a high-quality cleaned code search dataset *COFIC*, to facilitate relevant research in academia and industry.

References
----------

* [1]
* git [2021]2021.*Github*.Retrieved Sep 1, 2021 from <https://github.com/>
* jav [2021]2021.*How to Write Doc Comments for the Javadoc
Tool*.Retrieved Sep 1, 2021 from [https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html#styleguid](https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html#styleguid "")
* hqt [2021]2021.*On the Importance of Building High-quality
Training Datasets for Neural Code Search*.Retrieved Sep 1, 2021 from <https://sites.google.com/view/hqtd>
* sta [2021]2021.*StackOverflow*.Retrieved Sep 1, 2021 from <https://stackoverflow.com/>
* An and Cho [2015]Jinwon An and S. Cho.
2015.Variational Autoencoder based Anomaly Detection
using Reconstruction Probability.
* Barone and
Sennrich [2017]Antonio Valerio Miceli Barone and
Rico Sennrich. 2017.A Parallel Corpus of Python Functions and
Documentation Strings for Automated Code Documentation and Code Generation.*ArXiv* abs/1707.02275
(2017).
* Cambronero et al. [2019]José Cambronero,
Hongyu Li, S. Kim, K.
Sen, and S. Chandra. 2019.When deep learning met code search.*Proceedings of the 2019 27th ACM Joint
Meeting on European Software Engineering Conference and Symposium on the
Foundations of Software Engineering* (2019).
* Castellini
et al. [2017]Jacopo Castellini, V.
Poggioni, and Giulia Sorbi.
2017.Fake Twitter followers detection by denoising
autoencoder.*Proceedings of the International Conference
on Web Intelligence* (2017).
* Chen and Zhou [2018]Q. Chen and Minghui
Zhou. 2018.A Neural Framework for Retrieval and Summarization
of Source Code.*2018 33rd IEEE/ACM International Conference
on Automated Software Engineering (ASE)* (2018),
826–831.
* Chianucci and
Savakis [2016]Dan Chianucci and A.
Savakis. 2016.Unsupervised change detection using Spatial
Transformer Networks.*2016 IEEE Western New York Image and Signal
Processing Workshop (WNYISPW)* (2016),
1–5.
* Cho et al. [2014]Kyunghyun Cho, B. V.
Merrienboer, Çaglar Gülçehre,
Dzmitry Bahdanau, Fethi Bougares,
Holger Schwenk, and Yoshua Bengio.
2014.Learning Phrase Representations using RNN
Encoder-Decoder for Statistical Machine Translation.*ArXiv* abs/1406.1078
(2014).
* Cochran [1977]William G Cochran.
1977.*Sampling techniques*.Wiley Eastern Limited.
* Cohen and J. [1960]Cohen and J.
1960.A Coefficient of Agreement for Nominal Scales.*Educational $\\&$ Psychological Measurement*20, 1 (1960),
37–46.
* Dempster
et al. [1977]A. Dempster, N. Laird,
and D. Rubin. 1977.Maximum likelihood from incomplete data via the EM
- algorithm plus discussions on the paper.
* Gu et al. [2018]Xiaodong Gu, H. Zhang,
and S. Kim. 2018.Deep Code Search.*2018 IEEE/ACM 40th International Conference
on Software Engineering (ICSE)* (2018),
933–944.
* Haldar
et al. [2020]Rajarshi Haldar, L. Wu,
Jinjun Xiong, and J. Hockenmaier.
2020.A Multi-Perspective Architecture for Semantic Code
Search.*ArXiv* abs/2005.06980
(2020).
* Hu
et al. [2020]Gang Hu, Min Peng,
Yihan Zhang, Qianqian Xie, and
Mengting Yuan. 2020.Neural joint attention code search over structure
embeddings for software Q$\\&$A sites.*J. Syst. Softw.* 170
(2020), 110773.
* Hu
et al. [2018]X. Hu, G. Li,
Xin Xia, D. Lo, and
Zhi Jin. 2018.Deep Code Comment Generation.*2018 IEEE/ACM 26th International Conference
on Program Comprehension (ICPC)* (2018),
200–20010.
* Husain et al. [2019]H. Husain, Ho-Hsiang Wu,
Tiferet Gazit, Miltiadis Allamanis, and
Marc Brockschmidt. 2019.CodeSearchNet Challenge: Evaluating the State of
Semantic Code Search.*ArXiv* abs/1909.09436
(2019).
* Israel [1992]Glenn D Israel.
1992.Determining sample size.(1992).
* Kim et al. [2018]Kisub Kim, Dongsun Kim,
Tegawendé F Bissyandé, Eunjong
Choi, Li Li, Jacques Klein, and
Yves Le Traon. 2018.FaCoY - A Code-to-Code Search Engine. In*The 40th International Conference on Software
Engineering (ICSE 2018)*.
* Kingma and
Welling [2014]Diederik P. Kingma and
M. Welling. 2014.Auto-Encoding Variational Bayes.*CoRR* abs/1312.6114
(2014).
* Lawson
et al. [2017]W. Lawson, Esube Bekele,
and K. Sullivan. 2017.Finding Anomalies with Generative Adversarial
Networks for a Patrolbot.*2017 IEEE Conference on Computer Vision and
Pattern Recognition Workshops (CVPRW)* (2017),
484–485.
* Li
et al. [2019]Hongyu Li, S. Kim, and
S. Chandra. 2019.Neural Code Search Evaluation Dataset.*ArXiv* abs/1908.09804
(2019).
* Li
et al. [2016]Li Li, Tegawendé F
Bissyandé, Yves Le Traon, and
Jacques Klein. 2016.Accessing Inaccessible Android APIs: An Empirical
Study. In *The 32nd International Conference on
Software Maintenance and Evolution (ICSME 2016)*.
* Li
et al. [2018]Li Li, Jun Gao,
Tegawendé F Bissyandé, Lei Ma,
Xin Xia, and Jacques Klein.
2018.Characterising Deprecated Android APIs. In*The 15th International Conference on Mining
Software Repositories (MSR 2018)*.
* Li
et al. [2020a]Li Li, Jun Gao,
Tegawendé F Bissyandé, Lei Ma,
Xin Xia, and Jacques Klein.
2020a.CDA: Characterising Deprecated Android APIs.*Empirical Software Engineering (EMSE)*(2020).
* Li
et al. [2020b]W. Li, Haozhe Qin,
Shuhan Yan, Beijun Shen, and
Y. Chen. 2020b.Learning Code-Query Interaction for Enhancing Code
Searches.*2020 IEEE International Conference on
Software Maintenance and Evolution (ICSME)* (2020),
115–126.
* Lin et al. [2020]Shuyu Lin, R. Clark,
R. Birke, Sandro Schönborn,
Niki Trigoni, and S. Roberts.
2020.Anomaly Detection for Time Series Using VAE-LSTM
Hybrid Model.*ICASSP 2020 - 2020 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP)*(2020), 4322–4326.
* Ling
et al. [2020]Chunyang Ling, Zeqi Lin,
Yanzhen Zou, and Bing Xie.
2020.Adaptive Deep Code Search.*Proceedings of the 28th International
Conference on Program Comprehension* (2020).
* Liu
et al. [2020a]C. Liu, Xin Xia,
David Lo, Cuiyun Gao,
Xiaohu Yang, and J. Grundy.
2020a.Opportunities and Challenges in Code Search Tools.*ArXiv* abs/2011.02297
(2020).
* Liu
et al. [2020b]Chao Liu, Xin Xia,
David Lo, Zhiwei Liu, A.
Hassan, and Shanping Li.
2020b.Simplifying Deep-Learning-Based Model for Code
Search.*ArXiv* abs/2005.14373
(2020).
* Liu
et al. [2021]Pei Liu, Li Li,
Yichun Yan, Mattia Fazzini, and
John Grundy. 2021.Identifying and Characterizing Silently-Evolved
Methods in the Android API. In *The 43rd ACM/IEEE
International Conference on Software Engineering, SEIP Track (ICSE-SEIP
2021)*.
* Luo and Nagarajan [2018]Tie Luo and Sai Ganesh
Nagarajan. 2018.Distributed Anomaly Detection Using Autoencoder
Neural Networks in WSN for IoT.*2018 IEEE International Conference on
Communications (ICC)* (2018), 1–6.
* Mitra
et al. [2018]Bhaskar Mitra, Nick
Craswell, et al. 2018.*An introduction to neural information
retrieval*.Now Foundations and Trends.
* Nasehi
et al. [2012]Seyed Mehdi Nasehi,
Jonathan Sillito, F. Maurer, and
C. Burns. 2012.What makes a good code example?: A study of
programming Q$\\&$A in StackOverflow.*2012 28th IEEE International Conference on
Software Maintenance (ICSM)* (2012),
25–34.
* Nesbitt and
Nickolls [2017]A. Nesbitt and Benjamin
Nickolls. 2017.Libraries.io Open Source Repository and Dependency
Metadata.
* of Sydney et al. [2019]Raghavendra Chalapathy University of
Sydney, Capital Markets Cooperative Research Centre,
Sanjay Chawla Qatar Computing Research Institute, and
Hbku. 2019.Deep Learning for Anomaly Detection: A Survey.
* Pascarella et al. [2019]Luca Pascarella, Magiel
Bruntink, and Alberto Bacchelli.
2019.Classifying code comments in Java software
systems.*Empirical Software Engineering*24, 3 (June
2019), 1499–1537.[https://doi.org/10.1007/s10664-019-09694-w](https://doi.org/10.1007/s10664-019-09694-w "")
* Qihao
et al. [2020]Zhu Qihao, Sun Ze-yu,
Liang Xiran, Xiong Yingfei, and
Z. Lu. 2020.OCoR: An Overlapping-Aware Code Retriever.*arXiv: Computation and Language*(2020).
* Shuai
et al. [2020]Jianhang Shuai, Ling Xu,
Chao Liu, Meng Yan, Xin
Xia, and Yan Lei. 2020.Improving Code Search with Co-Attentive
Representation Learning.*Proceedings of the 28th International
Conference on Program Comprehension* (2020).
* Singh [2017]A. Singh. 2017.Anomaly Detection for Temporal Data using Long
Short-Term Memory (LSTM).
* Suh
et al. [2016]Suwon Suh, Daniel H.
Chae, Hyon-Goo Kang, and S. Choi.
2016.Echo-state conditional variational autoencoder for
anomaly detection.*2016 International Joint Conference on Neural
Networks (IJCNN)* (2016), 1015–1022.
* Sun
et al. [2020]Zhensu Sun, Yan Liu,
Chen Yang, and Yu Qian.
2020.PSCS: A Path-based Neural Model for Semantic Code
Search.*arXiv preprint arXiv:2008.03042*(2020).
* Terragni
et al. [2016]Valerio Terragni, Yepang
Liu, and S. C. Cheung. 2016.CSNIPPEX: automated synthesis of compilable code
snippets from Q\&A sites.*Proceedings of the 25th International
Symposium on Software Testing and Analysis* (2016).
* Wan
et al. [2019]Yao Wan, Jingdong Shu,
Yulei Sui, Guandong Xu,
Zhou Zhao, Jian Wu, and
Philip S. Yu. 2019.Multi-modal Attention Network Learning for Semantic
Source Code Retrieval.*2019 34th IEEE/ACM International Conference
on Automated Software Engineering (ASE)* (2019),
13–25.
* Wang
et al. [2020]W. Wang, Y. Zhang,
Zhengran Zeng, and Guandong Xu.
2020.TranS3̂: A Transformer-based Framework for
Unifying Code Summarization and Code Search.*ArXiv* abs/2003.03238
(2020).
* Wold
et al. [1987]S. Wold, K. Esbensen,
and P. Geladi. 1987.Principal component analysis.*Chemometrics and Intelligent Laboratory
Systems* 2 (1987),
37–52.
* Xu et al. [2018]Haowen Xu, Wenxiao Chen,
N. Zhao, Z. Li, Jiahao
Bu, Zhihan Li, Y. Liu,
Y. Zhao, D. Pei, Y.
Feng, Jian Jhen Chen, Zhaogang Wang,
and Honglin Qiao. 2018.Unsupervised Anomaly Detection via Variational
Auto-Encoder for Seasonal KPIs in Web Applications.*Proceedings of the 2018 World Wide Web
Conference* (2018).
* Yan
et al. [2020]Shuhan Yan, H. Yu,
Y. Chen, Beijun Shen, and
L. Jiang. 2020.Are the Code Snippets What We Are Searching for? A
Benchmark and an Empirical Study on Code Search with Natural-Language
Queries.*2020 IEEE 27th International Conference on
Software Analysis, Evolution and Reengineering (SANER)*(2020), 344–354.
* Yao
et al. [2019]Ziyu Yao,
Jayavardhan Reddy Peddamail, and Huan
Sun. 2019.CoaCor: Code Annotation for Code Retrieval with
Reinforcement Learning.*The World Wide Web Conference*(2019).
* Yao
et al. [2018]Ziyu Yao, Daniel S. Weld,
W. Chen, and Huan Sun.
2018.StaQC: A Systematically Mined Question-Code Dataset
from Stack Overflow.*Proceedings of the 2018 World Wide Web
Conference* (2018).
* Ye et al. [2020]Wei Ye, Rui Xie,
Jing lei Zhang, Tian xiang Hu,
Xiaoyin Wang, and Shikun Zhang.
2020.Leveraging Code Generation to Improve Code
Retrieval and Summarization via Dual Learning.*Proceedings of The Web Conference 2020*(2020).
* Yin
et al. [2018]Pengcheng Yin, Bowen
Deng, Edgar Chen, Bogdan Vasilescu,
and Graham Neubig. 2018.Learning to Mine Aligned Code and Natural Language
Pairs from Stack Overflow.*2018 IEEE/ACM 15th International Conference
on Mining Software Repositories (MSR)* (2018),
476–486.
* Zhang et al. [2018]Tianyi Zhang, Ganesha
Upadhyaya, Anastasia Reinhardt, Hridesh
Rajan, and Miryung Kim.
2018.Are Code Examples on an Online Q\&A Forum
Reliable?: A Study of API Misuse on Stack Overflow.*2018 IEEE/ACM 40th International Conference
on Software Engineering (ICSE)* (2018),
886–896.
* Zhang
et al. [2017]Y. Zhang, Weiling Chen,
C. Yeo, C. Lau, and B.
Lee. 2017.Detecting rumors on Online Social Networks using
multi-layer autoencoder.*2017 IEEE Technology $\\&$ Engineering
Management Conference (TEMSCON)* (2017),
437–441.
* Zhao et al. [2021]Yanjie Zhao, Li Li,
Haoyu Wang, Haipeng Cai,
Tegawende Bissyande, Jacques Klein, and
John Grundy. 2021.On the Impact of Sample Duplication in Machine
Learning based Android Malware Detection.*ACM Transactions on Software Engineering and
Methodology (TOSEM)* (2021).
