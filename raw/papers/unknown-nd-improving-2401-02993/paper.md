Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion
====================================================================================================

Shangyu Wu1, Ying Xiong2∗, Yufei Cui3∗  
Xue Liu3, Buzhou Tang2, Tei-Wei Kuo45, Chun Jason Xue1  
1 City University of Hong Kong2 Harbin Institute of Technology, Shenzhen  
3 MILA, McGill University4 National Taiwan University  
5 Mohamed bin Zayed University of Artificial Intelligence  
Authors contributed equally to this research

###### Abstract

Retrieval-based augmentations that aim to incorporate knowledge from an external database into language models have achieved great success in various knowledge-intensive (KI) tasks, such as question-answering and text generation.
However, integrating retrievals in non-knowledge-intensive (NKI) tasks, such as text classification, is still challenging.
Existing works focus on concatenating retrievals to inputs as context to form the prompt-based inputs.
Unfortunately, such methods require language models to have the capability to handle long texts.
Besides, inferring such concatenated data would also consume a significant amount of computational resources.

To solve these challenges, we propose ReFusion in this paper, a computation-efficient Retrieval representation Fusion with neural architecture search.
The main idea is to directly fuse the retrieval representations into the language models.
Specifically, we first propose an online retrieval module that retrieves representations of similar sentences.
Then, we present a retrieval fusion module including two effective ranking schemes, i.e., reranker-based scheme and ordered-mask-based scheme, to fuse the retrieval representations with hidden states.
Furthermore, we use Neural Architecture Search (NAS) to seek the optimal fusion structure across different layers.
Finally, we conduct comprehensive experiments, and the results demonstrate our ReFusion can achieve superior and robust performance on various NKI tasks.

1 Introduction
--------------

Recent advances in language models*(Khandelwal et al., [2020](#bib.bib23 ""); Borgeaud et al., [2022b](#bib.bib2 ""); Guu et al., [2020](#bib.bib18 ""); Lewis et al., [2020](#bib.bib24 ""); Li et al., [2022](#bib.bib25 ""))* have demonstrated that retrieval-based augmentations can achieve remarkable performance on a variety of knowledge-intensive (KI) tasks.
The basic idea of retrieval-based augmentations is to first leverage a dense vector indexing to retrieve the top-$k$ related knowledge from an external database, then incorporate the retrieved knowledge into language models.
For KI tasks such as question-answering and text generation, they have an inherent retrieval-based property*(Chen et al., [2017](#bib.bib3 ""); Karpukhin et al., [2020](#bib.bib22 ""))* as answers can be sourced or deduced from external knowledge databases.
For example, to answer a question, we can retrieve lots of relevant content for reference or even similar questions for answers.
Besides, such external databases containing question-answer pairs are quite common and easy to build.

However, retrieval-based augmentations in non-knowledge-intensive (NKI) tasks, such as text classification, are still challenging.
Different from KI tasks, NKI tasks often require understanding and categorizing given sentences rather than generating new sentences*Wang et al. ([2019](#bib.bib33 ""))*.
Previous works*Guo et al. ([2023b](#bib.bib17 ""))* have proposed to leverage plain texts such as Wikipedia to build the retrieval database.
They treat retrievals as the context of inputs and concatenate retrievals with inputs.
However, their methods demand language models to have the capability of handling long sequence data.
Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(a) shows that concatenating more retrievals would significantly increase the length of inputs, but the number of retrievals would be limited by the max sequence length of models.
This limitation would result in a performance drop as shown in the red line in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(b).
Besides, processing such long sequence inputs would also consume a substantial amount of computational resources as shown in the green line in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(b).

|  |
| --- |

<img src='x1.png' alt='Refer to caption' title='' width='230' height='171' />

*(a) Sequence length*

<img src='x2.png' alt='Refer to caption' title='' width='230' height='171' />

*(b) Accuracy and FLOPs*

*Figure 1: Impact of the number of retrievals on input sequence length and its effect on model’s accuracy and FLOPs. CA-A and RF-A refer to the accuracy of context-augmentation and retrieval representation fusion. CA-F and RF-F refer to the FLOPs of context-augmentation and retrieval-addition.*

In this paper, we introduce ReFusion as shown in Figure[3](#S2.F3 "Figure 3 ‣ 2 Background and Motivation ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion"), a computation-efficient Retrieval representation Fusion framework with neural architecture search.
Different from previous retrieval-based augmentations*(Borgeaud et al., [2022b](#bib.bib2 ""); Guo et al., [2023b](#bib.bib17 ""))*, ReFusion directly fuses the representations of retrievals into models.
ReFusion consists of three major modules, i.e., the retrieval module, the fusion module and the search module.
Specifically, the retrieval module is an online retrieval module.
It contains a query encoder to encode query texts and an in-memory cache or an on-disk database for retrieving the representations of similar texts.
The fusion module has two effective ranking schemes, i.e., the reranker-based scheme and the ordered-mask-based scheme, to refine the representation of retrievals.
The refined representations would be fused into the hidden representations.
The search module is based on the neural architecture search, which aims to find the optimal fusion structure.

Finally, we conducted a series of comprehensive experiments on 15 different NKI tasks.
These tasks vary from sentiment analysis, opinion polarity, natural language inference, and so on.
Our experimental setting mainly follows Gao et al.*Gao et al. ([2021](#bib.bib14 ""))*, i.e., prompt-based few-shot learning.
This can greatly reveal language models’ understanding capability.
Experimental results show that ReFusion outperforms other comparisons and achieves superior results on various tasks.
Besides, ReFusion is more robust than others.
Code is available at 111https://anonymous.4open.science/r/ReFusion-173F.

The main contributions of this paper are:

* •

    We are the first to propose fusing the representations of retrievals directly into models to solve the performance and efficiency bottleneck of prompt-based techniques.

* •

    Experimental results demonstrate that our ReFusion framework can significantly improve models’ understanding capability, and achieve a superior and robust performance.

2 Background and Motivation
---------------------------

<img src='x3.png' alt='Refer to caption' title='' width='461' height='270' />

*Figure 2: Conventional retrieval-augmented prompt-based fine-tuning.*

Retrieval-augmented Prompt-based Fine-tuning The common NKI tasks involves inputting a sentence as $X_{\texttt{single}}\={\texttt{[CLS]}x\texttt{[SEP]}}$ or two sentences as $X_{\texttt{pair}}\={\texttt{[CLS]}x_{1}\texttt{[SEP]}x_{2}\texttt{[SEP]}}$, and outputting a categorical label $y$.
In traditional methods, the [CLS] token is utilized to represent the overall contextual information of the input and to facilitate classification tasks, i.e., $y_{\texttt{logits}}\=\texttt{softmax}(W_{o}\cdot h_{\texttt{[CLS]}})$, where $h_{\texttt{[CLS]}}$ is the final hidden representation of [CLS] token.
To harness the potential of transformer-based models in terms of mask-prediction capabilities, which serve as the pre-training objective, recent studies*(Gao et al., [2021](#bib.bib14 ""); Zhang et al., [2022](#bib.bib35 ""); Guo et al., [2023b](#bib.bib17 ""))* suggest employing prompt-based fine-tuning.
These works involve transforming the original inputs into prompt-based inputs that incorporate a mask token for prediction.
A simple example of prompt-based inputs of a single sentence or a pair of sentences is shown below,

|  | $X_{\texttt{prompt-single}}\=\texttt{[CLS]}x\text{ It was }\texttt{[MASK]}\text{ . }\texttt{[SEP]}$ |  | (1) |
| --- | --- | --- | --- |

|  | $X_{\texttt{prompt-pair}}\=\texttt{[CLS]}x_{1}\texttt{[MASK]}\text{ ? }x_{2}\texttt{[SEP]}$ |  | (2) |
| --- | --- | --- | --- |

The categorical label $y$ is transformed into semantic token $y_{w}$, e.g., using ‘positive’ to replace ‘1’ and ‘negative’ to replace ‘0’.
Then, the objective is to maximize the probability of the label word corresponding to the [MASK] token,

|  | $Loss\=-\log p(y_{w}|X_{\texttt{prompt}})\=-\log p(\texttt{[MASK]}\=y_{w}|X_{\texttt{prompt}})$ |  | (3) |
| --- | --- | --- | --- |

To further improve prompt-based fine-tuning, recent works*(Guo et al., [2023b](#bib.bib17 ""); Chen et al., [2022c](#bib.bib6 ""))* concatenate retrieval information as contexts or demonstrations of the given inputs, thus helping language models have a better semantic understanding.
For example, as shown in Figure[2](#S2.F2 "Figure 2 ‣ 2 Background and Motivation ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(a), they first retrieve top-$k$ similar sentences $Z\={z_{1},\ldots,z_{k}}$ from an external key-value database for each input text $x_{i}$.
Then, they concatenate retrievals $Z$ and prompt-based inputs $X_{\texttt{prompt}}$.
The retrieval-augmented prompt-based input of a single sentence is:

|  | $X_{\texttt{retrieval-single}}\=\texttt{[CLS]}z_{1}\texttt{[SEP]}\ldots\texttt{[SEP]}z_{k}\texttt{[SEP]}x\text{ It was }\texttt{[MASK]}\text{ . }\texttt{[SEP]}$ |  | (4) |
| --- | --- | --- | --- |

The objective is then optimized based on the retrieval-augmented prompt-based inputs.

<img src='x4.png' alt='Refer to caption' title='' width='368' height='167' />

*Figure 3: The architecture of our ReFusion and the detailed structure of proposed modules.*

Motivations Retrieval-augmented prompt-based fine-tuning trades off the amount of contextual information against the hardware resources.
Since language models can only process a limited length of input (typically constrained by the hyperparameter max_length), the amount of contextual information concatenated into inputs is also limited, while longer inputs would be truncated before fed in models.
Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(a) illustrates the average sequence length of retrieval-augmented prompt-based inputs for both the single-sentence task (e.g., SST-2) and the two-sentence task (e.g., STS-B) across various numbers of retrievals.
Given that common language models have a maximum sequence length constraint, such as 512 for RoBERTa-large*Liu et al. ([2019b](#bib.bib28 ""))*, it is important to acknowledge that we can only concatenate a maximum of 16 retrievals.
Beyond this limit, additional retrievals would be truncated and not contribute to the model’s performance.
Although directly concatenating more retrievals to the inputs is a convenient way to supplement language models, it should be noted that such concatenation-based augmentations, including both textual and embedding concatenation, impose significant demands on hardware resources.

To quantify the impact of concatenation-based augmentations on hardware resources, we measure the number of floating-point operations (FLOPs) of two baseline methods, context-augmentation (CA) and retrieval-addition (RF), in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(b).
Specifically, CA refers to directly concatenating the retrievals with the input, while RF refers to directly adding the retrieval representations to the representation of the [CLS] token.
Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(b) demonstrates that increasing the number of retrievals can slightly improve CA’s accuracy (from 1 retrieval to 8 retrievals) while producing a significant amount of FLOPs.
The degradation in CA’s performance with 16 retrievals is due to the truncation of inputs, which results in a loss of semantic completeness.
Conversely, when increasing the number of retrievals, the FLOPs of RF remains almost unchanged, yet its accuracy continues to exhibit an upward trend.
Notably, RF achieves even higher accuracy than CA.

Consequently, we propose to directly integrate the representations of retrievals into transformer-based models.
As shown in Figure[2](#S2.F2 "Figure 2 ‣ 2 Background and Motivation ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(b), the basic idea is to add the mean representations of $k$ retrievals to the representation of the [CLS] token in each attention module.
The intuition is that the [CLS] token captures the overall semantics of the sentence, and by augmenting it with sentence-level representations, we can enhance context understanding and disambiguation.
To evaluate the effectiveness of the proposed idea, we conducted some preliminary experiments to show the potential of the proposed idea (RF).
Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion")(b) indicates that the increase in FLOPs for RF is negligible.
However, the performance improvement achieved by RF is also insignificant.
The reasons might be 1) The retrievals are obtained by a task-agnostic index based on a simple similarity metric, such as L2 norm, rather than a more sophisticated semantic similarity metric; 2) The retrieval representations across different dimensions should have different significance*(Rippel et al., [2014](#bib.bib30 ""); Cui et al., [2023](#bib.bib9 ""))*; 3) Not every layer of the model necessarily requires augmentation with retrievals.
These observations motivate us to propose a computation-efficient retrieval representation fusion with neural architecture search.

3 ReFusion: A Computation-Efficient Retrieval Representation Fusion with Neural Architecture Search
----------------------------------------------------------------------------------------------------

In this section, we introduce the details of our ReFusion framework.
Our framework can be adapted to any transformer-based architecture*Vaswani et al. ([2017](#bib.bib32 ""))*, or any architecture that contains the attention module.
As shown in Figure[3](#S2.F3 "Figure 3 ‣ 2 Background and Motivation ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion"), we first present the online retrieval module used in our framework, which retrieves the representations of top-$k$ similar sentences.
Then, we present the retrieval fusion module containing two ranking schemes, i.e., the reranker-based scheme and the ordered-mask scheme.
Finally, we propose the overall architecture-search-based framework, which learns to choose the best architecture for different ranking schemes.

### 3.1 The Online Retrieval Module

In the retrieval module, there is a query encoder for encoding query texts and a task-agnostic retriever built offline over billions of dense vectors.
The retriever consists of an efficient indexing like FAISS*Johnson et al. ([2019](#bib.bib21 ""))* or ScaNN*Guo et al. ([2020](#bib.bib15 ""))*, and a compressed key-value store database that contains all texts and embeddings.
The retrieving process in our framework is online performed, which means that for every forward, the query encoder first passes the representation $h_{x}$ of the input text $x$ to the retriever, then the retriever returns the representations $H_{Z}\={h_{z_{1}},\ldots,h_{z_{k}}}$ of top-$k$ similar sentences $Z\={z_{1},\ldots,z_{k}}$ to the fusion module.
For efficient retrieving, especially for the training, the retrieval module maintains an in-memory cache for the input text $x$ and corresponding representations $H_{Z}$ of similar sentences.

<img src='x5.png' alt='Refer to caption' title='' width='369' height='158' />

*Figure 4: Two different ranking schemes used in the fusion module.*

### 3.2 The Retrieval Fusion Module

The retrieval fusion module can be wrapped with any modules in the language models (LMs).
It takes the representations of top-$k$ similar sentences and the hidden representations of existing modules as inputs, and outputs the fused representations.
Specifically, we introduce two effective ranking schemes as shown in Figure[4](#S3.F4 "Figure 4 ‣ 3.1 The Online Retrieval Module ‣ 3 ReFusion: A Computation-Efficient Retrieval Representation Fusion with Neural Architecture Search ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion").

#### 3.2.1 Reranking the Retrievals

In the retrieval module, the retrievals are ranked by a task-agnostic similarity metric, e.g., L2 norm.
Directly adding the representations to the hidden representations would not improve LMs’ performance.
That is because 1) The retrievals are not optimally ranked for the existing module in LMs, which may introduce noise or irrelevant information; 2) The models should pay different attention to those retrievals in case of overemphasizing less relevant information.
Therefore, we aim to propose a learnable reranker to learn the ranking distribution tailored to each module in LMs.
As shown in the top of Figure[4](#S3.F4 "Figure 4 ‣ 3.1 The Online Retrieval Module ‣ 3 ReFusion: A Computation-Efficient Retrieval Representation Fusion with Neural Architecture Search ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion"), the significance of retrievals is re-assigned after reranking.

Specifically, the reranker is a 1D learnable vector of $k$ dimensions, i.e., $R\={r_{1},\ldots,r_{k}}$.
It is first normalized and then multiplied by the retrievals.
Finally, the averaged representation of all reranked retrievals is added to the sentence representation, e.g., [CLS] token in BERT-like models*(Liu et al., [2019b](#bib.bib28 ""); Devlin et al., [2019](#bib.bib11 ""))*.
The formal steps are as follows,

|  | $\displaystyle r_{i}\=\frac{\exp(r_{i})}{\sum_{j}\exp(r_{j})}$ |  | (5) |
| --- | --- | --- | --- |
|  | $\displaystyle h_{y_{\texttt{[CLS]}}}\=h_{x_{\texttt{[CLS]}}}+\frac{1}{k}\sum r_{i}\cdot h_{z_{i}}$ |  | (6) |
| --- | --- | --- | --- |

where $h_{x_{\texttt{[CLS]}}}$, $h_{y_{\texttt{[CLS]}}}$ are the sentence representations of inputs and outputs.

#### 3.2.2 Ordered Mask Over Retrieval Representations

Rippel et al.*Rippel et al. ([2014](#bib.bib30 ""))* proposed a nested dropout that directly drops the representation units from the sampled index $I$, thus yielding an inherent importance ranking of the representation dimensions.
This nested dropout can be implemented by a mask with leading $I$ ones then zeros.
Based on the nested dropout, recent works*(Cui et al., [2023](#bib.bib9 ""); [2020](#bib.bib7 ""); [2021](#bib.bib8 ""); Mao et al., [2022](#bib.bib29 ""))* proposed the ordered mask that modeled the dropping process with a chain of Bernoulli variables and made it differentiable using the re-parameterization trick.

As shown in the bottom of Figure[4](#S3.F4 "Figure 4 ‣ 3.1 The Online Retrieval Module ‣ 3 ReFusion: A Computation-Efficient Retrieval Representation Fusion with Neural Architecture Search ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion"), we apply the ordered mask over $k$ retrievals on each representation dimension.
This means that different from the reranker, the ordered mask trusts the ranking produced by the retriever and refines the ranking with training data.
Specifically, let $h_{z_{1}},\ldots,h_{z_{k}}$ be the top-$k$ $D$-dimensional retrieval representations.
For each dimension of retrieval representation (e.g., the dimension $d$), the ordered mask is modeled by a chain of Bernoulli variables $V\={v_{1}^{d},\ldots,v_{k}^{d}}$, where $v_{i}^{d}\sim\textbf{Bernoulli}(\pi_{i})$ indicates whether drop the $d$-th representation unit of the $i$-th retrieval.
Following the property of nested dropout, the variable $v_{i}^{d}$ is conditioned on $v_{i-1}^{d}$, thus we can obtain the marginal distribution $p(\mathbf{v_{i}^{d}})$ of $v_{i}^{d}$.

After that, the ordered mask uses the re-parameterization trick, e.g., choosing the Gumbel Softmax distribution*Jang et al. ([2017](#bib.bib20 ""))* as the tractable variational distribution $q(\mathbf{v_{i}^{d}})$.
With Gumbel Softmax distribution, if $\mathbf{c}^{d}\sim\textbf{Gumbel}(\beta,\tau)$, then $v_{i}^{d}\=1-\texttt{cumsum}_{i}(\mathbf{c}^{d})$, where $\mathbf{c}^{d}$ is a sample choice of the dropped index over $k$ retrievals on the dimension $d$, and $\texttt{cumsum}_{i}(\mathbf{c}^{d})\=\sum_{j\=0}^{i-1}c^{d}_{j}$.
In the Gumbel Softmax distribution, $\beta$ is a learnable parameter in the differentiable function $v_{i}^{d}\=g(\epsilon_{i};\beta)$ and $\tau$ is the temperature variable that controls the smoothness of the step at the dropped index.

Finally, we obtain the different ordered mask $V^{1},\ldots,V^{D}$ over representation dimensions. We use it to mask the retrievals in a fine-grained way.
Then, the masked retrievals would be fused into the sentence representations in the same way as Reranker.
The formal steps are as follows,

|  | $\displaystyle\mathbf{c}^{d}\sim\textbf{Gumbel}(\beta,\tau)$ |  | (7) |
| --- | --- | --- | --- |
|  | $\displaystyle v_{i}^{d}\=1-\texttt{cumsum}_{i}(c^{d})$ |  | (8) |
| --- | --- | --- | --- |
|  | $\displaystyle\hat{h}_{z_{i}}^{d}\=v^{d}_{i}\cdot h_{z_{i}}^{d}$ |  | (9) |
| --- | --- | --- | --- |
|  | $\displaystyle h_{y_{\texttt{[CLS]}}}\=h_{x_{\texttt{[CLS]}}}+\frac{1}{k}\sum\hat{h}_{z_{i}}$ |  | (10) |
| --- | --- | --- | --- |

where $\hat{h}_{z_{i}}^{d}$ is the $d$-th masked representation unit of $i$-th retrieval.

### 3.3 The Architecture Search Module

As shown in Figure[4](#S3.F4 "Figure 4 ‣ 3.1 The Online Retrieval Module ‣ 3 ReFusion: A Computation-Efficient Retrieval Representation Fusion with Neural Architecture Search ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion"), it is difficult to tell which ranking scheme is better on each layer in LMs.
Therefore, we propose an architecture search module, aiming to leverage neural architecture search (NAS) techniques to search to select the optimal ranking scheme.

#### 3.3.1 Search Space

In this work, we do not search for a totally new neural network architecture like previous NAS works*Liu et al. ([2019a](#bib.bib27 ""))* do.
Instead, we keep the main structure of transformer-based models unchanged and only replace several modules with our search modules.

A search module consists of multiple fusion modules with different ranking schemes and the original module.
For example, taking the linear module in LMs as an example, we replace the linear module with our linear search module, which includes three modules, the fusion module with reranker-based scheme, the fusion module with ordered-mask-based scheme, and the original linear module.

Although the number of candidate modules in the search module is small, the whole search space is quite large.
Given a transformer-based language model with $N$ hidden layers, assume that we only replace the linear module for the key and value in every attention module, we have at least $3\times 3\=9$ candidate modules and thus at least $9^{N}$ different retrieval-augmented transformer-based language models.
Taking the RoBERTa-large as an example, which has 24 layers, the search space can be septillion-level large.

#### 3.3.2 Searching Details

We follow the same searching strategies used in DARTS*Liu et al. ([2019a](#bib.bib27 ""))*.
Specifically, let $\alpha\={\alpha_{1},\ldots,\alpha_{l}}$ be the architectural weights, where $l$ is the number of candidate modules in each search module.
To make the search space continuous, we also relax the categorical choice of a particular candidate module to a softmax over all possible candidate modules within the search module,

|  | $\hat{o}(h)\=\sum_{i}\frac{\exp(\alpha_{i})}{\sum_{j}\exp(\alpha_{j})}o_{i}(h)$ |  | (11) |
| --- | --- | --- | --- |

where $o_{i}(h)$ represents the output of the $i$-th candidate module $o_{i}(\cdot)$ taking the hidden states $h$ as input, $\hat{o}(\cdot)$ indicates the output of the search module.

The goal of architecture searching is to jointly optimize the architectural weights $\alpha$ and the weights $\omega$ of all modules with LMs.
We update the weights $\omega$ based on the training loss, and the architectural weights based on the validation loss.
The updates of these two types of weights are done alternatively.
After training, we only choose the candidate module with the largest architectural weights for the inference.

4 Experiment
------------

*Table 1: Our main results with RoBERTa-large.*

| Methods | SST-2 | SST-5 | MR | CR | MPQA | SUBJ | TREC | CoLA | Avg-S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LM-BFF | $\text{92.7}_{\text{0.9}}$ | $\text{47.4}_{\text{2.5}}$ | $\text{87.0}_{\text{1.2}}$ | $\text{90.3}_{\text{1.0}}$ | $\text{84.7}_{\text{2.2}}$ | $\text{91.2}_{\text{1.1}}$ | $\text{84.8}_{\text{5.1}}$ | $\text{9.3}_{\text{7.3}}$ | 73.4 |
| DART | $\text{93.5}_{\text{0.5}}$ | - | $\text{88.2}_{\text{1.0}}$ | $\text{91.8}_{\text{0.5}}$ | - | $\text{90.7}_{\text{1.4}}$ | $\text{87.1}_{\text{3.8}}$ | - | - |
| KPT | $\text{90.3}_{\text{1.6}}$ | - | $\text{86.8}_{\text{1.8}}$ | $\text{88.8}_{\text{3.7}}$ | - | - | - | - | - |
| CA-512 | $\text{91.3}_{\text{1.4}}$ | $\text{46.7}_{\text{1.1}}$ | $\text{85.1}_{\text{1.4}}$ | $\text{88.3}_{\text{1.7}}$ | $\text{76.9}_{\text{2.8}}$ | $\text{88.0}_{\text{1.9}}$ | $\text{82.2}_{\text{4.4}}$ | $\text{7.4}_{\text{3.3}}$ | 70.7 |
| ReFusion | $\text{93.4}_{\text{0.6}}$ | $\text{49.8}_{\text{1.4}}$ | $\text{87.9}_{\text{1.1}}$ | $\text{91.7}_{\text{0.3}}$ | $\text{86.7}_{\text{1.1}}$ | $\text{92.5}_{\text{0.8}}$ | $\text{90.3}_{\text{3.7}}$ | $\text{11.4}_{\text{4.1}}$ | 75.5 |
| Methods | MNLI | MNLI-m | SNLI | QNLI | RTE | MRPC | QQP | Avg-P | Avg-all |
| LM-BFF | $\text{68.3}_{\text{2.3}}$ | $\text{70.5}_{\text{1.9}}$ | $\text{77.2}_{\text{3.7}}$ | $\text{64.5}_{\text{4.2}}$ | $\text{69.1}_{\text{3.6}}$ | $\text{74.5}_{\text{5.3}}$ | $\text{65.5}_{\text{5.3}}$ | 69.9 | 71.8 |
| DART | $\text{67.5}_{\text{2.6}}$ | - | $\text{75.8}_{\text{1.6}}$ | $\text{66.7}_{\text{3.7}}$ | - | $\text{78.3}_{\text{4.5}}$ | $\text{67.8}_{\text{3.2}}$ | - | - |
| KPT | $\text{61.4}_{\text{2.1}}$ | - | - | $\text{61.5}_{\text{2.8}}$ | - | - | $\text{71.6}_{\text{2.7}}$ | - | - |
| CA-512 | $\text{66.2}_{\text{1.0}}$ | $\text{67.8}_{\text{1.3}}$ | $\text{71.6}_{\text{2.2}}$ | $\text{66.9}_{\text{3.2}}$ | $\text{66.6}_{\text{3.1}}$ | $\text{73.5}_{\text{6.9}}$ | $\text{64.0}_{\text{1.9}}$ | 68.1 | 69.5 |
| ReFusion | $\text{69.3}_{\text{1.5}}$ | $\text{70.9}_{\text{1.5}}$ | $\text{80.6}_{\text{1.4}}$ | $\text{73.0}_{\text{1.1}}$ | $\text{70.9}_{\text{2.3}}$ | $\text{77.0}_{\text{3.6}}$ | $\text{68.9}_{\text{3.3}}$ | 72.9 | 74.3 |

The results of LM-BFF, DART refer to their original paper. The results of KPT refer to*Chen et al. ([2022c](#bib.bib6 ""))*.

### 4.1 Datasets and Experimental Setting

Datasets We conduct comprehensive experiments across 15 NKI tasks, including 8 tasks from GLUE benchmark*Wang et al. ([2019](#bib.bib33 ""))*, SNLI, SST-5, MR, CR, MNLI, MNLI-mm, Subj and TREC. There are 8 single-sentence tasks and 7 sentence-pair tasks. These tasks cover sentiment analysis, opinion polarity analysis, grammatical judgment, natural language inference, paraphrasing, etc. The configuration of the dataset is identical to that of LM-BFF*Gao et al. ([2021](#bib.bib14 ""))*.

Experimental Settings The proposed method was implemented using PyTorch framework, utilizing the computational power of a single Nvidia V100 GPU. Our experiments were conducted with the same settings as LM-BFF, which measures the average performance of five different sampled $D_{train}$ for each task with a fixed set of seed $S_{seed}\={13,21,42,87,100}$. In our model, there are 16 samples per class. The hyperparameters are listed as follows: the learning rate is 1e-5, the batch size is 32, the maximum sequence length is 128, the maximum steps are 1000, the number $k$ of similar sentences retrieved is set to 64 and we save the last checkpoint. We use AdamW as the optimizer. Our models are based on RoBERTa-large for fair comparison with LM-BFF.

To validate the effectiveness of our method, we compared ReFusion with several other models: (1) LM-BFF: a prompt-based fine-tuning approach; (2) DART*Zhang et al. ([2022](#bib.bib35 ""))*: a differentiable prompt-based model, which can automatically search for the optimal prompt; (3) KPT*Hu et al. ([2022](#bib.bib19 ""))*: a prompt-based approach incorporating knowledge into the prompt verbalizer; and (4) CA-512: a retrieval-augmented prompt-based method concatenating retrievals with inputs. The max length of model is set to 512. Unlike LM-BFF, DART, and KPT, we did not employ a grid search for the best parameters but instead chose a default template and parameters based on LM-BFF. Consequently, there is potential for our model to improve further through grid search. The templates are listed in Appendix[A](#A1 "Appendix A Templates on All Tasks ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion").

### 4.2 Main Results

Table [1](#S4.T1 "Table 1 ‣ 4 Experiment ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion") presents the main experimental results of our ReFusion and comparisons on 15 NKI tasks.
The results are shown in the form of means and variances, with the variance denoted by a subscript.

For tasks with single sentences (S-Task), ReFusion consistently demonstrates superior performance across almost all benchmarks.
ReFusion achieves state-of-the-art performance on 5 tasks over 8 tasks.
And ReFusion improves the average performance on the S-Task benchmark by about 2.1% than LM-BFF.
Specifically, on the TREC task, ReFusion (90.3%) exhibits the maximum improvements over LM-BFF (84.8%).

For tasks consisting of pair sentences (P-Task), ReFusion continues to demonstrate strong performance.
ReFusion also achieves the state-of-the-art on 5 tasks over 7 tasks.
And ReFusion can improve the average performance on the P-Task benchmark by about 3.0% than LM-BFF.
For instance, on the QNLI and SNLI benchmark, ReFusion (73% for QNLI, 80.6% for SNLI) significantly exceeds LM-BFF (64.5% for QNLI, 77.2% for SNLI).

The Avg-all represents the average performance of all 15 NKI tasks.
For overall average performance, ReFusion achieves a score of 74.3%, marginally surpassing LM-BFF’s 71.8%.
This further highlights ReFusion’s consistent and superior performance.
Besides, ReFusion surpasses other models like DART, CA-512 and KPT, delivering superior or comparable results.
Notably, the standard deviation of ReFusion is considerably smaller than that of other models, indicating that ReFusion produces stable results and offers superior robustness.

5 Ablation Study
----------------

We conduct ablation experiments on six representative tasks to show the contributions of each module to the overall performance.
On all tasks, ReFusion tends to produce better results than those just applying the retrieval fusion module.
The results of methods using NAS demonstrate that NAS can significantly boost performance.
Specifically, compared to the baseline, two ranking schemes can bring different but significant improvements.
This reveals that it is necessary to combine different ranking schemes on different tasks.
After using NAS, the performance of each ranking scheme is also significantly improved.
This suggests these two ranking schemes are not always suitable for every layer in LMs, thus we need to disable the fusion module at some layers.
Finally, our ReFusion integrating all effective candidate fusion modules using NAS achieves the best performance on three tasks.
We can infer that the combination of all candidate modules harnesses their strengths.

*Table 2: Ablation studies on different modules.*

| Methods | MPQA | SUBJ | TREC | SNLI | QNLI | RTE |
| --- | --- | --- | --- | --- | --- | --- |
| Roberta-Large | $\text{83.6}_{\text{2.5}}$ | $\text{90.3}_{\text{2.8}}$ | $\text{83.8}_{\text{5.2}}$ | $\text{73.5}_{\text{5.2}}$ | $\text{65.0}_{\text{3.0}}$ | $\text{64.1}_{\text{2.0}}$ |
| Reranker | $\text{84.2}_{\text{2.2}}$ | $\text{91.3}_{\text{1.3}}$ | $\text{85.0}_{\text{4.2}}$ | $\text{74.3}_{\text{4.6}}$ | $\text{68.8}_{\text{1.4}}$ | $\text{65.6}_{\text{3.1}}$ |
| Ordered Mask | $\text{83.3}_{\text{1.9}}$ | $\text{90.8}_{\text{1.4}}$ | $\text{83.0}_{5.8}$ | $\text{74.9}_{\text{4.0}}$ | $\text{68.3}_{\text{1.4}}$ | $\text{65.8}_{\text{3.1}}$ |
| NAS with Reranker | $\text{86.9}_{\text{1.3}}$ | $\text{92.4}_{\text{1.3}}$ | $\text{90.8}_{\text{2.5}}$ | $\text{80.3}_{\text{1.9}}$ | $\text{73.5}_{\text{1.8}}$ | $\text{69.2}_{\text{2.4}}$ |
| NAS with Ordered Mask | $\text{87.0}_{\text{1.5}}$ | $\text{92.4}_{\text{0.7}}$ | $\text{90.7}_{\text{3.0}}$ | $\text{80.3}_{\text{1.3}}$ | $\text{73.0}_{\text{1.0}}$ | $\text{70.4}_{\text{2.5}}$ |
| ReFusion | $\text{86.7}_{\text{1.1}}$ | $\text{92.5}_{\text{0.8}}$ | $\text{90.3}_{\text{3.7}}$ | $\text{80.6}_{\text{1.4}}$ | $\text{73.0}_{\text{1.1}}$ | $\text{70.9}_{\text{2.3}}$ |

6 Analysis
----------

The query used to retrieve the representations of similar sentences would affect the performance of models.
In this paper, we consider two types of queries, i.e., input texts or hidden states.
The “Input texts” refers to the method of encoding input texts and sentences used to build the retrieval database with a consistent encoder, then retrieving the representations of top $k$ similar sentences.
Alternatively, the “Hidden states” refers to the approach that directly leverages hidden states of LMs without the encoding process during training.
Table [3](#S6.T3 "Table 3 ‣ 6 Analysis ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion") displays the performance and the time cost of each training iteration for both methods.
Notably, the “Input texts” technique stands out for its impressive combination of performance and speed, making it ideal for real-time processing demands.
Conversely, the “Hidden states” method, which queries the hidden representations that dynamically change, provides heightened adaptability but at the expense of elongated computational times.
This approach is suitable for situations where adaptability is more important than time constraints, particularly in offline contexts.
In essence, the choice hinges on task-specific requirements: efficiency prioritizes “Input texts”, while adaptability leans towards the “Hidden states”, acknowledging its time trade-off.

*Table 3: What to use for retrieval.*

| Queries | MPQA | SUBJ | TREC |
| --- | --- | --- | --- |
| Input texts | 87.9 | 92.1 | 84.0 |
| | 3.7s | 3.8s | 11.1s |
| Hidden states | 88.7 | 91.7 | 85.6 |
| | 108.2s | 109.4s | 294.9s |

7 Related Work
--------------

Retrieval Augmentation Retrieval-based methods combine the benefits of information retrieval with modern deep learning models, like transformers, to improve the performance of related tasks.
Tasks usually retrieved “answers” to improve models’ performance *(Shi et al., [2022](#bib.bib31 ""); Zhou et al., [2023](#bib.bib36 ""); Frisoni et al., [2022](#bib.bib13 ""))*.
Borgeaud et al. *(Borgeaud et al., [2022a](#bib.bib1 ""))* proposed RETRO. They enhanced the auto-regressive language models by combining the next chunk of the found chunk, which is most similar to the current input chunk.
For knowledge-intensive tasks, retrieval-based approaches are also commonly used in question-answering (QA) systems, where the goal is to retrieve relevant information from a large corpus of documents or knowledge sources to answer user queries*(Das et al., [2019](#bib.bib10 ""); Chen et al., [2022a](#bib.bib4 ""); Lin \& Byrne, [2022](#bib.bib26 ""); Yu et al., [2023](#bib.bib34 ""))*.
They generated answers based on retrieval information.
Retrieval-augmented methods also greatly improved non-knowledge-intensive tasks*(Guo et al., [2023a](#bib.bib16 ""); Chen et al., [2022b](#bib.bib5 ""); Du \& Ji, [2022](#bib.bib12 ""))*.
They designed prompt-based methods and appended retrieval information into input text.
We focus on non-knowledge-intensive tasks as they are under exploration and propose a more memory-efficient and effective way to fuse retrieval representations into the models.

8 Conclusion
------------

In this paper, we aim to solve the bottleneck of prompt-based techniques by directly fusing retrieval representations into models.
We propose a computation-efficient retrieval representation fusion framework with neural architecture search.
Specifically, we first propose an online retrieval module to retrieve the representations of top-$k$ similar sentences.
Then, we introduce two ranking schemes from different perspectives in the retrieval fusion module, optimizing the retrieval representations, i.e., reranker-based scheme and ordered-mask-based scheme.
To harness the strengths of each ranking scheme, we present an architecture search module to seek the optimal structure of different fusion modules.
Experimental results demonstrate our fusion framework outperforms baselines and is robust on various tasks.

For future work, we plan to further investigate more ranking schemes and more architecture search frameworks.
Besides, it is also worthwhile to explore the effect of different hyper-parameters in our framework, such as the module to fuse retrievals and the number of retrieved representations.

References
----------

* Borgeaud et al. (2022a)Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre.RETRO-Improving Language Models by Retrieving from Trillions of Tokens.In *International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pp. 2206–2240. PMLR, 2022a.
* Borgeaud et al. (2022b)Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre.Improving language models by retrieving from trillions of tokens.In *International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pp. 2206–2240. PMLR, 2022b.
* Chen et al. (2017)Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes.Reading wikipedia to answer open-domain questions.In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers*, pp. 1870–1879, 2017.
* Chen et al. (2022a)Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen.MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 5558–5570. Association for Computational Linguistics, 2022a.
* Chen et al. (2022b)Xiang Chen, Lei Li, Ningyu Zhang, Xiaozhuan Liang, Shumin Deng, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen.RETROPROMPT-Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning.In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 23908–23922. Curran Associates, Inc., 2022b.
* Chen et al. (2022c)Xiang Chen, Lei Li, Ningyu Zhang, Xiaozhuan Liang, Shumin Deng, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen.RETROPROMPT-Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning.In *Advances in Neural Information Processing Systems*, volume 35, pp. 23908–23922. Curran Associates, Inc., 2022c.
* Cui et al. (2020)Yufei Cui, Ziquan Liu, Wuguannan Yao, Qiao Li, Antoni B. Chan, Tei-Wei Kuo, and Chun Jason Xue.Fully nested neural network for adaptive compression and quantization.In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020*, pp. 2080–2087, 2020.
* Cui et al. (2021)Yufei Cui, Ziquan Liu, Qiao Li, Antoni B. Chan, and Chun Jason Xue.Bayesian nested neural networks for uncertainty calibration and adaptive compression.In *IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021*, pp. 2392–2401, 2021.
* Cui et al. (2023)Yufei Cui, Yu Mao, Ziquan Liu, Qiao Li, Antoni B. Chan, Xue (Steve) Liu, Tei-Wei Kuo, and Chun Jason Xue.Variational nested dropout.*IEEE Trans. Pattern Anal. Mach. Intell.*, 45(8):10519–10534, 2023.
* Das et al. (2019)Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, and Andrew McCallum.Multi-step Retriever-Reader Interaction for Scalable Open-domain Question Answering.In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*, 2019.
* Devlin et al. (2019)Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.BERT: pre-training of deep bidirectional transformers for language understanding.In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)*, pp. 4171–4186. Association for Computational Linguistics, 2019.
* Du \& Ji (2022)Xinya Du and Heng Ji.Retrieval-Augmented Generative Question Answering for Event Argument Extraction.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 4649–4666. Association for Computational Linguistics, 2022.
* Frisoni et al. (2022)Giacomo Frisoni, Miki Mizutani, Gianluca Moro, and Lorenzo Valgimigli.BioReader: a Retrieval-Enhanced Text-to-Text Transformer for Biomedical Literature.In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 5770–5793. Association for Computational Linguistics, 2022.
* Gao et al. (2021)Tianyu Gao, Adam Fisch, and Danqi Chen.Making pre-trained language models better few-shot learners.In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pp. 3816–3830. Association for Computational Linguistics, 2021.
* Guo et al. (2020)Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar.Accelerating large-scale inference with anisotropic vector quantization.In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pp. 3887–3896. PMLR, 2020.
* Guo et al. (2023a)Zhicheng Guo, Sijie Cheng, Yile Wang, Peng Li, and Yang Liu.Prompt-Guided Retrieval Augmentation for Non-Knowledge-Intensive Tasks.In *Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 10896–10912. Association for Computational Linguistics, 2023a.
* Guo et al. (2023b)Zhicheng Guo, Sijie Cheng, Yile Wang, Peng Li, and Yang Liu.Prompt-guided retrieval augmentation for non-knowledge-intensive tasks.In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), *Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 10896–10912. Association for Computational Linguistics, 2023b.
* Guu et al. (2020)Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.REALM: retrieval-augmented language model pre-training.*CoRR*, abs/2002.08909, 2020.
* Hu et al. (2022)Shengding Hu, Ning Ding, Huadong Wang, Zhiyuan Liu, Jingang Wang, Juanzi Li, Wei Wu, and Maosong Sun.Knowledgeable prompt-tuning: Incorporating knowledge into prompt verbalizer for text classification.In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, pp. 2225–2240. Association for Computational Linguistics, 2022.
* Jang et al. (2017)Eric Jang, Shixiang Gu, and Ben Poole.Categorical reparameterization with gumbel-softmax.In *5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings*. OpenReview.net, 2017.
* Johnson et al. (2019)Jeff Johnson, Matthijs Douze, and Hervé Jégou.Billion-scale similarity search with GPUs.*IEEE Transactions on Big Data*, 7(3):535–547, 2019.
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.Dense passage retrieval for open-domain question answering.In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pp. 6769–6781, 2020.
* Khandelwal et al. (2020)Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis.Generalization through memorization: Nearest neighbor language models.In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*, 2020.
* Lewis et al. (2020)Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.Retrieval-augmented generation for knowledge-intensive NLP tasks.In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.
* Li et al. (2022)Zonglin Li, Ruiqi Guo, and Sanjiv Kumar.Decoupled context processing for context augmented language modeling.In *NeurIPS*, 2022.
* Lin \& Byrne (2022)Weizhe Lin and Bill Byrne.Retrieval Augmented Visual Question Answering with Outside Knowledge.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 11238–11254. Association for Computational Linguistics, 2022.
* Liu et al. (2019a)Hanxiao Liu, Karen Simonyan, and Yiming Yang.DARTS: differentiable architecture search.In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019a.
* Liu et al. (2019b)Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.Roberta: A robustly optimized BERT pretraining approach.*CoRR*, abs/1907.11692, 2019b.
* Mao et al. (2022)Yu Mao, Yufei Cui, Tei-Wei Kuo, and Chun Jason Xue.Accelerating general-purpose lossless compression via simple and scalable parameterization.In João Magalhães, Alberto Del Bimbo, Shin’ichi Satoh, Nicu Sebe, Xavier Alameda-Pineda, Qin Jin, Vincent Oria, and Laura Toni (eds.), *MM ’22: The 30th ACM International Conference on Multimedia, Lisboa, Portugal, October 10 - 14, 2022*, pp. 3205–3213. ACM, 2022.
* Rippel et al. (2014)Oren Rippel, Michael A. Gelbart, and Ryan P. Adams.Learning ordered representations with nested dropout.In *Proceedings of the 31th International Conference on Machine Learning, ICML 2014, Beijing, China, 21-26 June 2014*, volume 32 of *JMLR Workshop and Conference Proceedings*, pp. 1746–1754. JMLR.org, 2014.
* Shi et al. (2022)Ensheng Shi, Yanlin Wang, Wei Tao, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, and Hongbin Sun.RACE: Retrieval-augmented Commit Message Generation.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 5520–5530. Association for Computational Linguistics, 2022.
* Vaswani et al. (2017)Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.Attention is all you need.In *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pp. 5998–6008, 2017.
* Wang et al. (2019)Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman.GLUE: A multi-task benchmark and analysis platform for natural language understanding.In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019.
* Yu et al. (2023)Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang.GENREAD: Generate rather than Retrieve: Large Language Models are Strong Context Generators.In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*, 2023.
* Zhang et al. (2022)Ningyu Zhang, Luoqiu Li, Xiang Chen, Shumin Deng, Zhen Bi, Chuanqi Tan, Fei Huang, and Huajun Chen.Differentiable prompt makes pre-trained language models better few-shot learners.In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*, 2022.
* Zhou et al. (2023)Shuyan Zhou, Uri Alon, Frank F. Xu, Zhengbao Jiang, and Graham Neubig.DocPrompting: Generating Code by Retrieving the Docs.In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*, 2023.

Appendix A Templates on All Tasks
---------------------------------

Table [4](#A1.T4 "Table 4 ‣ Appendix A Templates on All Tasks ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion") provides an overview of the manual templates and selected label words used for each dataset in our experiments. These templates and label words were created following LM-BFF *Gao et al. ([2021](#bib.bib14 ""))*.

*Table 4: Templates and label words that we used in our experiments.*

| Task | Prompts | Label word |
| --- | --- | --- |
| SST-2 | [CLS] $x$ It was [MASK]. [SEP] | “0”:“terrible”, “1”:“great” |
| SST-5 | [CLS] $x$ It was [MASK]. [SEP] | “0”:“terrible”,“1”: “bad”, |
|  |  | “2”: “okay”,“3”: “good”,“4”: “great” |
| MR | [CLS] $x$ It was [MASK]. [SEP] | “0”:“terrible”, “1”:“great” |
| CR | [CLS] $x$ It was [MASK]. [SEP] | “0”:“terrible”, “1”:“great” |
| MPQA | [CLS] $x$ It was [MASK]. [SEP] | “0”:“terrible”, “1”:“great” |
| SUBJ | [CLS] $x$ This is [MASK]. [SEP] | “0”:“subjective”, “1”:“objective” |
| TREC | [CLS] [MASK] $x$ [SEP] | “0”:“Description”,“1”:“Entity”,“2”:“Expression”, |
|  |  | “3”:“Human”,“4”:“Location”,“5”:“Number” |
| CoLA | [CLS] $x$ It was [MASK]. [SEP] | “0”:“incorrect”, “1”:“correct” |
| MNLI | [CLS] $x_{1}$ ? [MASK], $x_{2}$ [SEP] | “contradiction”: “No”,“entailment”:“Yes”, |
|  |  | “neutral”: “Maybe” |
| MNLI-m | [CLS] $x_{1}$ ? [MASK], $x_{2}$ [SEP] | “contradiction”: “No”,“entailment”:“Yes”, |
|  |  | “neutral”: “Maybe” |
| SNLI | [CLS] $x_{1}$ ? [MASK], $x_{2}$ [SEP] | “contradiction”: “No”,“entailment”:“Yes”, |
|  |  | “neutral”: “Maybe” |
| QNLI | [CLS] $x_{1}$ ? [MASK], $x_{2}$ [SEP] | “not entailment”:“No ”,“entailment”:“Yes” |
| RTE | [CLS] $x_{1}$ ? [MASK], $x_{2}$ [SEP] | “not entailment”:“No ”,“entailment”:“Yes” |
| MRPC | [CLS] $x_{1}$ [MASK], $x_{2}$ [SEP] | “0”:“No”, “1”:“Yes” |
| QQP | [CLS] $x_{1}$ [MASK], $x_{2}$ [SEP] | “0”:“No”, “1”:“Yes” |

Appendix B Results on Full Training Set
---------------------------------------

We conduct experiments on several tasks under the prompt-based setting with the full training set. As shown in Table [5](#A2.T5 "Table 5 ‣ Appendix B Results on Full Training Set ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion"), across all datasets, ReFusion generally demonstrates either comparable or superior performance compared to LM-BFF. The average performance across all tasks in ReFusion surpasses that of LM-BFF by 1.4%. This suggests that ReFusion’s performance superiority is consistent and not dependent on the size of the dataset. This implies that ReFusion is robust and can generalize well across varying amounts of data.

*Table 5: Full training set results compared with LM-BFF.*

| Methods | SST-2 | SST-5 | MR | CR | MPQA | SUBJ | TREC | CoLA | RTE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LM-BFF | 95.0 | 58.7 | 90.8 | 89.4 | 87.8 | 97.0 | 97.4 | 62.6 | 80.9 |
| ReFusion | 95.6 | 61.0 | 92.3 | 91.4 | 84.4 | 97.1 | 97.6 | 62.8 | 85.2 |

Appendix C Where to fuse for retrieval
--------------------------------------

Table [6](#A3.T6 "Table 6 ‣ Appendix C Where to fuse for retrieval ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion") presents the results of experiments assessing where to fuse retrieved information in attention modules (query, key, or value).
The optimal module in which to fuse the retrieved representation varies based on the task. However, in general, all tested fusion modules – whether it be on the ’key and value’, ’query and key’, ’query and value’, or all three (’query, key, and value’) – yield competitive performances across the different benchmarks. This indicates that our model’s performance remains consistent regardless of the specific modules added, demonstrating its robustness.

*Table 6: Where to fuse for retrieval.*

| Module | MPQA | SUBJ | TREC | SNLI | QNLI | RTE |
| --- | --- | --- | --- | --- | --- | --- |
| key and value | 86.3 | 93.3 | 93.0 | 81.0 | 72.1 | 73.3 |
| query and key | 86.6 | 93.6 | 94.2 | 78.6 | 74.8 | 72.6 |
| query and value | 86.7 | 93.9 | 93.0 | 80.1 | 74.8 | 72.6 |
| query and key and value | 86.3 | 92.9 | 94.2 | 79.2 | 76.4 | 72.6 |

Appendix D Effect of retrieval sentences number $k$
---------------------------------------------------

Table [7](#A4.T7 "Table 7 ‣ Appendix D Effect of retrieval sentences number 𝑘 ‣ Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion") illustrates the impact of varying the number of retrieved similarity sentences, denoted as $k$, across different benchmarks. There isn’t a strict linear relationship between the number of sentences retrieved ($k$) and the performance across the benchmarks. The optimal value of $k$ appears to vary based on the dataset. However, in general, when $k$ exceeds 8, the performance achieved on each task becomes comparable. In practical applications, striking a balance between efficient computation and optimal performance is crucial.

*Table 7: The effect of retrieving $k$ sentences.*

| $k$ | 1 | 8 | 64 | 128 |
| --- | --- | --- | --- | --- |
| MPQA | 86.7 | 87.8 | 86.3 | 86.6 |
| SUBJ | 92.6 | 93.8 | 93.3 | 93.0 |
| TREC | 91.0 | 93.0 | 93.0 | 93.6 |
| SNLI | 78.2 | 79.1 | 79.6 | 79.4 |
| QNLI | 74.3 | 74.9 | 74.7 | 74.9 |
| RTE | 72.9 | 72.6 | 71.8 | 72.2 |
