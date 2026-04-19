Cross-Modal Adapter for Text-Video Retrieval
==============================================

Haojun Jiang1Jianke Zhang2Rui Huang1Chunjiang Ge1Zanlin Ni1  
Jiwen Lu1Jie Zhou1Shiji Song1Gao Huang1  
1 Tsinghua University, BNRist2 Beijing Institute of Technology  
{jhj20, hr20, gecj20, nzl22}@mails.tsinghua.edu.cn, zhangjianke53@gmail.com  
{lujiwen, jzhou, shijis, gaohuang}@tsinghua.edu.cnThis work was done during an internship at Tsinghua.Corresponding author.

###### Abstract

Text-video retrieval is an important multi-modal learning task, where the goal is to retrieve the most relevant video for a given text query.
Recently, pre-trained models, e.g., CLIP, show great potential on this task.
However, as pre-trained models are scaling up, fully fine-tuning them on text-video retrieval datasets has a high risk of overfitting.
Moreover, in practice, it would be costly to train and store a large model for each task.
To overcome the above issues, we present a novel Cross-Modal Adapter for parameter-efficient fine-tuning.
Inspired by adapter-based methods, we adjust the pre-trained model with a few parameterization layers.
However, there are two notable differences.
First, our method is designed for the multi-modal domain.
Secondly, it allows early cross-modal interactions between CLIP’s two encoders.
Although surprisingly simple, our approach has three notable benefits:
(1) reduces 99.6% of fine-tuned parameters, and alleviates the problem of overfitting,
(2) saves approximately 30% of training time, and
(3) allows all the pre-trained parameters to be fixed, enabling the pre-trained model to be shared across datasets.
Extensive experiments demonstrate that, without bells and whistles, it achieves superior or comparable performance compared to fully fine-tuned methods on MSR-VTT, MSVD, VATEX, ActivityNet, and DiDeMo datasets. The code will be available at [https://github.com/LeapLabTHU/Cross-Modal-Adapter](https://github.com/LeapLabTHU/Cross-Modal-Adapter "").

1 Introduction
--------------

Videos are playing an essential role in retaining and disseminating information.
Nowadays, on video platforms like YouTube, users search for the videos they are interested in through natural language queries.
Consequently, how to return the most relevant videos for a given text query is becoming an increasingly popular research topic in both the industry and academia.

<img src='x1.png' alt='Refer to caption' title='' width='181' height='96' />

*Figure 1: Comparison with fully fine-tuned method*[[35](#bib.bib35 "")]* and prompt-based methods*[[56](#bib.bib56 ""), [23](#bib.bib23 ""), [28](#bib.bib28 "")]*. Left: Text-to-video retrieval R@1 results on five datasets. Our method outperforms all the prompt-based methods and even surpasses the fully fine-tuned method. Right: On MSR-VTT, our approach reduces 99.6% trained parameters compared to the fully fine-tuned method.*

In the field of text-video retrieval, an important category of the method is pre-training*[[15](#bib.bib15 ""), [48](#bib.bib48 ""), [57](#bib.bib57 ""), [3](#bib.bib3 ""), [30](#bib.bib30 ""), [42](#bib.bib42 ""), [40](#bib.bib40 "")]* that aim at learning better transferable text-video representations.
A representative work is the Contrastive Language-Image Pre-training model*[[40](#bib.bib40 "")]*, i.e., CLIP, which exhibits great potential in handling this task.
Numerous works*[[35](#bib.bib35 ""), [16](#bib.bib16 ""), [12](#bib.bib12 ""), [36](#bib.bib36 ""), [4](#bib.bib4 ""), [14](#bib.bib14 ""), [9](#bib.bib9 ""), [34](#bib.bib34 "")]* have focused on adapting the CLIP to the text-video retrieval domain.
Although these methods achieve encouraging performance on multiple benchmarks*[[49](#bib.bib49 ""), [6](#bib.bib6 ""), [29](#bib.bib29 ""), [1](#bib.bib1 ""), [46](#bib.bib46 "")]*, they need to fully fine-tune the CLIP model.
As the current trend of scaling up pre-trained models, this paradigm leads to two issues:
(1) it has a high risk of overfitting on downstream datasets, and
(2) it is costly to train and store an entirely new large model for each dataset in practice.

An elegant solution to the above-mentioned problems is Adapter*[[20](#bib.bib20 "")]*, which has achieved great success in natural language processing.
For example, it attains comparable performance with fully fine-tuned methods by training only 3.6% parameters.
In the text-video retrieval task, there are two modalities and each with a feature encoder. Thus, a straightforward way is to design adapters separately for each encoder.
However, this naive scheme forbids any early cross-modal interaction, and may lead to a sub-optimal solution.
Meanwhile, introducing the early cross-modal interaction inside the CLIP’s encoders through adapters is non-trivial.
Since it requires interactions between every pair of video and text query, this will change the computational complexity from $O(N)$ to $O(N^{2})$ where $N$ is the number of text-video pairs.
Such a computational cost is unaffordable in practical applications.

To address the above challenge, we propose a novel method named Cross-Modal Adapter.
The key idea is to enable early cross-modal interactions by sharing adapters’ weights between two modalities rather than introducing explicit feature-level interactions.
By adopting this weight-sharing mechanism, computational complexity can be kept as $O(N)$, since each video or text feature can be obtained independently.
Furthermore, such a scheme allows an implicit cross-modal interaction, which will facilitate the re-alignment of CLIP’s vision and language feature spaces for the text-video retrieval task.

We conduct extensive experiments on five popular datasets, i.e., MSR-VTT*[[49](#bib.bib49 "")]* MSVD*[[6](#bib.bib6 "")]*, VATEX*[[46](#bib.bib46 "")]*, ActivityNet*[[29](#bib.bib29 "")]*, and DiDeMo*[[1](#bib.bib1 "")]*, to demonstrate the effectiveness of our method.
First, without bells and whistles, Cross-Modal Adapter attains comparable performance as the fully fine-tuned method*[[35](#bib.bib35 "")]* while optimizes much fewer parameters, e.g., 0.52M v.s. 123.52M on MSR-VTT.
Secondly, our method saves approximately 30% of training costs.
Last, our method outperforms prompt-based parameter-efficient methods*[[56](#bib.bib56 ""), [23](#bib.bib23 ""), [28](#bib.bib28 "")]* by a large margin.
In summary, this paper makes three-fold contributions:

* •

    We propose a parameter-efficient Cross-Modal Adapter for text-video retrieval task. To the best of our knowledge, we are the first to investigate adapter-based parameter-efficient transfer learning for the text-video retrieval domain.

* •

    We propose a weight-sharing mechanism to enable early cross-modal interaction without introducing huge additional computations.

* •

    Extensive experiments show that our approach can significantly reduce 99.6% of fine-tuned parameters without performance sacrifice compared to the fully fine-tuned method.

2 Related Work
--------------

### 2.1 Text-video Retrieval

Text-video retrieval is becoming one of the most important multi-modal learning topics*[[2](#bib.bib2 ""), [52](#bib.bib52 ""), [24](#bib.bib24 ""), [26](#bib.bib26 "")]* with advances in both computer vision*[[21](#bib.bib21 ""), [18](#bib.bib18 ""), [11](#bib.bib11 ""), [50](#bib.bib50 ""), [47](#bib.bib47 ""), [22](#bib.bib22 ""), [17](#bib.bib17 "")]* and natural language processing*[[10](#bib.bib10 ""), [44](#bib.bib44 ""), [5](#bib.bib5 ""), [41](#bib.bib41 ""), [25](#bib.bib25 "")]*.
Recently, great progress has been achieved in the text-video retrieval domain with the advances in pre-training*[[15](#bib.bib15 ""), [48](#bib.bib48 ""), [57](#bib.bib57 ""), [3](#bib.bib3 ""), [30](#bib.bib30 ""), [42](#bib.bib42 ""), [40](#bib.bib40 ""), [43](#bib.bib43 "")]*.
Among these works, Contrastive Language-Image Pre-training model*[[40](#bib.bib40 "")]*, i.e., CLIP, exhibits great potential on multiple downstream tasks.
Recently, CLIP4Clip*[[35](#bib.bib35 "")]* demonstrates that fully fine-tuning the CLIP with a similarity calculator can achieve promising performance on various text-video retrieval datasets.
As a result, a series of works focus on excavating the power of CLIP by designing parameter-rich cross-modal fusion modules*[[16](#bib.bib16 ""), [36](#bib.bib36 ""), [12](#bib.bib12 ""), [14](#bib.bib14 "")]* on top of it or plugging a token selection module*[[34](#bib.bib34 "")]* into it.
All these works adopt the fully fine-tuning paradigm following CLIP4Clip.
However, as the current trend of scaling up the pre-trained models, e.g., the latest BEIT-3*[[45](#bib.bib45 "")]* has 1.9B parameter, such a scheme is easy to overfit, computation-intensive, and time-consuming when applied to downstream datasets.
Contrary to these works, we make the first attempt to explore parameter-efficient transfer learning on the text-video retrieval domain and achieve comparable performance to fully fine-tuning with much fewer parameters.

### 2.2 Parameter Efficient Tuning

Pre-trained models become prevalent in recent years*[[10](#bib.bib10 ""), [40](#bib.bib40 "")]*. Since fine-tuning the whole model has a large computational cost and is time-consuming*[[35](#bib.bib35 "")]*, parameter efficient tuning (PET) has been proposed to efficiently adapt the pre-trained models to downstream tasks*[[39](#bib.bib39 ""), [54](#bib.bib54 "")]*. Among them, adapters*[[37](#bib.bib37 ""), [27](#bib.bib27 ""), [7](#bib.bib7 "")]* and prompt learning*[[56](#bib.bib56 ""), [23](#bib.bib23 ""), [28](#bib.bib28 "")]* are two dominant lines of research.

<img src='x2.png' alt='Refer to caption' title='' width='367' height='140' />

*Figure 2: Overview of the Cross-Modal Adapter. Left is the scheme that naively plugs independent uni-modal adapters into the video and text encoders. Note that there is no cross-modal interaction. Middle is our cross-modal adapter which enables early cross-modal interaction via a weight-sharing mechanism. Right is the implementation details of our cross-modal adapter.*

Adapters *[[20](#bib.bib20 "")]* are light-weighted layers inserted in the pre-trained model. During fine-tuning, only the weights of adapters are updated, and other parameters are frozen. Task-specific information is learned and knowledge learned in the pre-training phase is also preserved. In the NLP community, Mahabadi et al. *[[37](#bib.bib37 "")]* proposes to use hyper-networks to generate parameters for adapters in each layer. Furthermore, Mahabadi et al. *[[27](#bib.bib27 "")]* proposes Compactor, a parameterized hyper-complex multiplication layer, to reduce the parameter of adapter modules. Recently, in the CV community, Chen et al. *[[7](#bib.bib7 "")]* proposes a Conv-Adapter which consists of a point-wise convolution and depth-wise convolution in a bottleneck structure to enable adapters in convolutional neural networks. Previous works explore either vision or language modality, while our method investigates the effectiveness of adapters in the multi-modality domain.

Prompt learning *[[39](#bib.bib39 "")]*, first proposed in NLP, can probe pre-trained knowledge for downstream tasks with human-designed or learnable prompt input. Concretely, a natural language task instruction (prompt) is prepended in the input, and the whole pre-trained model is frozen. The prompt could be discrete or continuous embedding*[[39](#bib.bib39 ""), [32](#bib.bib32 ""), [31](#bib.bib31 "")]*.
Since multi-modal pre-trained models, e.g., CLIP*[[40](#bib.bib40 "")]*, have shown impressive performance on various tasks, context optimization*[[56](#bib.bib56 "")]*(CoOp), has been proposed to prompt CLIP on image classification tasks. CoOp optimizes continuous learnable prompts for the text encoder to learn better label distribution for image classification tasks. Visual prompt tuning*[[23](#bib.bib23 "")]*(VPT) leverages prompt in each layer of vision models. Recently, a multi-modal prompt tuning method, MaPLe*[[28](#bib.bib28 "")]*, has been proposed to fully exploit the knowledge of CLIP. Contrary to these works, our method is based on the adapter, and experiments demonstrate that ours outperforms prompt-based methods on various datasets.

3 Method
--------

In this section, we illustrate each component of our method in detail. First, we provide a thorough description of our video and text encoders in [Sec. 3.1](#S3.SS1 "3.1 Feature Encoders ‣ 3 Method ‣ Cross-Modal Adapter for Text-Video Retrieval"). Then, we explain the architecture of the cross-modal adapter in [Sec. 3.2](#S3.SS2 "3.2 Cross-Modal Adapter ‣ 3 Method ‣ Cross-Modal Adapter for Text-Video Retrieval"). Finally, we present the mechanism of a parameter-free similarity calculator in [Sec. 3.3](#S3.SS3 "3.3 Parameter-free Similarity Calculator ‣ 3 Method ‣ Cross-Modal Adapter for Text-Video Retrieval").

### 3.1 Feature Encoders

Given a set of videos $\mathcal{V}\=\left{v_{1},v_{2},\ldots,v_{n}\right}$ and its corresponding text queries $\mathcal{T}\=\left{t_{1},t_{2},\ldots,t_{n}\right}$, we perform text-video retrieval by calculating the similarity between the text query $t_{j}$ and each video $v_{i}$ which consists of $\left|v_{i}\right|$ frames $\left{v_{i,1},v_{i,2},\ldots,v_{i,|v_{i}|}\right}$, and return the video with the highest similarity score. To achieve this goal, we first need to obtain the feature of each video and text query. Thus, we will illustrate the video and text encoders in the following.

Video Encoder. We adopt the visual backbone of CLIP (ViT-B/32) as our video encoder, a 12-layer transformer.
Following CLIP4Clip*[[35](#bib.bib35 "")]*, we extract frames from a video and encode them with the video encoder separately.
For example, consider the video $v_{i}$, the $j$-th frame $\mathbf{v}_{i,j}\in\mathbb{R}^{H\times W\times C}$ is first split into non-overlapping patches with a size of $P\times P$.
Then the patches are mapped to embeddings with a linear projection and added with learnable positional embeddings.
Thus, a frame is transformed into a sequence of embeddings. Besides, a learnable [CLS] token, which represents the global information of the frame, is added at the start of the sequence.
Formally, we denote the input and features of the video encoder as $\mathbf{p}_{i,j,l}\in\mathbb{R}^{(M+1)\times D_{v}}$, where subscript $l$ means the layer index, $M$ is the total number of patches, and $D_{v}$ is the hidden dimension. Specifically, the input of the video encoder is denoted as $\mathbf{p}_{i,j,0}$.

Inside the video encoder, each layer consists of a self-attention module (MSA), a feedforward MLP, and layernomrs (LN).
For $j$-th frame of video $v_{i}$ , the modules in $l$-th layer perform calculations as follows:

|  | $\displaystyle\mathbf{\hat{p}}_{i,j,l}$ | $\displaystyle\=\mathrm{MSA}(\mathrm{LN}(\mathbf{p}_{i,j,l-1}))+\mathbf{p}_{i,j,l-1},\ l\=1,...,L,$ |  | (1) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\mathbf{p}_{i,j,l}$ | $\displaystyle\=\mathrm{MLP}(\mathrm{LN}(\mathbf{\hat{p}}_{i,j,l}))+\mathbf{\hat{p}}_{i,j,l},\ \ \ \ \ \mathbf{p}_{i,j,l}\in\mathbb{R}^{D_{v}}.$ |  | (2) |
| --- | --- | --- | --- | --- |

After the patches have been processed by all the layers, we regard the feature of the [CLS] token as the global representation for a frame. The final frame representation is obtained as follows:

|  | $\displaystyle\boldsymbol{f}_{i,j}$ | $\displaystyle\=\mathrm{Proj}(\mathrm{LN}(\mathbf{p}_{i,j,L}^{0})),\ \ \ \ \ \ \ \ \ \boldsymbol{f}_{i,j}\in\mathbb{R}^{D_{t}},$ |  | (3) |
| --- | --- | --- | --- | --- |

where $\mathrm{Proj}(\cdot)$ is a linear projection layer to handle the dimension mismatch between video and text encoders, and $D_{t}$ is the hidden dimension of the text encoder.
Therefore, for video $v_{i}$, we can get all the frame representations as $\left{\boldsymbol{f}_{i,1},\boldsymbol{f}_{i,2},\ldots,\boldsymbol{f}_{i,|v_{i}|}\right}$.
The video feature is generated based on the given text query feature, as detailed in [Sec. 3.3](#S3.SS3 "3.3 Parameter-free Similarity Calculator ‣ 3 Method ‣ Cross-Modal Adapter for Text-Video Retrieval").

Text Encoder. We use the linguistic backbone of CLIP as our text encoder, which is also a 12-layer transformer. The structure of the transformer layer is the same as the video encoder. The only difference is that the hidden dimension is 512 rather than 768.

Like the video encoder, the input text query $t_{i}\in\mathcal{T}$ is first transformed into a sequence of embeddings by a tokenizer.
Then, a [CLS] token and a [SEP] token are added to the start and end of the sequence, respectively.
Subsequently, this sequence is going through all the layers of the text encoder.
Finally, following CLIP, we adopt the feature of [SEP] token at the last layer as global representation $\boldsymbol{t}_{i}\in\mathbb{R}^{D_{t}}$ of a text query.

### 3.2 Cross-Modal Adapter

Pre-training methods*[[15](#bib.bib15 ""), [48](#bib.bib48 ""), [40](#bib.bib40 "")]* have shown great potential for transferring to downstream tasks.
Thus, previous works*[[35](#bib.bib35 ""), [16](#bib.bib16 ""), [34](#bib.bib34 "")]* mainly focus on adapting the powerful pre-trained CLIP*[[40](#bib.bib40 "")]* model to the text-video retrieval domain by fully fine-tuning the two encoders.
However, as the pre-trained models are scaling up quickly, this paradigm has a high risk of overfitting and is costly.
Inspired by recent parameter-efficient transfer learning works*[[20](#bib.bib20 ""), [37](#bib.bib37 "")]* in natural language processing, we proposed a novel Cross-Modal Adapter for parameter-efficient tuning on text-video retrieval task.
In the following paragraphs, we first introduce the uni-modal adapter.

Uni-modal Adapter. Our method is built based on Adapter*[[20](#bib.bib20 "")]* which is designed for parameter-efficient transfer learning in NLP. The authors adapt the large pre-trained model to downstream tasks by inserting a few learnable adapter modules inside each transformer layer. It attains good performance by fine-tuning only adapter modules, which contain merely 3.6% of the pre-trained model’s parameters. Specifically, the adapter module follows a bottleneck design to reduce the number of parameters, which consists of a down-projection linear layer, a non-linear layer, and an up-projection linear layer.

Formally, given an input feature $\mathbf{x}\in\mathbb{R}^{1\times d}$, the feed-forward function of the adapter module can be written as:

|  | $\text{Adapter}(\mathbf{x})\=\mathbf{x}+\sigma(\mathbf{x}\boldsymbol{\mathrm{W}}_{\text{down}})\boldsymbol{\mathrm{W}}_{\text{up}},$ |  | (4) |
| --- | --- | --- | --- |

where $\boldsymbol{\mathrm{W}}_{\text{down}}\in\mathbb{R}^{d\times r}$, $\boldsymbol{\mathrm{W}}_{\text{up}}\in\mathbb{R}^{r\times d}(r\ll d)$ are down-projection and up-projection weights respectively, and $\sigma(\cdot)$ is non-linear layer which is implemented as GELUs*[[19](#bib.bib19 "")]*. Besides, the adapter is initialized as a near-identity mapping to stabilize the training process.

Cross-Modal Adapter. A straightforward way to achieve parameter-efficient transfer learning for the multimodal model*[[40](#bib.bib40 "")]* is plugging independent adapter modules inside each encoder as shown in [Fig. 2](#S2.F2 "In 2.2 Parameter Efficient Tuning ‣ 2 Related Work ‣ Cross-Modal Adapter for Text-Video Retrieval") (left). However, this naive scheme forbids cross-modal communication in early layers, which might lead to a sub-optimal solution. Previous work*[[38](#bib.bib38 "")]* shows a model with early cross-modal interaction boosts retrieval performance significantly. Furthermore, we empirically demonstrate such a naive scheme is sub-optimal, as shown in [Tab. 2](#S4.T2 "In 4.1 Main Results ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval"). Meanwhile, introducing the cross-modal interaction among shallow layers in CLIP is non-trivial. It requires interactions between every pair of video and text query, which leads to quadratic computational costs with respect to the sample number. In practice, such a surge in computational cost will lead to a slow retrieval process and a bad user experience.

To address the above challenge, we propose a novel Cross-Modal Adapter to enable implicit early cross-modal interaction method via a weight-sharing mechanism as shown in [Fig. 2](#S2.F2 "In 2.2 Parameter Efficient Tuning ‣ 2 Related Work ‣ Cross-Modal Adapter for Text-Video Retrieval") (middle). Our inspiration comes from previous work*[[51](#bib.bib51 "")]* that facilitates the alignment of vision and text spaces by sharing the weights of self-attention and feedforward layers between two modalities when pre-training. Differently, we apply the weight-sharing mechanism in the up-projection layer of adapters, as shown in [Fig. 2](#S2.F2 "In 2.2 Parameter Efficient Tuning ‣ 2 Related Work ‣ Cross-Modal Adapter for Text-Video Retrieval") (right).

Formally, given an input feature $\mathbf{x}\in\mathbb{R}^{1\times d}$, where $\mathbf{x}$ can be a patch feature from a video frame or a token feature from a text query, the bottleneck feature $\mathbf{z}\in\mathbb{R}^{1\times r}$ can be obtained as follows:

|  | $\mathbf{z}\=\sigma(\mathbf{x}\boldsymbol{\mathrm{W}}_{\text{down}}),$ |  | (5) |
| --- | --- | --- | --- |

where $\boldsymbol{\mathrm{W}}_{\text{down}}\in\mathbb{R}^{d\times r}$ and $\sigma(\cdot)$ is non-linear layer. Then, the up-projection layer with a weight-sharing mechanism and the output of the cross-modal adapter can be written as:

|  | $\text{Adapter}_{\text{CM}}(\mathbf{x})\=\mathbf{x}+\text{Concat}\left[\mathbf{z}\boldsymbol{\mathrm{W}}_{\text{up,unique}},\text{ }\mathbf{z}\boldsymbol{\mathrm{W}}_{\text{up,share}}\right],$ |  | (6) |
| --- | --- | --- | --- |

where $\boldsymbol{\mathrm{W}}_{\text{up,share}}\in\mathbb{R}^{r\times d_{s}},0<d_{s}\leq d$ is modality-shared weights while $\boldsymbol{\mathrm{W}}_{\text{up,unique}}\in\mathbb{R}^{r\times(d-d_{s})}$ is modality-specific weights. Finally, the cross-modal adapter is inserted after self-attention and feedforward MLP modules for all layers.

<img src='x3.png' alt='Refer to caption' title='' width='172' height='122' />

*Figure 3: Parameter-free similarity calculator. The query-aware video feature is generated by weighted averaging of all the frame features based on the score between each frame and text query.*

### 3.3 Parameter-free Similarity Calculator

After extracting all the features, the last essential step is calculating the similarity between every pair of video and text query.
In contrast to the previous work*[[16](#bib.bib16 "")]* that introduces a parameter-rich similarity calculator, we apply a parameter-free one ([Fig. 3](#S3.F3 "In 3.2 Cross-Modal Adapter ‣ 3 Method ‣ Cross-Modal Adapter for Text-Video Retrieval")) to minimize the number of fine-tuned parameters following *[[4](#bib.bib4 "")]*.
First, we generate query-aware video features, which aggregate important information from each frame based on the given text query feature.
Concretely, we calculate the inner product $\alpha_{j}$ between the text query feature $\boldsymbol{t}$ and $j$-th frame feature $\boldsymbol{f}_{i,j}$ of video $v_{i}$.
Formally, we have the following equation:

|  | $\alpha_{j}\=\langle\,\boldsymbol{t},\;\boldsymbol{f}_{i,j}\rangle,\ \ \ j\in{1,\ldots,\left|v_{i}\right|},$ |  | (7) |
| --- | --- | --- | --- |

where $\langle\cdot,\cdot\rangle$ represents inner product.
Then, we obtain a final query-aware embedding $\boldsymbol{\hat{v}}_{i}$ for the video $v_{i}$ by weighted averaging all video frame features:

|  | $\hat{\alpha}_{j}\=\frac{e^{\alpha_{j}/\tau}}{\sum_{k\in{1,\ldots,\left|v_{i}\right|}}e^{\alpha_{k}/\tau}},\ \ \boldsymbol{\hat{v}}_{i}\=\sum_{j\in{1,\ldots,\left|v_{i}\right|}}\hat{\alpha}_{j}\boldsymbol{f}_{i,j},$ |  | (8) |
| --- | --- | --- | --- |

where $\tau$ is the temperature hyper-parameter, which determines the way of average weighting. For instance, if $\tau$ is very small, the embedding of the whole video will be almost the same as that of the most relevant frame to the query feature $\boldsymbol{t}$. Contrarily, if $\tau$ is very large, this simply becomes a uniform average without weights. Though its simplicity, this parameter-free aggregation method can effectively perform the temporal modeling for the video by considering different degrees of text-frame relevance.

For a batch with $n$ text-video pairs, we calculate the similarities between all the video features and all the text query features, resulting in a $n\times n$ similarity score map. Then, we adopt a symmetric cross-entropy loss to optimize the model. Formally, we have the following equations:

|  | $\displaystyle\mathcal{L}_{t2v}$ | $\displaystyle\=-\frac{1}{n}\sum_{i}^{n}\log\frac{\exp\left(s\left(\boldsymbol{\hat{v}}_{i},\boldsymbol{t}_{i}\right)\right)}{\sum_{j\=1}^{n}\exp\left(s\left(\boldsymbol{\hat{v}}_{j},\boldsymbol{t}_{i}\right))\right.},$ |  | (9) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\mathcal{L}_{v2t}$ | $\displaystyle\=-\frac{1}{n}\sum_{i}^{n}\log\frac{\exp\left(s\left(\boldsymbol{\hat{v}}_{i},\boldsymbol{t}_{i}\right)\right)}{\sum_{j\=1}^{n}\exp\left(s\left(\boldsymbol{\hat{v}}_{i},\boldsymbol{t}_{j}\right))\right.},$ |  | (10) |
| --- | --- | --- | --- | --- |

|  | $\mathcal{L}\=\frac{1}{2}(\mathcal{L}_{v2t}+\mathcal{L}_{t2v}),$ |  | (11) |
| --- | --- | --- | --- |

where $s(\cdot,\cdot)$ denotes cosine distance.

4 Experiments
-------------

We first introduce the five datasets, implementation details, evaluation metrics, and the baselines we compare.

MSR-VTT *[[49](#bib.bib49 "")]* contains 10,000 videos, each labeled with about 20 captions. The lengths of videos range from 10 to 32 seconds. We adopt the 9k training split. For the test data, we use the 1k-A test set from *[[53](#bib.bib53 "")]*, which is comprised of 1,000 text-video pairs.

MSVD *[[6](#bib.bib6 "")]* has 1,970 videos, each with approximately 40 associated captions in English. The duration of videos range from 1 to 62 seconds. We use the standard split of 1,200, 100, and 670 videos for training, validation, and test.

VATEX *[[46](#bib.bib46 "")]* is a multilingual dataset with 34,991 video clips.
We adopt the official training split with 25,991 videos and report the performance on the validation split with 1,500 videos following HGR*[[8](#bib.bib8 "")]*.

ActivityNet *[[29](#bib.bib29 "")]* consists of 20,000 YouTube videos. We concatenate all the descriptions of a video and conduct video-paragraph retrieval following the setting from *[[55](#bib.bib55 ""), [13](#bib.bib13 "")]*. The model is evaluated on the ‘val1’ split.

DiDeMo *[[1](#bib.bib1 "")]* contains over 10,000 videos with 40,000 sentences.
There are 8,395, 1,065, and 1,004 videos in the training, validation, and test set, respectively.
Following *[[33](#bib.bib33 ""), [30](#bib.bib30 ""), [3](#bib.bib3 "")]*, we concatenate all sentence descriptions of a video to evaluate video-paragraph retrieval.

Implementation details: Following CLIP4Clip*[[35](#bib.bib35 "")]*, we adopt the video encoder and text encoder from the pre-trained CLIP (ViT-B/32).
The cross-modal adapter is applied in every layer of two encoders.
We set the bottleneck dimension $r$ as {8, 8, 16, 16, 16} and weight-sharing dimension $d_{s}$ as {16, 8, 16, 64, 512} for MSR-VTT, DiDeMo, MSVD, VATEX, ActivityNet.
For training settings, we fine-tune our model with the Adam optimizer for 5 epochs (or 20 epochs for the ActivityNet).
Warm-up is applied in the first 10% of the training process following CLIP4Clip.
The learning rate is 1e-5 and decayed with a cosine schedule.
The batch size is set to 128, and the temperature $\tau$ for our parameter-free similarity calculator is set to 5.
We set the caption token length as 32 and the frame length as 12 by default.
For ActivityNet and DiDeMo, the caption token and frame length are set to 64.
More details are in the Appendix.

Evaluation metrics: To measure the performance of our proposed method, we use standard retrieval metrics: recall at rank K (R@K, higher is better) and mean rank (MnR, lower is better).
R@K calculates the proportion of test samples found in the top-K retrieved results. We report results for R@1, R@5, and R@10. MnR calculates the mean ranking of the test samples in the retrieval ranking list.

<img src='x4.png' alt='Refer to caption' title='' width='484' height='228' />

*Figure 4: Retrieval results on MSR-VTT dataset. The four curves above are the retrieval results for text-to-video, while below are those for video-to-text. The horizontal coordinate denotes the number of fine-tuned parameters. To make a fair comparison, all the prompt-based methods and $\text{CLIP4Clip}^{*}$ are implemented with our parameter-free similarity calculator. The curves of our method are realized mainly by changing the bottleneck dimension, while prompt-based methods are achieved by changing the prompt token number. Besides, the curves for the CLIP4Clip are implemented by freezing a varying number of layers. Please refer to the Appendix for more details.*

| Type | Methods | | Trained | | --- | | Params. | | R@1↑ | R@5↑ | R@10↑ | MnR↓ | Methods | | Trained | | --- | | Params. | | R@1↑ | R@5↑ | R@10↑ | MnR↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | MSVD | | | | | | DiDeMo | | | | | |
| ParameterEfficient | CoOp[[56](#bib.bib56 "")] | 0.002M | 42.1 | 70.7 | 80.9 | 12.8 | CoOp[[56](#bib.bib56 "")] | 0.002M | 32.5 | 60.4 | 71.7 | 29.2 |
| | VPT[[23](#bib.bib23 "")] | 0.18M | 44.2 | 72.6 | 81.8 | 12.6 | VPT[[23](#bib.bib23 "")] | 0.05M | 35.0 | 61.2 | 71.7 | 29.9 |
| VL-Prompt | 0.08M | 44.5 | 74.8 | 84.0 | 11.0 | VL-Prompt | 0.08M | 35.3 | 64.1 | 72.8 | 24.0 |
| MaPLe*[[28](#bib.bib28 "")] | 4.76M | 44.0 | 74.8 | 84.0 | 10.9 | MaPLe*[[28](#bib.bib28 "")] | 4.76M | 37.5 | 65.9 | 75.8 | 20.8 |
| Ours | 1.00M | 47.4 | 76.6 | 85.0 | 10.2 | Ours | 0.52M | 45.0 | 71.9 | 81.3 | 16.3 |
| FullyFine-tune | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 46.2 | 76.1 | 84.6 | 10.0 | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 43.4 | 70.2 | 80.6 | 17.5 |
| | CLIP2Video[[12](#bib.bib12 "")] | 142.46M | 47.0 | 76.8 | 85.9 | 9.6 | TS2-Net[[34](#bib.bib34 "")] | 139.35M | 41.8 | 71.6 | 82.0 | 14.8 |
| X-Pool[[16](#bib.bib16 "")] | 152.59M | 47.2 | 77.4 | 86.0 | 9.3 | X-CLIP[[36](#bib.bib36 "")] | 137.01M | 45.2 | 74.0 | - | 14.6 |
|  | VATEX | | | | | | ActivityNet | | | | | |
| ParameterEfficient | CoOp[[56](#bib.bib56 "")] | 0.033M | 50.0 | 82.4 | 90.5 | 5.8 | CoOp[[56](#bib.bib56 "")] | 0.008M | 28.1 | 56.4 | 70.2 | 17.5 |
| | VPT[[23](#bib.bib23 "")] | 0.05M | 52.8 | 83.9 | 91.6 | 5.7 | VPT[[23](#bib.bib23 "")] | 0.05M | 30.8 | 59.3 | 73.2 | 16.3 |
| VL-Prompt | 0.31M | 54.7 | 86.4 | 93.4 | 4.4 | VL-Prompt | 0.08M | 33.2 | 62.9 | 76.1 | 11.6 |
| MaPLe*[[28](#bib.bib28 "")] | 4.76M | 54.9 | 86.3 | 93.1 | 4.5 | MaPLe*[[28](#bib.bib28 "")] | 4.76M | 32.3 | 62.3 | 75.7 | 11.7 |
| Ours | 0.99M | 59.3 | 89.8 | 94.9 | 3.7 | Ours | 0.81M | 41.5 | 71.2 | 82.5 | 8.0 |
| FullyFine-tune | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 59.3 | 89.6 | 95.0 | 3.7 | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 41.4 | 72.7 | 83.6 | 8.0 |
| | CLIP2Video[[12](#bib.bib12 "")] | 142.46M | 57.4 | 90.0 | 95.5 | 3.6 | TS2-Net[[34](#bib.bib34 "")] | 139.35M | 41.0 | 73.6 | 84.5 | 8.4 |
| TS2-Net[[34](#bib.bib34 "")] | 139.35M | 59.1 | 90.0 | 95.2 | 3.5 | X-CLIP[[36](#bib.bib36 "")] | 137.01M | 44.3 | 74.1 | - | 7.9 |

*Table 1: Text-to-video retrieval results on MSVD, DiDeMo, VATEX, and ActivityNet datasets. We compare our method with other parameter-efficient transfer learning methods and list the CLIP-based fully fine-tuned methods for reference. The parameter differences on 4 datasets of our method are primarily due to the bottleneck dimension, while the prompt-based methods are impacted by the prompt token number. Please refer to the Appendix for the results of video-to-text retrieval.*

Baselines: There are 5 major baselines for comparison. More details about the baselines are in the Appendix.

1. CLIP4Clip*[[35](#bib.bib35 "")]* is a strong, fully fine-tuned baseline.

2. CoOp*[[56](#bib.bib56 "")]* applies learnable tokens as the input of the text encoder. Since only these tokens are updated during fine-tuning, it has very few trained parameters.

3. VPT*[[23](#bib.bib23 "")]*. Contrary to CoOp, Vision Prompt Tuning is proposed for visual backbones. VPT has two variants: VPT-shallow and VPT-deep. The VPT-deep leverages learnable prompts as the input of each transformer layer in the visual encoder. We use VPT-deep in the following experiments.

4. VL-Prompt. Since the previous prompt learning methods only tune vision or language prompts, we further applied learnable vision and language prompts into the vision and language encoder, which we call VL-Prompt. Following VPT, learnable tokens are applied as the input of each transformer block for both visual and text encoder.

5. MaPLe*[[28](#bib.bib28 "")]*. VL-Prompt is designed to learn vision and language prompts independently without cross-modal interaction, which might limit its performance. Hence, we compare our method with the latest multi-modality prompt learning method, namely MaPLe. It proposes a vision-language coupling function to link vision and language prompts and jointly optimizes them.

<img src='x5.png' alt='Refer to caption' title='' width='230' height='104' />

*Figure 5: Efficiency of cross-modal adapter. The horizontal axis is the relative training time (left) and relative memory cost (right) compared to the fully fine-tuned method i.e., CLIP4clip.*

### 4.1 Main Results

We compare our method with baseline methods on five datasets: MSR-VTT, MSVD, VATEX, ActivityNet, and DiDeMo. The results of MSR-VTT are shown in [Fig. 4](#S4.F4 "In 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval"), and results on other four datasets are shown in [Tab. 1](#S4.T1 "In 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval").

Results on MSR-VTT. [Fig. 4](#S4.F4 "In 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval") shows the trade-off between fine-tuned parameters and performances. Overall, our method achieves a better trade-off compared with the baselines for both text-to-video and video-to-text retrieval.

Compared with CLIP4Clip, a strong fully fine-tuned baseline, our method reduces 99.6% trained parameters without performance degradation. We think this is because fine-tuning with large parameters on downstream datasets has a high risk of overfitting, and this will be discussed in [Fig. 6](#S4.F6 "In 4.1 Main Results ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval").
Furthermore, we try to reduce the fine-tuned parameter of CLIP4Clip by freezing the first few layers in two encoders.
However, this leads to quite poor performance.

Compared with prompt-based methods, our method largely improves performance with comparable parameters.
First, the uni-modal prompt learning methods, i.e., CoOp and VPT, exhibit quite poor performances, which indicates that the uni-modal prompt is sub-optimal for the multimodal task.
Secondly, the multimodal prompt baseline VL-Prompt performs better than uni-modal prompt methods but worse than ours.
Last, the latest multi-modal prompt learning method MaPLe tunes much more parameters than ours but still performs poorly.
Although prompt learning achieves great success in uni-modal tasks, it is incapable of tackling the challenging multimodal task.

Results on other four datasets. [Tab. 1](#S4.T1 "In 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval") displays the results on MSVD, DiDeMo, VATEX, and ActivityNet.
Compared with the fully fine-tuned baseline CLIP4Clip, our method achieves comparable performance on VATEX and outperforms it by 1.2, 1.6, and 0.2 R@1 on MSVD, DiDeMo, and ActivityNet.
Meanwhile, our method is more parameter-efficient with less than 1M trained parameters.
Note that other fully fine-tuned methods, e.g., X-Pool*[[16](#bib.bib16 "")]* and TS2Net*[[34](#bib.bib34 "")]*, propose sophisticated similarity calculators or token selection modules, which is not applied in our method.
However, our method still achieves comparable R@1 performance on MSVD and VATEX.
It might indicate the potential of the pre-trained CLIP model is under-explored with the fully fine-tuning paradigm.
Additionally, we observe that our method outperforms existing prompt learning methods, regardless of whether they are based on multi-modal or uni-modal prompts.
Our method achieves 2.9, 7.5, 4.4, and 8.3 higher R@1 than the best of the prompt learning methods and has the lowest MnR on four datasets.

<img src='x6.png' alt='Refer to caption' title='' width='182' height='108' />

*Figure 6: Overfit phenomenon. Training losses and validation performances of fully fine-tuned method*[[35](#bib.bib35 "")]* and ours on MSR-VTT. The curves refer to the training loss of each epoch, while the histograms denote the R@1 on the validation set.*

| Dataset | Method | Text $\Longrightarrow$ Video | | Video $\Longrightarrow$ Text | |
| --- | --- | --- | --- | --- | --- |
| | | $\text{R@1}\uparrow$ | $\text{R@5}\uparrow$ | $\text{R@1}\uparrow$ | $\text{R@5}\uparrow$ |
| MSR-VTT | w$/$o Share | 44.9 | 72.8 | 45.2 | 73.3 |
| | Ours | 45.4($\uparrow$0.5) | 73.3($\uparrow$0.5) | 46.2($\uparrow$1.0) | 73.6($\uparrow$0.3) |
| ActivityNet | w$/$o Share | 41.2 | 70.9 | 39.1 | 71.1 |
| | Ours | 41.5($\uparrow$0.3) | 71.2($\uparrow$0.3) | 39.6($\uparrow$0.5) | 71.9($\uparrow$0.8) |
| DiDeMo | w$/$o Share | 44.1 | 72.0 | 44.2 | 72.0 |
| | Ours | 45.0($\uparrow$0.9) | 71.9 | 45.5($\uparrow$1.3) | 71.8 |
| MSVD | w$/$o Share | 46.7 | 76.7 | 63.1 | 90.6 |
| | Ours | 47.4($\uparrow$0.7) | 76.6 | 63.6($\uparrow$0.5) | 90.0 |
| VATEX | w$/$o Share | 58.9 | 89.9 | 74.3 | 97.1 |
| | Ours | 59.3($\uparrow$0.4) | 89.8 | 74.7($\uparrow$0.4) | 97.2 |

*Table 2: Ablations about the weight-sharing mechanism. Note that “w/o Share” variant also refers to the vanilla adapter.*

### 4.2 Efficiency of Cross-Modal Adapter

To demonstrate the efficiency of our method, we test the training speeds and memory occupations on 8 RTX-3090 GPUs.
As shown in [Fig. 5](#S4.F5 "In 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval"), compared to baselines, our method achieves a better trade-off between training speed and performance while with fewer memory occupations.
Specifically, our methods reduce 30% training time.
Although prompt-based methods, e.g., VPT and VL-Prompt, fine-tune a small number of parameters, the training time and memory cost are much larger than ours.

### 4.3 Overfit Phenomenon of Fully Fine-tuning.

We further investigate the training process of fully fine-tuned method*[[35](#bib.bib35 "")]* and our cross-modal adapter on MSR-VTT. As we can see from [Fig. 6](#S4.F6 "In 4.1 Main Results ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval"), the training loss of the fully fine-tuned method decreases rapidly due to its large capacity while our method’s loss declines much slower. However, lower training loss does not equate to better performance. The fully fine-tuned method achieves the highest performance at the second epoch and deteriorates later, which indicates a severe overfitting phenomenon. Since large models are data-hungry, MSR-VTT may not be adequate for fine-tuning a model with more than 123M parameters. In contrast, our method requires optimizing no more than 1M parameters. The training process of our method is healthier than the fully fine-tuned method as the performance peaks when the training loss is minimal, illustrated in [Fig. 6](#S4.F6 "In 4.1 Main Results ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval").

<img src='x7.png' alt='Refer to caption' title='' width='230' height='98' />

*Figure 7: Ablations on MSR-VTT. Left is the ablation of bottleneck dimension. The curve refers to the parameters, while the histogram denotes the R@1 on the validation set. Right is the ablation of the weight-sharing dimension.*

### 4.4 Ablation Study

Effectiveness of sharing mechanism. To demonstrate the effectiveness of our weight-sharing mechanism, we conduct thorough ablation studies on all five datasets. In [Tab. 2](#S4.T2 "In 4.1 Main Results ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval"), we show that the sharing weight not only boosts the performance of the text-to-video retrieval R@1 metric but also improves the video-to-text retrieval R@1 metric on all datasets. It indicates sharing weight facilitates the re-aligning of CLIP’s vision and language feature space on the text-video retrieval task. Note that on MSVD, VATEX, and DiDeMo, our method achieves comparable performance on R@5 while surpassing the baseline on the R@1 metric. Furthermore, the weight-sharing mechanism can potentially reduce the total parameters, e.g., the cross-modal adapter reduces 20.6% parameter on ActivityNet. In conclusion, sharing weight between video and text encoders helps to unleash the power of the pre-trained CLIP model and boosts retrieval performance.

<img src='x8.png' alt='Refer to caption' title='' width='185' height='108' />

*Figure 8: Visualizations of text-to-video retrieval results. Green boxes show our retrieval results, and they are also ground truth. Red boxes are CLIP4Clip’s retrieval results which are not correct.*

Bottleneck dimension $r$. The dimension of the bottleneck $r$ is an influential factor for the parameters and performance of the adapter.
Based on [Fig. 7](#S4.F7 "In 4.3 Overfit Phenomenon of Fully Fine-tuning. ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval") (left), it can be seen that the adapter’s total parameter increases with the bottleneck dimension.
In the case of a small $r$, the down-projection may lose a great deal of information regarding the original features, which causes performance degradation.
The performance of the adapter peaks at the bottleneck dimension of 8 and degrades thereafter.
Considering the trade-off between parameters and performance, we choose the bottleneck dimension as 8 for the MSR-VTT dataset.

Weight-sharing dimension $d_{s}$. With the bottleneck dimension set to 8, we further ablate the influence of changing the weight-sharing dimension $d_{s}$ on MSR-VTT dataset. As shown in [Fig. 7](#S4.F7 "In 4.3 Overfit Phenomenon of Fully Fine-tuning. ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval") (right), increasing the $d_{s}$ from 4 to 16 leads to a performance boost, and performance declines when $d_{s}$ exceeds 16. For the MSR-VTT dataset, sharing too much weight across two modalities has a negative effect on the performance. Therefore, the weight-sharing dimension is set to 16 on MSR-VTT.

### 4.5 Qualitative Results

We visualize multiple text-to-video retrieval results on the MSR-VTT test set in [Fig. 8](#S4.F8 "In 4.4 Ablation Study ‣ 4 Experiments ‣ Cross-Modal Adapter for Text-Video Retrieval"). Our method, based on the visualization, shows a better comprehension of temporal relationships, actions, and local visual objects. First, in the top left example, our method can understand the temporal relationship between the two kids and the woman. Second, in the bottom left example, our method can recognize the dog’s actions. Last, in the top right example, our method can distinguish the local “chart” from the video.

5 Conclusion
------------

In this paper, we explored adapter-based parameter-efficient transfer learning for the text-video retrieval task. To address the challenge of introducing feature-level interaction that causes a surge in computational cost, we proposed a novel Cross-Modal Adapter to enable early cross-modal interaction via a weight-sharing mechanism. Experiments showed that, compared to the fully fine-tuned method, our method can not only reduce 99.6% fine-tuned parameters without performance degradation but also requires fewer training resources. Furthermore, our approach surpassed prompt-based methods significantly, indicating that the adapters may offer a better parameter-efficient solution for transferring pre-trained models to the text-video retrieval task.

Acknowledgement
---------------

This work is supported in part by National Key R\&D Program of China (2021ZD0140407), the National Natural Science Foundation of China under Grant 62022048 and Guoqiang Institute of Tsinghua.

References
----------

* [1]Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell,
and Bryan Russell.Localizing moments in video with natural language.In ICCV, 2017.
* [2]Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra,
C Lawrence Zitnick, and Devi Parikh.Vqa: Visual question answering.In ICCV, 2015.
* [3]Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman.Frozen in time: A joint video and image encoder for end-to-end
retrieval.In ICCV, 2021.
* [4]Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman.A clip-hitchhiker’s guide to long video retrieval.arXiv preprint arXiv:2205.08508, 2022.
* [5]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
et al.Language models are few-shot learners.In NeurIPS, 2020.
* [6]David Chen and William B Dolan.Collecting highly parallel data for paraphrase evaluation.In ACL, 2011.
* [7]Hao Chen, Ran Tao, Han Zhang, Yidong Wang, Wei Ye, Jindong Wang, Guosheng Hu,
and Marios Savvides.Conv-adapter: Exploring parameter efficient transfer learning for
convnets.arXiv preprint arXiv:2208.07463, 2022.
* [8]Shizhe Chen, Yida Zhao, Qin Jin, and Qi Wu.Fine-grained video-text retrieval with hierarchical graph reasoning.In CVPR, 2020.
* [9]Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen.Improving video-text retrieval by multi-stream corpus alignment and
dual softmax loss.arXiv preprint arXiv:2109.04290, 2021.
* [10]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.Bert: Pre-training of deep bidirectional transformers for language
understanding.In NAACL, 2019.
* [11]Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, et al.An image is worth 16x16 words: Transformers for image recognition at
scale.In ICLR, 2020.
* [12]Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen.Clip2video: Mastering video-text retrieval via image clip.arXiv preprint arXiv:2106.11097, 2021.
* [13]Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid.Multi-modal transformer for video retrieval.In ECCV, 2020.
* [14]Zijian Gao, Jingyu Liu, Sheng Chen, Dedan Chang, Hao Zhang, and Jinwei Yuan.Clip2tv: An empirical study on transformer-based methods for
video-text retrieval.arXiv preprint arXiv:2111.05610, 2021.
* [15]Yuying Ge, Yixiao Ge, Xihui Liu, Dian Li, Ying Shan, Xiaohu Qie, and Ping Luo.Bridging video-text retrieval with multiple choice questions.In CVPR, 2022.
* [16]Satya Krishna Gorti, Noël Vouitsis, Junwei Ma, Keyvan Golestan, Maksims
Volkovs, Animesh Garg, and Guangwei Yu.X-pool: Cross-modal language-video attention for text-video
retrieval.In CVPR, 2022.
* [17]Yizeng Han, Gao Huang, Shiji Song, Le Yang, Yitian Zhang, and Haojun Jiang.Spatially adaptive feature refinement for efficient inference.IEEE TIP, 2021.
* [18]Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition.In CVPR, 2016.
* [19]Dan Hendrycks and Kevin Gimpel.Gaussian error linear units (gelus).arXiv preprint arXiv:1606.08415, 2016.
* [20]Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin
De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly.Parameter-efficient transfer learning for nlp.In ICML, 2019.
* [21]Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger.Densely connected convolutional networks.In CVPR, 2017.
* [22]Gao Huang, Yulin Wang, Kangchen Lv, Haojun Jiang, Wenhui Huang, Pengfei Qi, and
Shiji Song.Glance and focus networks for dynamic visual recognition.IEEE TPAMI, 2022.
* [23]Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath
Hariharan, and Ser-Nam Lim.Visual prompt tuning.In ECCV, 2022.
* [24]Haojun Jiang, Yuanze Lin, Dongchen Han, Shiji Song, and Gao Huang.Pseudo-q: Generating pseudo language queries for visual grounding.In CVPR, 2022.
* [25]Lan Jiang, Hao Zhou, Yankai Lin, Peng Li, Jie Zhou, and Rui Jiang.Rose: Robust selective fine-tuning for pre-trained language models.arXiv preprint arXiv:2210.09658, 2022.
* [26]Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C
Lawrence Zitnick, and Ross Girshick.Clevr: A diagnostic dataset for compositional language and elementary
visual reasoning.In CVPR, 2017.
* [27]Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder.Compacter: Efficient low-rank hypercomplex adapter layers.In NeurIPS, 2021.
* [28]Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and
Fahad Shahbaz Khan.Maple: Multi-modal prompt learning.arXiv preprint arXiv:2210.03117, 2022.
* [29]Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles.Dense-captioning events in videos.In ICCV, 2017.
* [30]Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and
Jingjing Liu.Less is more: Clipbert for video-and-language learning via sparse
sampling.In CVPR, 2021.
* [31]Brian Lester, Rami Al-Rfou, and Noah Constant.The power of scale for parameter-efficient prompt tuning.In EMNLP, 2021.
* [32]Xiang Lisa Li and Percy Liang.Prefix-tuning: Optimizing continuous prompts for generation.In ACL, 2021.
* [33]Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman.Use what you have: Video retrieval using representations from
collaborative experts.In BMVC, 2019.
* [34]Yuqi Liu, Pengfei Xiong, Luhui Xu, Shengming Cao, and Qin Jin.Ts2-net: Token shift and selection transformer for text-video
retrieval.In ECCV, 2022.
* [35]Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li.Clip4clip: An empirical study of clip for end to end video clip
retrieval.arXiv preprint arXiv:2104.08860, 2021.
* [36]Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji.X-clip: End-to-end multi-grained contrastive learning for video-text
retrieval.In ACMMM, 2022.
* [37]Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, and James Henderson.Parameter-efficient multi-task fine-tuning for transformers via
shared hypernetworks.In ACL, 2021.
* [38]Antoine Miech, Jean-Baptiste Alayrac, Ivan Laptev, Josef Sivic, and Andrew
Zisserman.Thinking fast and slow: Efficient text-to-visual retrieval with
transformers.In CVPR, 2021.
* [39]Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu,
Alexander H Miller, and Sebastian Riedel.Language models as knowledge bases?In EMNLP, 2019.
* [40]Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.Learning transferable visual models from natural language
supervision.In ICML, 2021.
* [41]Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al.Exploring the limits of transfer learning with a unified text-to-text
transformer.JMLR, 2020.
* [42]Andrew Rouditchenko, Angie Boggust, David Harwath, Brian Chen, Dhiraj Joshi,
Samuel Thomas, Kartik Audhkhasi, Hilde Kuehne, Rameswar Panda, Rogerio Feris,
et al.Avlnet: Learning audio-visual language representations from
instructional videos.In INTERSPEECH, 2021.
* [43]Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid.Videobert: A joint model for video and language representation
learning.In CVPR, 2019.
* [44]Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.Attention is all you need.In NeurIPS, 2017.
* [45]Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti
Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al.Image as a foreign language: Beit pretraining for all vision and
vision-language tasks.arXiv preprint arXiv:2208.10442, 2022.
* [46]Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang
Wang.Vatex: A large-scale, high-quality multilingual dataset for
video-and-language research.In ICCV, 2019.
* [47]Yulin Wang, Zhaoxi Chen, Haojun Jiang, Shiji Song, Yizeng Han, and Gao Huang.Adaptive focus for efficient video recognition.In ICCV, 2021.
* [48]Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian
Metze, Luke Zettlemoyer, and Christoph Feichtenhofer.Videoclip: Contrastive pre-training for zero-shot video-text
understanding.In EMNLP, 2021.
* [49]Jun Xu, Tao Mei, Ting Yao, and Yong Rui.Msr-vtt: A large video description dataset for bridging video and
language.In CVPR, 2016.
* [50]Le Yang, Haojun Jiang, Ruojin Cai, Yulin Wang, Shiji Song, Gao Huang, and Qi
Tian.Condensenet v2: Sparse feature reactivation for deep networks.In CVPR, 2021.
* [51]Haoxuan You, Luowei Zhou, Bin Xiao, Noel Codella, Yu Cheng, Ruochen Xu, Shih-Fu
Chang, and Lu Yuan.Learning visual representation from modality-shared contrastive
language-image pre-training.In ECCV, 2022.
* [52]Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and
Tamara L Berg.Mattnet: Modular attention network for referring expression
comprehension.In CVPR, 2018.
* [53]Youngjae Yu, Jongseok Kim, and Gunhee Kim.A joint sequence fusion model for video question answering and
retrieval.In ECCV, 2018.
* [54]Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg.Bitfit: Simple parameter-efficient fine-tuning for transformer-based
masked language-models.In ACL, 2021.
* [55]Bowen Zhang, Hexiang Hu, and Fei Sha.Cross-modal and hierarchical modeling of video and text.In ECCV, 2018.
* [56]Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu.Learning to prompt for vision-language models.IJCV, 2022.
* [57]Linchao Zhu and Yi Yang.Actbert: Learning global-local video-text representations.In CVPR, 2020.

Appendix
--------

Appendix A Main Results
-----------------------

Video-to-text retrieval results on other four datasets. We show the video-to-text retrieval results on MSVD*[[6](#bib.bib6 "")]*, VATEX*[[46](#bib.bib46 "")]*, ActivityNet*[[29](#bib.bib29 "")]*, and DiDeMo*[[1](#bib.bib1 "")]* in [Tab. 3](#A1.T3 "In Appendix A Main Results ‣ Cross-Modal Adapter for Text-Video Retrieval").

Compared to the fully fine-tuned method*[[35](#bib.bib35 "")]*, i.e., CLIP4Clip, our method achieves superior or comparable performance on MSVD, DiDeMo, and VATEX datasets.
Although our approach’s R@1 is worse than CLIP4Clip on ActivityNet, we attain comparable performance on the other three metrics.
Furthermore, our approach also surpasses CLIP2Video*[[12](#bib.bib12 "")]* on MSVD and outperforms X-CLIP*[[36](#bib.bib36 "")]* on DiDeMo.
Note that those fully fine-tuned methods*[[12](#bib.bib12 ""), [36](#bib.bib36 "")]* introduce sophisticated parameter-rich similarity calculators which are not applied in our model.
Compared to the prompt-based methods*[[56](#bib.bib56 ""), [23](#bib.bib23 ""), [28](#bib.bib28 "")]*, our method surpasses them significantly on all datasets, demonstrating the effectiveness of cross-modal adapter.

| Type | Methods | | Trained | | --- | | Params. | | R@1↑ | R@5↑ | R@10↑ | MnR↓ | Methods | | Trained | | --- | | Params. | | R@1↑ | R@5↑ | R@10↑ | MnR↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | MSVD | | | | | | DiDeMo | | | | | |
| ParameterEfficient | CoOp[[56](#bib.bib56 "")] | 0.002M | 58.3 | 83.3 | 89.2 | 7.2 | CoOp[[56](#bib.bib56 "")] | 0.002M | 34.9 | 63.3 | 75.0 | 16.1 |
| | VPT[[23](#bib.bib23 "")] | 0.18M | 59.1 | 89.9 | 93.6 | 4.3 | VPT[[23](#bib.bib23 "")] | 0.05M | 33.3 | 60.1 | 70.7 | 19.8 |
| VL-Prompt | 0.08M | 63.1 | 87.8 | 94.3 | 3.3 | VL-Prompt | 0.08M | 34.2 | 62.4 | 73.7 | 17.1 |
| MaPLe*[[28](#bib.bib28 "")] | 4.76M | 60.7 | 86.4 | 92.8 | 4.1 | MaPLe*[[28](#bib.bib28 "")] | 4.76M | 36.4 | 65.6 | 74.8 | 13.8 |
| Ours | 1.00M | 63.6 | 90.0 | 94.7 | 3.0 | Ours | 0.52M | 45.5 | 71.8 | 82.0 | 9.8 |
| FullyFine-tune | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 56.6 | 79.7 | 84.3 | 7.6 | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 42.5 | 70.6 | 80.2 | 11.6 |
| | CLIP2Video[[12](#bib.bib12 "")] | 142.46M | 58.7 | 85.6 | 91.6 | 4.3 | TS2-Net[[34](#bib.bib34 "")] | 139.35M | - | - | - | - |
| X-Pool[[16](#bib.bib16 "")] | 152.59M | 66.4 | 90.0 | 94.2 | 3.3 | X-CLIP[[36](#bib.bib36 "")] | 137.01M | 43.1 | 72.2 | - | 10.9 |
|  | VATEX | | | | | | ActivityNet | | | | | |
| ParameterEfficient | CoOp[[56](#bib.bib56 "")] | 0.033M | 66.7 | 94.5 | 97.9 | 2.1 | CoOp[[56](#bib.bib56 "")] | 0.008M | 29.0 | 57.7 | 72.4 | 14.0 |
| | VPT[[23](#bib.bib23 "")] | 0.05M | 69.5 | 94.5 | 98.5 | 1.9 | VPT[[23](#bib.bib23 "")] | 0.05M | 28.0 | 56.5 | 71.9 | 14.7 |
| VL-Prompt | 0.31M | 71.5 | 96.7 | 98.7 | 1.7 | VL-Prompt | 0.08M | 31.1 | 62.6 | 76.7 | 10.8 |
| MaPLe*[[28](#bib.bib28 "")] | 4.76M | 70.9 | 96.5 | 98.7 | 1.8 | MaPLe*[[28](#bib.bib28 "")] | 4.76M | 31.6 | 63.1 | 77.3 | 10.2 |
| Ours | 0.99M | 74.7 | 97.2 | 99.1 | 1.6 | Ours | 0.81M | 39.6 | 71.9 | 84.2 | 7.1 |
| FullyFine-tune | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 75.2 | 96.9 | 98.9 | 1.7 | CLIP4Clip[[35](#bib.bib35 "")] | 123.52M | 41.6 | 72.3 | 84.8 | 7.5 |
| | CLIP2Video[[12](#bib.bib12 "")] | 142.46M | 76.0 | 97.7 | 99.9 | 1.5 | TS2-Net[[34](#bib.bib34 "")] | 139.35M | - | - | - | - |
| TS2-Net[[34](#bib.bib34 "")] | 139.35M | - | - | - | - | X-CLIP[[36](#bib.bib36 "")] | 137.01M | 43.9 | 73.9 | - | 7.6 |

*Table 3: Video-to-text retrieval results on MSVD, DiDeMo, VATEX, and ActivityNet datasets, where higher R@K and lower MnR indicate better performance. We compare our method with other parameter-efficient transfer learning methods and list the CLIP-based fully fine-tuned methods for reference. Some fully fine-tuned baselines do not report video-to-text retrieval results.*

Results on MSR-VTT. To clearly show the trade-off between parameters and performances, we choose to present results on MSR-VTT with curves in the main text. Here, we provide the exact numbers of all the baselines in [Tab. 4](#A2.T4 "In Appendix B Implementation Details ‣ Cross-Modal Adapter for Text-Video Retrieval").
For the fully fine-tuned method*[[35](#bib.bib35 "")]*, we reduce fine-tuned parameters by freezing the first $n$ layers inside the video and text encoders. For prompt-based methods, we increase fine-tuned parameters by enlarging the prompt token number.

Appendix B Implementation Details
---------------------------------

For all datasets, we first sample frames with 1 FPS for videos.
Then, uniformly sample 12 frames for MSR-VTT*[[49](#bib.bib49 "")]*, MSVD*[[6](#bib.bib6 "")]*, and VATEX*[[46](#bib.bib46 "")]* while 64 frames for ActivityNet*[[29](#bib.bib29 "")]* and DiDeMo*[[1](#bib.bib1 "")]*.
The data augmentations for each frame are only resizing and normalization.
Concretely, we resize frames to a size of $224\times 224$.
In our method, we apply a GELU approximation as the non-linear layer because it calculates faster.
Besides, we empirically find that applying a dropout layer after the non-linear layer boost performance.
Before training, we initialize the cross-modal adapter with a normal distribution $\mathcal{N}(0,0.01)$.
In training, following CLIP4Clip*[[35](#bib.bib35 "")]*, we set the weight decay as 0.2, and our seed is 42.

The following are the implementation details for the prompt-based baselines. For CoOp*[[56](#bib.bib56 "")]*, we set the prompt token number as {4, 16, 64}.
For VPT*[[23](#bib.bib23 "")]* and VL-Prompt, we set the prompt token number as {5, 20, 40, 80}.
For MaPLe*[[28](#bib.bib28 "")]*, we set the prompt token number as {5, 20, 40}.
We report results for the different prompt token numbers on MSR-VTT.
On the other four datasets, we report the best results for different token numbers.

| MSR-VTT | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methods | Trained | Percent | Text$\Longrightarrow$Video | | | | Video$\Longrightarrow$Text | | | |
| | Params. | | $\text{R@1}\uparrow$ | $\text{R@5}\uparrow$ | $\text{R@10}\uparrow$ | $\text{MnR}\downarrow$ | $\text{R@1}\uparrow$ | $\text{R@5}\uparrow$ | $\text{R@10}\uparrow$ | $\text{MnR}\downarrow$ |
| CLIP4Clip[[35](#bib.bib35 "")] |  |  |  |  |  |  |  |  |  |  |
| - Freeze none | 123.52M | 81.65% | 43.1 | 70.4 | 80.8 | 16.2 | 43.1 | 70.5 | 81.2 | 12.4 |
| - Freeze 6 layers | 62.10M | 41.05% | 43.2 | 70.0 | 79.7 | 16.3 | 42.4 | 69.9 | 81.0 | 11.9 |
| - Freeze 9 layers | 31.38M | 20.74% | 42.4 | 68.9 | 79.1 | 17.3 | 43.2 | 70.0 | 80.8 | 12.9 |
| - Freeze 11 layers | 10.90M | 7.21% | 41.0 | 67.0 | 78.2 | 17.9 | 41.2 | 69.0 | 79.3 | 13.7 |
| - Freeze 12 layers | 0.66M | 0.44% | 31.4 | 54.2 | 63.1 | 40.8 | 26.5 | 52.3 | 62.3 | 36.2 |
| $\text{CLIP4Clip}^{*}$ |  |  |  |  |  |  |  |  |  |  |
| - Freeze none | 123.52M | 81.65% | 44.8 | 71.2 | 81.1 | 15.2 | 42.3 | 71.8 | 82.5 | 10.1 |
| - Freeze 6 layers | 62.10M | 41.05% | 45.1 | 72.4 | 81.3 | 14.2 | 44.3 | 73.3 | 82.8 | 9.0 |
| - Freeze 9 layers | 31.38M | 20.74% | 44.3 | 70.1 | 79.9 | 16.2 | 43.8 | 71.0 | 82.9 | 10.9 |
| - Freeze 11 layers | 10.90M | 7.21% | 42.5 | 69.7 | 78.8 | 17.0 | 43.4 | 70.4 | 81.0 | 11.3 |
| - Freeze 12 layers | 0.66M | 0.44% | 31.9 | 54.3 | 64.2 | 40.6 | 28.2 | 53.6 | 63.5 | 33.9 |
| CoOp[[56](#bib.bib56 "")] |  |  |  |  |  |  |  |  |  |  |
| - 4 tokens | 0.002M | 0.0014% | 38.6 | 63.5 | 74.4 | 18.0 | 41.9 | 68.0 | 78.4 | 11.6 |
| - 16 tokens | 0.008M | 0.0054% | 38.6 | 65.0 | 76.2 | 17.8 | 39.8 | 68.5 | 78.9 | 11.2 |
| - 64 tokens | 0.033M | 0.0217% | 38.9 | 64.2 | 75.8 | 17.6 | 39.5 | 67.5 | 78.4 | 11.3 |
| VPT[[23](#bib.bib23 "")] |  |  |  |  |  |  |  |  |  |  |
| - 5 tokens | 0.05M | 0.03% | 40.6 | 66.2 | 75.3 | 19.2 | 40.5 | 66.1 | 76.8 | 17.4 |
| - 20 tokens | 0.18M | 0.12% | 42.0 | 66.6 | 77.3 | 19.2 | 39.4 | 66.8 | 77.2 | 16.2 |
| - 40 tokens | 0.37M | 0.24% | 41.6 | 65.2 | 75.8 | 19.3 | 37.7 | 66.8 | 77.1 | 16.9 |
| - 80 tokens | 0.73M | 0.49% | 40.1 | 65.3 | 76.2 | 19.5 | 37.4 | 66.2 | 76.6 | 18.0 |
| VT-Prompt |  |  |  |  |  |  |  |  |  |  |
| - 5 tokens | 0.08M | 0.05% | 42.0 | 70.7 | 80.2 | 13.7 | 42.5 | 71.9 | 81.4 | 10.1 |
| - 20 tokens | 0.31M | 0.20% | 42.6 | 70.1 | 80.4 | 13.2 | 42.7 | 71.4 | 81.6 | 10.8 |
| - 40 tokens | 0.61M | 0.41% | 42.2 | 70.9 | 80.5 | 14.4 | 42.2 | 71.0 | 82.7 | 10.0 |
| - 80 tokens | 1.23M | 0.81% | 37.6 | 67.2 | 78.1 | 17.1 | 38.4 | 68.5 | 78.4 | 12.1 |
| $\text{MaPLe}^{*}$[[28](#bib.bib28 "")] |  |  |  |  |  |  |  |  |  |  |
| - 5 tokens | 4.76M | 3.15% | 42.6 | 69.6 | 80.1 | 14.0 | 43.1 | 71.7 | 81.2 | 9.6 |
| - 20 tokens | 4.85M | 3.21% | 42.2 | 68.5 | 78.4 | 14.9 | 41.9 | 70.2 | 79.9 | 10.6 |
| - 40 tokens | 4.97M | 3.29% | 41.7 | 68.2 | 79.0 | 15.1 | 40.5 | 71.0 | 80.5 | 10.1 |
| Ours | 0.52M | 0.34% | 45.4 | 73.3 | 82.3 | 12.8 | 46.2 | 73.6 | 83.8 | 8.6 |

*Table 4: Retrieval results on MSR-VTT, where higher R@K and lower MnR indicate better performance. “Percent” refers to the percentage of fine-tuned parameters to the total parameters of the model. $\text{CLIP4Clip}^{*}$ is our implementation with our parameter-free similarity calculator. Besides, all the prompt-based methods are implemented with our parameter-free similarity calculator.*
