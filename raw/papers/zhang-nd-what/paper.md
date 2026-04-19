What Makes Good Examples for Visual In-Context Learning?
=========================================================

Yuanhan ZhangKaiyang ZhouZiwei Liu

###### Abstract

Large-scale models trained on broad data have recently become the mainstream architecture in computer vision due to their strong generalization performance. In this paper, the main focus is on an emergent ability in large vision models, known as in-context learning, which allows inference on unseen tasks by conditioning on in-context examples (a.k.a. prompt) without updating the model parameters. This concept has been well-known in natural language processing but has only been studied very recently for large vision models. We for the first time provide a comprehensive investigation on the impact of in-context examples in computer vision, and find that the performance is highly sensitive to the choice of in-context examples. To overcome the problem, we propose a prompt retrieval framework to automate the selection of in-context examples. Specifically, we present (1) an unsupervised prompt retrieval method based on nearest example search using an off-the-shelf model, and (2) a supervised prompt retrieval method, which trains a neural network to choose examples that directly maximize in-context learning performance. The results demonstrate that our methods can bring non-trivial improvements to visual in-context learning in comparison to the commonly-used random selection. The code and models are available at [https://github.com/ZhangYuanhan-AI/visual_prompt_retrieval](https://github.com/ZhangYuanhan-AI/visual_prompt_retrieval "").

Machine Learning, ICML

1 Introduction
--------------

<img src='x1.png' alt='Refer to caption' title='' width='461' height='213' />

*Figure 1: (a) Different choices of in-context examples (outlined in green) often lead to significantly different results. Here we show 30 random query images (x-axis) from Pascal-$5^{i}$*(Shaban et al., [2017](#bib.bib23 ""))* split 0, and measure the performance range using 50 different in-context examples. (b) We propose a prompt retrieval framework aiming to automate the selection of in-context examples. We provide two implementations of the idea: one is unsupervised while the other is supervised, both outperforming random selection by a clear margin.*

In recent years, large-scale models have emerged in computer vision: they have enormous parameter size and are pre-trained on broad data to gain wide-ranging knowledge. These models have demonstrated remarkable generalization performance and have great potential for numerous downstream applications*(Bommasani et al., [2021](#bib.bib4 ""))*. However, due to the large model size and the potentially proprietary data used for training, entities able to develop large-scale models typically only provide users with APIs, known as Model-as-a-Service (Maas). Representative examples include the prominent text-to-image generation models, DALL$\cdot$E*(Ramesh et al., [2021](#bib.bib19 ""))* and Imagen*(Saharia et al., [2022](#bib.bib22 ""))*, and OpenAI’s powerful language models like GPT-3/ChatGPT*(Radford et al., [2021](#bib.bib18 ""))*. As a result, users are unable to apply full fine-tuning or some parameter-efficient tuning techniques, such as prompt learning*(Li \& Liang, [2021](#bib.bib11 ""); Lester et al., [2021](#bib.bib10 ""); Zhou et al., [2022c](#bib.bib27 ""), [b](#bib.bib26 ""); Zhang et al., [2022](#bib.bib24 ""); Pan et al., [2022](#bib.bib16 ""))*, for model adaptation, largely limiting downstream performance.

In-context learning, which is a “hidden” capability originally found in large autoregressive language models*(Radford et al., [2021](#bib.bib18 ""))*, has recently been investigated for large vision models*(Bar et al., [2022](#bib.bib3 ""))*, and more importantly, has the potential to become the mainstream approach for MaaS applications in the near future. Without the need to update any parameter for previously unseen tasks, in-context learning simply prepends some domain-specific input-output pairs, called in-context examples or prompt,111These two terms are used interchangeably in this paper. to a test example, which together guide the model to produce an ideal result. For instance, in natural language processing one could prepend a French-English sentence pair to a French sentence, and the model would produce an English translation of the French sentence. In computer vision, *Bar et al. ([2022](#bib.bib3 ""))* pre-trained a neural network to fill missing patches in grid-like images, which allows the model to perform in-context learning for unseen tasks like image segmentation (see the grid images in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ What Makes Good Examples for Visual In-Context Learning?")(a) bottom).

In this work, we focus on visual in-context learning, a relatively new concept with little existing research regarding how to better apply it in practice. We for the first time conduct a comprehensive investigation on the impact of in-context examples for large vision models, and identify a critical issue: downstream performance is highly sensitive to the choice of in-context examples. This is evidenced by the large variances observed for a variety of test examples shown in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ What Makes Good Examples for Visual In-Context Learning?")(a) top. By visualizing the results in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ What Makes Good Examples for Visual In-Context Learning?")(a) bottom, it seems to suggest that the closer the in-context example to the query, the better the result. For example, the best prompt image is closer to the query as they are similar in object pose and background; on the other hand, the worst prompt image has a drastically different style than the query image, which might explain why the predicted mask focuses on the wrong region, i.e., the white pillar instead of the cat.

Clearly, designing a proper prompt containing the optimal in-context example(s) by hand would be extremely difficult. To overcome the problem, we propose a prompt retrieval framework where the core component is a score function, which aims to give each source instance a score to indicate the level of suitability for being included in the prompt. Once the scoring process is done, we can simply pick one or multiple examples with the highest score(s) to construct a prompt. An overview of our framework is depicted in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ What Makes Good Examples for Visual In-Context Learning?")(b).

We provide two implementations for the prompt retrieval framework, both interpreting the score as the cosine distance measuring similarity between a query and a source example. The first is an unsupervised method based on nearest example search using an off-the-shelf model. The second is a supervised method, which learns a neural network to choose examples that directly maximize in-context learning performance. Since there is no ground-truth score to be used as the supervisory signal, we resort to a contrastive learning paradigm: source examples that result in better (or worse) in-context learning performance should get closer (or farther) to the query in feature space.

Our contributions and the main findings are summarized as follows. (1) We present the first comprehensive study concerning how to select good examples for the emerging visual in-context learning, and reveal a critical issue that the choice of in-context examples has a huge impact on performance. (2) From the technical perspective, we present a prompt retrieval framework that can automate the prompt selection process, and provide two simple implementations: an unsupervised method and a supervised method. (3) By conducting extensive experiments on three visual in-context learning tasks (which have not been seen during pre-training), namely foreground segmentation, single object detection and image colorization, we share valuable insights with the community on how to find good visual in-context examples, e.g., the supervised method performs the best and often finds examples that are both semantically close and spatially similar to a query.

2 Methods
---------

### 2.1 Visual In-Context Learning

<img src='x2.png' alt='Refer to caption' title='' width='438' height='182' />

*Figure 2: Overview of the supervised prompt retrieval method. The main idea is to compute the in-context learning result for each source example, and pick those with the highest/lowest results to form a positive/negative set for contrastive learning.*

In-context learning is a new paradigm that originally emerged from large autoregressive language models pre-trained on broad data, such as GPT-3*(Brown et al., [2020](#bib.bib5 ""))*. Unlike traditional learning methods, in-context learning does not require any parameter update and instead conditions prediction on some in-context examples in the form of input-output pairs. For example, in natural language processing one might give a French-English sentence pair and a test French sentence as input to the model, which then produces the English version of the sentence. In computer vision, such a paradigm has only been studied very recently. For example, *Bar et al. ([2022](#bib.bib3 ""))* trained a neural network to fill missing patches in grid-like images, which in turn allows the model to perform in-context learning on unseen tasks.

Formally, given a dataset $\mathcal{D}\={(x_{n},y_{n})}_{n\=1}^{N}$ containing $N$ image-label pairs (e.g., an image and its segmentation mask), a query example $x_{q}$, and a model $g_{\tau}$, in-context learning can be formulated as:

|  | $y_{q}\=g_{\tau}(\mathcal{P},x_{q}),$ |  | (1) |
| --- | --- | --- | --- |

where $\mathcal{P}$ is called a prompt, which consists of $K$ input-output pairs, $\mathcal{P}\={x_{c_{1}},y_{c_{1}},...,x_{c_{K}},y_{c_{K}}}\subset\mathcal{D}$. In particular, the prompt $\mathcal{P}$ provides some context for guiding the model to produce the ideal $y_{q}$ for $x_{q}$ without updating the large model’s parameters $\tau$.

##### Problem.

The most common approach for designing the prompt $\mathcal{P}$ in the vision domain is (within-class) random selection proposed by*Bar et al. ([2022](#bib.bib3 ""))*: one or multiple image-label pairs (with the same label as the test example) are randomly chosen from the training dataset. As illustrated in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ What Makes Good Examples for Visual In-Context Learning?")(a), the performance is highly sensitive to the selection of in-context examples—the gap between the best and worst prompt could reach over 70% mIoU. Below we propose two automatic prompt selection methods to tackle this problem.

### 2.2 Prompt Retrieval

Our goal is to automatically select the most suitable example(s) from the training dataset for a query $x_{q}$. To this end, we propose a prompt retrieval framework in the following form,

|  | $x^{*}\=\arg\max_{x_{n}\in\mathcal{D}}f_{\theta}(x_{n},x_{q}),$ |  | (2) |
| --- | --- | --- | --- |

where $f_{\theta}$ is a function parameterized by $\theta$, aiming to produce a score for a pair of $x_{n}$ and $x_{q}$. When $K\=1$, we choose the optimal example pair as the prompt, $\mathcal{P}\={x^{*},y^{*}}$. When $K>1$, we rank the training examples by their scores and choose the top-$K$ example pairs. An overview of our methods is provided in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ What Makes Good Examples for Visual In-Context Learning?")(b).

In this work, we implement $f_{\theta}$ as a combination of a neural network for feature extraction and the cosine distance function for measuring similarity between two feature vectors.

#### 2.2.1 Unsupervised Prompt Retrieval

Our first method is unsupervised prompt retrieval where the key idea is to use an off-the-shelf feature extractor for extracting image features so that we can compare the cosine distance between the query $x_{q}$ and each training example $x_{n}\in\mathcal{D}$. In this case, the parameters $\theta$ for the score function $f_{\theta}$ correspond to the off-the-shelf feature extractor, which are kept fixed.

#### 2.2.2 Supervised Prompt Retrieval

The unsupervised method discussed above is not explicitly optimized for in-context learning; instead, it relies on how the feature extractor was pre-trained and the objective (function) used in pre-training may well not align with that of in-context learning. We propose a second method based on supervised prompt retrieval where we assume the source data contains labels. The goal is to directly optimize the score function $f_{\theta}$ such that the chosen in-context example(s) can maximize the log-likelihood,

|  | $\max_{\mathcal{P}}\quad\log p(y_{q}|\mathcal{P},x_{q}).$ |  | (3) |
| --- | --- | --- | --- |

In this work, we present a simple implementation for the supervised method, which simply turns the unsupervised method into a supervised one by making the feature extractor learnable. In other words, we directly optimize Eq.[3](#S2.E3 "Equation 3 ‣ 2.2.2 Supervised Prompt Retrieval ‣ 2.2 Prompt Retrieval ‣ 2 Methods ‣ What Makes Good Examples for Visual In-Context Learning?") with respect to the feature extractor. Below we explain in detail how we train the feature extractor (see Fig.[2](#S2.F2 "Figure 2 ‣ 2.1 Visual In-Context Learning ‣ 2 Methods ‣ What Makes Good Examples for Visual In-Context Learning?") for an overview).

##### Data.

Recall that we interpret the score $f_{\theta}(\cdot,\cdot)$ as the cosine distance between two images in feature space. We would like to learn a space such that an image pair $(x_{n},x_{q})$ with high in-context learning performance is close to each other, or far away from each other if the performance is low. Since there is no label defining how close a distance should be, we resort to contrastive learning for training the feature extractor. The goal is then to find a positive and a negative set for each training example $x_{n}\in\mathcal{D}$ treated as a query. Specifically, for each example $x_{n}$ we compute the prediction $\hat{y}_{n}\=g_{\tau}((x_{m},y_{m}),x_{n})$ where $g_{\tau}$ is the large vision model defined in Sec.[2.1](#S2.SS1 "2.1 Visual In-Context Learning ‣ 2 Methods ‣ What Makes Good Examples for Visual In-Context Learning?") and $x_{m}\in\mathcal{D}$ but $x_{m}\neq x_{n}$. Since we have the ground truth $y_{n}$ for $x_{n}$, we can measure the performance by comparing the prediction $\hat{y}_{n}$ with the ground truth $y_{n}$. Then, for each $x_{n}$ we choose the top-5 examples with the highest/lowest performance to form a positive/negative set.

##### Training.

Let $z_{n}$ denote the features of $x_{n}$ extracted by the neural network we aim to optimize. At each iteraction, we sample a mini-batch $\mathcal{B}$ from the training dataset. Then, for each example in $\mathcal{B}$, we sample one example from the top-5 positive and negative sets, respectively. The contrastive loss is computed as

|  | $\ell\=-\frac{1}{|\mathcal{B}|}\sum_{x_{n}\sim\mathcal{B}}\log\frac{e^{cos(z_{n},z_{n}^{+})}}{e^{cos(z_{n},z_{n}^{+})}+\sum\limits_{z_{n}^{-}\in\mathcal{N}}e^{cos(z_{n},z_{n}^{-})}},$ |  | (4) |
| --- | --- | --- | --- |

where $cos(\cdot,\cdot)$ is the cosine distance function, $z_{n}^{+}$ denotes the feature representation of a positive example, and $z_{n}^{-}$ denotes the feature representation of a negative example. It is worth noting that for mini-batch training, the negative set $\mathcal{N}$ contains a negative example of $x_{n}$ sampled from the top-5 negative set and other examples within the same mini-batch.

3 Experiments
-------------

In this section we conduct a comprehensive evaluation using different prompt selection methods (Sec.[3.1](#S3.SS1 "3.1 Main Results ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?")) and compare their robustness to distribution shifts (Sec.[3.2](#S3.SS2 "3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?")). We also provide extensive quantitative and qualitative analyses in Sec.[3.3](#S3.SS3 "3.3 Further Analysis ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") to help understand why our methods work and how to better apply them in practice. Source code will be released to the community for reproducing the full experiments.

##### Methods.

All experiments are based on the image inpainting model pre-trained by*Bar et al. ([2022](#bib.bib3 ""))* on a dataset consisting of academic figures.222<https://github.com/amirbar/visual_prompting> We mainly compare the following methods: (1) Random, the baseline method that randomly samples in-context examples from the source training dataset; (2) Unsupervised prompt retrieval (UnsupPR), our first proposed method that uses off-the-shelf features for nearest example search. The main experiments are based on CLIP’s vision encoder*(Radford et al., [2021](#bib.bib18 ""))*, which was pre-trained using multimodal contrastive learning; (3) Supervised prompt retrieval (SupPR), our second proposed method that fine-tunes CLIP’s vision encoder by directly optimizing in-context learning performance on downstream datasets. A variety of backbones are evaluated in Sec.[3.3](#S3.SS3 "3.3 Further Analysis ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?").

##### Training details for the supervised model.

The supervised model is trained for 200 epochs using SGD. The initial learning rate is set to 0.005, decayed by the cosine annealing rule.

*Table 1:  Main results. The two prompt retrieval methods outperform random selection, and the supervised method achieves the best performance.*

|  | Seg. (mIoU) $\uparrow$ | | | | | Det. (mIoU) $\uparrow$ | Color. (mse) $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | Split-0 | Split-1 | Split-2 | Split-3 | Avg | | |
| Random | 28.66 | 30.21 | 27.81 | 23.55 | 27.56 | 25.45 | 0.67 |
| UnsupPR | 34.75 | 35.92 | 32.41 | 31.16 | 33.56 | 26.84 | 0.63 |
| SupPR | 37.08 | 38.43 | 34.40 | 32.32 | 35.56 | 28.22 | 0.63 |

### 3.1 Main Results

##### Setup.

Following*Bar et al. ([2022](#bib.bib3 ""))*, we evaluate our methods on three computer vision tasks, which have not been seen during the training of the image inpainting model. We provide the details about the datasets used for these tasks as follows. (1) Foreground segmentation: We use Pascal-$5^{i}$*(Shaban et al., [2017](#bib.bib23 ""))*, which has four non-overlapping splits each containing five categories. The results are averaged over all splits. (2) Single object detection: The experiments are done on Pascal VOC*(Everingham et al., [2015](#bib.bib7 ""))*. (3) Colorization: We use ImageNet-2012*(Russakovsky et al., [2015](#bib.bib21 ""))*, where the original validation set containing 50,000 images is used as our test set. The training data used to learn our supervised prompt retrieval model is created by randomly sampling 50,000 images from ImageNet’s 1.2M training set. For all experiments, in-context examples come from the training set.

##### Results.

Table[1](#S3.T1 "Table 1 ‣ Training details for the supervised model. ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") shows the results on the three benchmarks covering foreground segmentation, single object detection, and colorization. We summarize our findings as follows. First, prompt retrieval clearly outperforms random selection. In particular, the improvements of prompt retrieval over random selection are significant in foreground segmentation and single object detection: more than 6% on the former and 1% on the latter. However, the gains on colorization are only marginal (0.63 vs. 0.67), suggesting that the image inpainting model is probably weak at image colorization. Second, the supervised prompt retrieval method performs the best. This is not surprising as the supervised method optimizes in-context learning performance concerning the prompt selection module. In contrast, the unsupervised method relies more on the off-the-shelf feature extractor. Overall, the results well justify the design of the prompt retrieval framework, which can serve as a strong baseline for future research.

### 3.2 Experiments on Distribution Shifts

*Table 2:  Results on distribution shifts (from Pascal to MSCOCO). Despite being a learning-based approach, SupPR shows stronger robustness than UnsupPR and Random, which do not require any training.*

|  | Seg. (mIoU) $\uparrow$ | | | | |
| --- | --- | --- | --- | --- | --- |
|  | Split-0 | Split-1 | Split-2 | Split-3 | Avg |
| Random | 12.17 | 18.47 | 20.55 | 15.94 | 16.78 |
| UnsupPR | 12.67 | 19.62 | 21.33 | 18.44 | 18.02 |
| SupPR | 13.62 | 21.25 | 24.46 | 20.44 | 19.95 |

##### Setup.

Distribution shifts are commonly seen in real-world applications, and therefore AI models need to be robust to distribution shifts*(Zhou et al., [2022a](#bib.bib25 ""))*. To test this ability in visual in-context learning, we create a new protocol focusing on foreground segmentation where the source dataset is Pascal while the target dataset is MSCOCO*(Lin et al., [2014](#bib.bib12 ""))*. Specifically, we follow the design of Pascal-$5^{i}$ and create MSCOCO-$5^{i}$, which also has four splits, each having the same set of categories as in the corresponding split in Pascal-$5^{i}$. Note that such a shift mainly affects the supervised prompt retrieval method that requires training but not the unsupervised UnsupPR and Random.

##### Results.

The results are shown in Table[2](#S3.T2 "Table 2 ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?"). First of all, the unsupervised prompt retrieval method beats the random selection method by a clear margin. By comparing the two prompt retrieval methods, we find that the supervised method again performs better than the unsupervised one despite being a learning-based approach—this is an exciting finding as it means the supervised method does not have the overfitting problem here. Nonetheless, we observe that the gains achieved by the prompt retrieval methods here are generally smaller than the gains achieved on the standard foreground segmentation benchmark: here SupPR is only around 3% better on average than Random (19.95% vs. 16.78%) while the improvement in Table[1](#S3.T1 "Table 1 ‣ Training details for the supervised model. ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") reaches 8% (35.56% vs. 27.56%). One potential solution to reduce the gap might be to improve the image inpainting model, which is beyond the scope of this paper.

<img src='x3.png' alt='Refer to caption' title='' width='461' height='373' />

*Figure 3: In-context examples retrieved by UnsupPR and SupPR. In each grid, the first row contains the prompt while the second row contains the query and prediction. The in-context examples found by SupPR are more similar than those found by UnsupPR to the queries in a numer of ways: semantics (e.g., (e)), background (e.g., (a)), object pose (e.g., (b), object appearance (e.g., (i)), viewpoint (e.g., (k)), etc. More examples can be found in the supplementary.*

<img src='x4.png' alt='Refer to caption' title='' width='461' height='132' />

*Figure 4: (Left) Impact of the size of retrieval set. (Right) Ablation study on distance metric used to compute the score function in Eq.[2](#S2.E2 "Equation 2 ‣ 2.2 Prompt Retrieval ‣ 2 Methods ‣ What Makes Good Examples for Visual In-Context Learning?"). It can be observed that different metrics perform similarly.*

*Table 3: Comparison between different backbones pre-trained using different methods: multimodal contrastive learning for CLIP, self-supervised learning for EVA, and supervised learning for ViT. Overall, the performance is insensitive to the choice of different backbones.*

|  |  | Seg. (mIoU) $\uparrow$ | | | | |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | Split-0 | Split-1 | Split-2 | Split-3 | Avg |
| UnsupPR | CLIP | 34.75 | 35.92 | 32.41 | 31.16 | 33.56 |
| | EVA | 34.75 | 36.09 | 32.11 | 31.61 | 33.64 |
| ViT | 35.10 | 37.37 | 32.05 | 30.80 | 33.83 |
| SupPR | CLIP | 37.08 | 38.43 | 34.40 | 32.32 | 35.56 |
| | EVA | 36.11 | 39.14 | 34.31 | 33.30 | 35.71 |
| ViT | 36.80 | 39.70 | 34.71 | 33.25 | 36.12 |

*Table 4: Impact of the order of in-context examples.*

|  | Seg. (mIoU) $\uparrow$ | | | | |
| --- | --- | --- | --- | --- | --- |
|  | Split-0 | Split-1 | Split-2 | Split-3 | Avg |
| Random | 17.93 $\pm$ 0.20 | 25.48 $\pm$ 0.27 | 21.34 $\pm$ 0.73 | 21.12 $\pm$ 0.53 | 21.46 $\pm$ 0.43 |
| UnsupPR | 20.22 $\pm$ 0.31 | 27.58 $\pm$ 0.40 | 22.42 $\pm$ 0.38 | 23.36 $\pm$ 0.42 | 23.39 $\pm$ 0.37 |
| SupPR | 20.74$\pm$ 0.40 | 28.19$\pm$ 0.37 | 23.09$\pm$ 0.34 | 24.22$\pm$ 0.48 | 24.06$\pm$ 0.40 |

### 3.3 Further Analysis

##### What are good in-context examples?

To answer this question, we visualize the in-context examples found by UnsupPR and SupPR in Fig.[3](#S3.F3 "Figure 3 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?"). We focus on foreground segmentation and choose two categories from Pascal (person and cow).333The results of the remaining categories of Pascal and the results on other tasks are provided in the supplementary. In each grid, the first row corresponds to the retrieved in-context example (i.e., an input-output pair) while the second row contains the query and model prediction. By comparing the in-context examples picked by UnsupPR and those picked by SupPR, we find the reason why SupPR performs better than UnsupPR: the examples found by SupPR are more similar to the queries in terms of semantics (e.g., Fig.[3](#S3.F3 "Figure 3 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?")(e)), background (e.g., Fig.[3](#S3.F3 "Figure 3 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?")(a)), object pose (e.g., Fig.[3](#S3.F3 "Figure 3 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?")(b), object appearance (e.g., Fig.[3](#S3.F3 "Figure 3 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?")(i)), viewpoint (e.g., Fig.[3](#S3.F3 "Figure 3 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?")(k)), and so on. We also observe similar patterns in other categories/tasks (please refer to the supplementary).

##### Backbone.

To understand if using a different backbone than CLIP would make a big difference, we further evaluate our prompt retrieval methods, UnsupPR and SupPR, on the foreground segmentation benchmark using two other backbones: EVA*(Fang et al., [2022](#bib.bib8 ""))* pre-trained using self-supervised learning (i.e., masked image modeling) and ViT*(Dosovitskiy et al., [2020](#bib.bib6 ""))* pre-trained using supervised learning. The results are reported in Table[3](#S3.T3 "Table 3 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?"). Although these three backbones perform differently on image recognition under the fine-tuning setting—EVA performed the best—the gap between them for both UnsupPR and SupPR is less than 1%. Therefore, we can conclude that the backbone for visual in-context learning does not matter much.

##### Size of retrieval set.

Recall that in-context examples are sampled from the training dataset, namely the retrieval set. We are interested to know whether the size has any impact on performance, especially for the supervised prompt retrieval method. To this end, we build seven subsets for each split in Pascal-$5^{i}$, which cover a wide range of sizes (see the x-axis in Fig.[4](#S3.F4 "Figure 4 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") left). The results are plotted in Fig.[4](#S3.F4 "Figure 4 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") left. For random selection, the size does not matter at all. In contrast, the two prompt retrieval methods clearly benefit from a bigger size. But their performance plateaus when the size reaches a certain level. It is worth noting that for the supervised method, 20% of the total data is sufficient for achieving a decent performance.

##### Number of in-context examples.

We follow*Bar et al. ([2022](#bib.bib3 ""))* and create a large grid enough to fit 8 examples at maximum (as shown in Fig.[5](#S3.F5 "Figure 5 ‣ Distance metric. ‣ 3.3 Further Analysis ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") right). By varying the number of in-context examples from 1 to 7, we obtain a set of results and plot them in Fig.[5](#S3.F5 "Figure 5 ‣ Distance metric. ‣ 3.3 Further Analysis ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") left. Clearly, more in-context examples lead to better performance for all three methods, including SupPR, UnsupPR, and Random. This is probably because in-context examples can be viewed as “training data”, and having more training data typically benefits performance—in visual in-context learning, more training data gives a more comprehensive “context.” We show a few example cases in Fig.[5](#S3.F5 "Figure 5 ‣ Distance metric. ‣ 3.3 Further Analysis ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") right to explain this observation.

##### Order of in-context examples.

To understand if changing the order of in-context examples makes a difference, we fix the number of in-context examples to 3, evaluate all possible combinations, and compute the mean and standard deviation. As shown in Table[4](#S3.T4 "Table 4 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?"), the standard deviation is generally small, so the order is not a concern as long as good examples are chosen.

##### Distance metric.

We use the cosine distance by default to compute the score function in Eq.[2](#S2.E2 "Equation 2 ‣ 2.2 Prompt Retrieval ‣ 2 Methods ‣ What Makes Good Examples for Visual In-Context Learning?"). Here we evaluate other design choices including Euclidean distance and Manhattan distance. As shown in Fig.[4](#S3.F4 "Figure 4 ‣ Results. ‣ 3.2 Experiments on Distribution Shifts ‣ 3 Experiments ‣ What Makes Good Examples for Visual In-Context Learning?") right, the results are very similar for different distance metrics.

<img src='x5.png' alt='Refer to caption' title='' width='461' height='245' />

*Figure 5: (Left) Impact of the number of in-context examples. (Right) More in-context examples can lead to better performance. The query in each grid is shown in the bottom right.*

4 Related Work
--------------

### 4.1 In-Context Learning

In-context learning is a novel paradigm that emerged in large language models, such as GPT-3*(Brown et al., [2020](#bib.bib5 ""))*. It allows an autoregressive language model to perform inference on unseen tasks by conditioning the input on some target-specific input-output pairs serving as “context.” Such a powerful paradigm allows users to customize a model’s output according to their downstream datasets without changing the internal model parameters, which are often inaccessible. Recent research in natural language processing has shown that in-context learning can be applied to numerous language tasks, such as machine translation*(Garcia \& Firat, [2022](#bib.bib9 ""))*, sentiment analysis*(Min et al., [2021](#bib.bib15 ""))*, and question answering*(Press et al., [2022](#bib.bib17 ""))*.

In computer vision, in-context learning is still a relatively new concept. One of the earliest works tackling in-context learning is Flamingo*(Alayrac et al., [2022](#bib.bib2 ""))*, a large visual language model taking language as instruction and allowing the processing of both images and videos. More relevant to our work is a pure vision model developed by*Bar et al. ([2022](#bib.bib3 ""))*, which was pre-trained to fill missing patches in images made of academic figures and infographics. *Bar et al. ([2022](#bib.bib3 ""))* found that such an image inpainting model can solve problems unseen during training, like foreground segmentation and image colorization.

Our work follows*Bar et al. ([2022](#bib.bib3 ""))* but studies visual in-context learning from a different dimension: how to find good visual in-context examples that benefit downstream performance.

### 4.2 Prompt Retrieval in NLP

The natural language processing community has found that the choice of in-context examples has a huge impact on performance*(Agrawal et al., [2022](#bib.bib1 ""); Liu et al., [2021](#bib.bib13 ""))*. Moreover, the way how in-context examples, also called prompts, are constructed can also affect performance, e.g., prompt length and the order of in-context examples, as reported in the literature*(Agrawal et al., [2022](#bib.bib1 ""))*. These findings prompted the community to study how to find good in-context examples for large language models, which has inspired our research.

*Liu et al. ([2021](#bib.bib13 ""))* assumed that good in-context examples should be semantically close to query sentences, based on which they proposed to select nearest neighbors in the training set measured by a sentence encoder like RoBERTa*(Liu et al., [2019](#bib.bib14 ""))*. *Rubin et al. ([2021](#bib.bib20 ""))* first used an unsupervised method to retrieve some candidates, among which top examples were chosen using a supervised prompt retriever to maximize downstream performance.

5 Discussion and Conclusion
---------------------------

Our research presents a timely study on an emergent ability termed in-context learning for large vision models. We systematically investigate how the choice of in-context examples impacts downstream performance, exposing a critical issue that different in-context examples could lead to drastically different results. We then propose an effective prompt retrieval framework for visual in-context learning, with two simple implementations provided: one based on unsupervised learning and the other based on supervised learning. Our methods obtain significant improvements over random selection under various problem settings, showing the potential of using prompt retrieval in vision applications with a Model-as-a-Service (MaaS) business structure.

Our research also unveils some intriguing phenomena. For instance, we show that a good in-context example should be semantically similar to the query and closer in context, e.g., viewpoint, background, and appearance. As such, state-of-the-art vision models like CLIP would not be sufficient because these models often emphasize semantics but not other elements critical to finding good visual in-context examples. A model that can better balance spatial and semantic closedness in feature space would be more ideal for visual in-context learning. We hope the insights presented in this work could pave the way for developing more effective prompt retrieval methods.

Our experiments show that our methods are not strong enough to cope with distribution shifts. Though our methods outperform random selection under distribution shifts, the gap is much smaller than that on a standard benchmark, suggesting huge room for improvement.

References
----------

* Agrawal et al. (2022)Agrawal, S., Zhou, C., Lewis, M., Zettlemoyer, L., and Ghazvininejad, M.In-context examples selection for machine translation.*arXiv preprint arXiv:2212.02437*, 2022.
* Alayrac et al. (2022)Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc,
K., Mensch, A., Millican, K., Reynolds, M., et al.Flamingo: a visual language model for few-shot learning.*arXiv preprint arXiv:2204.14198*, 2022.
* Bar et al. (2022)Bar, A., Gandelsman, Y., Darrell, T., Globerson, A., and Efros, A. A.Visual prompting via image inpainting.*arXiv preprint arXiv:2209.00647*, 2022.
* Bommasani et al. (2021)Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S.,
Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al.On the opportunities and risks of foundation models.*arXiv preprint arXiv:2108.07258*, 2021.
* Brown et al. (2020)Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P.,
Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.Language models are few-shot learners.*Advances in neural information processing systems (NeurIPS)*,
33:1877–1901, 2020.
* Dosovitskiy et al. (2020)Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.An image is worth 16x16 words: Transformers for image recognition at
scale.*arXiv preprint arXiv:2010.11929*, 2020.
* Everingham et al. (2015)Everingham, M., Eslami, S., Van Gool, L., Williams, C. K., Winn, J., and
Zisserman, A.The pascal visual object classes challenge: A retrospective.*International journal of computer vision (ICCV)*, 111(1):98–136, 2015.
* Fang et al. (2022)Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X.,
and Cao, Y.Eva: Exploring the limits of masked visual representation learning at
scale.*arXiv preprint arXiv:2211.07636*, 2022.
* Garcia \& Firat (2022)Garcia, X. and Firat, O.Using natural language prompts for machine translation.*arXiv preprint arXiv:2202.11822*, 2022.
* Lester et al. (2021)Lester, B., Al-Rfou, R., and Constant, N.The power of scale for parameter-efficient prompt tuning.*arXiv preprint arXiv:2104.08691*, 2021.
* Li \& Liang (2021)Li, X. L. and Liang, P.Prefix-tuning: Optimizing continuous prompts for generation.*arXiv preprint arXiv:2101.00190*, 2021.
* Lin et al. (2014)Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D.,
Dollár, P., and Zitnick, C. L.Microsoft coco: Common objects in context.In *European conference on computer vision (ECCV)*, pp. 740–755. Springer, 2014.
* Liu et al. (2021)Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and Chen, W.What makes good in-context examples for gpt-$3$?*arXiv preprint arXiv:2101.06804*, 2021.
* Liu et al. (2019)Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M.,
Zettlemoyer, L., and Stoyanov, V.Roberta: A robustly optimized bert pretraining approach.*arXiv preprint arXiv:1907.11692*, 2019.
* Min et al. (2021)Min, S., Lewis, M., Zettlemoyer, L., and Hajishirzi, H.Metaicl: Learning to learn in context.*arXiv preprint arXiv:2110.15943*, 2021.
* Pan et al. (2022)Pan, J., Lin, Z., Zhu, X., Shao, J., and Li, H.St-adapter: Parameter-efficient image-to-video transfer learning for
action recognition.*arXiv preprint arXiv:2206.13559*, 2022.
* Press et al. (2022)Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., and Lewis, M.Measuring and narrowing the compositionality gap in language models.*arXiv preprint arXiv:2210.03350*, 2022.
* Radford et al. (2021)Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry,
G., Askell, A., Mishkin, P., Clark, J., et al.Learning transferable visual models from natural language
supervision.In *International Conference on Machine Learning (ICML)*, pp. 8748–8763. PMLR, 2021.
* Ramesh et al. (2021)Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and
Sutskever, I.Zero-shot text-to-image generation.In *International Conference on Machine Learning (ICML)*, pp. 8821–8831. PMLR, 2021.
* Rubin et al. (2021)Rubin, O., Herzig, J., and Berant, J.Learning to retrieve prompts for in-context learning.*arXiv preprint arXiv:2112.08633*, 2021.
* Russakovsky et al. (2015)Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z.,
Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L.ImageNet Large Scale Visual Recognition Challenge.*International Journal of Computer Vision (IJCV)*, 115(3):211–252, 2015.doi: 10.1007/s11263-015-0816-y.
* Saharia et al. (2022)Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour,
S. K. S., Ayan, B. K., Mahdavi, S. S., Lopes, R. G., et al.Photorealistic text-to-image diffusion models with deep language
understanding.*arXiv preprint arXiv:2205.11487*, 2022.
* Shaban et al. (2017)Shaban, A., Bansal, S., Liu, Z., Essa, I., and Boots, B.One-shot learning for semantic segmentation.*british machine vision conference (BMVC)*, 2017.
* Zhang et al. (2022)Zhang, Y., Zhou, K., and Liu, Z.Neural prompt search.*arXiv preprint arXiv:2206.04673*, 2022.
* Zhou et al. (2022a)Zhou, K., Liu, Z., Qiao, Y., Xiang, T., and Loy, C. C.Domain generalization: A survey.*IEEE Transactions on Pattern Analysis and Machine Intelligence
(TPAMI)*, 2022a.
* Zhou et al. (2022b)Zhou, K., Yang, J., Loy, C. C., and Liu, Z.Conditional prompt learning for vision-language models.In *Conference on Computer Vision and Pattern Recognition
(CVPR)*, 2022b.
* Zhou et al. (2022c)Zhou, K., Yang, J., Loy, C. C., and Liu, Z.Learning to prompt for vision-language models.*International Journal of Computer Vision (IJCV)*,
2022c.

Appendix A Illustration of In-context Examples
-----------------------------------------------

In the supplementary material, we illustrate more in-context learning results of foreground segmentation, single object detection, and colorization tasks.

### A.1 Foreground Segmentation

<img src='x6.png' alt='Refer to caption' title='' width='461' height='550' />

*Figure 6: In-context examples, which are from the foreground segmentation task, retrieved by UnsupPR and SupPR. These grids show examples from the train, tv, and bus categories.*

<img src='x7.png' alt='Refer to caption' title='' width='461' height='557' />

*Figure 7: In-context examples, which are from the foreground segmentation task, retrieved by UnsupPR and SupPR. These grids show examples from the bottle, sheep, and bird categories.*

<img src='x8.png' alt='Refer to caption' title='' width='461' height='559' />

*Figure 8: In-context examples, which are from the foreground segmentation task, retrieved by UnsupPR and SupPR. These grids show examples from the boat, airplane, and bicycle categories.*

<img src='x9.png' alt='Refer to caption' title='' width='461' height='551' />

*Figure 9: In-context examples, which are from the foreground segmentation task, retrieved by UnsupPR and SupPR. These grids show examples from the car, cat, and chair categories.*

<img src='x10.png' alt='Refer to caption' title='' width='461' height='547' />

*Figure 10: In-context examples, which are from the foreground segmentation task, retrieved by UnsupPR and SupPR. These grids show examples from the dog, horse, and motorbike categories.*

<img src='x11.png' alt='Refer to caption' title='' width='461' height='551' />

*Figure 11: In-context examples, which are from the foreground segmentation task, retrieved by UnsupPR and SupPR. These grids show examples from the table, plant, and sofa categories.*

The main paper presents the in-context examples from the person and cow categories. In the supplementary, as shown in Fig.[6](#A1.F6 "Figure 6 ‣ A.1 Foreground Segmentation ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?")-[11](#A1.F11 "Figure 11 ‣ A.1 Foreground Segmentation ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?"), we present examples from the remained 18 categories in Pascal-5i.

### A.2 Single Object Detection

<img src='x12.png' alt='Refer to caption' title='' width='461' height='563' />

*Figure 12: In-context examples, which are from the single object detection task, retrieved by UnsupPR and SupPR. We find the examples found by SupPR are more similar to the queries in terms of object pose (e.g., (f)), viewpoint (e.g., (r))*

<img src='x13.png' alt='Refer to caption' title='' width='461' height='563' />

*Figure 13: In-context examples, which are from the single object detection task, retrieved by UnsupPR and SupPR. We find the examples found by SupPR are more similar to the queries in terms of object pose (e.g., (l)), viewpoint (e.g., (m))*

As shown in Fig.[12](#A1.F12 "Figure 12 ‣ A.2 Single Object Detection ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?")-[13](#A1.F13 "Figure 13 ‣ A.2 Single Object Detection ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?"), we illustrate the in-context examples from the single object detection task. By comparing the in-context examples picked by UnsupPR and those picked by SupPR, we find the examples found by SupPR are more similar to the queries in terms of object pose (e.g., Fig.[12](#A1.F12 "Figure 12 ‣ A.2 Single Object Detection ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?")(f)), viewpoint (e.g., Fig.[12](#A1.F12 "Figure 12 ‣ A.2 Single Object Detection ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?")(r).

### A.3 Coloralization

As shown in Fig.[14](#A1.F14 "Figure 14 ‣ A.3 Coloralization ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?")-[15](#A1.F15 "Figure 15 ‣ A.3 Coloralization ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?"), we illustrate the in-context examples from the colorization task. This task aims to map a gray-scale image to a color image. By comparing the in-context examples picked by UnsupPR and those picked by SupPR, we find the ground truth images of examples found by SupPR are more similar to that of the queries in terms of image style, e.g. the background color (e.g., Fig.[14](#A1.F14 "Figure 14 ‣ A.3 Coloralization ‣ Appendix A Illustration of In-context Examples ‣ What Makes Good Examples for Visual In-Context Learning?")(g)(h)).

<img src='x14.png' alt='Refer to caption' title='' width='461' height='526' />

*Figure 14: In-context examples, which are from the colorization task, retrieved by UnsupPR and SupPR. We also show the ground truth of the query image. The query image is the gray-scale version of its ground truth. The ground truth images of the in-context examples found by SupPR are more similar than those found by UnsupPR to the ground truth images of queries in terms of image style, e.g. the background color (g).*

<img src='x15.png' alt='Refer to caption' title='' width='461' height='515' />

*Figure 15: In-context examples, which are from the colorization task, retrieved by UnsupPR and SupPR. We also show the ground truth of the query image. The query image is the gray-scale version of its ground truth. The ground truth images of the in-context examples found by SupPR are more similar than those found by UnsupPR to the ground truth images of queries in terms of image style, e.g. the background color (h).*
