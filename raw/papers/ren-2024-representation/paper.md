Representation Learning with Large Language Models  for Recommendation
======================================================================

Xubin Ren1, Wei Wei1, Lianghao Xia1, Lixin Su2, Suqi Cheng2  
Junfeng Wang2, Dawei Yin2 and Chao Huang1∗1University of Hong Kong2Baidu Inc.

###### Abstract.

Recommender systems have seen significant advancements with the influence of deep learning and graph neural networks, particularly in capturing complex user-item relationships. However, these graph-based recommenders heavily depend on ID-based data, potentially disregarding valuable textual information associated with users and items, resulting in less informative learned representations. Moreover, the utilization of implicit feedback data introduces potential noise and bias, posing challenges for the effectiveness of user preference learning. While the integration of large language models (LLMs) into traditional ID-based recommenders has gained attention, challenges such as scalability issues, limitations in text-only reliance, and prompt input constraints need to be addressed for effective implementation in practical recommender systems. To address these challenges, we propose a model-agnostic framework RLMRec that aims to enhance existing recommenders with LLM-empowered representation learning. It proposes a recommendation paradigm that integrates representation learning with LLMs to capture intricate semantic aspects of user behaviors and preferences. RLMRec incorporates auxiliary textual signals, develops a user/item profiling paradigm empowered by LLMs, and aligns the semantic space of LLMs with the representation space of collaborative relational signals through a cross-view alignment framework. This work further establish a theoretical foundation demonstrating that incorporating textual signals through mutual information maximization enhances the quality of representations. In our evaluation, we integrate RLMRec with state-of-the-art recommender models, while also analyzing its efficiency and robustness to noise data. Our implementation codes are available at <https://github.com/HKUDS/RLMRec>.

$*$ Corresponding author, chaohuang75@gmail.com

††copyright: none

1. Introduction
----------------

Recommender systems have evolved to provide personalized item recommendations based on user interactions, with deep learning and graph neural networks playing a significant role*(Chang et al., [2021](#bib.bib5 ""); Wu et al., [2022](#bib.bib40 ""))*. Graph-based recommenders like NGCF*(Wang et al., [2019](#bib.bib36 ""))* and LightGCN*(He et al., [2020](#bib.bib12 ""))* have demonstrated impressive capabilities in capturing complex user-item relationships, making them state-of-the-art approaches.

However, it is important to acknowledge that recent graph-based recommenders heavily rely on ID-corresponding information for learning. In this line, the training data consists exclusively of mapped user/item indices, and their interactions are represented in an interaction matrix using binary values (1 indicating an interaction and 0 indicating no interaction). While this data arrangement has demonstrated effectiveness, one significant limitation is that it primarily relies on ID-based information, thereby potentially overlooking other valuable data, such as rich textual information associated with users and items. The absence of this additional information can lead to reduced informativeness in the learned representations. Furthermore, it is worth noting that a substantial portion of the data in these graph-based recommenders consists of implicit feedback*(Rendle et al., [2012](#bib.bib28 ""); Wang et al., [2021a](#bib.bib34 ""))*, which can introduce noise from false negatives or bias (e.g., misclicks*(Wang et al., [2021b](#bib.bib35 ""))* or popularity bias*(Chen et al., [2023](#bib.bib6 ""))*). Consequently, the learned representations of these GNN-based models heavily rely on the inherent quality of the data. This heavy reliance on the data quality poses a potential challenge as it can lead to detrimental representations that hinder the effectiveness of recommendation systems, especially when the data contains noise.

In recent times, there have been several endeavors to leverage diverse data modalities in order to enhance traditional ID-based recommenders*(Yuan et al., [2023](#bib.bib47 ""); Li et al., [2023](#bib.bib18 ""); Geng et al., [2022](#bib.bib10 ""))*. Particularly interesting is the emergence of large language models (LLMs) like GPT-4*(OpenAI, [2023](#bib.bib25 ""))* and LLaMA*(Touvron et al., [2023](#bib.bib33 ""))*, which have demonstrated impressive capabilities in neural language understanding tasks. This development has sparked significant interest among researchers, who are actively exploring how LLMs, with their proficiency in handling textual content, can extend the capabilities of recommendation systems beyond the original data*(Fan et al., [2023](#bib.bib8 ""); Lin et al., [2023](#bib.bib19 ""); Liu et al., [2023b](#bib.bib22 ""))*. A primary focus of current research in this field revolves around aligning recommendation approaches with the characteristics of language models through prompt design. Methods like InstructRec*(Zhang et al., [2023](#bib.bib48 ""))* structure the recommendation task in an instruction-question-answering format, enabling LLMs to simultaneously address the recommendation objective and respond to intricately designed questions*(Geng et al., [2022](#bib.bib10 ""); Bao et al., [2023](#bib.bib2 ""))*. However, despite displaying some recommendation capabilities, these methods still fall behind existing recommenders in terms of efficiency and precision. This can be attributed to inherent shortcomings associated with this approach, including the following key aspects:

i) Scalability issues in practical recommenders. The utilization of large language models (LLMs) in personalized user behavior modeling, comes with inherent computational demands. As the scale of user behavior data increases, the computational requirements and associated inference time costs also tend to rise significantly. For instance, in the case of TALLRec*(Bao et al., [2023](#bib.bib2 ""))*, where recommendations are generated based on an instruction-question-answering format, the response time for LLaMA2-13B to provide recommendations to individual users stands at approximately 3.6 seconds, based on an input size of around 800 tokens (equivalent to approximately 5 users). However, this poses significant challenges when attempting to scale up the LLM-based approach for practical recommender systems with a substantial user base and extensive item catalog.

ii) Limitations stemming from text-only reliance. LLMs have the potential to generate text answers that may include recommendations for non-existent items due to hallucination issues*(Liu et al., [2023a](#bib.bib21 ""))*. This poses a challenge in ensuring the accuracy and reliability of the generated recommendations. Additionally, the limited capacity of prompt inputs, constrained by the maximum number of tokens (e.g., 2048 tokens for LLaMA), hinders the effective modeling of comprehensive collaborative signals with global user dependencies.

<img src='x1.png' alt='Refer to caption' title='' width='461' height='124' />

*Figure 1. LLM’s performance for recommendation reranking when dealing with different sizes of candidate items.*

To validate the aforementioned limitations, we conduct an evaluation to assess the effectiveness of directly using LLM in enhancing the re-ranking task*(Hou et al., [2023](#bib.bib13 ""); Sun et al., [2023](#bib.bib32 ""))* for recommendation on the Amazon dataset. Specifically, we utilize LightGCN*(He et al., [2020](#bib.bib12 ""))* as the underlying backbone recommender model, which generate a ranking list of 50 candidate items preferred by each user based on learned user-item interaction probabilities. To further refine the recommendations, we integrate the textual information of each item using our custom prompts (for prompt format details, please refer to Appendix[A.3](#A1.SS3 "A.3. Analysis on the Reranking Task with LLMs ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation")). These prompts are then processed by ChatGPT platform (i.e., gpt-3.5-turbo). The objective of this evaluation is to identify the Top-10 and Top-20 most relevant items from the ranking list generated by LightGCN for each user through the re-ranking task.

It is evident from the results in Figure[1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Representation Learning with Large Language Models for Recommendation") that the recommendations refined by the ChatGPT perform worse than the original results provided by LightGCN. This indicates that there are limitations when blindly using LLMs to improve the re-ranking process in recommendation systems that heavily rely on textual information. These limitations can be attributed to three key factors: i) The hallucination issue of LLMs, where they may suggest recommended items that are not included in the candidate set; ii) The lack of a comprehensive global text-based collaborative relationship input due to the token limit of LLMs; iii) Additionally, it is worth noting that the reranking process using LLM takes several hours to complete, which poses a challenge when dealing with large-scale data in real-world recommendation scenarios. Due to page limit, we delve deeper into this experiment and present real cases to illustrate the hallucination phenomenon for the reranking in the Appendix.

Contributions. In light of the aforementioned limitations, our work aims to leverage the power of LLMs to seamlessly enhance existing recommender systems. To accomplish this, we propose a model-agnostic framework called RLMRec (Representation Learning with Large Language Models for Recommendation). The core idea of RLMRec is to utilize representation learning as a bridge between ID-based recommenders and LLMs. Our new recommendation paradigm aims to preserve the accuracy and efficiency of existing recommenders while harnessing the powerful text comprehension capabilities of LLMs to understand the intricate semantic aspects of user behaviors and preferences. To begin, we lay the theoretical groundwork by modeling the benefits of incorporating auxiliary textual signals for representation learning. This involves transforming the textual signals into meaningful representations and establishing a theoretical foundation for maximizing mutual information within general recommendation models. Furthermore, we develop a user/item profiling paradigm empowered by LLMs. This paradigm enables us to incorporate the comprehensive semantic understanding derived from LLMs into the encoded representations of users and items. By leveraging the global knowledge space of LLMs, we enhance the representation expressiveness.

Furthermore, we propose to align the semantic space of LLMs and the representation space of collaborative relational signals through a cross-view alignment framework. This alignment is achieved through a cross-view mutual information maximization scheme, which allows us to find a common semantic subspace where the textual and collaborative relational embeddings are well aligned from the contrastive and generative modeling, respectively. This typically involves jointly optimizing the recommendation model parameters and the cross-modal alignment paradigm. In a nutshell, our main contributions can be summarized as follows:

* •

    This work aims to explore the potential of enhancing the recommendation performance of existing recommender systems, by leveraging LLMs and aligning their semantic space with collaborative relation modeling for better representation learning.

* •

    We propose a model-agnostic representation learning framework called RLMRec, which is guided by our theoretical findings. This framework leverages contrastive or generative modeling techniques to enhance the quality of learned representations.

* •

    We establish a theoretical foundation to demonstrate the effectiveness of incorporating textual signals in enhancing the representation learning of existing recommenders. By utilizing mutual information maximization as the optimization direction, we show how textual signals can improve the representation quality.

* •

    We integrate RLMRec with various state-of-the-art recommender models and validate the effectiveness of our method. Additionally, we analyze the framework’s robustness to noise and incomplete data, showcasing its ability to handle real-world challenges.

2. Related Work
----------------

GNN-enhanced Collaborative Filtering. Collaborative Filtering (CF), which is a fundamental technique in recommendation systems, has been extensively studied over the years*(Su and Khoshgoftaar, [2009](#bib.bib31 ""); Koren et al., [2021](#bib.bib17 ""))*. Recently, an emerging research direction involves leveraging historical user-item interactions to construct a bipartite graph and utilizing graph neural networks (GNNs) to capture high-order collaborative relationships. These graph-based methods, such as NGCF*(Wang et al., [2019](#bib.bib36 ""))*, GCCF*(Chen et al., [2020](#bib.bib7 ""))*, LightGCN*(He et al., [2020](#bib.bib12 ""))*, have demonstrated state-of-the-art performance, improving recommendation effectiveness. However, the sparsity and noise in implicit feedback data pose challenges to graph-based methods. To address these challenges, researchers have started exploring the use of self-supervised learning (SSL) techniques as auxiliary learning objectives to enhance robustness in recommendations*(Yu et al., [2023](#bib.bib46 ""); Yang et al., [2023](#bib.bib44 ""))*. Among various SSL techniques, contrastive learning has emerged as a prominent solution in collaborative filtering models. Methods like SGL*(Wu et al., [2021](#bib.bib38 ""))*, SimGCL*(Yu et al., [2022](#bib.bib45 ""))*, NCL*(Lin et al., [2022](#bib.bib20 ""))*, LightGCL*(Cai et al., [2023](#bib.bib4 ""))* leverage contrastive data augmentation to improve recommendation performance. In this work, we take a step further by integrating LLMs with existing CF models to effectively align the knowledge and reasoning abilities of LLMs with the collaborative relation learning for enhancing recommendation performance.

Large Language Models for Recommendation. Recently, there has been a growing interest in leveraging LLMs for recommendation systems.*(Fan et al., [2023](#bib.bib8 ""); Lin et al., [2023](#bib.bib19 ""); Liu et al., [2023b](#bib.bib22 ""); Wu et al., [2023](#bib.bib39 ""))*. Several studies have leveraged LLMs as inference models by designing prompts that align them with recommendation tasks. For example, P5*(Geng et al., [2022](#bib.bib10 ""))* converts the user interaction data into textual prompts using item indexes, which are then used for language model training. Chat-REC*(Gao et al., [2023](#bib.bib9 ""))* builds a conversational recommender by transforming user profiles and interactions into prompts for LLMs to generate recommendations. InstructRec*(Zhang et al., [2023](#bib.bib48 ""))* and TALLRec*(Bao et al., [2023](#bib.bib2 ""))* employ instructional designs to define recommendation tasks and fine-tune LLMs to align with these instructions for generating recommendations. However, using LLMs directly as inference models for recommendation tasks presents challenges, such as high computational costs and slow inference times. These challenges hinder the practical deployment of such models in real-world recommender systems. To address this gap, the proposed approach adopts a theoretically grounded paradigm of mutual information maximization to align the knowledge of LLMs with collaborative relation modeling, enabling scalable and effective recommendations.

3. Methodology
---------------

### 3.1. Theoretical Basis of RLMRec

Collaborative Filtering. In our recommendation scenario, we have a set of users $\mathcal{U}\={u_{1},...,u_{I}}$ and a set of items $\mathcal{V}\={v_{1},...,v_{J}}$. The observed user-item interactions are represented by $\mathcal{X}$. In learning-based recommenders, each user and item is assigned initial embeddings $\textbf{x}_{u}$ and $\textbf{x}_{v}$. The goal is to learn user and item representations $\textbf{e}_{u},\textbf{e}_{v}$ through a recommender model (i.e., $\textbf{e}_{u},\textbf{e}_{v}\=\mathcal{R}(\textbf{x}_{u},\textbf{x}_{v})$) that maximizes the posterior distribution shown below:

| (1) |  | $\displaystyle p(\mathbf{e}|\mathcal{X})\propto p(\mathcal{X}|\mathbf{e})p(\mathbf{e}).$ |  |
| --- | --- | --- | --- |

In practical recommendation scenarios, the observed user-item interactions $\mathcal{X}$ often contain noise, including false positives (e.g., misclicks or interactions influenced by popularity bias) and false negatives (e.g., users do not interact with unseen but interested items). As a result, the learned representation $\mathbf{e}$ can also be affected by this noise, which negatively impacts recommendation accuracy. In this work, we introduce a hidden prior belief $\mathbf{z}$ that is inherently beneficial for recommendation. This prior belief helps identify the true positive samples in $\mathcal{X}$. Hence, the generation of representation $\mathbf{e}$ involves a combination of the advantageous prior belief $\mathbf{z}$ and the unavoidable noise present during the learning process.

Text-enhanced User Preference Learning. To mitigate the impact of irrelevant signals on the representation, it is necessary to incorporate auxiliary informative cues. One approach is to introduce textual information, e.g., user and item profiles, which provide insights for user preference learning. These profiles can be encoded using language models to generate representations $\mathbf{s}\in\mathbb{R}^{d_{s}}$ that effectively capture the semantic aspects of user preferences. Importantly, both $\mathbf{s}$ and $\mathbf{e}$ capture shared information that is relevant to the aspects associated with user-item interactions. This shared information is crucial as it indicates the inclusion of beneficial aspects for recommendation, aligning with the prior belief $\mathbf{z}$.

<img src='x2.png' alt='Refer to caption' title='' width='438' height='194' />

*Figure 2. The type of directed graph model under consideration. As the alignment between CF-side representation and LLM-enhanced representation, the noisy effects in the learned representations $e$ are alleviated in RLMRec.*

<img src='x3.png' alt='Refer to caption' title='' width='461' height='88' />

*Figure 3. The overall framework of our proposed LLM-enhanced representation learning framework RLMRec.*

With the collaborative-side representation $\mathbf{e}$ and textual-side representation $\mathbf{s}$, both of which contain recommendation-beneficial information generated from $\mathbf{z}$, our objective is to learn the optimal value of $\mathbf{e}$ denoted as $\mathbf{e}^{*}$, by maximizing the conditional probability:

| (2) |  | $\displaystyle\mathbf{e}^{*}\=\mathop{\arg\max}\limits_{\mathbf{e}}\mathbb{E}_{p(\mathbf{e},\mathbf{s})}[p(\mathbf{z},\mathbf{s}|\mathbf{e})].$ |  |
| --- | --- | --- | --- |

The underlying intuition behind maximizing the conditional probability is to ensure that the learnable representation $\mathbf{e}$ from recommender models incorporates purer information generated from the prior belief $\mathbf{z}$ and the shared information with the semantic representation $\mathbf{s}$. By doing so, the relevance and benefits of the learned representations $\mathbf{e}$ for recommendation are enhanced.

Theorem 1. Maximizing the posteriori probability $\mathbb{E}_{p(\mathbf{e},\mathbf{s})}[p(\mathbf{z},\mathbf{s}|\mathbf{e})]$ given the hidden prior belief $\mathbf{z}$, is equivalent to maximizing the mutual information $I(e;s)$ between the CF-side relational representation $\mathbf{e}$ and LLM-side semantic representation $\mathbf{s}$.

Proof. It is important to note that since the profiles of users and items are fixed, the probability $p(\mathbf{s})$ remains constant during the learning process. Therefore, we can deduce the following:

| (3) |  | $\displaystyle\mathbb{E}_{p(\mathbf{e},\mathbf{s})}[p(\mathbf{z},\mathbf{s}|\mathbf{e})]$ | $\displaystyle\propto\mathbb{E}_{p(\mathbf{e},\mathbf{s})}\log[\int_{\mathbf{z}}\ \frac{p(\mathbf{z},\mathbf{s}|\mathbf{e})}{p(\mathbf{s})}\,d\mathbf{z}\,]$ |  |
| --- | --- | --- | --- | --- |
| (4) |  |  | $\displaystyle\=\mathbb{E}_{p(\mathbf{e},\mathbf{s})}\log[\frac{\int_{\mathbf{z}}\ p(\mathbf{z},\mathbf{e}|\mathbf{s})\,d\mathbf{z}\,}{p(\mathbf{e})}]$ |  |
| --- | --- | --- | --- | --- |
| (5) |  |  | $\displaystyle\=\mathbb{E}_{p(\mathbf{e},\mathbf{s})}\log[\frac{p(\mathbf{e}|\mathbf{s})}{p(\mathbf{e})}]\=I(\mathbf{e},\mathbf{s}).$ |  |
| --- | --- | --- | --- | --- |

Let’s consider $\mathbf{e}$ and $\mathbf{s}$ as data samples, assuming that we have $N$ pairwise corresponding elements of $\mathbf{e}$ and $\mathbf{s}$, forming the sets $\mathbf{E}\={\mathbf{e}_{1},\ldots,\mathbf{e}_{i},\ldots,\mathbf{e}_{N}}$ and $\mathbf{S}\={\mathbf{s}_{1},\ldots,\mathbf{s}_{i},\ldots,\mathbf{s}_{N}}$, respectively. Based on this, we optimize the mutual information as follows.

Theorem 2. By introducing the density ratio to preserve mutal information*(Oord et al., [2018](#bib.bib24 ""))* $f(\mathbf{s},\mathbf{e})\propto p(\mathbf{s}|\mathbf{e})/p(\mathbf{s})$, the maximization of $I(\mathbf{e}_{i};\mathbf{s}_{i})$ can be reformulated as maximizing the following lower bound:

| (6) |  | $\displaystyle\mathbb{E}\log[\frac{f(\mathbf{s}_{i},\mathbf{e}_{i})}{\sum_{\mathbf{s}_{j}\in\mathbf{S}}f(\mathbf{s}_{j},\mathbf{e}_{i})}].$ |  |
| --- | --- | --- | --- |

Proof. Based on the property of mutual information, we have $I(\textbf{e}_{i},\textbf{s}_{i})\=I(\textbf{s}_{i},\textbf{e}_{i})$. With this in mind, we make the deductions as:

| (7) |  | $\displaystyle I(\mathbf{s}_{i},\mathbf{e}_{i})$ | $\displaystyle\geq I(\mathbf{s}_{i},\mathbf{e}_{i})-\log(N)\=-\mathbb{E}\log[\frac{p(\mathbf{s}_{i})}{p(\mathbf{{s}_{i}}|\mathbf{e}_{i})}N]$ |  |
| --- | --- | --- | --- | --- |
| (8) |  |  | $\displaystyle\geq-\mathbb{E}\log[1+\frac{p(\mathbf{s}_{i})}{p(\mathbf{{s}_{i}}|\mathbf{e}_{i})}(N-1)]$ |  |
| --- | --- | --- | --- | --- |
| (9) |  |  | $\displaystyle\=-\mathbb{E}\log[1+\frac{p(\mathbf{s}_{i})}{p(\mathbf{{s}_{i}}|\mathbf{e}_{i})}(N-1)\mathbb{E}_{\mathbf{s}_{j}\in\mathbf{S}_{neg}}\frac{p(\mathbf{s}_{j}|\mathbf{e}_{i})}{p(\mathbf{s}_{j})}]$ |  |
| --- | --- | --- | --- | --- |
| (10) |  |  | $\displaystyle\approx-\mathbb{E}\log[1+\frac{p(\mathbf{s}_{i})}{p(\mathbf{{s}_{i}}|\mathbf{e}_{i})}\sum_{\mathbf{s_{j}}\in\mathbf{S}_{neg}}\frac{p(\mathbf{s}_{j}|\mathbf{e}_{i})}{p(\mathbf{s}_{j})}]$ |  |
| --- | --- | --- | --- | --- |
| (11) |  |  | $\displaystyle\=\mathbb{E}\log[\frac{f(\mathbf{s}_{i},\mathbf{e}_{i})}{\sum_{\mathbf{s}_{j}\in\mathbf{S}}f(\mathbf{s}_{j},\mathbf{e}_{i})}].$ |  |
| --- | --- | --- | --- | --- |

Here, $\mathbf{S}_{\text{neg}}$ represents the negative samples when considering the $i$-th sample (i.e., $\mathbf{S}_{neg}\=\mathbf{S}\setminus{s_{i}}$).
Up to this point, we have derived, from a theoretical perspective, how to alleivate noisy effects in representations by introducing external knowledge. However, this approach also presents two challenges: i) Challenge 1: How to obtain effective descriptions of users and items that capture their interaction preferences. ii) Challenge 2: How to involves effectively modeling the density ratio $f(\mathbf{s},\mathbf{e})$ to maximize the mutual information between $\mathbf{e}$ and $\mathbf{s}$. In the following sections, we discuss potential solutions to address these two challenges.

### 3.2. User/Item Profiling Paradigm

In our previous derivation, we emphasize the importance of obtaining textual descriptions, referred to as profiles, for users and items. These profiles play a crucial role in mitigating the impact of noise in the learned representations of recommenders and enable a semantic understanding of users’ and items’ interaction preferences. Ideally, user and item profiles should exhibit the following characteristics:

* •

    User profile: should effectively encapsulate the particular types of items that users are inclined to favor, allowing for a comprehensive representation of their personalized tastes and preferences.

* •

    Item profile: It should eloquently articulate the specific types of users that the item is apt to attract, providing a clear representation of the item’s characteristics and qualities that align with the preferences and interests of those users.

In some cases, the original data may include textual properties related to users and items. For example, in the Yelp dataset, users provide reviews for visited businesses, and businesses have attributes such as location and category. However, such textual data often contains extraneous noise, leading to common predicaments: i) Missing Attributes: Some attributes of certain items or users may be missing; ii) Noisy Textual Data: The text itself may be contaminated with a plethora of noise that is irrelevant to users’ preferences. For instance, in the Steam dataset, user reviews for games may contain numerous special symbols or irrelevant information. These challenges make it difficult to distill useful user and item profiles from text. As a result, prevailing models often convert low-noise attributes into one-hot encodings without effectively leveraging the semantic information present in the textual data.

Fortunately, recent advancements in Large Language Models (LLMs) have unleashed their remarkable text processing capabilities, enabling them to address a wide range of NLP tasks, including text denoising and summarization. This pivotal development opens up new possibilities for generating user and item profiles from the noisy textual features inherent in the dataset. Leveraging the tremendous potential of LLMs, we propose a paradigm for profile generation that capitalizes on collaborative information. Considering that datasets often contain a higher proportion of textual descriptions for item attributes compared to user attributes, our approach takes an item-to-user perspective, as outlined below.

#### 3.2.1. Profile Generation via Reasoning.

Recent research has demonstrated the effectiveness of incorporating process reasoning in LLMs to mitigate hallucination and improve the quality of generated outputs. Building upon these findings, we have meticulously designed the system prompt $\mathcal{S}_{u/v}$ as part of the input provided to LLMs. The objective is to clearly define its functionality in generating user profile for user $u$ or item profile for item $v$ by precisely specifying the input-output content and desired output format. Importantly, we explicitly emphasize the inclusion of reasoning processes as an integral part of the generated output. By combining this system prompt with user/item profile generation prompts $\mathcal{Q}_{u}$ and $\mathcal{Q}_{v}$, we can leverage LLMs to generate accurate profiles. The specific process is outlined as follows:

| (12) |  | $\displaystyle\mathcal{P}_{u}\=LLMs(\mathcal{S}_{u},\mathcal{Q}_{u}),\,\,\,\ \mathcal{P}_{v}\=LLMs(\mathcal{S}_{v},\mathcal{Q}_{v})$ |  |
| --- | --- | --- | --- |

#### 3.2.2. Item Prompt Construction.

we categorize the textual information of an item $v\in\mathcal{V}$ into four types: title $\alpha$, original description $\beta$, dataset-specific attributes $\bm{\gamma}\={\gamma_{1},...,\gamma_{|\bm{\gamma}|}}$, and a collection of $n$ reviews from users $\mathbf{r}\={r_{1},...,r_{n}}$. Based on these categories, we can formally outline the arrangement of the input prompt $\mathcal{Q}_{v}$ for item-profile generation as follows:

| (13) |  | $\displaystyle\mathcal{Q}_{v}\=f_{v}(\textbf{x})\,\,\,\,w.r.t.\,\,\,\,\textbf{x}\=\begin{cases}[\alpha,\beta],\&\text{if }\beta\,\,\text{exists},\\ [\alpha,\bm{\gamma},\hat{\textbf{r}}\subset\mathbf{r}],\&\text{other wise}.\end{cases}$ |  |
| --- | --- | --- | --- |

In our approach, we use a function $f_{v}(\cdot)$ specific to each item, which combines various text features into a single string. If the original description $\beta$ is missing, we randomly sample a subset of reviews $\hat{\textbf{r}}$ and combine them with the attributes for input. By incorporating item descriptions or user reviews, our prompts provide precise information to Large Language Models, ensuring that the generated item profiles accurately reflect appealing characteristics.

#### 3.2.3. User Prompt Construction.

To generate the profile of user $u$, we leverage collaborative information, assuming that we have already generated the item profiles beforehand. Specifically, we consider the items interacted with by user $u$ as $\mathcal{I}_{u}$ and uniformly sample a subset of items $\hat{\mathcal{I}}_{u}\subset\mathcal{I}_{u}$. For each item $v$ in $\hat{\mathcal{I}}_{u}$, we concatenate its textual attributes as $\textbf{c}_{v}\=[\alpha,\mathcal{P}_{v},r_{u}^{v}]$, where $r_{u}^{v}$ represents the review provided by user $u$. The input prompt $\mathcal{Q}_{u}$ for user-profile generation can be defined as follows:

| (14) |  | $\displaystyle\mathcal{Q}_{u}\=f_{u}({\mathbf{c}_{v}|v\in\hat{\mathcal{I}}_{u}}).$ |  |
| --- | --- | --- | --- |

The function $f_{u}(\cdot)$ serves a similar purpose to $f_{v}(\cdot)$ by organizing the textual content into a coherent string. Each textual attribute $\textbf{c}_{v}$ includes user reviews, which authentically reflect their genuine opinions. This construction of the user prompt provides valuable insights into their true preferences. Due to space constraints, we have included the detailed design of the prompt, including $\mathcal{S}$, $\mathcal{Q}$, and $f_{u/v}(\cdot)$, along with sample examples in Appendix[A.2](#A1.SS2 "A.2. Details of Profile Generation ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation").

### 3.3. Density Ratio Modeling for Mutual Information Maximization

In this section, we outline the process of modeling the density ratio, denoted as $f(\textbf{s}_{i},\textbf{e}_{i})$, with the objective of maximizing the mutual information $I(\textbf{s}_{i},\textbf{e}_{i})$. First of all, it is important to note that we have previously generated user/item profiles $\mathcal{P}_{u/v}$ that showcase their interaction preferences. As such, it is logical to encode the semantic representation s based on these profiles as follow:

| (15) |  | $\displaystyle\textbf{s}_{u}\=\mathcal{T}(\mathcal{P}_{u}),\,\,\,\,\textbf{s}_{v}\=\mathcal{T}(\mathcal{P}_{v}).$ |  |
| --- | --- | --- | --- |

Here, $\mathcal{T}(\cdot)$ refers to a cutting-edge technology known as a text embedding model*(Su et al., [2023](#bib.bib30 ""); Izacard et al., [2021](#bib.bib15 ""))*, which has been shown to effectively transform diverse text inputs into fixed-length vectors that retain their inherent meaning and contextual information.

According to*(Oord et al., [2018](#bib.bib24 ""))*, the density ratio $f(\textbf{s}_{i},\textbf{e}_{i})$ can be interpreted as a positive real-valued score measurement function that captures the similarity between $\textbf{s}_{i}$ and $\textbf{e}_{i}$. A more accurate modeling of the density ratio*(Sasaki and Takenouchi, [2022](#bib.bib29 ""))* can have a positive impact on the alignment between CF-side rational representations and LLMs-enhanced semantic representations, helping to mitigate the influence of noisy signals in representation learning. In this context, we propose two types of modeling approaches that are well-suited for achieving this alignment. The first approach is contrastive modeling, which has been extensively validated*(Wu et al., [2021](#bib.bib38 ""); Khosla et al., [2020](#bib.bib16 ""))* for effectively aligning different views bidirectionally, such as through pull and push pairs. The second approach is mask-reconstruction generative modeling, which is widely used as a self-supervised mechanism for reconstructing the partially masked input from data itself*(He et al., [2022](#bib.bib11 ""); Hou et al., [2022](#bib.bib14 ""))*. By employing CF-side representations to reconstruct the semantic representations, we can effectively align these two forms of information.

#### 3.3.1. Contrastive Alignment

As depicted in Fig[3](#S3.F3 "Figure 3 ‣ 3.1. Theoretical Basis of RLMRec ‣ 3. Methodology ‣ Representation Learning with Large Language Models for Recommendation") (b), we denote the specific implementation of $f(\textbf{s}_{i},\textbf{e}_{i})$ as contrastive alignment.

| (16) |  | $\displaystyle f(\textbf{s}_{i},\textbf{e}_{i})\=exp(sim(\sigma_{\downarrow}(\textbf{s}_{i}),\textbf{e}_{i})).$ |  |
| --- | --- | --- | --- |

The function $sim(\cdot)$ represents the cosine similarity, while $\sigma_{\downarrow}$ denotes a multi-layer perception that maps the semantic representation $\textbf{s}_{i}$ into the feature space of $\textbf{e}_{i}$. In our contrastive alignment, we treat $\textbf{e}_{i}$ and $\textbf{s}_{i}$ as positive sample pairs. During the learning process, these pairs are pulled towards each other to align their representations. In the specific implementation, the objective is to bring positive sample pairs closer within a batch while considering the remaining samples as negatives.

#### 3.3.2. Generative Alignment

Taking inspiration from recent research on the masked autoencoder (MAE), which is considered a paradigm of generative self-supervised learning, we propose an additional modeling approach for the density ratio within the MAE.

| (17) |  | $\displaystyle f(\textbf{s}_{i},\textbf{e}_{i})\=exp(sim(\textbf{s}_{i},\sigma_{\uparrow}(\hat{\textbf{e}_{u}})))\,\,\,\,w.r.t.\,\,\,\,\hat{\textbf{e}_{i}}\=\mathcal{R}({\textbf{x}}\setminus\textbf{x}_{i}).$ |  |
| --- | --- | --- | --- |

We employ $\sigma_{\uparrow}$ as a multi-layer perception model to map the representations to the semantic feature space. ${{\textbf{x}}}\setminus\textbf{x}_{i}$ represents the initial embedding of the $i$-th sample with masking applied. The generative process follows a single-direction reconstruction approach, focusing on reconstructing the semantic representations exclusively for the masked samples. Specifically, the masking operation involves replacing the initial embedding with a designated mask token (i.e., $[MASK]$), and a random subset of users/items is masked and subsequently reconstructed. This allows us to explore the reconstruction capabilities within the semantic feature space.

With our contrastive and generative alignment method, we effectively align the knowledge of the LLM with the domain of understanding user preferences. This is achieved by combining id-based collaborative relational signals with text-based behavior semantics. We have given the names RLMRec-Con and RLMRec-Gen to our two proposed modeling approaches, respectively. In our experiments conducted on real-world data, we will comprehensively evaluate the performance of these two models across various tasks, each showcasing its unique advantages and disadvantages.

### 3.4. Model-agnostic Learning

Up until this point, our focus has been on optimizing the CF-side relational representation e and LLM-side semantic representation s. Any model that can perform representation learning for users/items can undergo the optimization process described earlier. Hence, our approach is model-agnostic and can seamlessly enhance existing collaborative filtering recommenders. Assuming that the optimization objective of the recommender $\mathcal{R}$ is denoted as $\mathcal{L}_{\mathcal{R}}$, our overall optimization function $\mathcal{L}$ can be formulated as follows:

| (18) |  | $\displaystyle\mathcal{L}\=\mathcal{L}_{\mathcal{R}}+\mathcal{L}_{info}\,\,\,\,w.r.t.\,\,\,\mathcal{L}_{info}\=-\mathbb{E}\log[\frac{f(\mathbf{s}_{i},\mathbf{e}_{i})}{\sum_{\mathbf{s}_{j}\in\mathbf{S}}f(\mathbf{s}_{j},\mathbf{e}_{i})}],$ |  |
| --- | --- | --- | --- |

Minimizing the overall optimization function $\mathcal{L}$ corresponds to maximizing the mutual information mentioned earlier.

4. Evaluation
--------------

*Table 1. Recommendation performance Imprvement of all backbone methods on different datasets in terms of Recall and NDCG. The superscript * indicates the Imprvement is statistically significant where the p-value is less than $0.05$.*

| Data | | Amazon-book | | | | | | Yelp | | | | | | Steam | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Backbone | Variants | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
| Semantic Embeddings Only | | 0.0081 | 0.0125 | 0.0199 | 0.0072 | 0.0088 | 0.0112 | 0.0013 | 0.0022 | 0.0047 | 0.0014 | 0.0018 | 0.0026 | 0.0033 | 0.0062 | 0.0120 | 0.0031 | 0.0043 | 0.0064 |
| GCCF | Base | 0.0537 | 0.0872 | 0.1343 | 0.0537 | 0.0653 | 0.0807 | 0.0390 | 0.0652 | 0.1084 | 0.0451 | 0.0534 | 0.0680 | 0.0500 | 0.0826 | 0.1313 | 0.0556 | 0.0665 | 0.0830 |
| | RLMRec-Con | 0.0561* | 0.0899* | 0.1395* | 0.0562* | 0.0679* | 0.0842* | 0.0409* | 0.0685* | 0.1144* | 0.0474* | 0.0562* | 0.0719* | 0.0538* | 0.0883* | 0.1398* | 0.0597* | 0.0713* | 0.0888* |
| RLMRec-Gen | 0.0551* | 0.0891* | 0.1372* | 0.0559* | 0.0675* | 0.0832* | 0.0393 | 0.0654 | 0.1074 | 0.0454 | 0.0535 | 0.0678 | 0.0532* | 0.0874* | 0.1385* | 0.0588* | 0.0702* | 0.0875* |
| Best Imprv. | $\uparrow$4.28% | $\uparrow$3.10% | $\uparrow$3.87% | $\uparrow$4.66% | $\uparrow$3.98% | $\uparrow$4.34% | $\uparrow$4.87% | $\uparrow$5.06% | $\uparrow$5.54% | $\uparrow$5.10% | $\uparrow$5.24% | $\uparrow$5.74% | $\uparrow$7.60% | $\uparrow$6.90% | $\uparrow$6.47% | $\uparrow$7.37% | $\uparrow$7.22% | $\uparrow$6.99% |
| LightGCN | Base | 0.0570 | 0.0915 | 0.1411 | 0.0574 | 0.0694 | 0.0856 | 0.0421 | 0.0706 | 0.1157 | 0.0491 | 0.0580 | 0.0733 | 0.0518 | 0.0852 | 0.1348 | 0.0575 | 0.0687 | 0.0855 |
| | RLMRec-Con | 0.0608* | 0.0969* | 0.1483* | 0.0606* | 0.0734* | 0.0903* | 0.0445* | 0.0754* | 0.1230* | 0.0518* | 0.0614* | 0.0776* | 0.0548* | 0.0895* | 0.1421* | 0.0608* | 0.0724* | 0.0902* |
| RLMRec-Gen | 0.0596* | 0.0948* | 0.1446* | 0.0605* | 0.0724* | 0.0887* | 0.0435* | 0.0734* | 0.1209* | 0.0505 | 0.0600* | 0.0761* | 0.0550* | 0.0907* | 0.1433* | 0.0607* | 0.0729* | 0.0907* |
| Best Imprv. | $\uparrow$6.67% | $\uparrow$5.90% | $\uparrow$5.10% | $\uparrow$5.57% | $\uparrow$5.76% | $\uparrow$5.49% | $\uparrow$5.70% | $\uparrow$6.80% | $\uparrow$6.31% | $\uparrow$5.50% | $\uparrow$5.86% | $\uparrow$5.87% | $\uparrow$6.18% | $\uparrow$6.46% | $\uparrow$6.31% | $\uparrow$5.74% | $\uparrow$6.11% | $\uparrow$6.08% |
| SGL | Base | 0.0637 | 0.0994 | 0.1473 | 0.0632 | 0.0756 | 0.0913 | 0.0432 | 0.0722 | 0.1197 | 0.0501 | 0.0592 | 0.0753 | 0.0565 | 0.0919 | 0.1444 | 0.0618 | 0.0738 | 0.0917 |
| | RLMRec-Con | 0.0655* | 0.1017* | 0.1528* | 0.0652* | 0.0778* | 0.0945* | 0.0452* | 0.0763* | 0.1248* | 0.0530* | 0.0626* | 0.0790* | 0.0589* | 0.0956* | 0.1489* | 0.0645* | 0.0768* | 0.0950* |
| RLMRec-Gen | 0.0644 | 0.1015 | 0.1537* | 0.0648* | 0.0777* | 0.0947* | 0.0467* | 0.0771* | 0.1263* | 0.0537* | 0.0631* | 0.0798* | 0.0574* | 0.0940* | 0.1476* | 0.0629* | 0.0752* | 0.0934* |
| Best Imprv. | $\uparrow$2.83% | $\uparrow$2.31% | $\uparrow$4.34% | $\uparrow$3.16% | $\uparrow$2.91% | $\uparrow$3.72% | $\uparrow$8.10% | $\uparrow$6.79% | $\uparrow$5.51% | $\uparrow$7.19% | $\uparrow$6.59% | $\uparrow$5.98% | $\uparrow$5.20% | $\uparrow$4.03% | $\uparrow$3.12% | $\uparrow$4.37% | $\uparrow$4.07% | $\uparrow$3.60% |
| SimGCL | Base | 0.0618 | 0.0992 | 0.1512 | 0.0619 | 0.0749 | 0.0919 | 0.0467 | 0.0772 | 0.1254 | 0.0546 | 0.0638 | 0.0801 | 0.0564 | 0.0918 | 0.1436 | 0.0618 | 0.0738 | 0.0915 |
| | RLMRec-Con | 0.0633* | 0.1011* | 0.1552* | 0.0633* | 0.0765* | 0.0942* | 0.0470 | 0.0784* | 0.1292* | 0.0546 | 0.0642 | 0.0814* | 0.0582* | 0.0945* | 0.1482* | 0.0638* | 0.0760* | 0.0942* |
| RLMRec-Gen | 0.0617 | 0.0991 | 0.1524* | 0.0622 | 0.0752 | 0.0925* | 0.0464 | 0.0767 | 0.1267 | 0.0541 | 0.0634 | 0.0803 | 0.0572 | 0.0929 | 0.1456* | 0.0627* | 0.0747* | 0.0926* |
| Best Imprv. | $\uparrow$2.43% | $\uparrow$1.92% | $\uparrow$2.65% | $\uparrow$2.26% | $\uparrow$2.14% | $\uparrow$2.50% | $\uparrow$0.64% | $\uparrow$1.55% | $\uparrow$3.03% | $-$ | $\uparrow$0.63% | $\uparrow$1.62% | $\uparrow$3.19% | $\uparrow$2.94% | $\uparrow$1.53% | $\uparrow$3.24% | $\uparrow$2.98% | $\uparrow$2.95% |
| DCCF | Base | 0.0662 | 0.1019 | 0.1517 | 0.0658 | 0.0780 | 0.0943 | 0.0468 | 0.0778 | 0.1249 | 0.0543 | 0.0640 | 0.0800 | 0.0561 | 0.0915 | 0.1437 | 0.0618 | 0.0736 | 0.0914 |
| | RLMRec-Con | 0.0665 | 0.1040* | 0.1563* | 0.0668 | 0.0798* | 0.0968* | 0.0486* | 0.0813* | 0.1321* | 0.0561* | 0.0663* | 0.0836* | 0.0572* | 0.0929* | 0.1459* | 0.0627* | 0.0747* | 0.0927* |
| RLMRec-Gen | 0.0666 | 0.1046* | 0.1559* | 0.0670* | 0.0801* | 0.0969* | 0.0475 | 0.0785 | 0.1281* | 0.0549 | 0.0646 | 0.0815 | 0.0570* | 0.0918 | 0.1430 | 0.0625 | 0.0741 | 0.0915 |
| Best Imprv. | $\uparrow$0.60% | $\uparrow$2.65% | $\uparrow$3.03% | $\uparrow$1.82% | $\uparrow$2.69% | $\uparrow$2.76% | $\uparrow$3.85% | $\uparrow$4.50% | $\uparrow$5.76% | $\uparrow$3.31% | $\uparrow$3.59% | $\uparrow$4.50% | $\uparrow$2.14% | $\uparrow$1.53% | $\uparrow$1.53% | $\uparrow$1.46% | $\uparrow$1.49% | $\uparrow$1.42% |
| AutoCF | Base | 0.0689 | 0.1055 | 0.1536 | 0.0705 | 0.0828 | 0.0984 | 0.0469 | 0.0789 | 0.1280 | 0.0547 | 0.0647 | 0.0813 | 0.0519 | 0.0853 | 0.1358 | 0.0572 | 0.0684 | 0.0855 |
| | RLMRec-Con | 0.0695 | 0.1083* | 0.1586* | 0.0704 | 0.0837 | 0.1001* | 0.0488* | 0.0814* | 0.1319* | 0.0562* | 0.0663* | 0.0835* | 0.0540* | 0.0876* | 0.1372* | 0.0593* | 0.0704* | 0.0872* |
| RLMRec-Gen | 0.0693 | 0.1069* | 0.1581* | 0.0701 | 0.0830 | 0.0996 | 0.0493* | 0.0828* | 0.1330* | 0.0572* | 0.0677* | 0.0848* | 0.0539* | 0.0888* | 0.1410* | 0.0593* | 0.0710* | 0.0886* |
| Best Imprv. | $\uparrow$0.87% | $\uparrow$2.65% | $\uparrow$3.26% | $\downarrow$0.14% | $\uparrow$1.87% | $\uparrow$1.73% | $\uparrow$5.12% | $\uparrow$4.94% | $\uparrow$3.91% | $\uparrow$4.57% | $\uparrow$4.64% | $\uparrow$4.31% | $\uparrow$4.05% | $\uparrow$4.10% | $\uparrow$3.83% | $\uparrow$3.67% | $\uparrow$3.80% | $\uparrow$3.63% |

This section presents the experimental evaluation of our RLMRec on multiple datasets to address the following research questions:

* •

    RQ1: Does our proposed RLMRec improve upon existing state-of-the-art recommenders across various experimental settings?

* •

    RQ2: Do the LLM-enhanced semantic representations contribute to the recommendation performance improvement?

* •

    RQ3: Does our proposed framework effectively tackle the issue of noisy data through cross-view semantic alignment?

* •

    RQ4: What is the potential of our model as a pre-training framework for enhancing the performance of recommender systems?

* •

    RQ5: How does our RLMRec perform w.r.ttraining efficiency?

### 4.1. Experimental Settings

#### 4.1.1. Datasets

We conduct evaluations of our RLMRec on three public datasets: Amazon-book: This dataset contains user ratings and corresponding reviews for books sold on Amazon. Yelp: This dataset is a user-business dataset that provides extensive textual category information about various businesses. Steam: This dataset consists of textual feedback given by users for electronic games available on the Steam platform. Following the similar settings in*(Wang et al., [2019](#bib.bib36 ""); Xia et al., [2022](#bib.bib43 ""); Yu et al., [2022](#bib.bib45 ""))* for data preprocessing, we filter out interactions with ratings below 3 for both the Amazon-book and Yelp data. No filtering is applied to the Steam dataset due to the absence of rating scores. We then perform k-core filtering and divided each dataset into training, validation, and testing sets using a 3:1:1 ratio. Please refer to Table [5](#A1.T5 "Table 5 ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation") in Appendix for a summary of the dataset statistics.

*Table 2. Comparison with LLMs-enhanced Approaches.*

| Data | | Amazon-book | | Yelp | |
| --- | --- | --- | --- | --- | --- |
| Backb. | Variants | R@20 | N@20 | R@20 | N@20 |
| Light-GCN | Base | 0.1411 | 0.0856 | 0.1157 | 0.0733 |
| | KAR | $0.1416^{+0.3\%}$ | $0.0863^{{+0.8\%}}$ | $0.1194^{{+3.2\%}}$ | $0.0756^{{+3.1\%}}$ |
| RLMRec-Con | $0.1483^{{+\underline{\textbf{5.1\%}}}}$ | $0.0903^{{+\underline{\textbf{5.5\%}}}}$ | $0.1230^{{+\underline{\textbf{6.3\%}}}}$ | $0.0776^{{+\underline{\textbf{5.9\%}}}}$ |
| RLMRec-Gen | $0.1446^{{+2.5\%}}$ | $0.0887^{{+3.6\%}}$ | $0.1209^{{+4.5\%}}$ | $0.0761^{{+3.8\%}}$ |
| SGL | Base | 0.1473 | 0.0913 | 0.1197 | 0.0753 |
| | KAR | $0.1372^{{-6.9\%}}$ | $0.0875^{{-4.2\%}}$ | $0.1208^{{+0.9\%}}$ | $0.0761^{{+1.1\%}}$ |
| RLMRec-Con | $0.1528^{{+3.7\%}}$ | $0.0945^{{+3.5\%}}$ | $0.1248^{{+4.3\%}}$ | $0.0790^{{+4.9\%}}$ |
| RLMRec-Gen | $0.1537^{{+\underline{\textbf{4.3\%}}}}$ | $0.0947^{{+\underline{\textbf{3.7\%}}}}$ | $0.1263^{{+\underline{\textbf{5.5\%}}}}$ | $0.0798^{{+\underline{\textbf{6.0\%}}}}$ |

#### 4.1.2. Evaluation Protocols and Metrics

To ensure comprehensive evaluation and mitigate bias, we adopt the all-rank protocol*(He et al., [2020](#bib.bib12 ""); Wu et al., [2021](#bib.bib38 ""); Wang et al., [2020](#bib.bib37 ""))* across all items to accurately assess our recommendations. We use two widely adopted ranking-based metrics: Recall@N and NDCG@N, which measure the model effectiveness.

#### 4.1.3. Base Models

We evaluate the effectiveness of our RLMRec by integrating it with state-of-the-art representation-based recommenders based on SSLRec*(Ren et al., [2023a](#bib.bib26 ""))*. This approach allows us to assess its model-agnostic performance gain in comparison to base models.

* •

    GCCF *(Chen et al., [2020](#bib.bib7 ""))*: It simplifies graph-based recommender design by re-evaluating the role of non-linear operations in GNNs.

* •

    LightGCN *(He et al., [2020](#bib.bib12 ""))*: It creates a lightweight recommender by streamlining redundant neural modules in graph message passing.

* •

    SGL *(Wu et al., [2021](#bib.bib38 ""))*: It utilizes node/edge dropout as a data augmentator to generate diverse perspectives for contrastive learning.

* •

    SimGCL *(Yu et al., [2022](#bib.bib45 ""))*: It enhances recommendation performance by introducing an augmentation-free view generation technique.

* •

    DCCF *(Ren et al., [2023b](#bib.bib27 ""))*: It captures intent-wise relationships for recommendation purposes using disentangled contrastive learning.

* •

    AutoCF *(Xia et al., [2023](#bib.bib42 ""))*: It is a self-supervised masked autoencoder to automate the process of data augmentation for recommendation.

#### 4.1.4. Implementation Details

To ensure a fair comparison, all baseline models are evaluated using the same evaluation protocol. The dimension of representations (i.e., x and e) is set to 32 for all base models. We determine the hyperparameters specific to each model through grid search under various settings. To generate user and item profiles, we leverage the ChatGPT model (specifically, gpt-3.5-turbo) provided by OpenAI. We use the text-embedding-ada-002*(Neelakantan et al., [2022](#bib.bib23 ""))* to generate semantic representations s. During training, all methods are trained with a fixed batch size of 4096 and a learning rate of 1e-3 using the Adam optimizer. We adopt the early stop technique based on the model’s performance on the validation set. For detailed implementation of our RLMRec, we provide the source code for reference.

### 4.2. Performance Comparison (RQ1)

Model-agnostic Performance Gain. To demonstrate the effectiveness of RLMRec in improving recommendation performance, we integrate it into six state-of-the-art collaborative filtering models. We conduct experiments using 5 random initializations and report the average results in Table[1](#S4.T1 "Table 1 ‣ 4. Evaluation ‣ Representation Learning with Large Language Models for Recommendation"). The evaluation results reveal several interesting observations, as outlined below:

* •

    Overall, we consistently observe that integrating RLMRec with the backbone recommenders leads to improved performance compared to the original versions. This provides compelling evidence for the effectiveness of RLMRec. We attribute these improvements to two key factors: i) RLMRec enables the accurate user/item profiling paradigm empowered by LLMs, enhancing the representation of rich semantic information derived from user interaction behaviors. ii) Through our cross-view mutual information maximization, the CF-side relational embeddings and LLM-side semantic representations work cooperatively to enhance each other. This collaborative effort effectively filters out irrelevant noise in the recommendation features.

* •

    It is clear that both contrastive and generative modeling approaches generally improve performance. However, it is important to note that the contrastive approach exhibits superior performance when combined with various backbones like GCCF and SimGCL. Conversely, when applied to AutoCF, which involves masked reconstruction, RLMRec-Gen shows more significant improvements. We speculate that the mask operation functions as a form of regularization, leading to better results when used in conjunction with methods that employ a generative approach.

Superiority over LLM-enhanced Approach. In addition, we conduct a comparative evaluation of the effectiveness of RLMRec in comparison to KAR*(Xi et al., [2023](#bib.bib41 ""))*, a recent LLM-enhanced user behavior modeling approach. KAR aims to generate textual user/item descriptions to enhance the learning of user preferences for the CTR task. To ensure a fair comparison, we utilized the same semantic representation as in our approach and employed two classic methods (LightGCN and SGL) as the backbone models. This could be attributed to the fact that, while KAR incorporates textual information into the learning of user preferences, it treats the semantic representation as input features for the model. As a result, it may not effectively align the textual knowledge with the user behavior representations and could be more susceptible to irrelevant noise from either user behaviors or the LLM knowledge base.

<img src='x4.png' alt='Refer to caption' title='' width='461' height='217' />

*Figure 4. Ablation study on variant text embedding models conducted on the Amazon-book dataset. Shuffling involves reordering user/item embeddings.*

### 4.3. Ablation Study (RQ2)

In this section, we examine the impact of integrating semantic representations on performance. To do this, we shuffle the acquired semantic representations, creating a misalignment with collaborative relational representation and LLM’s knowledge. We use the default semantic encoding model, text-embedding-ada-002*(Brown et al., [2020](#bib.bib3 ""))*, and also experiment with advanced models like Contriever*(Izacard et al., [2021](#bib.bib15 ""))* and Instructor*(Su et al., [2023](#bib.bib30 ""))*. We evaluate our approach on four backbone methods (i.e., LightGCN, GCCF, SimGCL, and DCCF). The results are summarized in Figure[4](#S4.F4 "Figure 4 ‣ 4.2. Performance Comparison (RQ1) ‣ 4. Evaluation ‣ Representation Learning with Large Language Models for Recommendation"), leading to two key observations.

* •

    After randomly rearranging the semantic representations to disrupt the correlation between collaborative and semantic signals, we observe a decrease in performance for both RLMRec-Con and RLMRec-Gen on the evaluated backbone models compared to their original performance. This suggests that the shuffled representations introduce noise due to the mismatch between semantic and collaborative information. It provides evidence that accurate alignment between the semantic knowledge of the LLM and collaborative relationships among users is crucial for enhancing recommendation performance.

* •

    When we utilize variant text embedding models like Contriever and Instructor, our RLMRec still enhances the base performance, similar to the default setting with text-embedding-ada-002. This indicates that our RLMRec can effectively leverage an appropriate text encoder capable of transferring textual semantics into preference representations to improve the performance of the recommender backbone. Moreover, the ability of text embedding models to capture semantic information with higher accuracy can lead to even more significant improvements.

### 4.4. In-depth Analysis of RLMRec (RQ3 – RQ5)

<img src='x5.png' alt='Refer to caption' title='' width='438' height='137' />

*Figure 5. Comparing performance on different noise ratios in the Amazon-book dataset with LightGCN as the base model.*

#### 4.4.1. Performance w.r.t. Noisy Data (RQ3)

We assess the robustness of RLMRec to data noise by adding non-existent interactions to the original training data. Noise levels range from 5% to 25% relative to the training set size. Using the Amazon dataset, we compare the performance of vanilla LightGCN with LightGCN enhanced by our RLMRec-Con/Gen. Key findings from Fig[5](#S4.F5 "Figure 5 ‣ 4.4. In-depth Analysis of RLMRec (RQ3 – RQ5) ‣ 4. Evaluation ‣ Representation Learning with Large Language Models for Recommendation") are:

* •

    (i) Both RLMRec-Con and RLMRec-Gen consistently outperform the LightGCN backbone model at all noise levels. This highlights the advantages of incorporating semantic information and leveraging mutual information to filter out irrelevant data, resulting in improved recommendations and robustness over noise.

* •

    (ii) RLMRec-Con has shown better resistance to data noise compared to RLMRec-Gen. This is likely due to the inherent noise introduced by the generative method through node masking. In contrast, contrastive methods encounter less noise, leading to superior performance under the same noise ratio.

*Table 3. Performance comparison with different initialized parameters from various pre-training methods on the Yelp.*

| Metric | Recall | | | NDCG | | |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained Params | @5 | @10 | @20 | @5 | @10 | @20 |
| None | 0.0274 | 0.0462 | 0.0820 | 0.0203 | 0.0270 | 0.0375 |
| Base | 0.0304 | 0.0557 | 0.0971 | 0.0229 | 0.0319 | 0.0439 |
| RLMRec-Con | 0.0359 | 0.0613 | 0.1034 | 0.0261 | 0.0352 | 0.0475 |
| RLMRec-Gen | 0.0362 | 0.0612 | 0.1068 | 0.0263 | 0.0353 | 0.0484 |

#### 4.4.2. Performance in Pre-training Scenarios (RQ4)

We investigate the potential of our semantically involved training mechanism as a pre-training technique to provide informative representations as initial embeddings for downstream models. Using the Yelp dataset, we utilize data from 2012 to 2017 for pre-training and divide the data from 2018 to 2019 into a training set, a validation set, and a test set (the downstream dataset). Both datasets contain the same users and items. We train vanilla LightGCN and our model on the pre-training dataset. The learned parameters are used to initialize the embeddings for vanilla LightGCN, which is then trained on the downstream dataset. Key findings from Table[3](#S4.T3 "Table 3 ‣ 4.4.1. Performance w.r.t. Noisy Data (RQ3) ‣ 4.4. In-depth Analysis of RLMRec (RQ3 – RQ5) ‣ 4. Evaluation ‣ Representation Learning with Large Language Models for Recommendation") are:

* •

    Pre-training with parameters yields superior results compared to no pre-training, regardless of whether it was done with the base model or our RLMRec. This suggests that the pre-training dataset contains valuable collaborative information that helps predict user/item preferences and benefits downstream tasks.

* •

    Both RLMRec-Con and RLMRec-Gen provide better pre-training benefits compared to pre-training with the base model alone, with RLMRec-Gen achieving the best results. This highlights the advantage of incorporating semantic information and the effectiveness of generative methods in pre-training scenarios, potentially due to the regulatory function of the mask operation, preventing overfitting on the pre-training dataset.

*Table 4. RLMRec’s efficiency with various recommenders.*

| Ama-Variants | GCCF | LightGCN | SGL | SimGCL | DCCF | AutoCF |
| --- | --- | --- | --- | --- | --- | --- |
| Base | 0.88s | 1.01s | 2.18s | 2.62s | 2.26s | 2.73s |
| RLMRec-Con | 1.95s | 1.94s | 2.58s | 3.02s | 2.49s | 2.96s |
| RLMRec-Gen | 1.72s | 1.76s | 2.36s | 2.69s | 2.29s | 2.96s |
| Yelp-Variants | GCCF | LightGCN | SGL | SimGCL | DCCF | AutoCF |
| Base | 1.11s | 1.26s | 2.80s | 3.35s | 3.02s | 3.96s |
| RLMRec-Con | 2.39s | 2.57s | 3.27s | 3.95s | 3.42s | 4.41s |
| RLMRec-Gen | 2.03s | 2.12s | 3.20s | 3.50s | 3.24s | 4.39s |
| Steam-Variants | GCCF | LightGCN | SGL | SimGCL | DCCF | AutoCF |
| Base | 2.05s | 2.27s | 5.42s | 6.47s | 9.31s | 8.44s |
| RLMRec-Con | 4.32s | 4.67s | 6.77s | 7.88s | 10.18s | 10.06s |
| RLMRec-Gen | 3.33s | 3.81s | 6.10s | 6.89s | 9.57s | 9.89s |

#### 4.4.3. Analysis of Training Efficiency (RQ5)

We analyze the time complexity of using RLMRec. The theoretical time complexity of the multi-layer perception ($\sigma_{\uparrow}$ and $\sigma_{\downarrow}$) for both RLMRec-Con and RLMRec-Gen is $\mathcal{O}(N\times d_{s}\times d_{e})$. For RLMRec-Con, the loss computation introduces an additional complexity of $O(N^{2}\times d)$. For RLMRec-Gen, the time complexity is $O(M\times d+M\times N\times d)$, where the masking operation accounts for $M\times d$, with $M$ representing the number of masked nodes. In Table[4](#S4.T4 "Table 4 ‣ 4.4.2. Performance in Pre-training Scenarios (RQ4) ‣ 4.4. In-depth Analysis of RLMRec (RQ3 – RQ5) ‣ 4. Evaluation ‣ Representation Learning with Large Language Models for Recommendation"), we present the epoch time of training on a server with an Intel Xeon Silver 4314 CPU and an NVIDIA RTX 3090 GPU. The results show that the time cost of RLMRec-Gen is consistently lower than that of RLMRec-Con. This is primarily because the value of $N$ in RLMRec-Con is determined by the batch size, which tends to be larger than the number of masked nodes M in RLMRec-Gen. Additionally, for larger models with improved performance, the additional time complexity is only around 10% to 20% compared to the original time.

### 4.5. Case Study

In our investigation, we explore the integration of LLM-enhanced semantics to capture global user relationships that are not easily captured through direct message passing on the graph. Figure[6](#S4.F6 "Figure 6 ‣ 4.5. Case Study ‣ 4. Evaluation ‣ Representation Learning with Large Language Models for Recommendation") presents a case study where the distance between user $u_{1998}$ and $u_{227}$ exceeds 3 hops. To evaluate the models’ ability to capture their relationship, we examine the similarity of user representations. We compared LightGCN and RLMRec-Con, both using the same backbone. Two metrics were introduced: a relevance score for user $u_{1998}$ and the ranking of its long-distance neighbors (greater than 3 hops) based on the score. By incorporating semantic information, derived from language models that highlight shared interests between $u_{1998}$ and $u_{227}$ (e.g., their preference for friendly service), both the relevance score and ranking increased. This indicates that the learned representations from RLMRec effectively capture global collaborative relationships beyond ID-based recommendation techniques.

<img src='x6.png' alt='Refer to caption' title='' width='415' height='231' />

*Figure 6. Case study on capturing global user dependencies.*

5. Conclusion
--------------

This paper presents RLMRec, a model-agnostic framework that leverages Large Language Models (LLMs) to improve the representation performance of recommender systems. We introduce a collaborative profile generation paradigm and a reasoning-driven system prompt, emphasizing the inclusion of reasoning processes in the generated output. RLMRec utilizes contrastive and generative alignment techniques to align CF-side relational embeddings with LLM-side semantic representations, effectively reducing feature noise. The framework combines the strengths of general recommenders and LLMs, supported by robust theoretical guarantees, and is extensively evaluated on real-world datasets. Our future investigations will focus on advancing LLM-based reasoning results in recommender systems by providing more insightful explanations.

References
----------

* (1)
* Bao et al. (2023)Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023.Tallrec: An effective and efficient tuning framework to align large language model with recommendation.*arXiv preprint arXiv:2305.00447* (2023).
* Brown et al. (2020)Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.Language models are few-shot learners. In *NeurIPS*, Vol. 33. 1877–1901.
* Cai et al. (2023)Xuheng Cai, Chao Huang, Lianghao Xia, and Xubin Ren. 2023.LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation. In *ICLR*.
* Chang et al. (2021)Jianxin Chang, Chen Gao, Yu Zheng, Yiqun Hui, Yanan Niu, Yang Song, Depeng Jin, and Yong Li. 2021.Sequential recommendation with graph neural networks. In *SIGIR*. 378–387.
* Chen et al. (2023)Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2023.Bias and debias in recommender system: A survey and future directions.*ACM Transactions on Information Systems (TOIS)* 41, 3 (2023), 1–39.
* Chen et al. (2020)Lei Chen, Le Wu, Richang Hong, Kun Zhang, and Meng Wang. 2020.Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach. In *AAAI*, Vol. 34. 27–34.
* Fan et al. (2023)Wenqi Fan, Zihuai Zhao, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Jiliang Tang, and Qing Li. 2023.Recommender systems in the era of large language models (llms).*arXiv preprint arXiv:2307.02046* (2023).
* Gao et al. (2023)Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023.Chat-rec: Towards interactive and explainable llms-augmented recommender system.*arXiv preprint arXiv:2303.14524* (2023).
* Geng et al. (2022)Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022.Recommendation as language processing (rlp): A unified pretrain, personalized prompt \& predict paradigm (p5). In *RecSys*. 299–315.
* He et al. (2022)Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022.Masked autoencoders are scalable vision learners. In *CVPR*. 16000–16009.
* He et al. (2020)Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020.Lightgcn: Simplifying and powering graph convolution network for recommendation. In *SIGIR*. 639–648.
* Hou et al. (2023)Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2023.Large language models are zero-shot rankers for recommender systems.*arXiv preprint arXiv:2305.08845* (2023).
* Hou et al. (2022)Zhenyu Hou, Xiao Liu, Yuxiao Dong, Chunjie Wang, Jie Tang, et al. 2022.Graphmae: Self-supervised masked graph autoencoders.*arXiv preprint arXiv:2205.10803* (2022).
* Izacard et al. (2021)Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021.Unsupervised dense information retrieval with contrastive learning.*arXiv preprint arXiv:2112.09118* (2021).
* Khosla et al. (2020)Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020.Supervised contrastive learning.*NeurIPS* (2020).
* Koren et al. (2021)Yehuda Koren, Steffen Rendle, and Robert Bell. 2021.Advances in collaborative filtering.*Recommender systems handbook* (2021), 91–142.
* Li et al. (2023)Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. 2023.Text Is All You Need: Learning Language Representations for Sequential Recommendation.*arXiv preprint arXiv:2305.13731* (2023).
* Lin et al. (2023)Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, et al. 2023.How Can Recommender Systems Benefit from Large Language Models: A Survey.*arXiv preprint arXiv:2306.05817* (2023).
* Lin et al. (2022)Zihan Lin, Changxin Tian, Yupeng Hou, and Wayne Xin Zhao. 2022.Improving graph collaborative filtering with neighborhood-enriched contrastive learning. In *WWW*. 2320–2329.
* Liu et al. (2023a)Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023a.Is chatgpt a good recommender? a preliminary study.*arXiv preprint arXiv:2304.10149* (2023).
* Liu et al. (2023b)Peng Liu, Lemei Zhang, and Jon Atle Gulla. 2023b.Pre-train, prompt and recommendation: A comprehensive survey of language modelling paradigm adaptations in recommender systems.*arXiv preprint arXiv:2302.03735* (2023).
* Neelakantan et al. (2022)Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022.Text and code embeddings by contrastive pre-training.*arXiv preprint arXiv:2201.10005* (2022).
* Oord et al. (2018)Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.Representation learning with contrastive predictive coding.*arXiv preprint arXiv:1807.03748* (2018).
* OpenAI (2023)OpenAI. 2023.GPT-4 Technical Report.*arXiv preprint arXiv:2303.08774* (2023).
* Ren et al. (2023a)Xubin Ren, Lianghao Xia, Yuhao Yang, Wei Wei, Tianle Wang, Xuheng Cai, and Chao Huang. 2023a.SSLRec: A Self-Supervised Learning Framework for Recommendation.*arXiv preprint arXiv:2308.05697* (2023).
* Ren et al. (2023b)Xubin Ren, Lianghao Xia, Jiashu Zhao, Dawei Yin, and Chao Huang. 2023b.Disentangled Contrastive Collaborative Filtering. In *SIGIR*. 1137–1146.
* Rendle et al. (2012)Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2012.BPR: Bayesian personalized ranking from implicit feedback.*arXiv preprint arXiv:1205.2618* (2012).
* Sasaki and Takenouchi (2022)Hiroaki Sasaki and Takashi Takenouchi. 2022.Representation learning for maximization of MI, nonlinear ICA and nonlinear subspaces with robust density ratio estimation.*JMLR* (2022).
* Su et al. (2023)Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2023.One Embedder, Any Task: Instruction-Finetuned Text Embeddings. In *Findings of the ACL*. 1102–1121.
* Su and Khoshgoftaar (2009)Xiaoyuan Su and Taghi M Khoshgoftaar. 2009.A survey of collaborative filtering techniques.*Advances in artificial intelligence* 2009 (2009).
* Sun et al. (2023)Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. 2023.Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent.*arXiv preprint arXiv:2304.09542* (2023).
* Touvron et al. (2023)Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023.Llama: Open and efficient foundation language models.*arXiv preprint arXiv:2302.13971* (2023).
* Wang et al. (2021a)Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. 2021a.Denoising implicit feedback for recommendation. In *WSDM*. 373–381.
* Wang et al. (2021b)Wenjie Wang, Fuli Feng, Xiangnan He, Xiang Wang, and Tat-Seng Chua. 2021b.Deconfounded recommendation for alleviating bias amplification. In *KDD*. 1717–1725.
* Wang et al. (2019)Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019.Neural graph collaborative filtering. In *SIGIR*. 165–174.
* Wang et al. (2020)Yifan Wang, Suyao Tang, Yuntong Lei, Weiping Song, Sheng Wang, and Ming Zhang. 2020.Disenhan: Disentangled heterogeneous graph attention network for recommendation. In *CIKM*. 1605–1614.
* Wu et al. (2021)Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021.Self-supervised graph learning for recommendation. In *SIGIR*. 726–735.
* Wu et al. (2023)Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, et al. 2023.A Survey on Large Language Models for Recommendation.*arXiv preprint arXiv:2305.19860* (2023).
* Wu et al. (2022)Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2022.Graph neural networks in recommender systems: a survey.*ACM Computing Surveys (CSUR)* 55, 5 (2022), 1–37.
* Xi et al. (2023)Yunjia Xi, Weiwen Liu, Jianghao Lin, Jieming Zhu, Bo Chen, Ruiming Tang, Weinan Zhang, Rui Zhang, and Yong Yu. 2023.Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models.*arXiv preprint arXiv:2306.10933* (2023).
* Xia et al. (2023)Lianghao Xia, Chao Huang, Chunzhen Huang, Kangyi Lin, Tao Yu, and Ben Kao. 2023.Automated Self-Supervised Learning for Recommendation. In *WWW*.
* Xia et al. (2022)Lianghao Xia, Chao Huang, Yong Xu, Jiashu Zhao, Dawei Yin, and Jimmy Huang. 2022.Hypergraph contrastive collaborative filtering. In *SIGIR*. 70–79.
* Yang et al. (2023)Yuhao Yang, Chao Huang, Lianghao Xia, Chunzhen Huang, Da Luo, and Kangyi Lin. 2023.Debiased Contrastive Learning for Sequential Recommendation. In *WWW*. 1063–1073.
* Yu et al. (2022)Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. 2022.Are graph augmentations necessary? simple graph contrastive learning for recommendation. In *SIGIR*. 1294–1303.
* Yu et al. (2023)Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Jundong Li, and Zi Huang. 2023.Self-supervised learning for recommender systems: A survey.*TKDE* (2023).
* Yuan et al. (2023)Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023.Where to go next for recommender systems? id-vs. modality-based recommender models revisited. In *SIGIR*. 2639–2649.
* Zhang et al. (2023)Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023.Recommendation as instruction following: A large language model empowered recommendation approach.*arXiv preprint arXiv:2305.07001* (2023).

Appendix A Supplementary Material
---------------------------------

In the supplementary materials, we provide the training procedure of our proposed framework, RLMRec, through pseudocode. We also offer detailed insights into the design of prompts, accompanied by examples, to visualize the profile generation process within our item-to-user generation paradigm. Finally, we present experiment details for the reranking task mentioned in Section[1](#S1 "1. Introduction ‣ Representation Learning with Large Language Models for Recommendation"), where we analyze specific examples within the task.

*Table 5. Statistics of the experimental datasets.*

| Dataset | #Users | #Items | #Interactions | Density |
| --- | --- | --- | --- | --- |
| Amazon-book | 11,000 | 9,332 | 120,464 | 1.2$e^{-3}$ |
| Yelp | 11,091 | 11,010 | 166,620 | 1.4$e^{-3}$ |
| Steam | 23,310 | 5,237 | 316,190 | 2.6$e^{-3}$ |

input :Base model $\mathcal{R}$, implicit feedback $\mathcal{X}$, semantic representation s for each user \& item and learning rate $\eta$

Result: Trained model parameters $\mathbf{\Theta}$

1repeat

2 uniformly sample batch data $\mathcal{B}\={(u,v_{pos},v_{neg})}\in\mathcal{X}$;

3 inference collaborative-side representation $\textbf{e}_{u/v}$ with $\mathcal{R}$;

4 calculate model optimization objective $\mathcal{L}_{\mathcal{R}}$ based on $\mathcal{B}$;

5 calculate $L_{info}$ w.r.t. Eq ([16](#S3.E16 "In 3.3.1. Contrastive Alignment ‣ 3.3. Density Ratio Modeling for Mutual Information Maximization ‣ 3. Methodology ‣ Representation Learning with Large Language Models for Recommendation") \& [18](#S3.E18 "In 3.4. Model-agnostic Learning ‣ 3. Methodology ‣ Representation Learning with Large Language Models for Recommendation")) for all $u/v$ in $\mathcal{B}$;

6 $\mathcal{L}\=\mathcal{L}_{\mathcal{R}}+\mathcal{L}_{info}$;

7 $\mathbf{\Theta}\leftarrow\mathbf{\Theta}-\eta\nabla_{\mathbf{\Theta}}\mathcal{L}$;

8

9until *convergence*;

*Algorithm 1 Training Procedure in RLMRec-Con*

input :Base model $\mathcal{R}$, implicit feedback $\mathcal{X}$, semantic representation s for each user \& item, learning rate $\eta$ and masking ratio $\alpha$

Result: Trained model parameters $\mathbf{\Theta}$

1repeat

2 uniformly sample batch data $\mathcal{B}\={(u,v_{pos},v_{neg})}\in\mathcal{X}$;

3 randomly sample a subset of users \& items with ratio $\alpha$;

4 replace initial embeddings of masked $u/v$ with $[MASK]$;

5 inference collaborative-side representation $\textbf{e}_{u/v}$ with $\mathcal{R}$;

6 calculate model optimization objective $\mathcal{L}_{\mathcal{R}}$ based on $\mathcal{B}$;

7 calculate $L_{info}$ w.r.t. Eq ([17](#S3.E17 "In 3.3.2. Generative Alignment ‣ 3.3. Density Ratio Modeling for Mutual Information Maximization ‣ 3. Methodology ‣ Representation Learning with Large Language Models for Recommendation") \& [18](#S3.E18 "In 3.4. Model-agnostic Learning ‣ 3. Methodology ‣ Representation Learning with Large Language Models for Recommendation")) for masked $u/v$ in $\mathcal{B}$;

8 $\mathcal{L}\=\mathcal{L}_{\mathcal{R}}+\mathcal{L}_{info}$;

9 $\mathbf{\Theta}\leftarrow\mathbf{\Theta}-\eta\nabla_{\mathbf{\Theta}}\mathcal{L}$;

10

11until *convergence*;

*Algorithm 2 Training Procedure in RLMRec-Gen*

### A.1. Pseudocode of RLMRec

This section introduces the pseudocode for our model-agnostic RLMRec framework implementations, namely RLMRec-Con and RLMRec-Gen. The focus is on the training procedure of these implementations. Prior to training, user and item profiles are preprocessed, and their semantic embeddings s are generated using text models. Algorithm[1](#alg1 "In Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation") presents the training procedure for RLMRec-Con, while Algorithm[2](#alg2 "In Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation") outlines the process for RLMRec-Gen.

The difference between RLMRec-Con and RLMRec-Gen is that RLMRec-Gen randomly masks a portion of users/items before the base model encodes the CF-side relational representations. The objective function $\mathcal{L}_{info}$ for mutual information maximization is then computed based on the representations of the masked users and items. In contrast, RLMRec-Con models the density ratio in a contrastive manner and calculates the $\mathcal{L}_{info}$ objective for all users and items in the batch data, including both positives and negatives.

### A.2. Details of Profile Generation

In this section, we offer a comprehensive explanation of the generation process for both user and item profiles. Real examples from the Amazon-book dataset are used to illustrate this process, as depicted in Figure[7](#A1.F7 "Figure 7 ‣ A.2.2. Example of the Generated User Profile ‣ A.2. Details of Profile Generation ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation") and Figure[8](#A1.F8 "Figure 8 ‣ A.2.2. Example of the Generated User Profile ‣ A.2. Details of Profile Generation ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation"). We adopt a general interaction paradigm with large language models (LLMs), where the system prompt serves as an instruction to guide the profile generation task. While the Amazon-book dataset is specifically showcased, the overall generation process remains consistent for the Yelp and Steam datasets as well, with minor differences in the instructions provided to represent the data information.

#### A.2.1. Example of the Generated Item Profile

Figure[7](#A1.F7 "Figure 7 ‣ A.2.2. Example of the Generated User Profile ‣ A.2. Details of Profile Generation ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation") showcases an example of item profile generation specifically for the Amazon-book dataset. The instruction prompt provided to the language models for all items remains consistent, directing them to summarize the types of books that would appeal to users, thus offering valuable information for recommendation purposes. The input information consists of the book’s title and original description from the dataset. To maintain consistency and facilitate parsing, we enforce the requirement that the output of the language models adhere to the JSON format. Furthermore, it is essential for the language models to provide their reasoning behind the generated profile, ensuring high-quality summarization while preventing any potential hallucinations. The generated results demonstrate that the language model, in this case ChatGPT, accurately captures from the book description that the book is likely to attract readers interested in mental health and women’s experiences.

#### A.2.2. Example of the Generated User Profile

Figure[8](#A1.F8 "Figure 8 ‣ A.2.2. Example of the Generated User Profile ‣ A.2. Details of Profile Generation ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation") illustrates the process of generating user profiles using the Amazon-book dataset as an example. Our approach adopts an item-to-user generation paradigm, which allows us to leverage the previously generated profiles that describe the interaction preferences of items. To accomplish this, our prompt methodology incorporates not only users’ feedback information on items but also the profiles of the items themselves. By utilizing both sources of information comprehensively, large language models are empowered to capture users’ true preferences with enhanced accuracy. In the presented example, leveraging both the book descriptions and users’ review text, the language models distill the user’s preference for young adult fiction that seamlessly combines paranormal or supernatural elements.

<img src='x7.png' alt='Refer to caption' title='' width='415' height='215' />

*Figure 7. Case study on item profile generation in Amazon-book data.*

<img src='x8.png' alt='Refer to caption' title='' width='415' height='323' />

*Figure 8. Case study on user profile generation in Amazon-book data.*

<img src='x9.png' alt='Refer to caption' title='' width='415' height='358' />

*Figure 9. Case study on LLMs-based reranking. The candidate items are retrieved by LightGCN.*

### A.3. Analysis on the Reranking Task with LLMs

In this section, we explore real-world scenarios where LLMs are employed for reranking tasks on the Amazon-book dataset, as introduced in Section[1](#S1 "1. Introduction ‣ Representation Learning with Large Language Models for Recommendation"). Firstly, as depicted in Figure[9](#A1.F9 "Figure 9 ‣ A.2.2. Example of the Generated User Profile ‣ A.2. Details of Profile Generation ‣ Appendix A Supplementary Material ‣ Representation Learning with Large Language Models for Recommendation"), we carefully design the instruction and input prompt. We utilize the item ID as a key to index the items, along with their textual information such as the book title. This approach aims to enhance the accuracy of the generated outputs by avoiding direct generation of item titles that may not precisely match the original titles.

In the prompt, we include the historical purchased books of the user as a reference, along with a list of 40 candidate items retrieved by LightGCN. The goal is for the LLM to select the top 20 items for recommendation. However, upon examining the output results, it becomes apparent that the language models have recommended some non-existent items (highlighted in green) within the provided list. Despite utilizing item IDs for indexing, this occurrence is common in many reranking examples, and the presence of non-existent items can undoubtedly impact the overall reranking performance.

Additionally, the number of correctly recommended items from the language models is lower than the retrieved items (highlighted in red). This discrepancy is primarily attributed to the limited textual information available for the language models to effectively exploit users’ preferences. Moreover, the retrieved item list, learned by the state-of-the-art method LightGCN, benefits from collaborative information beyond just the textual content. This collaborative information contributes to the improved performance of the retrieval process compared to the language models’ recommendations.

Incorporating other raw textual information from the datasets to improve performance may have some anticipated limitations: i) The limitation of input token numbers may constrain the size of candidate items, as many raw descriptions can be excessively lengthy. ii) Raw descriptions may be missing or contain substantial noise in certain datasets. The absence of descriptions or the presence of noisy information can hinder the language models’ comprehension of users’ preferences. iii) Including a larger amount of input data, such as additional raw textual information, can increase the computational cost, which impacts the system’s scalability.
