Disentangled Contrastive Collaborative Filtering
================================================

Xubin RenUniversity of Hong KongHong KongChina[xubinrencs@gmail.com](mailto:xubinrencs@gmail.com),Lianghao XiaUniversity of Hong KongHong KongChina[aka_xia@foxmail.com](mailto:aka_xia@foxmail.com),Jiashu ZhaoWilfrid Laurier UniversityWaterlooCanada[jzhao@wlu.ca](mailto:jzhao@wlu.ca),Dawei YinBaidu IncBeijingChina[yindawei@acm.org](mailto:yindawei@acm.org)andChao HuangUniversity of Hong KongHong KongChina[chaohuang75@gmail.com](mailto:chaohuang75@gmail.com)

(2023)

###### Abstract.

Recent studies show that graph neural networks (GNNs) are prevalent to model high-order relationships for collaborative filtering (CF). Towards this research line, graph contrastive learning (GCL) has exhibited powerful performance in addressing the supervision label shortage issue by learning augmented user and item representations. While many of them show their effectiveness, two key questions still remain unexplored: i) Most existing GCL-based CF models are still limited by ignoring the fact that user-item interaction behaviors are often driven by diverse latent intent factors (e.g., shopping for family party, preferred color or brand of products); ii) Their introduced non-adaptive augmentation techniques are vulnerable to noisy information, which raises concerns about the model’s robustness and the risk of incorporating misleading self-supervised signals. In light of these limitations, we propose a Disentangled Contrastive Collaborative Filtering framework (DCCF) to realize intent disentanglement with self-supervised augmentation in an adaptive fashion. With the learned disentangled representations with global context, our DCCF is able to not only distill finer-grained latent factors from the entangled self-supervision signals but also alleviate the augmentation-induced noise. Finally, the cross-view contrastive learning task is introduced to enable adaptive augmentation with our parameterized interaction mask generator. Experiments on various public datasets demonstrate the superiority of our method compared to existing solutions. Our model implementation is released at the link <https://github.com/HKUDS/DCCF>.

Collaborative Filtering, Contrastive Learning, Disentangled Representation, Graph Neural Networks, Recommendation

††journalyear: 2023††copyright: acmlicensed††conference: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval; July 23–27, 2023; Taipei, Taiwan††booktitle: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR’23), July 23–27, 2023, Taipei, Taiwan††price: 15.00††doi: 10.1145/3539618.3591665††isbn: 978-1-4503-9408-6/23/07††ccs: Information systems Recommender systems

1. Introduction
----------------

Recommender systems have become fundamental services for suggesting personalized items to users by learning their preference from historical interactions*(Wu et al., [2020](#bib.bib41 ""); Chang et al., [2021](#bib.bib4 ""))*. Graph neural networks have recently achieved remarkable success in collaborative filtering (CF) modeling user-item interaction with high-order connectivity, such as NGCF*(Wang et al., [2019b](#bib.bib33 ""))*,MCCF*(Wang et al., [2020d](#bib.bib36 ""))*, LightGCN*(He et al., [2020](#bib.bib14 ""))*, and GCCF*(Chen et al., [2020c](#bib.bib8 ""))*. Those GNN-based CF models encode user-item bipartite graph structures into representations via iterative message passing for collaborative information aggregation*(Wang et al., [2022](#bib.bib38 ""))*. By capturing the high-order user (item) similarity in latent embedding space, graph neural CF methods have provided state-of-the-art recommendation performance.

However, user-item interactions, which serve as important labels for supervised recommendation models, are often highly sparse in real-world recommender systems*(Zhou et al., [2020](#bib.bib50 ""); Xia et al., [2023](#bib.bib42 ""); Zhu and Chen, [2022](#bib.bib51 ""))*. To address the issue of supervision shortage in recommendations, recent works*(Wu et al., [2021](#bib.bib40 ""); Xia et al., [2022](#bib.bib43 ""))* attempt to marry the power of contrastive learning with GNNs to explore the unlabeled information and offer self-supervision signals. These graph contrastive learning (GCL) methods propose to learn invariant user (item) representations by maximizing agreement between established contrastive augmentation views. In general, by following the mutual information maximization principle*(Velickovic et al., [2019](#bib.bib29 ""); Peng et al., [2020](#bib.bib23 ""))*, the agreements of pre-defined positive pairs are achieved, and embeddings of negative pairs are pushed apart. Two key research lines of augmentation schemes have recently emerged in GCL-based collaborative filtering. To be specific, SGL*(Wu et al., [2021](#bib.bib40 ""))* generates contrastive views with stochastic augmentors, e.g., random node/edge dropout. To supplement the direct graph connections, HCCF*(Xia et al., [2022](#bib.bib43 ""))* and MHCN*(Yu et al., [2021](#bib.bib47 ""))* propose to purse the consistency between node-level representations and graph-level semantic embeddings.

Although promising results have been achieved, we argue that two key limitations exist in current GCL recommender systems.

*First*, most previous studies have ignored the fact that the latent factors behind user-item interactions are highly entangled due to preference diversity, resulting in suboptimal augmentation-induced user representations. In real-life applications, the formation of user-item interactions is driven by many intent factors*(Wang et al., [2020b](#bib.bib35 ""), [c](#bib.bib37 ""))*, such as purchasing products for a family party or being attracted to certain clothing characteristics. However, the learned user preferences with the encoded invariant representations in current GCL-based recommendation approaches are entangled, making it difficult to capture the finer-grained interaction patterns between users and items. This hinders the recommender’s ability to capture genuine user preferences and provide accurate intent-aware self-supervision. Therefore, there is an urgent need for a new method that can generate disentangled contrastive signals for informative augmentation.

*Second*, many existing GCL-based methods still struggle to provide accurate self-supervised learning (SSL) signals against data noise, which makes it difficult to adapt contrastive learning to user-item interaction graphs with diverse structures. Specifically, the introduced stochastic augmentation strategy*(Wu et al., [2021](#bib.bib40 ""))* may not preserve the original semantic relationships well, as they use random dropout operators. For example, For instance, dropping hub nodes can damage important inter-community connection structures, resulting in an augmented user-item relation graph that may not be positively related to the original interaction structures. Additionally, Additionally, although some methods incorporate graph-level semantics into auxiliary self-supervised signals*(Xia et al., [2022](#bib.bib43 ""); Yu et al., [2021](#bib.bib47 ""))* via self-discrimination over all nodes, their model performance is vulnerable to user interaction data noise, such as misclicks or popularity bias. Under a contrastive augmentation framework, if the importance of node- or edge-wise SSL signals is not differentiated, methods can be easily biased by supplementing the main recommendation task with self-supervised signals derived from noisy nodes, e.g., users with many misclick behaviors or high conformity to popularity bias*(Wang et al., [2021a](#bib.bib31 ""))*.

In this paper, we propose a new disentangled contrastive learning-based collaborative filtering model, called DCCF, to address the limitations of existing methods. Specifically, Our model encodes multi-intent representations by considering the global dependencies between users and items. We achieve this by designing intent-aware information passing and aggregation between patch-level nodes and global-level intent prototypes. We aim to identify important graph structural information that captures accurate and helpful environment-invariant patterns with intent disentanglement. In this way, we can prevent the distillation of self-supervised information with severe noisy signals. To achieve our goal, we create parameterized edge mask generators that capture implicit relationships among users and items, and we inject intent-aware global dependencies. As a result, the graph structure masker can naturally capture the importance of each interaction for contrastive augmentation, which is adaptive to the user-item relations.

To sum up, the main contributions of this work are as follows:

* •

    In this work, we study the generalization problem of GCL-based recommender systems in a more challenging yet practical scenario: adapting graph contrastive learning to intent disentanglement with self-supervision noise for collaborative filtering.

* •

    We develop a new recommendation model called DCCF, with parameterized mask generators that are adaptive to build over the global context-enhanced disentangled GNN architecture. This enhances recommender robustness and generalization ability.

* •

    Extensive experimental results demonstrate that our new method achieves superior recommendation performance compared to more than 10 existing solutions. Furthermore, the effectiveness of our disentangled adaptive augmentation is justified by studies of model ablation, robustness, and interpretability.

2. Related Work
----------------

GNNs-based Recommender Systems.
Graph neural networks (GNNs) have demonstrated strong performance in representation learning of user preference for recommendation. These GNN-based recommenders perform recursive message passing over graph structures to model high-order collaborative relations*(Wu et al., [2020](#bib.bib41 ""); Yang et al., [2021](#bib.bib45 ""); Chen et al., [2023](#bib.bib9 ""))*. Towards this line, Many efforts have been made to build recommender systems based on various graph neural techniques. For instance, graph convolutional networks have been widely adopted as encoders to model the user-item interaction graph, such as LightGCN, LR-GCCF*(Chen et al., [2020c](#bib.bib8 ""))*, and HGCF*(Sun et al., [2021](#bib.bib26 ""))*. Additionally, graph-enhanced attention mechanisms explicitly distinguish influence for embedding propagation among neighboring nodes, and serve as important components in various recommenders, including social relation learning DGRec*(Song et al., [2019](#bib.bib25 ""))*, multi-behavior recommendation*(Yang et al., [2022](#bib.bib44 ""))*, knowledge graph-based recommenders KGAT*(Wang et al., [2019a](#bib.bib32 ""))*, JNSKR*(Chen et al., [2020d](#bib.bib5 ""))*.

Recommendation with Disentangled Representations.
Learning disentangled representations of user latent intents from implicit feedback has been a popular topic in recent years. Various approaches have been proposed, such as using variational auto-encoders to encode high-level user intentions for improving recommendation*(Ma et al., [2019b](#bib.bib21 ""))*. DGCF*(Wang et al., [2020b](#bib.bib35 ""))* builds upon this idea of intent disentanglement, and performs disentangled representation learning over graph neural network with embedding splitting. To incorporate side information from user or item domain into recommendation, DisenHAN*(Wang et al., [2020c](#bib.bib37 ""))* attempts to learn disentangled user/item representations with heterogeneous graph attention network. KGIN*(Wang et al., [2021b](#bib.bib34 ""))* is a method that aims to encode latent user intents using item knowledge graph to improve recommendation performance. DCF*(Chen et al., [2021b](#bib.bib11 ""))* decomposes users and items into factor-level representations and using a factor-level attention mechanism to capture the underlying intents. In CDR*(Chen et al., [2021a](#bib.bib7 ""))*, a dynamic routing mechanism is designed to characterize correlations among user intentions for embedding denoising. However, most existing disentangled recommender systems are built in a fully supervised manner, which can be limited by the sparsity of user-item interactions in real-world scenarios. To address this challenge, we propose a new model that leverages self-supervised learning for intent-aware augmentation.

Contrastive Learning in Recommendation.
Recently, contrastive learning (CL) has gained considerable attention in various recommendation scenarios, such as sequential recommendation*(Chen et al., [2022](#bib.bib10 ""))*, knowledge graph-enhanced recommendation*(Zou et al., [2022](#bib.bib52 ""))*, multi-interest recommendation*(Zhang et al., [2022](#bib.bib48 ""))* and multi-behavior recommendation*(Wei et al., [2022](#bib.bib39 ""))*. The most relevant research line in recommendation systems is to enhance graph neural network (GNN)-based collaborative filtering with contrastive learning. To this end, several recently proposed models, such as SGL*(Wu et al., [2021](#bib.bib40 ""))*, NCL*(Lin et al., [2022](#bib.bib18 ""))*, and HCCF*(Xia et al., [2022](#bib.bib43 ""))*, have achieved state-of-the-art performance by leveraging contrastive augmentation. For example, SGL*(Wu et al., [2021](#bib.bib40 ""))* uses random dropout operators to corrupt interaction graph structures for augmentation. In NCL*(Lin et al., [2022](#bib.bib18 ""))*, representation alignment is performed between individual users and semantic-centric nodes. While these models have been effective in improving recommendation accuracy, they may fall short in encoding latent factors behind user-item interactions, which can result in suboptimal representations with coarse-grained user preference modeling for recommendation.

3. Methodology
---------------

<img src='x1.png' alt='Refer to caption' title='' width='461' height='97' />

*Figure 1. The overall framework of our proposed DCCF model involves adaptive augmentation through the integration of global intent disentanglement and interaction pattern encoding, resulting in disentangled environment-invariant representations.*

### 3.1. Disentangled Intent Representation

#### 3.1.1. Modeling Latent Intent Factors

In our recommendation scenario, we represent the interaction matrix between the user set $\mathcal{U}\={u_{1},...,u_{i},...,u_{I}}$ (with size $I$) and item set $\mathcal{I}\={v_{1},...,v_{j},...,v_{J}}$ (with size $J$) as $\mathcal{A}\in\mathbb{R}^{I\times J}$. The entry $\mathcal{A}_{i,j}\in\mathcal{A}$ is set to 1 if user $u_{i}$ has adopted item $v_{j}$ before, and $\mathcal{A}_{i,j}\=0$ otherwise. Our model aims to predict the likelihood that a candidate user will adopt an item given their observed interactions. From a probabilistic perspective, our predictive model aims to estimate the conditional probability $P(y|u_{i},v_{j})$ for the interaction between user $u_{i}$ and item $v_{j}$, where $y$ is the learned preference score.

When interacting with items, users often have diverse intents, such as preferences for specific brands or interests in the genres and actors of movies*(Zhao et al., [2022](#bib.bib49 ""); Mu et al., [2021](#bib.bib22 ""))*. To capture these diverse intents, we assume $K$ different intents $c_{u}$ and $c_{v}$ from the user and item sides, respectively. The intent on the item side can also be understood as the context of the item, for example, a user who intends to shop for Valentine’s Day may have a preference for items that have a “romantic” context. Our predictive objective of user-item preference can be presented as follows:

| (1) |  | $\displaystyle\int_{c_{u}}\ \int_{c_{v}}\ P(y,c_{u},c_{v}|u,v)\,dc_{v}\,dc_{u}\=\sum_{k}^{K}P(y,c_{u}^{k},c_{v}^{k}|u,v)$ |  |
| --- | --- | --- | --- |

The user-item interaction probability $y$ is determined by the latent intents $c_{u}$ and $c_{v}$ and can be derived using the formulas:

| (2) |  | $\displaystyle\sum_{k}^{K}P(y,c_{u}^{k},c_{v}^{k}|u,v)$ | $\displaystyle\=\sum_{k}^{K}P(y|c_{u}^{k},c_{v}^{k})P(c_{u}^{k}|u)P(c_{v}^{k}|v)$ |  |
| --- | --- | --- | --- | --- |
| (3) |  |  | $\displaystyle\=\mathbb{E}_{P(c_{u}|u)P(c_{v}|v)}[P(y|c_{u},c_{v})].$ |  |
| --- | --- | --- | --- | --- |

Here, we use $f(\cdot)$ to denote the forecasting function over the encoded intents. Following the statistical theory in*(Wang et al., [2020a](#bib.bib30 ""), [2021a](#bib.bib31 ""))*, we make the following approximation to derive our prediction objective:

| (4) |  | $\displaystyle\mathbb{E}_{P(c_{u}|u)P(c_{v}|v)}[f(c_{u},c_{v})]\approx f(\mathbb{E}_{P(c_{u}|u)}[c_{u}],\mathbb{E}_{P(c_{v}|v)}[c_{v}]).$ |  |
| --- | --- | --- | --- |

With the above inference, the approximation error, known as *Jensen gap* *(Abramovich and Persson, [2016](#bib.bib2 ""))*, can be well bounded in our forecasting function $f(\cdot)$*(Gao et al., [2019](#bib.bib13 ""))*.

#### 3.1.2. Multi-Intent Representation with Global Context

While intent diversity has been encoded in existing recommender systems through disentangled representations, global-level intent-aware collaborative relations have been largely overlooked. Global-level user (item) dependency modeling can enhance the robustness of GNN-based message passing models against sparsity and over-smoothing issue, via propagating information without the limitation of direct local connections*(Xia et al., [2022](#bib.bib43 ""))*. Towards this end, we propose to disentangle collaborative relations among users and items with both local- and global-level embedding for information propagation.

Graph-based Message Passing.
Owing to the strength of graph neural networks, GNNs has become the prevalent learning paradigm to capture collaborative filtering signals in state-of-the-art recommender systems. Examples include LightGCN*(He et al., [2020](#bib.bib14 ""))*, LR-GCCF*(Chen et al., [2020c](#bib.bib8 ""))*, and HGCF*(Sun et al., [2021](#bib.bib26 ""))*. The insights offered by these studies have inspired us to build our DCCF model using a graph-based message passing framework for user representations. In general, our message propagation layer is formally presented with the user/item embedding matrix $\textbf{E}^{(u)}\in\mathbb{R}^{I\times d}$ and $\textbf{E}^{(v)}\in\mathbb{R}^{J\times d}$ as follows:

| (5) |  | $\displaystyle\textbf{Z}^{(u)}\=\bar{\mathcal{A}}\cdot\textbf{E}^{(v)},\leavevmode\nobreak\ \leavevmode\nobreak\ \leavevmode\nobreak\ \textbf{Z}^{(v)}\=\bar{\mathcal{A}}^{T}\cdot\textbf{E}^{(u)},$ |  |
| --- | --- | --- | --- |

The aggregated representations from neighboring nodes to the target ones are denoted by $\textbf{Z}^{(u)}\in\mathbb{R}^{I\times d}$ and $\textbf{Z}^{(v)}\in\mathbb{R}^{J\times d}$. Here, $\bar{\mathcal{A}}\in\mathbb{R}^{I\times J}$ denotes the normalized adjacent matrix which is derived from the user-item interaction matrix $\mathcal{A}$ as $\bar{\mathcal{A}}\=\textbf{D}^{-1/2}_{(u)}\cdot\mathcal{A}\cdot\textbf{D}^{-1/2}_{(v)}$.

where $\textbf{D}_{(u)}\in\mathbb{R}^{I\times I}$ and $\textbf{D}_{(v)}\in\mathbb{R}^{J\times J}$ are diagonal degree matrices.

To exploit high-order collaborative filtering signals, we perform GNN-based embedding propagation across different graph layers, such as from the $(l-1)$-th to the $(l)$-th layer, as follows:

| (6) |  | $\displaystyle\textbf{E}_{l}^{(u)}\=\textbf{E}_{l-1}^{(u)}+\textbf{Z}_{l-1}^{(u)},\leavevmode\nobreak\ \leavevmode\nobreak\ \leavevmode\nobreak\ \textbf{E}_{l}^{(v)}\=\textbf{E}_{l-1}^{(v)}+\textbf{Z}_{l-1}^{(v)},$ |  |
| --- | --- | --- | --- |

To suppress the over-smoothing effect, residual connections are applied to the aggregation phase*(Chen et al., [2020c](#bib.bib8 ""); Xia et al., [2022](#bib.bib43 ""))*.

Intent-aware Information Aggregation. We will describe how to incorporate intent-aware global user (item) dependencies into our GNN-based collaborative filtering framework. In our multi-intent encoder, disentangled user-item preferences are preserved in $\mathbb{E}_{P(c_{u}|u)}[c_{u}]$ and $\mathbb{E}_{P(c_{v}|v)}[c_{v}]$. In our DCCF, we define $K$ global intent prototypes ${\textbf{c}_{u}^{k}\in\mathbb{R}^{d}}_{k\=1}^{K}$ and ${\textbf{c}_{v}^{k}\in\mathbb{R}^{d}}_{k\=1}^{K}$ for user and item, respectively. With these learnable intent embeddings, we generate user and item representations by aggregating information across different $K$ intent prototypes with the global context at the $l$-th graph embedding layer, using the following design:

| (7) |  | $\displaystyle\textbf{r}_{i,l}^{(u)}\=\mathbb{E}_{P(\textbf{c}_{u}|\textbf{e}_{i,l}^{(u)})}[\textbf{c}_{u}]\=\sum_{k}^{K}\textbf{c}_{u}^{k}P(\textbf{c}_{u}^{k}|\textbf{e}_{i,l}^{(u)}),$ |  |
| --- | --- | --- | --- |
| (8) |  | $\displaystyle\textbf{r}_{j,l}^{(v)}\=\mathbb{E}_{P(\textbf{c}_{v}|\textbf{e}_{j,l}^{(v)})}[\textbf{c}_{v}]\=\sum_{k}^{K}\textbf{c}_{v}^{k}P(\textbf{c}_{v}^{k}|\textbf{e}_{j,l}^{(v)}),$ |  |
| --- | --- | --- | --- |

The $l$-th layer-specific user and item embeddings are denoted by $\textbf{e}_{i,l}^{(u)}\in\textbf{E}_{l}^{(u)}$ and $\textbf{e}_{j,l}^{(v)}\in\textbf{E}_{l}^{(v)}$, respectively. The relevance score between user $u_{i}$ and each intent prototype $\textbf{c}_{u}$ is defined as $P(\textbf{c}_{u}^{k}|\textbf{e}_{i,l}^{(u)})$, which can be derived as follows:

|  | $\displaystyle P(\textbf{c}_{u}^{k}|\textbf{e}_{i,l}^{(u)})\=\frac{\eta(\textbf{e}_{i,l-1}^{(u)\top}\textbf{c}_{u}^{k})}{\sum_{k^{\prime}}^{K}\eta(\textbf{e}_{i,l-1}^{(u)\top}\textbf{c}_{u}^{k^{\prime}})},P(\textbf{c}_{v}^{k}|\textbf{e}_{j,l}^{(v)})\=\frac{\eta(\textbf{e}_{j,l-1}^{(v)\top}\textbf{c}_{v}^{k})}{\sum_{k^{\prime}}^{K}\eta(\textbf{e}_{j,l-1}^{(v)\top}\textbf{c}_{v}^{k^{\prime}})}$ |  |
| --- | --- | --- |

Here, $\eta(\cdot)\=\exp(\cdot)$. After generating the propagated message, we refine it by integrating the local collaborative filtering signals with the global disentangled collaborative relations, as follows:

| (9) |  | $\displaystyle\textbf{E}_{l}^{(u)}\=\textbf{E}_{l-1}^{(u)}+\textbf{Z}_{l-1}^{(u)}+\textbf{R}_{l-1}^{(u)},\leavevmode\nobreak\ \leavevmode\nobreak\ \leavevmode\nobreak\ \textbf{E}_{l}^{(v)}\=\textbf{E}_{l-1}^{(v)}+\textbf{Z}_{l-1}^{(v)}+\textbf{R}_{l-1}^{(v)}.$ |  |
| --- | --- | --- | --- |

In this equation, $\textbf{R}_{l-1}^{(u)}\in\mathbb{R}^{I\times d}$ and $\textbf{R}_{l-1}^{(v)}\in\mathbb{R}^{J\times d}$ represent the stacked intent-aware user embeddings ($\textbf{r}_{i,l-1}^{(u)}$) and item embeddings ($\textbf{r}_{j,l-1}^{(v)}$), respectively. Incorporating intent disentanglement into the graph neural architecture enables our learned representations to effectively disentangle the latent factors driving complex user-item interaction behaviors.

### 3.2. Disentangled Contrastive Learning

Taking inspiration from recent developments in contrastive learning, we explore the potential of contrastive augmentation with intent disentanglement to address the data sparsity issue in recommender systems. Although self-supervision signals can be generated by maximizing the consistency between positive pairs among contrastive views, we argue that such augmentation is susceptible to data noise, such as misclicks. Noisy contrastive regularization may mislead the self-supervised learning process. For instance, reinforcing the model to achieve embedding agreement via node self-discrimination on noisy interaction edges may involve noisy self-supervised signals and lead to suboptimal representations.

To address this challenge, we design learnable augmenters that consider both local collaborative relations and global disentangled user (item) dependencies. By doing so, the learnable contrastive augmenters can adaptively learn disentangled SSL signals.

#### 3.2.1. Disentangled Data Augmentation

To enable the augmentation to be adaptive to each connection hop, we introduce a learnable relation matrix $\mathcal{G}^{l}\in\mathbb{R}^{I\times J}$ for each $(l)$-th GNN layer to encode the implicit relationships between users and items. Inspired by previous work on graph denoising*(Luo et al., [2021](#bib.bib19 ""); Tian et al., [2022](#bib.bib27 ""))*, we aim to generate a graph mask $\mathcal{M}^{l}\in\mathbb{R}^{I\times J}$, which can be used to obtain the relation matrix through element-wise multiplication: $\mathcal{G}^{l}\=\mathcal{M}^{l}\odot\mathcal{A}$.

Learning Graph Mask.
Each entry $\mathcal{M}^{l}_{i,j}\in[0,1]$ in the graph mask $\mathcal{M}^{l}$ reflects the degree to which the interaction between user $i$ and item $j$ is masked. The closer the value is to <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S3.SS2.SSS1.p2.5.m5.1"><semantics id="S3.SS2.SSS1.p2.5.m5.1a"><mn id="S3.SS2.SSS1.p2.5.m5.1.1" xref="S3.SS2.SSS1.p2.5.m5.1.1.cmml">0</mn><annotation-xml encoding="MathML-Content" id="S3.SS2.SSS1.p2.5.m5.1b"><cn id="S3.SS2.SSS1.p2.5.m5.1.1.cmml" type="integer" xref="S3.SS2.SSS1.p2.5.m5.1.1">0</cn></annotation-xml></semantics></math> -->00, the less important the interaction is, and vice versa. In our DCCF model, we derive $\mathcal{M}^{l}_{i,j}$ based on the disentangled embeddings of user ($\textbf{r}_{i,l}^{(u)}$) and item ($\textbf{r}_{i,l}^{(v)}$) to preserve the intent-aware interaction patterns. Specifically, we use cosine similarity*(Chen et al., [2020b](#bib.bib12 ""); Tian et al., [2022](#bib.bib27 ""))* between node embeddings to measure the importance of interactions:

| (10) |  | $\displaystyle s(\textbf{r}_{i,l}^{(u)},\mathbf{r}_{j,l}^{(v)})\=\frac{{\textbf{r}_{i,l}^{(u)}}^{T}{\mathbf{r}_{j,l}^{(v)}}}{\|{\textbf{r}_{i,l}^{(u)}}\|_{2}\|{\mathbf{r}_{j,l}^{(v)}}\|_{2}}.$ |  |
| --- | --- | --- | --- |

The mask value is obtained by linearly transforming the range of the similarity to $[0,1]$, using the formula: $\mathcal{M}^{l}_{i,j}\=(s(\textbf{r}_{i,l}^{(u)},\mathbf{r}_{j,l}^{(v)})+1)/2$.

Learnable Augmentation.
$\mathcal{A}_{i,j}$ is <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S3.SS2.SSS1.p3.2.m2.1"><semantics id="S3.SS2.SSS1.p3.2.m2.1a"><mn id="S3.SS2.SSS1.p3.2.m2.1.1" xref="S3.SS2.SSS1.p3.2.m2.1.1.cmml">0</mn><annotation-xml encoding="MathML-Content" id="S3.SS2.SSS1.p3.2.m2.1b"><cn id="S3.SS2.SSS1.p3.2.m2.1.1.cmml" type="integer" xref="S3.SS2.SSS1.p3.2.m2.1.1">0</cn></annotation-xml></semantics></math> -->00 when there is no interaction between user $i$ and item $j$. $\mathcal{G}^{l}$ is obtained by element-wise multiplication of $\mathcal{M}^{l}$ and $\mathcal{A}$. only the mask values of observed interactions are calculated for computational simplicity. With the learned relation matrix, we then normalize it with the degree of the node as follows (layer index is omitted for simplicity):

| (11) |  | $\displaystyle\bar{\mathcal{G}}_{i,j}\=\mathcal{G}_{i,j}/\sum_{j^{\prime}}^{J}\mathcal{G}_{i,j},\leavevmode\nobreak\ \leavevmode\nobreak\ \leavevmode\nobreak\ \bar{\mathcal{G}}^{T}_{j,i}\=\mathcal{G}^{T}_{j,i}/\sum_{i^{\prime}}^{I}\mathcal{G}^{T}_{j,i^{\prime}}.$ |  |
| --- | --- | --- | --- |

To integrate our adaptive augmentation with the message passing scheme, we apply our normalized learned relation matrix $\bar{\mathcal{G}}^{l}$ over the messages of nodes for learnable propagation. With this design, we perturb the graph structure to generate contrastive learning views with adaptive augmentation. The augmentation with adaptive masking can be formally presented as follows:

| (12) |  | $\displaystyle\textbf{H}_{l}^{(u)}\=\bar{\mathcal{G}}\cdot\textbf{E}_{l}^{(v)},\leavevmode\nobreak\ \leavevmode\nobreak\ \leavevmode\nobreak\ \textbf{H}_{l}^{(v)}\=\bar{\mathcal{G}}^{T}\cdot\textbf{E}_{l}^{(u)},$ |  |
| --- | --- | --- | --- |

To generate multiple contrastive views, we consider both local collaborative signals and global disentangled relationships. In particular, we perform augmentation using two learnable mask matrices over encoded local embeddings ($\textbf{Z}_{l}^{(u)}$ and $\textbf{Z}_{l}^{(v)}$ in Eq.[5](#S3.E5 "In 3.1.2. Multi-Intent Representation with Global Context ‣ 3.1. Disentangled Intent Representation ‣ 3. Methodology ‣ Disentangled Contrastive Collaborative Filtering")), and global embeddings with intent disentanglement ($\textbf{R}_{l}^{(u)}$ and $\textbf{R}_{l}^{(v)}$ in Eq.[7](#S3.E7 "In 3.1.2. Multi-Intent Representation with Global Context ‣ 3.1. Disentangled Intent Representation ‣ 3. Methodology ‣ Disentangled Contrastive Collaborative Filtering")). We derive two mask values $\mathcal{M}_{i,j}^{l}$ separately using the following formulas: $\mathcal{M}_{i,j}^{l}\=(s(\textbf{r}_{i,l}^{(u)},\mathbf{r}_{j,l}^{(v)})+1)/2$ and $\mathcal{M}_{i,j}^{{}^{\prime},l}\=(s(\textbf{z}_{i,l}^{(u)},\mathbf{z}_{j,l}^{(v)})+1)/2$. After that, our augmentation-aware message passing paradigm can be described with the following embedding refinement details:

| (13) |  | $\displaystyle\textbf{E}_{l}^{(u)}\=\textbf{E}_{l-1}^{(u)}+\textbf{Z}_{l-1}^{(u)}+\textbf{R}_{l-1}^{(u)}+\textbf{H}_{l-1}^{\beta,(u)}+\textbf{H}_{l-1}^{\gamma,(u)}$ |  |
| --- | --- | --- | --- |

Here, $\textbf{H}_{l-1}^{\beta,(u)}$ and $\textbf{H}_{l-1}^{\gamma,(u)}$ represent the local- and global-level augmented representations, respectively. Similarly, item embeddings are fused in an analogous manner.

#### 3.2.2. Contrastive Learning

Using the above augmented representation views, we conduct contrastive learning across different view-specific embeddings of users and items. Following the approach of supervised contrastive signals in*(Wu et al., [2021](#bib.bib40 ""); Xia et al., [2022](#bib.bib43 ""))*, we generate each positive pair using the embeddings of the same user (item) from the original CF view and each of the augmented views. The encoded representations of different nodes are treated as negative pairs. Specifically, we generate three augmented views using our augmenters: i) the local collaborative view with adaptive augmentation ($\textbf{H}^{\beta,(u)}$); ii) the disentangled global collaborative view ($\textbf{R}^{(u)}$); and iii) the adaptive augmented view ($\textbf{H}^{\gamma,(u)}$). We generate contrastive self-supervision signals using InfoNCE loss as follows:

| (14) |  | $\displaystyle\mathcal{I}(\textbf{m},\textbf{n})\=\frac{1}{I}\sum_{i\=0}^{I}\sum_{l\=0}^{L}-\log\frac{\exp(s(\textbf{m}_{i,l}^{(u)},\mathbf{n}_{i,l}^{(u)})/\tau)}{\sum_{i^{\prime}\=0}^{I}\exp(s({\textbf{m}_{i,l}^{(u)},\mathbf{n}_{i^{\prime},l}^{(u)})/\tau})},$ |  |
| --- | --- | --- | --- |

Here, m denotes the original view with vanilla embeddings ($\textbf{z}\in\textbf{Z}^{(u)}$) encoded from GNN. n is sampled from one of three augmented embeddings $\textbf{h}^{\beta}\in\textbf{H}^{\beta,(u)}$, $\textbf{R}^{(u)}$, and $\textbf{h}^{\gamma}\in\textbf{H}^{\gamma,(u)}$. The cosine similarity function is denoted by $s(\cdot)$. The contrastive learning loss from the user side can be formalized as follows:

| (15) |  | $\displaystyle\mathcal{L}_{cl}^{(u)}\=\mathcal{I}(\textbf{z},\textbf{r})+\mathcal{I}(\textbf{z},\textbf{h}^{\beta})+\mathcal{I}(\textbf{z},\textbf{h}^{\gamma})$ |  |
| --- | --- | --- | --- |

By stacking $L$ graph neural layers, the layer-specific embeddings are aggregated across different layers as follows: $\textbf{E}^{(u)}\=\sum_{l\=0}^{L}\textbf{E}_{l}^{(u)}$ and $\textbf{E}^{(v)}\=\sum_{l\=0}^{L}\textbf{E}_{l}^{(v)}$. The user-item preference score is derived as:

| (16) |  | $\displaystyle\textbf{Y}\=\textbf{E}^{(u)}(\textbf{E}^{(v)})^{T},\leavevmode\nobreak\ \leavevmode\nobreak\ \leavevmode\nobreak\ \textbf{Y}_{i,j}\=(\textbf{e}_{i}^{(u)})^{T}\textbf{e}_{j}^{(v)}.$ |  |
| --- | --- | --- | --- |

To optimize the classical supervised recommendation task using the estimated preference score, we use the following Bayesian Personalized Ranking (BPR) loss:

| (17) |  | $\displaystyle\mathcal{L}_{bpr}\=-\frac{1}{|\mathcal{R}|}\sum_{(i,p_{s},n_{s})\in\mathcal{R}}ln\sigma(\textbf{Y}_{i,p_{s}}-\textbf{Y}_{i,n_{s}}),$ |  |
| --- | --- | --- | --- |

where $\mathcal{R}$ is the set of sampled interactions in each mini-batch*(He et al., [2020](#bib.bib14 ""))*. For each user $u_{i}$, we sample $S$ positive items (indexed by $p_{s}$) and $S$ negative items (indexed by $n_{s}$) from the training data.

Finally, we integrate the self-supervised loss with our classical recommendation loss into a multi-task learning objective as follows:

| (18) |  | $\displaystyle\mathcal{L}\=\mathcal{L}_{bpr}+\lambda_{1}\cdot(\mathcal{L}_{cl}^{(u)}+\mathcal{L}_{cl}^{(v)})+\lambda_{2}\cdot\|\mathbf{\Theta_{1}}\|_{\text{F}}^{2}+\lambda_{3}\cdot\|\mathbf{\Theta_{2}}\|_{\text{F}}^{2}$ |  |
| --- | --- | --- | --- |

where $\lambda_{1}$, $\lambda_{2}$ and $\lambda_{3}$ are tunable weights. $\mathbf{\Theta_{1}}\={\textbf{E}_{0}^{(u)},\textbf{E}_{0}^{(v)}}$ and $\mathbf{\Theta_{2}}\={{\textbf{c}_{u}^{k}}_{k\=1}^{K},{\textbf{c}_{v}^{k}}_{k\=1}^{K}}$ are trainable parameters in our model.

### 3.3. Discussions on DCCF Model

In this section, we present theoretical analyses of the benefits of our disentangled contrastive learning paradigm. Initially, for a specific user $u_{i}$, the corresponding contrastive self-supervised learning signals are incorporated with $\mathcal{I}(\textbf{r}_{i}^{(u)},\textbf{z}_{i}^{(u)})$, where $\textbf{r}_{i}^{(u)}$ is the encoded embedding of $u_{i}$ from the augmentation with intent-aware user global dependency. The gradients of $\mathcal{I}(\textbf{r}_{i}^{(u)},\textbf{z}_{i}^{(u)})$ with respect to the disentangled representation $\textbf{r}_{i}^{(u)}$ contributed by negative samples can be derived as follows:

| (19) |  | $\displaystyle c(i^{\prime})$ | $\displaystyle\=\left(\frac{\textbf{r}_{i}^{(u)}}{\|\textbf{r}_{i}^{(u)}\|_{2}}-s(\textbf{r}_{i}^{(u)},\mathbf{z}_{i^{\prime}}^{(u)})\frac{\textbf{z}_{i}^{(u)}}{\|\textbf{z}_{i}^{\prime(u)}\|_{2}}\right)$ |  |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle\times\frac{\exp(s(\textbf{r}_{i}^{(u)},\textbf{z}_{i}^{\prime(u)})/\tau)}{\sum_{i^{\prime}}\exp(s(\textbf{r}_{i}^{(u)},\mathbf{z}_{i^{\prime}}^{(u)}/\tau)}$ |  |
| --- | --- | --- | --- |

Without loss of generality, we omit the index of graph layers. Here, $i^{\prime}$ denotes the negative sample $u_{i}^{\prime}$ for $u_{i}$ ($i^{\prime}\neq i$ \& $1\leq i\leq I$). The L2 norm of $c(i^{\prime})$ is proportional to a special function as follows:

| (20) |  | $\displaystyle\|c(i^{\prime})\|_{2}\propto\sqrt{1-s(\textbf{r}_{i}^{(u)},\mathbf{z}_{i^{\prime}}^{(u)})^{2}}\cdot\exp(\frac{s(\textbf{r}_{i}^{(u)},\mathbf{z}_{i^{\prime}}^{(u)})}{\tau})$ |  |
| --- | --- | --- | --- |

In the above equation, $s(\textbf{r}_{i}^{(u)},\mathbf{z}_{i^{\prime}}^{(u)})\in[-1,1]$. For hard negative samples, the corresponding embedding similarity score is close to 1, and the L2 norm of $c(i^{\prime})$ increases significantly*(Wu et al., [2021](#bib.bib40 ""); Xia et al., [2022](#bib.bib43 ""))*. Similar observations can be made for the contrastive augmentations $\mathcal{I}(\textbf{z},\textbf{h}^{\alpha})$ and $\mathcal{I}(\textbf{z},\textbf{h}^{\beta})$ using the learnable augmenter. Thus, our disentangled contrastive learning paradigm is capable of seeking hard negative samples to enhance model optimization.

In addition, we further justify the effectiveness of our model design for capturing the implicit cross-intent dependency via the gradient propagation. Here, we discuss how the encoding process of disentangled representation $\textbf{r}_{i}^{(u)}$ can propagate gradients to latent intent prototypes ${\textbf{c}_{u}^{k}}_{k\=1}^{K}$. Referring to Equation ([7](#S3.E7 "In 3.1.2. Multi-Intent Representation with Global Context ‣ 3.1. Disentangled Intent Representation ‣ 3. Methodology ‣ Disentangled Contrastive Collaborative Filtering")) and ([3.1.2](#S3.Ex1 "3.1.2. Multi-Intent Representation with Global Context ‣ 3.1. Disentangled Intent Representation ‣ 3. Methodology ‣ Disentangled Contrastive Collaborative Filtering")), we have the following partial derivative:

| (21) |  | $\displaystyle\frac{\partial\textbf{r}_{i}^{(u)}}{\partial\textbf{c}_{u}^{t}}\=\begin{bmatrix}\frac{\partial(\textbf{r}_{i}^{(u)})_{1}}{\partial(\textbf{c}_{u}^{t})_{1}}\&\cdots\&\frac{\partial(\textbf{r}_{i}^{(u)})_{1}}{\partial(\textbf{c}_{u}^{t})_{d}}\\ \vdots\&\ddots\&\vdots\\ \frac{\partial(\textbf{r}_{i}^{(u)})_{d}}{\partial(\textbf{c}_{u}^{t})_{1}}\&\cdots\&\frac{\partial(\textbf{r}_{i}^{(u)})_{d}}{\partial(\textbf{c}_{u}^{t})_{d}}\end{bmatrix},$ |  |
| --- | --- | --- | --- |

| (22) |  | $\displaystyle\frac{\partial(\textbf{r}_{i}^{(u)})_{m}}{\partial(\textbf{c}_{u}^{t})_{n}}\=P_{t}\sum_{k\=1}^{K}P_{k}[(\textbf{e}_{i}^{(u)})_{n}((\textbf{c}_{u}^{t})_{m}-(\textbf{c}_{u}^{k})_{m})+\mathbb{I}(m\=n)].$ |  |
| --- | --- | --- | --- |

$P_{t}$ is short for $P(\textbf{c}_{u}^{t}|\textbf{e}_{i}^{(u)})$. As can be seen from the partial derivatives, the intent-aware representations $\textbf{r}_{i}^{(u)}$ propagate gradients to the latent intent prototype via the estimation of conditional probability $\sum_{k}^{K}P(y,c_{u}^{k},c_{v}^{k}|u,v)$. During the backward propagation process, the cross-intent embedding aggregation can propagate gradients to all latent intents with the learned relevance weights. Therefore, the gradient learning enhanced by our auxiliary contrastive learning tasks is appropriately distributed to all latent intents, which facilitates the cross-intent dependency modeling and helps to capture accurate user preferences for recommendation.

Time Complexity Analysis.
We analyze the time complexity of different components in our DCCF from the following aspects: i) The graph-based message passing procedure takes $\mathcal{O}(L\times|\mathcal{A}|\times d)$ time, where $L$ denotes the number of graph neural layers for message passing. $|\mathcal{A}|$ represents the number of edges in the graph and $d$ is the dimensionality of user/item representations. ii) The intent-aware information aggregation component takes $\mathcal{O}(L\times(I+J)\times K\times d)$ time complexity, where $K$ denotes the number of latent intents. iii) Due to local- and global-based adaptive augmentation, it takes $\mathcal{O}(2\times L\times|\mathcal{A}|\times d)$ time complexity to generate two augmented views for self-supervision. iv) To calculate the contrastive learning objective, the cost is $\mathcal{O}(L\times B\times(I+J)\times d)$, where $B$ is the number of users/items included in a single mini-batch.

4. Evaluation
--------------

In this section, we perform experiments to evaluate our DCCF on different datasets by answering the following research questions:

* •

    RQ1: Does our proposed DCCF outperform various recommendation solutions under different experimental settings?

* •

    RQ2: Do the designed key components benefit the representation learning of our DCCF in achieving performance improvement?

* •

    RQ3: Is our proposed model effective in alleviating the data sparsity issues with our disentangled self-supervised signals?

* •

    RQ4: What is the impact of the number of latent intents?

* •

    RQ5: How does our DCCF perform w.r.ttraining efficiency?

*Table 1. Statistics of the experimental datasets.*

| Dataset | #Users | #Items | #Interactions | Density |
| --- | --- | --- | --- | --- |
| Gowalla | 50,821 | 57,440 | 1,172,425 | 4.0$e^{-4}$ |
| Amazon-book | 78,578 | 77,801 | 2,240,156 | 3.7$e^{-4}$ |
| Tmall | 47,939 | 41,390 | 2,357,450 | 1.2$e^{-3}$ |

### 4.1. Experimental Settings

#### 4.1.1. Datasets

We evaluate our model performance on public datasets: Gowalla: This dataset is collected from the Gowalla platform to record check-in relations between users and different locations based on mobility traces. Amazon-book: This dataset includes rating behaviors of users over products with book category on Amazon. Tmall: It contains customer purchase behaviors from the online retailer Tmall. Table [1](#S4.T1 "Table 1 ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering") summarizes the dataset statistics.

*Table 2. Recommendation performance of all compared methods on different datasets in terms of Recall and NDCG.*

| Data | Gowalla | | | | Amazon-book | | | | Tmall | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metrics | Recall@20 | Recall@40 | NDCG@20 | NDCG@40 | Recall@20 | Recall@40 | NDCG@20 | NDCG@40 | Recall@20 | Recall@40 | NDCG@20 | NDCG@40 |
| NCF | 0.1247 | 0.1910 | 0.0659 | 0.0832 | 0.0468 | 0.0771 | 0.0336 | 0.0438 | 0.0383 | 0.0647 | 0.0252 | 0.0344 |
| AutoR | 0.1409 | 0.2142 | 0.0716 | 0.0905 | 0.0546 | 0.0914 | 0.0354 | 0.0482 | 0.0336 | 0.0611 | 0.0203 | 0.0295 |
| NGCF | 0.1413 | 0.2072 | 0.0813 | 0.0987 | 0.0532 | 0.0866 | 0.0388 | 0.0501 | 0.0420 | 0.0751 | 0.0250 | 0.0365 |
| LightGCN | 0.1799 | 0.2577 | 0.1053 | 0.1255 | 0.0732 | 0.1148 | 0.0544 | 0.0681 | 0.0555 | 0.0895 | 0.0381 | 0.0499 |
| DisenGCN | 0.1379 | 0.2003 | 0.0798 | 0.0961 | 0.0481 | 0.0776 | 0.0353 | 0.0451 | 0.0422 | 0.0688 | 0.0285 | 0.0377 |
| DisenHAN | 0.1437 | 0.2079 | 0.0829 | 0.0997 | 0.0542 | 0.0865 | 0.0407 | 0.0513 | 0.0416 | 0.0682 | 0.0283 | 0.0376 |
| CDR | 0.1364 | 0.1943 | 0.0812 | 0.0963 | 0.0564 | 0.0887 | 0.0419 | 0.0526 | 0.0520 | 0.0833 | 0.0356 | 0.0465 |
| DGCF | 0.1784 | 0.2515 | 0.1069 | 0.1259 | 0.0688 | 0.1073 | 0.0513 | 0.0640 | 0.0544 | 0.0867 | 0.0372 | 0.0484 |
| DGCL | 0.1793 | 0.2483 | 0.1067 | 0.1247 | 0.0677 | 0.1057 | 0.0506 | 0.0631 | 0.0526 | 0.0845 | 0.0359 | 0.0469 |
| SLRec | 0.1529 | 0.2200 | 0.0926 | 0.1102 | 0.0544 | 0.0879 | 0.0374 | 0.0490 | 0.0549 | 0.0888 | 0.0375 | 0.0492 |
| SGL-ED | 0.1809 | 0.2559 | 0.1067 | 0.1262 | 0.0774 | 0.1204 | 0.0578 | 0.0719 | 0.0574 | 0.0919 | 0.0393 | 0.0513 |
| SGL-ND | 0.1814 | 0.2589 | 0.1065 | 0.1267 | 0.0722 | 0.1121 | 0.0542 | 0.0674 | 0.0553 | 0.0885 | 0.0379 | 0.0494 |
| HCCF | 0.1818 | 0.2601 | 0.1061 | 0.1265 | 0.0824 | 0.1282 | 0.0625 | 0.0776 | 0.0623 | 0.0986 | 0.0425 | 0.0552 |
| LightGCL | 0.1825 | 0.2601 | 0.1077 | 0.1280 | 0.0836 | 0.1280 | 0.0643 | 0.0790 | 0.0632 | 0.0971 | 0.0444 | 0.0562 |
| DCCF | 0.1876 | 0.2644 | 0.1123 | 0.1323 | 0.0889 | 0.1343 | 0.0680 | 0.0829 | 0.0668 | 0.1042 | 0.0469 | 0.0598 |
| p-val. | $8.9e^{-6}$ | $1.3e^{-3}$ | $2.6e^{-6}$ | $8.1e^{-6}$ | $8.6e^{-7}$ | $2.2e^{-6}$ | $8.6e^{-6}$ | $2.2e^{-6}$ | $2.6e^{-7}$ | $1.4e^{-7}$ | $8.6e^{-7}$ | $1.8e^{-7}$ |

#### 4.1.2. Evaluation Protocols and Metrics

To alleviate the bias of negative item instance sampling, we follow the all-rank protocol*(He et al., [2020](#bib.bib14 ""); Wu et al., [2021](#bib.bib40 ""); Wang et al., [2020c](#bib.bib37 ""))* over all items to measure the accuracy of our recommendation results. We use two widely adopted ranking-based metrics to evaluate the performance of all methods, namely Recall@N and NDCG (Normalized Discounted Cumulative Gain)@N.

#### 4.1.3. Baseline Methods

We include five groups of baseline methods for comprehensive comparison, as detailed below.

(i) Factorization-based Method.

* •

    NCF *(He et al., [2017](#bib.bib15 ""))*. This method replaces the inner product in MF with a multi-layer perceptron to estimate user-item interactions. For comparison, we include the NeuMF version.

(ii) Autoencoder-based Method.

* •

    AutoR *(Sedhain et al., [2015](#bib.bib24 ""))*. It reconstructs user-item interactions based on the autoencoder to obtain user preference for non-interacted items.

(iii) Recommendation with Graph Neural Network.

* •

    NGCF *(Wang et al., [2019b](#bib.bib33 ""))*. This method designs the propagation rule to inject collaborative signals into the embedding process of recommendation, which is beneficial for capturing higher-order connectivity.

* •

    LightGCN *(He et al., [2020](#bib.bib14 ""))*. This method simplifies the message passing rule of GCN by linearly propagate user/item embeddings on the interaction graph for collaborative filtering.

(iv) Disentangled Multi-Intent Recommender Systems.

* •

    DisenGCN *(Ma et al., [2019a](#bib.bib20 ""))*. This method proposes a neighborhood routing mechanism to learn disentangled node representation. The dot-product is used to predict the interaction likelihood.

* •

    DisenHAN *(Wang et al., [2020c](#bib.bib37 ""))*. It disentangles user/item representations into different aspects (i.e., latent intents) and then aggregates information from various aspects with attention for recommendation.

* •

    CDR *(Chen et al., [2021a](#bib.bib7 ""))*. This method utilizes a user’s noisy multi-feedback to mine user intentions and improves the training process through curriculum learning. We implement it with implicit feedback.

* •

    DGCF *(Wang et al., [2020b](#bib.bib35 ""))*. This method generates the intent-aware graph by modeling a distribution over intents for each interaction and thus learns disentangled representations.

* •

    DGCL *(Li et al., [2021](#bib.bib17 ""))*. This method proposes a factor-wise discrimination objective to learn disentangled representations. We implement it to learn disentangled representations of nodes and make user-item interaction prediction using inner products.

(v) Self-Supervised Learning for Recommendation.

* •

    SLRec *(Yao et al., [2021](#bib.bib46 ""))*. This method proposes a multi-task self-supervised learning framework to address the label sparsity problem in large-scale item recommender system.

* •

    SGL-ED/ND *(Wu et al., [2021](#bib.bib40 ""))*. This method reinforces user/item representation learning with GNNs by applying an auxiliary self-supervised contrastive learning task through data augmentation, namely edge drop (ED) or node drop (ND).

* •

    HCCF *(Xia et al., [2022](#bib.bib43 ""))*. It jointly captures local and global collaborative relations under a hypergraph neural network, and designs cross-view contrastive learning for augmentation.

* •

    LightGCL *(Cai et al., [2023](#bib.bib3 ""))*. It is a lightweight graph contrastive learning framework by leveraging singular value decomposition to generate augmented view for embedding contrasting.

#### 4.1.4. Hyperparameter Settings.

We implement our DCCF using PyTorch and use Adam*(Kingma and Ba, [2015](#bib.bib16 ""))* as optimizer with learning rate $1e^{-3}$. The number of latent intent prototypes $K$ is selected from the range of ${32,64,128,256}$ with $K\=128$ by default. $\lambda_{1}$, $\lambda_{2}$ and $\lambda_{3}$ are tuned from the range of $[0.001,0.025,0.1,0.2]$, $[2.5e^{-5},5e^{-4},5e^{-3}]$, $[2.5e^{-5},5e^{-4},5e^{-3}]$, respectively. To evaluate baseline performance with fair settings, latent embedding dimensionality $d$ and batchsize is set as 32 and 10240 for all compared methods. For graph-based models, the number of propagation layers is chosen from {1,2,3}. Detailed model implementation of our DCCF can be found in our released source code in the Abstract Section.

*Table 3. Ablation study on key components of DCCF (measured by $Recall@20$ and $NDCG@20$) on different datasets.*

| Category | Data | Gowalla | | Amazon-book | | Tmall | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | Variants | Recall | NDCG | Recall | NDCG | Recall | NDCG |
| DME | -Disen | 0.1637 | 0.0975 | 0.0772 | 0.0580 | 0.0629 | 0.0437 |
| PAM | -LocalR | 0.1719 | 0.1015 | 0.0786 | 0.0593 | 0.0638 | 0.0446 |
| | -DisenR | 0.1718 | 0.1016 | 0.0793 | 0.0597 | 0.0640 | 0.0447 |
| SSL | -DisenG | 0.1763 | 0.1053 | 0.0829 | 0.0635 | 0.0644 | 0.0449 |
| | -AllAda | 0.1845 | 0.1096 | 0.0833 | 0.0632 | 0.0651 | 0.0452 |
| *DCCF* | | 0.1876 | 0.1123 | 0.0889 | 0.0680 | 0.0668 | 0.0469 |

### 4.2. Performance Comparison (RQ1)

Table [2](#S4.T2 "Table 2 ‣ 4.1.1. Datasets ‣ 4.1. Experimental Settings ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering") shows the performance comparison of different methods on all datasets. To validate the significant performance improvement achieved by our DCCF model, the p-value is provided. From evaluation results, we summarize the following observations:

* •

    DCCF consistently outperforms all baselines on all three datasets. Through disentangled contrastive learning, DCCF improves the generalization and robustness of recommenders by offering more informative representations. We attribute the significant performance gain of DCCF to two key aspects: (i) DCCF effectively alleviates the data sparsity issue by distilling disentangled self-supervised signals as supplementary training tasks. (ii) Our proposed parameterized graph mask generator is beneficial for achieving adaptive self-supervision against data noise redundancy, which further improves the representation robustness.

* •

    Although data augmentation techniques are also proposed in current SSL-based methods (e.g., SGL, HCCF), our DCCF still outperforms them by a large margin. This is because simply learning augmented representations at coarse-grained level cannot disentangle latent intention factors behind user-item interactions. In addition, we notice that most SSL-based methods perform better than conventional GNN-based approaches (e.g., LightGCN, NGCF), which suggests the positive effects of SSL brings to GNN-based CF models. With our disentangled adaptive augmentation, DCCF still pushes that boundary forward, achieving state-of-the-art performance across all datasets.

* •

    The performance improvement of DCCF over other disentangled recommender systems (e.g., DGCF, DisenGCN, CDR) verifies that our approach is not limited to the label shortage issue. The integration of disentangled multi-intent encoding and contrastive learning results in better performance. Existing disentangled learning solutions struggle to generate informative embeddings in the face of insufficient training labels due to the overfitting effect. Although DGCL attempts to use contrastive learning to encode latent factors into augmented representations, its non-adaptive contrastive view generation makes it easily influenced by noise perturbation.

<img src='x2.png' alt='Refer to caption' title='' width='461' height='127' />

*(a) Performance w.r.t. different item groups*

<img src='x3.png' alt='Refer to caption' title='' width='461' height='127' />

*(b) Performance w.r.t. different user groups*

*Figure 2. Performance comparison w.r.t. data sparsity over different user/item groups on Gowalla data.*

### 4.3. Ablation Study (RQ2)

In this section, to verify the effectiveness of each component, we conduct an ablation study to examine the component-specific benefits of our DCCF framework from three perspectives: (i) Disentangled Multi-intent Encoding (DME); (ii) Parameterized Adaptive Masking (PAM); (iii) Self-supervised Learning (SSL). The performance results are reported in Table[3](#S4.T3 "Table 3 ‣ 4.1.4. Hyperparameter Settings. ‣ 4.1. Experimental Settings ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering"), and the variant details and impact study are presented as follows:

* •

    Disentangled Multi-intent Encoding (DME). We generate the ablation model (-Disen) by removing the disentangled multi-intent encoding module. The performance gap between DCCF and -Disen indicates the contribution of multi-intent representation encoding to the overall performance.

* •

    Parameterized Adaptive Masking (PAM). To investigate the effect of our parameterized adaptive masking, we create two variants: (i) -LocalR which removes implicit user-item relation learning based on local relation embeddings; and (ii) -DisenR, which removes the intent-based graph structure learning process. The results show that both variants lead to a performance degradation, indicating the necessity of adaptive self-supervised signal distillation for contrastive augmentation.

* •

    Self-Supervised Learning (SSL). We also examine the influence of our disentangled contrastive learning on performance by adjusting the incorporated self-supervised optimization objectives. Specifically, we creat two variants by removing agreements between the original graph representations with auxiliary augmented views: (i) disentangled global collaborative view (-DisenG) and (ii) all augmented views with adaptive masking (-AllAda). Our results show that DCCF achieves the best performance compared to these variants, further emphasizing the benefits of integrating auxiliary self-supervised learning signals from the global view of intent-aware collaborative relationships for adaptive data augmentation.

*Table 4. The embedding smoothness on Amazon-book and Tmall data measured by MAD metric (the smaller the MAD indicates more obvious the over-smoothing phenomenon).*

| EmbeddingType | DCCF | DCCF-CL | DGCL | DisenGCN | LightGCN |
| --- | --- | --- | --- | --- | --- |
| | Amazon-book | | | | |
| User | 0.999 | 0.902 | 0.980 | 0.961 | 0.984 |
| Item | 0.990 | 0.961 | 0.989 | 0.986 | 0.944 |
|  | Tmall | | | | |
| User | 0.999 | 0.800 | 0.897 | 0.876 | 0.910 |
| Item | 0.998 | 0.873 | 0.920 | 0.992 | 0.927 |

### 4.4. In-Depth Analysis of DCCF (RQ3 \& RQ4)

#### 4.4.1. Performance w.r.t Data Sparsity

We further verify if DCCF is robust to data sparsity issue. To do this, we divide users and items into different groups based on the number of their interactions, and separately measured recommendation accuracy for each group. From the results in Figure [2](#S4.F2 "Figure 2 ‣ 4.2. Performance Comparison (RQ1) ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering"), we make two main observations: (i) DCCF consistently outperforms several representative baselines (i.e., LightGCN, DGCL, DGCF) by providing better recommendation results for both inactive and active users. This indicates the benefits of our generated self-supervised signals in alleviating sparse data issues. While DGCL conducts factor-wise alignment with contrastive learning, the interaction noise and bias can still impair the disentangled representation learning for latent factors.(ii) We notice that the performance gap between DCCF and the compared methods is still apparent on low-degree items. This is because the baseline DGCF only focuses on splitting the user representation into multiple intent-aware embeddings, which can easily lead to recommending high-degree items and neglect the long-tail items. In contrast, our DCCF enhances the interaction modeling on long-tail items through effective self-supervised information.

#### 4.4.2. Impact of the Number of Intent Prototypes.

To investigate the impact of the number of latent intents on model performance, we select this parameter from the range ${32,64,128,256}$ and re-train the model. The results are shown in Figure[3](#S4.F3 "Figure 3 ‣ 4.4.3. Robustness of DCCF in Alleviating Over-Smoothing ‣ 4.4. In-Depth Analysis of DCCF (RQ3 & RQ4) ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering"). It is clear that as the number of intents increases, the performance of the model also improves. However, when the number of intents increases from 128 to 256, the performance improvement is limited, and even degrades on the Tmall dataset. To further understand this phenomenon, we transform the intent prototypes into 2D space for visualization using t-SNE*(Van der Maaten and Hinton, [2008](#bib.bib28 ""))* and then clustered them. As shown in Figure[4](#S4.F4 "Figure 4 ‣ 4.4.3. Robustness of DCCF in Alleviating Over-Smoothing ‣ 4.4. In-Depth Analysis of DCCF (RQ3 & RQ4) ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering"), when the number of intents is 128, some latent intents have begun to cluster together. Further increasing the number of intents causes intent redundancy with too fine-grained latent factor granularity and introduces noise into learning representations.

#### 4.4.3. Robustness of DCCF in Alleviating Over-Smoothing

To validate the effectiveness of DCCF in alleviating over-smoothing, we calculate the Mean Average Distance (MAD)*(Chen et al., [2020a](#bib.bib6 ""); Xia et al., [2022](#bib.bib43 ""))* over encoded user/item embeddings of DCCF and the variant DCCF-CL, which disables the cross-view contrastive learning module. We also calculate the MAD of several representative baseline methods (i.e., DGCL, DisenGCN, LightGCN) for comparison. Note that all the embeddings were normalized before calculating MAD for fair comparison. The results are shown in Table[4](#S4.T4 "Table 4 ‣ 4.3. Ablation Study (RQ2) ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering"). We notice that by removing the SSL objective, the over-smoothing phenomenon becomes more pronounced, which suggests the effectiveness of our contrastive learning component in addressing the over-smoothing problem. Moreover, all the baselines have lower MAD than our DCCF, indicating that DCCF is capable of alleviating the over-smoothing issue in the widely-adopted GNN architecture. Our disentangled contrastive learning approach achieves better representation uniformity in recommendation compared to the baselines.

<img src='x4.png' alt='Refer to caption' title='' width='484' height='230' />

*Figure 3. Performance w.r.tthe number of latent intents.*

<img src='x5.png' alt='Refer to caption' title='' width='447' height='123' />

*(a) User intent prototypes*

<img src='x6.png' alt='Refer to caption' title='' width='447' height='123' />

*(b) Item intent prototypes*

*Figure 4. Distribution of latent intent prototypes.*

*Table 5. Computational cost evaluation in terms of per-epoch training time (seconds) on Gowalla, Amazon, and Tmall data.*

| Model | DisenGCN | DGCF | DisenHAN | DGCL | Ours |
| --- | --- | --- | --- | --- | --- |
| Gowalla | 19.1s | 25.1s | 16.8s | 9.3s | 12.4s |
| Amazon-book | 42.2s | 49.6s | 30.6s | 12.4s | 18.9s |
| Tmall | 43.5s | 51.6s | 29.8s | 12.0s | 18.8s |

### 4.5. Model Training Efficiency Study (RQ5)

In this section, we investigate the model efficiency of our DCCF in terms of training computational cost on all datasets. The experiments were conducted on a server with system configurations of an Intel Xeon Gold 6330 CPU, NVIDIA RTX 3090. As shown in Table[5](#S4.T5 "Table 5 ‣ 4.4.3. Robustness of DCCF in Alleviating Over-Smoothing ‣ 4.4. In-Depth Analysis of DCCF (RQ3 & RQ4) ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering"), we compare our DCCF with disentangled recommender systems (e.g., DGCF and DisenHAN) and found that our DCCF achieves comparable training efficiency in all cases. Specifically, while DGCF splits the user embedding into intent-aware vectors to reduce embedding size, the heavy cost of DGCF stems from the recursively routing mechanism for information propagation. It requires extra time to process multiple iterations to obtain intent-relevant weights. In DisenHAN, the time-consuming graph attention network brings high cost due to the need for computing the attention weights.

<img src='x7.png' alt='Refer to caption' title='' width='424' height='216' />

*Figure 5. Case study of intent-aware global user relations. Non-locally connected users ($u_{1155}$ and $u_{32856}$) can be identified with similar user preference (large item category overlap) via our learned disentangled representations.*

### 4.6. Case Study

Global Intent-aware Semantic Dependency.
In this section, we examine the potential ability of our DCCF in capturing the global intent-aware semantic dependencies among users. To achieve this goal, we showe some concrete examples in Figure[5](#S4.F5 "Figure 5 ‣ 4.5. Model Training Efficiency Study (RQ5) ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering") to visualize the intent-aware user preferences learned by our DCCF. We observe that $u_{1155}$ and $u_{32856}$ share very similar intent-aware preferences, as shown with intent prototype-specific user weights, despite not being locally connected on the interaction graph. After investigating their interaction patterns, we observe a significant overlap between the categories (categories $29,36,$ and $69$) of the items they interacted with, indicating the high semantic relatedness of their interaction behaviors. Therefore, in addition to local collaborative relations, the global intent-aware user dependencies can be preserved with our encoded disentangled user representations.

<img src='x8.png' alt='Refer to caption' title='' width='424' height='260' />

*Figure 6. Case study of intent-aware adaptive augmentation over the user-item relation matrix. User interacted items are grouped in terms of their categories. The value of the learned user-item connectivity weight is consistent with the user preference degree, i.e., the higher user-item weight encoded by DCCF indicates stronger user preference.*

Intent-aware Adaptive Augmentation We further analyze the rationality of our intent-aware adaptive augmentation over user-item relations. As shown in Figure[6](#S4.F6 "Figure 6 ‣ 4.6. Case Study ‣ 4. Evaluation ‣ Disentangled Contrastive Collaborative Filtering"), we grouped the interacted items of user $u_{22567}$ based on categories (e.g., category 1024, 645). After performing adaptive augmentation over the user-item relation matrix, the implicit dependency weight between each user-item pair was learned through our contrastive intent disentanglement. The value of the learned user-item connectivity weight determines the user’s preference degree over this item. We notice that a higher user-item relation weight (e.g., 0.0282 or 0.0273) indicates a stronger interaction preference over category-specific items, which is consistent with the observation of the category-specific interaction frequency of $u_{22567}$. For example, the highest item correlation weight (i.e., 0.0282) is generated from the categorical items that $u_{22567}$ interacted with the most. This observation further demonstrates the effectiveness of our disentangled contrastive augmentation, which is easily adaptable to different user-item interaction environments.

5. Conclusion
--------------

This paper proposes a disentangled contrastive learning method for recommendation that explores latent factors underlying implicit intents for interactions. We introduce a graph structure learning layer that enables adaptive interaction augmentation based on learned disentangled user (item) intent-aware dependencies. Along the augmented intent-aware graph structures, we propose an intent-aware contrastive learning scheme that brings the benefits of disentangled self-supervision signals. Our extensive experiments validate the effectiveness of our proposed model on different recommendation datasets. For future work, one potential extension is to integrate disentangled representation learning with causal analysis to address the bias issues of noisy interaction data. Additionally, by considering the diverse nature of user characteristics, personalized augmentation may further enhance the power of contrastive learning for customized graph perturbing operations in recommenders. By tailoring the augmentation operations to the specific user characteristics, we may better capture the individual preferences.

References
----------

* (1)
* Abramovich and Persson (2016)Shoshana Abramovich and
Lars-Erik Persson. 2016.Some new estimates of the ’Jensen gap’.*Journal of Inequalities and Applications*2016, 1 (2016),
1–9.
* Cai et al. (2023)Xuheng Cai, Chao Huang,
Lianghao Xia, and Xubin Ren.
2023.LightGCL: Simple Yet Effective Graph Contrastive
Learning for Recommendation. In *ICLR*.
* Chang et al. (2021)Jianxin Chang, Chen Gao,
Yu Zheng, Yiqun Hui,
Yanan Niu, Yang Song,
Depeng Jin, and Yong Li.
2021.Sequential recommendation with graph neural
networks. In *SIGIR*. 378–387.
* Chen et al. (2020d)Chong Chen, Min Zhang,
Weizhi Ma, Yiqun Liu, and
Shaoping Ma. 2020d.Jointly non-sampling learning for knowledge graph
enhanced recommendation. In *SIGIR*.
189–198.
* Chen et al. (2020a)Deli Chen, Yankai Lin,
Wei Li, Peng Li, Jie
Zhou, and Xu Sun. 2020a.Measuring and relieving the over-smoothing problem
for graph neural networks from the topological view. In*AAAI*, Vol. 34.
3438–3445.
* Chen et al. (2021a)Hong Chen, Yudong Chen,
Xin Wang, Ruobing Xie,
Rui Wang, Feng Xia, and
Wenwu Zhu. 2021a.Curriculum Disentangled Recommendation with Noisy
Multi-feedback.*NeurIPS* 34,
26924–26936.
* Chen et al. (2020c)Lei Chen, Le Wu,
Richang Hong, Kun Zhang, and
Meng Wang. 2020c.Revisiting Graph Based Collaborative Filtering: A
Linear Residual Graph Convolutional Network Approach. In*AAAI*, Vol. 34.
27–34.
* Chen et al. (2023)Mengru Chen, Chao Huang,
Lianghao Xia, Wei Wei,
Yong Xu, and Ronghua Luo.
2023.Heterogeneous Graph Contrastive Learning for
Recommendation. In *WSDM*.
544–552.
* Chen et al. (2022)Yongjun Chen, Zhiwei Liu,
Jia Li, Julian McAuley, and
Caiming Xiong. 2022.Intent contrastive learning for sequential
recommendation. In *WWW*.
2172–2182.
* Chen et al. (2021b)Yudong Chen, Xin Wang,
Miao Fan, Jizhou Huang,
Shengwen Yang, et al.2021b.Curriculum meta-learning for next POI
recommendation. In *KDD*.
2692–2702.
* Chen et al. (2020b)Yu Chen, Lingfei Wu,
and Mohammed Zaki. 2020b.Iterative deep graph learning for graph neural
networks: Better and robust node embeddings.*NeurIPS* (2020),
19314–19326.
* Gao et al. (2019)Xiang Gao, Meera
Sitharam, and Adrian E Roitberg.
2019.Bounds on the Jensen gap, and implications for
mean-concentrated distributions.*AJMAA* 16,
14 (2019), 1–16.
* He et al. (2020)Xiangnan He, Kuan Deng,
Xiang Wang, et al.2020.Lightgcn: Simplifying and powering graph
convolution network for recommendation. In*SIGIR*. 639–648.
* He et al. (2017)Xiangnan He, Lizi Liao,
Hanwang Zhang, Liqiang Nie,
Xia Hu, and Tat-Seng Chua.
2017.Neural collaborative filtering. In*WWW*. 173–182.
* Kingma and Ba (2015)Diederik P. Kingma and
Jimmy Ba. 2015.Adam: A Method for Stochastic Optimization. In*ICLR*.
* Li et al. (2021)Haoyang Li, Xin Wang,
Ziwei Zhang, Zehuan Yuan,
Hang Li, and Wenwu Zhu.
2021.Disentangled contrastive learning on graphs.*NeurIPS* 34
(2021), 21872–21884.
* Lin et al. (2022)Zihan Lin, Changxin Tian,
Yupeng Hou, and Wayne Xin Zhao.
2022.Improving Graph Collaborative Filtering with
Neighborhood-enriched Contrastive Learning. In*WWW*. 2320–2329.
* Luo et al. (2021)Dongsheng Luo, Wei Cheng,
Wenchao Yu, Bo Zong,
Jingchao Ni, Haifeng Chen, and
Xiang Zhang. 2021.Learning to drop: Robust graph neural network via
topological denoising. In *WSDM*.
779–787.
* Ma et al. (2019a)Jianxin Ma, Peng Cui,
Kun Kuang, Xin Wang, and
Wenwu Zhu. 2019a.Disentangled graph convolutional networks. In*ICML*. PMLR, 4212–4221.
* Ma et al. (2019b)Jianxin Ma, Chang Zhou,
Peng Cui, Hongxia Yang, and
Wenwu Zhu. 2019b.Learning disentangled representations for
recommendation. In *NeurIPS*.
5711–5722.
* Mu et al. (2021)Shanlei Mu, Yaliang Li,
Wayne Xin Zhao, Siqing Li, and
Ji-Rong Wen. 2021.Knowledge-Guided Disentangled Representation
Learning for Recommender Systems.*Transactions on Information Systems (TOIS)*40, 1 (2021),
1–26.
* Peng et al. (2020)Zhen Peng, Wenbing Huang,
Minnan Luo, et al.2020.Graph representation learning via graphical mutual
information maximization. In *WWW*.
259–270.
* Sedhain et al. (2015)Suvash Sedhain,
Aditya Krishna Menon, Scott Sanner, and
Lexing Xie. 2015.Autorec: Autoencoders meet collaborative
filtering. In *WWW*. 111–112.
* Song et al. (2019)Weiping Song, Zhiping
Xiao, Yifan Wang, Laurent Charlin,
Ming Zhang, and Jian Tang.
2019.Session-based social recommendation via dynamic
graph attention networks. In *WSDM*.
555–563.
* Sun et al. (2021)Jianing Sun, Zhaoyue
Cheng, Saba Zuberi, Felipe Pérez,
and Maksims Volkovs. 2021.HGCF: Hyperbolic Graph Convolution Networks for
Collaborative Filtering. In *WWW*.
593–601.
* Tian et al. (2022)Changxin Tian, Yuexiang
Xie, Yaliang Li, Nan Yang, and
Wayne Xin Zhao. 2022.Learning to Denoise Unreliable Interactions for
Graph Collaborative Filtering. In *SIGIR*.
122–132.
* Van der Maaten and Hinton (2008)Laurens Van der Maaten and
Geoffrey Hinton. 2008.Visualizing data using t-SNE.*Journal of machine learning research*9, 11 (2008).
* Velickovic et al. (2019)Petar Velickovic, William
Fedus, William L Hamilton, Pietro
Liò, Yoshua Bengio, and R Devon
Hjelm. 2019.Deep Graph Infomax.. In*ICLR*.
* Wang et al. (2020a)Tan Wang, Jianqiang
Huang, Hanwang Zhang, and Qianru Sun.
2020a.Visual commonsense r-cnn. In*CVPR*. 10760–10770.
* Wang et al. (2021a)Wenjie Wang, Fuli Feng,
Xiangnan He, Xiang Wang, and
Tat-Seng Chua. 2021a.Deconfounded recommendation for alleviating bias
amplification. In *KDD*.
1717–1725.
* Wang et al. (2019a)Xiang Wang, Xiangnan He,
Yixin Cao, Meng Liu, and
Tat-Seng Chua. 2019a.Kgat: Knowledge graph attention network for
recommendation. In *KDD*.
950–958.
* Wang et al. (2019b)Xiang Wang, Xiangnan He,
Meng Wang, Fuli Feng, and
Tat-Seng Chua. 2019b.Neural Graph Collaborative Filtering. In*SIGIR*.
* Wang et al. (2021b)Xiang Wang, Tinglin
Huang, Dingxian Wang, Yancheng Yuan,
Zhenguang Liu, Xiangnan He, and
Tat-Seng Chua. 2021b.Learning intents behind interactions with knowledge
graph for recommendation. In *WWW*.
878–887.
* Wang et al. (2020b)Xiang Wang, Hongye Jin,
An Zhang, Xiangnan He,
Tong Xu, and Tat-Seng Chua.
2020b.Disentangled graph collaborative filtering. In*SIGIR*. 1001–1010.
* Wang et al. (2020d)Xiao Wang, Ruijia Wang,
Chuan Shi, Guojie Song, et al.2020d.Multi-component graph convolutional collaborative
filtering. In *AAAI*, Vol. 34.
6267–6274.
* Wang et al. (2020c)Yifan Wang, Suyao Tang,
et al. 2020c.Disenhan: Disentangled heterogeneous graph
attention network for recommendation. In *CIKM*.
1605–1614.
* Wang et al. (2022)Zhenyi Wang, Huan Zhao,
and Chuan Shi. 2022.Profiling the Design Space for Graph Neural
Networks based Collaborative Filtering. In *WSDM*.
1109–1119.
* Wei et al. (2022)Wei Wei, Chao Huang,
Lianghao Xia, Yong Xu,
Jiashu Zhao, and Dawei Yin.
2022.Contrastive meta learning with behavior
multiplicity for recommendation. In *WSDM*.
1120–1128.
* Wu et al. (2021)Jiancan Wu, Xiang Wang,
Fuli Feng, Xiangnan He,
Liang Chen, Jianxun Lian,
et al. 2021.Self-supervised graph learning for recommendation.
In *SIGIR*. 726–735.
* Wu et al. (2020)Shiwen Wu, Fei Sun,
Wentao Zhang, Xu Xie, et al.2020.Graph neural networks in recommender systems: a
survey.*ACM Computing Surveys (CSUR)*(2020).
* Xia et al. (2023)Lianghao Xia, Chao Huang,
Chunzhen Huang, Kangyi Lin,
Tao Yu, and Ben Kao.
2023.Automated Self-Supervised Learning for
Recommendation. In *WWW*.
992–1002.
* Xia et al. (2022)Lianghao Xia, Chao Huang,
Yong Xu, Jiashu Zhao,
Dawei Yin, and Jimmy Huang.
2022.Hypergraph contrastive collaborative filtering. In*SIGIR*. 70–79.
* Yang et al. (2022)Yuhao Yang, Chao Huang,
Lianghao Xia, Yuxuan Liang,
Yanwei Yu, and Chenliang Li.
2022.Multi-behavior hypergraph-enhanced transformer for
sequential recommendation. In *KDD*.
2263–2274.
* Yang et al. (2021)Yonghui Yang, Le Wu,
Richang Hong, Kun Zhang, and
Meng Wang. 2021.Enhanced graph learning for collaborative filtering
via mutual information maximization. In *SIGIR*.
71–80.
* Yao et al. (2021)Tiansheng Yao, Xinyang
Yi, Derek Zhiyuan Cheng, et al.2021.Self-supervised Learning for Large-scale Item
Recommendations. In *CIKM*.
4321–4330.
* Yu et al. (2021)Junliang Yu, Hongzhi Yin,
Jundong Li, Qinyong Wang,
Nguyen Quoc Viet Hung, and Xiangliang
Zhang. 2021.Self-Supervised Multi-Channel Hypergraph
Convolutional Network for Social Recommendation. In*WWW*. 413–424.
* Zhang et al. (2022)Shengyu Zhang, Lingxiao
Yang, Dong Yao, Yujie Lu,
Fuli Feng, Zhou Zhao,
Tat-seng Chua, and Fei Wu.
2022.Re4: Learning to Re-contrast, Re-attend,
Re-construct for Multi-interest Recommendation. In*WWW*. 2216–2226.
* Zhao et al. (2022)Sen Zhao, Wei Wei,
Ding Zou, and Xianling Mao.
2022.Multi-view intent disentangle graph networks for
bundle recommendation.*AAAI* (2022).
* Zhou et al. (2020)Kun Zhou, Hui Wang,
Wayne Xin Zhao, Yutao Zhu,
Sirui Wang, Fuzheng Zhang,
Zhongyuan Wang, et al.2020.S3-rec: Self-supervised learning for sequential
recommendation with mutual information maximization. In*CIKM*. 1893–1902.
* Zhu and Chen (2022)Yaochen Zhu and
Zhenzhong Chen. 2022.Mutually-regularized dual collaborative variational
auto-encoder for recommendation systems. In *WWW*.
2379–2387.
* Zou et al. (2022)Ding Zou, Wei Wei,
Ziyang Wang, Xian-Ling Mao,
Feida Zhu, Rui Fang, and
Dangyang Chen. 2022.Improving knowledge-aware recommendation with
multi-level interactive contrastive learning. In*CIKM*. 2817–2826.
