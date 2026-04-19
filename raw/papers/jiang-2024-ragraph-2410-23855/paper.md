Xinke Jiang\*, Rihong Qiu\*, Yongxin Xu\*, Wentao Zhang, Yichen Zhu, Ruizhe Zhang\* Yuchen Fang, Xu Chu\*, Junfeng Zhao\*†, Yasha Wang\*†

$\boxed{=}$  {xinkejiang, rihongqiu, xuyx, ruizhezhang}@stu.pku.edu.cn

$\equiv \boxtimes$  [wentaozh2001, yichenzhu2014, fyclmiss]@gmail.com

$\boxed{\boxtimes}\{chu_{-}xu, zhaojf, wangyasha\} @pku.edu.cn$

$\spadesuit$ Key Laboratory of High Confidence Software Technologies (Peking University),

School of Computer Science, Peking University, China

No Affiliation, University of Electronic Science and Technology of China  
$\diamond$  Center on Frontiers of Computing Studies, Peking University, Beijing, China  
\*Big Data Technology Research Center, Nanhu Laboratory, Jiaxing, China  
$^{\star}$ Peking University Information Technology Institute, Tianjin Binhai, China

https://github.com/Artessay/RAGraph/

# Abstract

Graph Neural Networks (GNNs) have become essential in interpreting relational data across various domains, yet, they often struggle to generalize to unseen graph data that differs markedly from training instances. In this paper, we introduce a novel framework called General Retrieval-Augmented Graph Learning (RAGRAP), which brings external graph data into the general graph foundation model to improve model generalization on unseen scenarios. On the top of our framework is a toy graph vector library that we established, which captures key attributes, such as features and task-specific label information. During inference, the RAGRAP adeptly retrieves similar toy graphs based on key similarities in downstream tasks, integrating the retrieved data to enrich the learning context via the message-passing prompting mechanism. Our extensive experimental evaluations demonstrate that RAGRAP significantly outperforms state-of-the-art graph learning methods in multiple tasks such as node classification, link prediction, and graph classification across both dynamic and static datasets. Furthermore, extensive testing confirms that RA-GRAPH consistently maintains high performance without the need for task-specific fine-tuning, highlighting its adaptability, robustness, and broad applicability.

# 1 Introduction

Graph Neural Networks (GNNs) [5, 50, 98, 65, 127] have recently burgeoning a surge of interest in both academic and industry communities due to their robust capability to model complex, real-world data in diverse domains, including societal [74, 57, 81], biochemical [17, 112, 108], and traffic-related [56, 24, 46, 22, 21] fields and etc [55, 38, 70, 15, 26, 25]. Utilizing a message-passing mechanism [50, 30], GNNs have transcended traditional node embedding approaches [29, 80, 96], enabling the capture of intricate relationships within data through sophisticated architectures and advanced graph representation learning techniques [50, 52, 56, 18, 98]. However, the challenge of generalizing GNNs across different modalities, domains [64, 63], and tasks remains largely unexplored [58, 114]. This is in stark contrast to the significant successes of large models such as GPTs [75, 76] in NLP and Sora [66] in CV, presenting a crucial frontier for further research and realms for graph data generalizing.

In graph learning tasks, providing the necessary context is crucial for graph generalization [130, 53, 39, 135], i.e., retrieve similar shopping context as illustrated in Figure 1 (c). Therefore, our insight is to enhance the model's generalization ability and prediction accuracy by retrieving necessary contexts during graph learning through retrieval. Retrieval-Augmented Generation (RAG) represents a prominent methodology, significantly augmenting language model functionalities through the integration of a dynamic retrieval mechanism during the generation process [136, 78] (e.g., a person asks what animal it is, and we use some visual [139] or text retrieval [2] methods to retrieve more descriptive features or even the wanted category). RAG enriches not only accurate and reliable content but also reduces factual errors, addressing challenges such as incorrect answers, hallucinations, and limited interpretability in knowledge-intensive tasks [42, 2, 1], obviating the need for updating model parameters and could be generalized even in unseen scenarios.

However, how to enable retrieval-augmented generation for graph learning, i.e., retrieving the user's historical purchasing behavior to enhance recommendation ability [31, 114, 36] and identifying fraud crimes by searching for similar fraudulent relationship behaviors [86, 65], still remains unexplored and faces the following challenges  $C1 \& C2$ .

C1. The first challenge is how to leverage the retrieved context i.e., features  $(X)$  and labels  $(Y)$  into the GNNs model under dynamic changing scenarios. Previous studies, such as PRODIGY [39], have adopted the concept of in-context learning (ICL) by constructing consistent and static task graphs for each specific task or dataset. These task graphs determine labels through the calculation of similarities using hidden vectors, employing a few-shot learning approach. However, PRODIGY's reliance on a fixed set of examples as rules may not sufficiently address and generalize the variety of scenarios encountered in real-world settings, which is particularly problematic in dynamically changing environments, as the system focuses primarily on teaching the direct mapping paradigm from inputs to outputs  $(X\rightarrow Y)$ , rather than truly integrate the input  $(X)$  and output  $(Y)$  data into the analysis. In contrast to RAG, PRODIGY struggles to incorporate external information  $(X$  and  $Y$  ) related to data nodes, which is crucial for enriching the learning process in graph-based systems.  
C2. Moreover, it is challenging to develop a tune-free prompt mechanism to support retrieved knowledge and be applicable to seamlessly switch unseen scenarios and multi-tasks. Numerous initiatives have been undertaken in the realm of graph pre-training [34, 117, 35, 82, 99, 37, 7, 89, 126], however, the challenge persists in designing a plug-and-play RAG module that can seamlessly interface with already pre-trained models. Insights derived from prior investigations into the graph prompt [9, 27, 91, 67, 114, 20, 95], the knowledge obtained by RAG can be facilitated and injected into prompt via a plug-and-play manner.

For endeavoring to address these two challenges previously mentioned, we put forward the General Retrieval-Augmented Graph Learning Framework (RAGRAP). Drawing inspiration from the success of RAG on LLMs [136] and the ICL on GNNs [39] (we detail the difference between RAG and ICL in Appendix E), we constructed a toy graphs vector library by chunking from resource graphs, where the library key stores key information, including environmental, historical, structural, and semantic details, while

(a) NLP

(b) CV

(c) GNNs  
Figure 1: (a) RAG in NLP utilizes retrieval to enhance model responses, based on a query to retrieve related features (e.g., a tail, primarily feeds on mice) and answers (e.g., Cat). (b) In CV, RAG employs similar photo retrieval to enhance model comprehension, assisting in downstream tasks such as inpainting or image question answering. (c) For GNNs, RAG could leverage retrieval of similar historical subgraphs or scenarios to aid in graph-based tasks (e.g., recommendations or fraud detection).

node features and label information (task-specific output vector) are stored as values. For downstream tasks, the key value of the query node would be leveraged to retrieve toy graphs by the key similarities,

and the stored features (X) and labels (Y) would be aggregated structurally to provide essential knowledge to the query node, instead of the mapping paradigm, to address challenge  $C1$ . In prompt mechanism design, we start by transferring features and task-specific output from the toy graphs to their master nodes (the central node of the toy graph) via message-passing. Subsequently, features from the master nodes and the query node's neighbors are aggregated to the query node, along with the task-specific output from master nodes. This process could be parameter-free, indicating that our model can be applied across different tasks and datasets without the need to fine-tune for downstream tasks, effectively addressing challenge  $C2$ .

In summary, our contributions are listed as follows:

- To the best of our knowledge, our proposed framework, RAGRAPH, is the first to integrate RAG with pre-trained GNNs. By constructing a key-value vector library for toy graphs, RAGRAPH facilitates explicit plug-and-play access to pre-trained GNNs, achieving commendable performance even without fine-tuning, demonstrating its superiority on cross-task and cross-dataset capabilities.  
- Our RAGRAPH employs a classic message-passing mechanism and introduces a well-designed prompt mechanism to integrate knowledge. This approach effectively incorporates the retrieved knowledge  $X$  and  $Y$  from toy graphs, into the pre-trained GNNs model, enhancing the accuracy and relevance of the model's outputs.  
- We have extensively tested RAGraph on both static and dynamic graphs across multiple levels of graph tasks (node, edge, and graph). The results validate the effectiveness of our model, showing significant improvements over state-of-the-art baselines in both fine-tuned and tuning-free scenarios, particularly in cross-dataset validations.

# 2 Related Work

# 2.1 Retrieval-Augmented Generation on Large Language Models

RAG integrates an external knowledge retrieval component and through prompt engineering into pretrained language models to enhance factual consistency, thus improving the reliability and interpretability of LLM responses [132, 136, 51, 23, 45, 119, 59, 111, 128]. Traditional RAG approaches utilize retriever models to source relevant documents from extensive knowledge corpora [107, 83, 71, 49], which are then processed further by reader models—primarily LLMs [77, 85]. Furthermore, several studies focus on fine-tuning reader LLMs by applying prompt-tuning with retrieved knowledge or using RAG API calls [69, 42, 2, 116, 102, 129, 62]. While RAG has seen considerable success in the NLP field, it has also been applied to tasks involving joint visual and text retrieval [139, 61, 60, 8, 125, 10], code retrieval [68, 134], audio retrieval [6, 32] and video retrieval [3, 101]. Although there have been applications of RAG on structured data such as KG-RAG for knowledge graphs [48, 45, 87, 93, 94, 40], these primarily leverage the text information of knowledge graph nodes to enhance language or graph models. In contrast, there are no significant studies utilizing RAG on structured graphs without text information to enhance pre-trained GNNs. Our work aims to extend this successful approach similarly to graph data, to enhance the capabilities of pre-trained GNNs, and can be adapted to various tasks and across different graphs without additional fine-tuning by integrating a plug-and-play RAG module.

# 2.2 Graph Prompt Learning

Inspired by the application of pre-training models [75, 76] and prompt learning [103, 134, 43] in NLP, recently, learning on the graph has been divided into pre-training models on large-scale graph data [34, 117, 35, 82, 39, 131, 105, 90, 120, 122, 123, 121, 124], with or without labels, followed by fine-tuning model parameters via prompts for diverse downstream tasks [67, 114, 39, 138, 90, 95]. The adoption of prompting mechanisms in graph learning represents a promising avenue to overcome the constraints of traditional graph representation methods, striking a balance between flexibility and expressiveness [92]. For instance, VNT [95] utilizes virtual nodes as prompts to refine the application of pre-trained graph models. GraphPrompt [67] introduces a task-specific readout mechanism to tailor models for various tasks, while GraphPro [114] implements spatial- and temporal-based gating mechanisms suited for dynamic recommendation systems. Furthermore, PRODIGY [39] constructs task graphs (prompts) and data graphs to enhance the model's ICL capabilities. Leveraging the successes in graph prompt learning, we aim to inject retrieved knowledge via prompt into pre-trained GNNs to support downstream tasks.

# 3 Preliminaries

In RAGraph, we focus on RAG on multi-level graph tasks. For consistency, we define the graphs as dynamic graphs, considering static graphs as the special cases within this framework. The subsequent definition provides a detailed description of toy graphs, including the definitions of keys and values utilized in RAGraph. Additionally, inspired by GraphPrompt [67], we have unified node-level, edge-level, and graph-level tasks into a cohesive framework, and employ query graphs to tackle downstream tasks with precision.

Definition 1. (Dynamic Graph) Let  $\mathcal{G} = \{G_t\}_{t=1}^T$  denote a dynamic graph comprising a sequence of graph snapshots, each represented as a static graph  $G_t = (V_t, E_t, X_t, A_t, Y_t)$ .  $\mathcal{V} = \bigcup_{t=1}^T V_t = \{v_1, \dots, v_n\}$  defines the combined set of nodes across all snapshots and  $\mathcal{E} = \bigcup_{t=1}^T E_t \subseteq \mathcal{V} \times \mathcal{V}$  is the edge set, where  $V_t$  and  $E_t$  represent the nodes and edges of the  $t$ -th snapshot, respectively. Feature matrix  $X_t = \{x_v | v \in \mathcal{V}\} \in \mathbb{R}^{n \times d}$  contains the feature vectors for the nodes in the  $t$ -th snapshot, where  $d$  is the feature dimension.  $A_t$  denotes the edge weight matrix at time  $t$ , where edge weight  $A_t[i,j] \in (0,1]$  if  $v_i, v_j \in V_t$  and  $(v_i, v_j) \in E_t$ , and 0 otherwise. Furthermore,  $Y_t$  represents the task-related labels associated with nodes, edges, or the graph at time  $t$ . Note that a graph is static if  $T = 1$  and for consistency in terminology, we unify static graphs as a particular instance of dynamic graphs.

Definition 2. (Toy Graph Vector Base) Let  $\mathcal{G}^{\mathcal{R}} = \{G_t^{\mathcal{R}}\}_{t=1}^T$  denote a dynamic resource graph. We chunk  $\mathcal{G}^{\mathcal{R}}$  into snapshots and take each node in  $\mathcal{G}^{\mathcal{R}}$  as the master node  $v_m$  of the corresponding toy graph, and then store  $v_m$  with its neighbors within  $k$  hops as subgraphs. Data augmentation techniques [56, 133] such as node dropout, edge dropout, and random noise addition are employed on subgraphs to enhance the robustness and variability when generating each toy graph  $G^{\mathcal{T}}$  (c.f. Section 4.1 for details). Each toy graph  $G^{\mathcal{T}} \subseteq \mathcal{G}^{\mathcal{R}}$  is associated with a specific timestamp  $\tau$  and master node  $v_m \in \mathcal{V}$  with each toy graph's scale being considerably smaller in scale compared to their corresponding  $\mathcal{G}^{\mathcal{R}}$ . Toy graphs can be retrieved using keys that include the timestamp  $\tau$ , the hidden embedding of the master node  $h_m^\tau \in \mathbb{R}^{f_1}$  (e.g., embedded by pre-trained GNNs in RAGRAPH), the environmental key (e.g., the neighbors set  $\mathcal{N}(v_m^\tau) = \{v_i^\tau | A_\tau[m,i] > 0, v_i^\tau \in G^{\mathcal{T}}\}$ ) and the structure-based position-aware code  $s_m^\tau$  (cf. Appendix C.2 for details). By retrieving based on key similarity (c.f. Section 4.2 for details), we can obtain the required values of  $G^{\mathcal{T}}$ , i.e. task-specific output vector  $\{o_i^\tau \in \mathbb{R}^{f_2} | v_i \in G^{\mathcal{T}}\}$  and hidden embeddings  $\{h_i^\tau \in \mathbb{R}^{f_1} | v_i \in G^{\mathcal{T}}\}$  of the master node and its neighbors, where  $f_1$  and  $f_2$  represent the dimensions. Finally, we denote the key-value vector base for the toy graph as  $\mathcal{G}^{\mathcal{T}}$ .

Definition 3. (A Unified Graph Task Definition) Given a dynamic graph  $\mathcal{G}$ , it can be divided into training and testing subsets, i.e.  $\mathcal{G} = \mathcal{G}_{\mathrm{train}} \cup \mathcal{G}_{\mathrm{test}}$  based on either snapshot or node set partitioning. The label  $y_{i}$  of a node  $v_{i}$ , edge  $(v_{i}, v_{j})$  or subgraph  $G_{i}$  can be observed only if they belong to  $\mathcal{G}_{\mathrm{train}}$ . The objective of label prediction is to predict test labels  $Y_{\mathrm{test}} \in \mathcal{G}_{\mathrm{test}}$ . Following GraphPrompt [67], we unify the three types of graph learning tasks (node-level, edge-level, and graph-level) into a single framework via similarity comparison  $\operatorname{sim}(\cdot, \cdot)$  of the task-specific output vector (abbreviated as  $\bar{O}$ , where each entry is  $o$ ) with the ground-truth (i.e., the one-hot vector or the prototype embedding under few-shot setting). It's noted that  $o$  can be either low-dimensional (with the dimension equal to the number of predicted classes) under normal settings [50, 127], or high-dimensional under few-shot settings [67] or in link prediction tasks [114, 31]. In our experiment,  $\Phi$  for node-level and graph-level tasks, the downstream tasks are given in few-shot settings following [67]: For node / graph classification on a node / graph set, let  $\mathcal{C}$  be the set of classes with  $y_{i} \in \mathcal{C}$  denoting the class label of node / graph. For each node / graph class, the class prototypical output vector is calculated by the mean value of the  $\kappa$ -shot set  $\mathcal{D}$ :  $\tilde{o}_c = \frac{1}{\kappa} \sum_{(i, y_i) \in \mathcal{D}, y_i = c} o_i$ . The class  $y_{i}$  of the node or graph is determined by calculating similarity with the class prototype as:  $y_{i} = \operatorname{argmax}_{c \in \mathcal{C}} \operatorname{sim}(o_i, \tilde{o}_c)$ .  $\Phi$  For edge-level tasks, to predict a link between nodes  $v_{i}$  and  $v_{q}$ , if  $\exists v_{j}, (v_{i}, v_{j}) \in \mathcal{E}_{\mathrm{train}} \in \mathcal{G}_{\mathrm{train}}$  and  $\operatorname{sim}(o_i, o_q) \geq \operatorname{sim}(o_i, o_j) + \epsilon$ , we regard  $(v_{i}, v_{q})$  as linked. Following PRODIGY [39] and GraphPrompt [67], we also apply a query graph  $G^{\mathcal{Q}}$  that includes the center node and its neighbors within  $k$  hops. Specifically, for graph-level task, we apply a full-link virtual node as the center node inside the query graph  $G^{\mathcal{Q}}$ .

# 4 RAGRAPH Framework

In this section, we introduce RAGRAP, a general and novel retrieval-augmented graph learning framework that can operate on arbitrary graphs with or without additional fine-tuning, as illustrated in Figure 2. Initially, in Section 4.1, we elucidate the methodology for constructing the Resource Toy Graphs. Subsequently, in Section 4.2 we detail the Toy Graphs Retrieval Process. Finally, the Training and Inference processes are elaborated in Section 4.3, which utilize retrieved toy graphs from two propagation views—intra and inter-propagation—and handle two types of information: hidden embeddings and task-specific output vectors in two techniques (noisy trainable approach or parameter-free approach). The main notations of RAGRAP are summarized in Table 3, Appendix A. For enhanced clarity, the Toy Graph Construction is outlined in Algorithm 1 (cf. Appendix C.5) and the Training and Inference with Toy Graphs Retrieval are detailed in Algorithm 2 (cf. Appendix C.5). Moreover, in Appendix C.4, we theoretically prove the effectiveness of applying RAG on GNNs from the perspective of mutual information gain.


Figure 2: The overall framework of RAGRAP. ① Given resource graph  $\mathcal{G}^{\mathcal{R}}$ , we chunk it and augment toy graphs  $\{G^{\mathcal{T}}\}$ , and feed them into pre-trained GNNs to generate hidden embeddings via the encoder and task-specific output vectors via decoder, which are stored as values. Keys such as environment, history, position-aware, and hidden embeddings are stored to form the key-value database of toy graphs  $\mathcal{G}^{\mathcal{T}}$ . ② For a given query graph  $G^{\mathcal{Q}}$ , the keys are fetched to retrieve the topK toy graphs  $G_{\mathrm{topK}}^{\mathcal{T}}$  from the database. ③ Leveraging  $G_{\mathrm{topK}}^{\mathcal{T}}$ , intra- and inter-propagation are performed to propagate hidden embeddings and task-specific output vectors to pass retrieved knowledge to center node  $v_{c}$ . Through a weighted fusion, the aggregated output is used to perform graph-, node- and edge-level tasks.

# 4.1 Toy Graphs Embedding Pipeline

In graph-based learning, nodes with higher connectivity—typically with higher degrees—often hold more significance, meaning their information is more extensively learned during graph-pre-training processes. Conversely, less important nodes—those in the long tail—often have their features over

looked. This issue is particularly pronounced in LLMs performing RAG, where the predominance of common knowledge overshadows the long-tail knowledge that RAG is meant to leverage. To tackle this, we construct toy graphs using an inverse importance sampling strategy, thereby countering this bias by preferentially sampling and augmenting toy graphs that accentuate the long-tail knowledge.

Inverse Importance Sampling Strategy. To achieve this, we calculate each node's importance  $I(v)$  for node  $v \in G_{\tau}^{\mathcal{R}}$  by combining PageRank PR(v) and Degree Centrality DC(v) using the formula  $I(v) = \alpha \mathrm{PR}(v) + (1 - \alpha)\mathrm{DC}(v)$ , where  $\alpha \in (0,1)$  is the balance weight. We reverse the node importance with  $I'(v) = \frac{1}{I(v) + \epsilon}, \epsilon \to 0$ , normalize it to obtain node  $v_i$ 's sampling probabilities  $p_i = \frac{I'(v_i)}{\sum_{j=1}^{n} I'(v_j)}$ , and perform weighted sampling function WEIGHTED SAMPLING( $G_{\tau}^{\mathcal{R}}, p_i$ ) to prioritize nodes with higher sampling probability (lower importance) according to  $p_i$ . When sampling, for each master node  $v_m$ , we generate its  $k$ -hop neighbors, termed an ego net  $G_{\tau}^e(v_m)$ . Given the constrained size of the resource graph, we adopt data augmentation techniques commonly used in contrastive learning [56, 118, 117] to enhance the representativeness and diversity of the resultant toy graphs.

Toy Graphs Augmentation Strategy. For augmentation, we first calculate the average reversed importance  $\bar{I}^{\prime}(G_{\tau}^{e}(v_m))$  of the nodes within an ego graph as  $\bar{I}^{\prime}(G_{\tau}^{e}(v_m)) = \frac{1}{|G_{\tau}^{e}(v_m)|}\sum_{v\in G_{\tau}^{e}(v_m)}I^{\prime}(v)$ , which then determines the number of augmentations  $n_{\mathrm{aug}}(G_{\tau}^{e}(v_m)) = \lfloor K\cdot \bar{I}^{\prime}(G_{\tau}^{e}(v_m))\rfloor$ , where  $K$  is a scaling constant that adjusts the intensity of the augmentation. For node  $v_{i}, v_{j}\in G_{\tau}^{e}(v_{m})$ , the augmentation techniques DATA AUGMENTATION  $(G_{\tau}^{e}(v_m), n_{\mathrm{aug}})$  employed include:

-  $\bullet$  Node Dropout:  $v_{i} \in G_{\tau}^{e}(v_{m})$  has a probability of being dropped:  $p(v_{i}$  being dropped) = 1 -  $p_{i}$ .  
- ② Addition of Gaussian Noise: we add gaussian noise to node features as augmentation  $X'(v_i) = X(v_i) + \mathcal{N}(0, \sigma^2)$ .  
-  $③$  Node Interpolation: a new node feature  $X^{\prime}(v_{new})$  is created by linearly combining the features of two existing nodes  $v_{i}$  and  $v_{j}$ , calculated as  $X^{\prime}(v_{new}) = \lambda X(v_{i}) + (1 - \lambda)X(v_{j}), v_{i}, v_{j} \in G^{\mathcal{T}}$ . And the edge weight between the new node  $v_{new}$  and node  $v_{i}$  is updated to  $\lambda A[i,j]$  and node  $v_{j}$  is  $(1 - \lambda)A[i,j]$  accordingly [109].  
- 4 Edge Rewriting: we alter connections based on the average of the involved nodes' sampling probabilities, expressed as  $p((v_i, v_j)$  being rewired) =  $\frac{p_i + p_j}{2}$ .

Key-Value Pairs Construction. After completing the sampling and augmentation procedures, the generated toy graphs are transformed into key-value pairs for storage [110]. Specifically, we collect each master node's  $v_{m}$  historical information (t timestamps  $\tau$ ), environmental information (neighbors  $\mathcal{N}(v_m^\tau)$ ), structural encodings  $s_m^\tau$  (as described in the Appendix C.2), and the hidden embeddings  $h_m^\tau$  (obtained by processing the toy graph through the frozen pre-trained GNNs) and store them as keys at the master node  $v_{m}$  of the toy graph. Additionally, we store task-specific output vectors  $\{o_i^\tau |v_i\in G^T\}$  and hidden embeddings  $\{h_i^\tau |v_i\in G^T\}$  as values at each node of the toy graph. For storage of these key-value pairs, we utilize the FAISS vector library [14] to facilitate accelerated retrieval and storage.

# 4.2 Toy Graphs Retrieval Process

After constructing the key-value toy graphs vector database, we proceed with the retrieval process for sub-tasks according to the four sub-similarities between the key values of the master node  $v_{m}$  in the toy graph and the center node  $v_{c}$  in the query graph, as detailed in Appendix C.3. The final similarity score is a weighted combination of these factors, and the top  $K$  toy graphs are selected as the retrieval results:

$$
S \left(v _ {c}, v _ {m}\right) = \mathbf {w} \times \left[ S _ {\text {t i m e}} \left(v _ {c}, v _ {m}\right), S _ {\text {s t r u c t u r e}} \left(v _ {c}, v _ {m}\right), S _ {\text {e n v i r o n m e n t}} \left(v _ {c}, v _ {m}\right), S _ {\text {s e m a n t i c}} \left(v _ {c}, v _ {m}\right) \right] ^ {\mathrm {T}}, \tag {1}
$$

where  $\mathbf{w} = [w_1, w_2, w_3, w_4]$  are the hyper-parameterized weights attributed to the time, structure, environment, and semantic similarities, respectively. Using this composite similarity, we rank and retrieve the top  $K$  toy graphs:

$$
G _ {\mathrm {T o p K}} ^ {\mathcal {T}} = \operatorname {T o p} \mathrm {K} _ {G ^ {\mathcal {T}} \in \mathcal {G} ^ {\mathcal {T}}} S \left(v _ {c}, v _ {m}\right), \tag {2}
$$

where  $G_{\mathrm{topK}}^T$  represents the subset of toy graphs that best match the query based on the combined criteria. This process ensures that we retrieve the most relevant toy graphs based on a comprehensive similarity measure, incorporating historical, structural, and environmental information.

# 4.3 Training and Inference

In Section 4.3.1, we detail the Knowledge Injection Propagation process, which includes two distinct propagation manners. Next, in Section 4.3.2, we present our approach for combining the retrieved hidden embeddings with the task-specific output vectors. Additionally, to enhance the robustness of RAGRAPH, a noise-based prompt tuning strategy is introduced in Section 4.3.3.

# 4.3.1 Knowledge Injection Propagation

After retrieving the top  $K$  toy graphs  $G_{\mathrm{TopK}}^T$ , knowledge, specifically the task-specific output vectors  $O$  and hidden embeddings  $H$ , is propagated from these toy graphs to the master nodes (Toy Graph Intra Propagation) and then to the center node  $v_{c}$  (Query-Toy Graph Inter Propagation). This propagation utilizes message-passing mechanisms via GNNs (cf. Appendix C.1). Each master node  $v_{m}$  in the toy graphs is connected to the center node  $v_{c}$  of the query graph based on the similarity scores  $S(v_{c},v_{m})$  computed in Eq.(1) and the connection weights dictate the influence of each toy graph, ensuring that graphs with higher similarity have a more substantial impact. This process can be implemented using either a parameter-free or a learnable approach. Moreover, it is worth noting that for learnable methods, the parameters of GNN are different.

$\bullet$  Toy Graph Intra Propagation Within each toy graph, information  $\mathbf{z}$  is propagated from neighbors to the master node using pre-trained GNNs. The task-specific output vectors  $o$  and hidden embeddings  $h$  from the neighbors are aggregated and transmitted to the master node. For each node  $v_{i}$  in a toy graph  $G^T$ , the GNN aggregates information from its neighbors  $\mathcal{N}(v_i)$  to update the master node  $v_{m}$ :

$$
\mathbf {z} _ {m} = \operatorname {G N N} \left(\left\{\mathbf {z} _ {i} \mid v _ {i} \in \mathcal {N} \left(v _ {m}\right) \right\}\right), \tag {3}
$$

where  $\mathbf{z}_i$  and  $\mathbf{z}_m$  represent the hidden embeddings  $h_i, h_m$  or task-specific output vectors  $o_i, o_m$  of the neighbor nodes and master node, respectively. For parameter-free situations, we can prepare  $\mathbf{z}_m$  in advance when constructing the toy graph to improve inference efficiency.

$\Theta$  Query-Toy Graph Inter Propagation Next, information from the toy graphs is aggregated to the query graph. Specifically, during propagation, information  $\mathbf{z}$  from the neighbors and master node of the toy graph is propagated to the center node using the same pre-trained GNNs. For a center node  $v_{c}$  in the query graph  $G^{\mathcal{Q}}$ , the GNN aggregates hidden embeddings  $H$  from its neighbors  $\mathcal{N}(v_c)$  and the master node  $v_{m}$  from the toy graph:

$$
h _ {c} = \operatorname {G N N} \left(\left\{h _ {i} \mid v _ {i} \in \mathcal {N} \left(v _ {c}\right) \cup \left\{v _ {m} \right\} \right\}\right). \tag {4}
$$

When propagating the task-specific output vector  $O$ , only the master node's information is passed to the center node:

$$
o _ {c} = \mathrm {G N N} \left(\left\{o _ {i} \mid v _ {i} \in \left\{v _ {m} \right\} \right\}\right). \tag {5}
$$

For scenarios where the propagation mechanism is learnable, attention mechanisms can be adapted on the edges. In parameter-free scenarios—where there are no learnable weights—the attention on the edges is determined based on the edge weights from the previous resource graph.

# 4.3.2 Knowledge Fusion Layer

Finally, at the data fusion layer, the aggregated hidden embeddings  $H$  of the center node  $v_{c}$  are processed through the pre-trained GNN's decoder  $\mathrm{DECODER}(\cdot)$  to obtain an output vector  $O$ . This output vector is then combined with the aggregated task-specific output vector in a weighted manner to produce the final output for downstream tasks as illustrated in Definition 3. The combined output is formulated as follows:

$$
\hat {o} _ {c} = \gamma o _ {c} + (1 - \gamma) \mathrm {D E C O D E R} \left(h _ {c}\right), \tag {6}
$$

where  $\gamma$  is a reweighting hyper-parameter. The resulting vector  $\hat{o}_c$  is then utilized to perform node-, graph-, or edge-level tasks via a similarity function.

For the same task, the decoder can be directly used to generate outputs. For different tasks, the decoder can be masked, allowing the model to utilize pre-computed embeddings without additional training. Furthermore, the decoder can be fine-tuned to better meet the specific requirements of each task, providing both flexibility and optimized performance. This approach ensures that the model effectively integrates and leverages information from both the toy graphs and the query graph, enhancing its effectiveness in various downstream tasks through the use of the aggregated task-specific output vector.

# 4.3.3 Noise-based Graph Prompting Tuning

When prompt tuning, RAGraph employs the same prompt loss function  $\mathcal{L}_{\mathrm{prompt}}$  as the backbone model (e.g., GraphPro, GraphPrompt). However, to mitigate the challenge of noise retrieval—a common issue in traditional RAG where highly related but irrelevant data is often retrieved—we enhance the training process by incorporating noise data to bolster model robustness, motivated by [55]. Specifically, we implement two types of noise integration strategies:

- Inner-Toy-Graph Noise: This strategy involves artificially introducing irrelevant nodes ( $v_{j} \notin G_{\tau}^{e}(v_{m})$ ) into the toy graph during its construction, complementing other augmentation techniques.  
-  $②$  Toy-Graph Noise: Throughout the training phase, we not only retrieve the top  $K$  toy graphs that are most relevant but also deliberately include the bottom  $K$  toy graphs to incorporate noise knowledge.

The integration of these noise elements is intended to enhance the model's ability to distinguish relevant information from irrelevant information, significantly improving its robustness and overall performance in downstream tasks by noise training. However, during the inference stage, we do not incorporate the noise.

# 5 Experiments

In this section, we conduct a series of experiments to evaluate the performance of RAGRAPH against state-of-the-art baselines on three dynamic and five static datasets on three-level graph tasks. Further details and experiment results are provided in Appendix D.

# 5.1 Experimental Setup

Datasets. We use four static datasets PROTEINS, COX2, ENZYMES and BZR for graph classification and node classification, as well as three dynamic datasets TAOBAO, KOUBEI and AMAZON for link prediction. More details about these datasets can be found in Table 4 in Appendix D.1.

Methods and Baselines. We consider three versions of our proposed framework RAGRAPH: 1) RAGRAPH/NF, which indicates we utilize the plug-and-plag RAGRAPH without fine-tuning on the train set; 2) RAGRAPH/FT, which employs prompt tuning on the train set with RAG; and 3) RAGRAPH/NFT, which applies noise prompt tuning on the train set with RAG. For the baseline of the dynamic graph, we choose LightGCN [31], SGL [104], MixGCF [41], SimGCL [118], GraphPro [114] and GraphPro+PRODIGY [39]. For the static graph, we choose GCN [50], GraphSAGE [30], GAT [98], GIN [106], GraphPrompt [67], GraphPrompt+PRODIGY [39] as baselines. In addition, we denote 'NF' and '/FT' respectively to represent without fine-tuning and fine-tuning. A detailed description of baselines can be referred to in Appendix D.3.

Settings and Evaluation. We establish a training-resource split with the remainder of the data reserved as unseen during fine-tuning. For static graphs, the split is based on node partitioning with the ratio of  $50\%:30\%$  [67], while for dynamic graphs, it is based on partitioning snapshots with the history snapshots as resource graph [39]. For fair comparisons, for methods employing PRODIGY and RAGRAPH, we fine-tune models using the training set while retrieving the resource graph to prevent information leakage and over-fitting; when testing, we retrieve the combined training and resource graphs. For other methods, fine-tuning was directly performed on the combined train and resource set for fairness. For the evaluation of static graphs, we refer GraphPrompt, utilizing pre-trained GNNs for both node- and graph-level tasks within a  $k$ -shot classification framework. For dynamic graphs, we follow GraphPro to employ pre-trained GNNs on a substantial dataset fraction, with fine-tuning and testing conducted on later snapshots. Moreover, we pre-train GraphPro and GraphPrompt unsupervised on other datasets within the similar domain following [67, 39] to avoid information leakage. For classification tasks, we utilize the accuracy as evaluation matric; For link prediction tasks, we use standard metrics Recall@k and nDCG@k at  $k = 20$ , in line with existing methodologies [31, 104, 118]. The metrics used in the experiment are detailed in Appendix D.2 and the implementation details of RAGRAPH and baselines are in Appendix D.4.

Table 1: Accuracy evaluation on node and graph classification. All tabular results  $(\%)$  are in mean  $\pm$  standard deviation across five seeds run, with best bolded and runner-up underlined.  

<table><tr><td rowspan="2">Methods</td><td colspan="2">Node Classification</td><td colspan="4">Graph Classification</td></tr><tr><td>PROTEINS (5-shot)</td><td>ENZYMES (5-shot)</td><td>PROTEINS (5-shot)</td><td>COX2 (5-shot)</td><td>ENZYMES (5-shot)</td><td>BZR (5-shot)</td></tr><tr><td>GCN</td><td>46.63±03.04</td><td>52.80±12.89</td><td>54.80±06.64</td><td>67.87±03.39</td><td>22.67±05.20</td><td>58.76±05.08</td></tr><tr><td>GraphSAGE</td><td>48.87±02.64</td><td>48.75±01.59</td><td>52.99±10.57</td><td>67.02±05.42</td><td>21.17±05.49</td><td>58.27±04.79</td></tr><tr><td>GAT</td><td>48.13±07.90</td><td>47.75±01.23</td><td>55.82±07.31</td><td>64.89±03.23</td><td>20.67±03.27</td><td>57.04±06.70</td></tr><tr><td>GIN</td><td>49.61±01.58</td><td>48.82±01.58</td><td>56.17±08.58</td><td>62.77±02.85</td><td>19.00±03.74</td><td>56.54±04.20</td></tr><tr><td colspan="7">GraphPrompt+</td></tr><tr><td>Vanilla/NF</td><td>44.88±13.17</td><td>48.81±01.88</td><td>56.68±03.63</td><td>53.04±04.13</td><td>36.50±03.31</td><td>68.77±03.44</td></tr><tr><td>Vanilla/FT</td><td>48.99±01.88</td><td>51.99±01.36</td><td>57.04±03.88</td><td>64.04±08.20</td><td>40.00±04.36</td><td>69.01±02.21</td></tr><tr><td>PRODIGY/NF</td><td>47.32±08.12</td><td>43.80±14.03</td><td>53.48±06.72</td><td>53.97±10.34</td><td>22.12±13.84</td><td>67.18±08.93</td></tr><tr><td>PRODIGY/FT</td><td>53.26±06.42</td><td>57.98±12.37</td><td>57.14±10.34</td><td>65.31±04.28</td><td>25.94±05.12</td><td>68.08±06.68</td></tr><tr><td>RAGRAPH/NF</td><td>56.12±04.11</td><td>75.92±01.72</td><td>58.48±03.93</td><td>55.32±04.15</td><td>38.17±03.39</td><td>77.53±05.26</td></tr><tr><td>RAGRAPH/FT</td><td>58.74±00.87</td><td>75.74±01.92</td><td>62.33±02.52</td><td>76.60±02.30</td><td>47.71±06.88</td><td>76.79±05.02</td></tr><tr><td>RAGRAPH/NFT</td><td>59.83±00.40</td><td>76.23±01.63</td><td>59.08±03.50</td><td>71.70±04.29</td><td>49.17±04.64</td><td>74.81±04.25</td></tr></table>

# 5.2 Retrieval-Augmented Graph Results

As discussed, we conduct experiments and report the results of the three graph tasks for static graph and dynamic graph, as illustrated in Table 1 and Table 2. From the reported accuracy, we can find the following observations:

Outperforming SOTA Methods. First, our proposed RAGRAPH outperforms almost all the baselines across the three graph tasks, demonstrating the effectiveness of RAGRAPH in transferring knowledge from the pre-training to downstream tasks compared to traditional GNNs i.e., GCN and GraphSAGE. It achieves the highest average accuracy across almost all tasks on ENZYMES, with an improvement of at least  $5.19\%$  in the static graph, and up to  $11.78\%$  on the dynamic graph over the best baseline PRODIGY/FT. We argue that by virtue of the integration of hidden embedding and task-specific output vector, RAGRAPH is able to comprehend more knowledge than simply learns the paradigm from  $X \rightarrow Y$ . Second, compared with the models of PRODIGY/NF and RAGRAPH/NF, the introduction of noise training in noise prompt tuning also improves the robustness of the model, avoiding the influence of a large amount of noise on the information aggregation inside the query graph.

Strong Retrieval-Augmented Performance on Unseen Datasets. We observe that PRODIGY/NF and RAGRAPH/NF are better to Vanilla/NF, indicating that the retrieval knowledge truly works when testing on unseen datasets. Moreover, the difference between PRODIGY/NF and PRODIGY/FT is much greater than that of RAGRAPH, which also indicates that a simple learning paradigm for ICL is not enough and that RAGRAPH can achieve acceptable results even on unseen downstream datasets without the need for sophisticated fine-tuning.

Figure 3: Hyper-parameter study with hopsk (Left) from 1 to 5 andtopk from 1 to 20 (Right) on node classification with PROTEINS, and ENZYMES datasets with the setting in Table 1.


Table 2: Performance evaluation  $(\%)$  on link prediction.  

<table><tr><td rowspan="2">Method</td><td colspan="2">TAOBAO</td><td colspan="2">KOUBEI</td><td colspan="2">AMAZON</td></tr><tr><td>Recall</td><td>nDCG</td><td>Recall</td><td>nDCG</td><td>Recall</td><td>nDCG</td></tr><tr><td>LightGCN</td><td>22.47±02.53</td><td>21.89±02.80</td><td>30.21±06.45</td><td>22.24±05.83</td><td>15.07±06.48</td><td>06.53±02.66</td></tr><tr><td>SGL</td><td>22.15±02.20</td><td>22.12±03.09</td><td>32.61±04.27</td><td>22.36±04.82</td><td>15.78±07.12</td><td>07.90±02.49</td></tr><tr><td>MixGCF</td><td>22.84±02.15</td><td>23.05±03.87</td><td>32.06±04.20</td><td>22.49±06.91</td><td>15.24±08.98</td><td>07.40±03.44</td></tr><tr><td>SimGCL</td><td>22.18±02.22</td><td>23.15±02.75</td><td>33.07±05.28</td><td>23.08±05.55</td><td>16.10±07.91</td><td>07.58±03.51</td></tr><tr><td>GraphPro+</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla/NF</td><td>20.10±01.50</td><td>20.12±01.30</td><td>21.31±04.59</td><td>15.31±03.11</td><td>12.56±07.45</td><td>06.31±03.92</td></tr><tr><td>Vanilla/FT</td><td>23.99±02.11</td><td>23.26±01.42</td><td>33.96±04.13</td><td>24.66±02.78</td><td>18.14±07.55</td><td>08.73±03.74</td></tr><tr><td>PRODIGY/NF</td><td>21.67±01.42</td><td>23.15±03.20</td><td>21.66±03.21</td><td>14.82±03.92</td><td>11.88±02.61</td><td>05.84±01.84</td></tr><tr><td>PRODIGY/FT</td><td>23.74±01.22</td><td>23.65±02.31</td><td>33.46±04.70</td><td>23.28±03.40</td><td>16.72±04.28</td><td>08.09±02.66</td></tr><tr><td>RAGRAPH/NF</td><td>20.31±01.60</td><td>20.45±01.44</td><td>22.86±03.44</td><td>16.68±02.48</td><td>13.78±05.54</td><td>06.52±02.69</td></tr><tr><td>RAGRAPH/FT</td><td>24.78±01.93</td><td>24.35±01.34</td><td>34.27±03.93</td><td>24.82±02.69</td><td>18.69±07.45</td><td>09.09±03.89</td></tr><tr><td>RAGRAPH/NFT</td><td>24.89±02.01</td><td>24.51±01.44</td><td>33.27±04.37</td><td>24.09±03.14</td><td>18.32±06.91</td><td>09.01±03.67</td></tr></table>

# 5.3 Hyper-parameter Study

In this section, we examine the impact of various hyper-parameters on RAGRAPH. We specifically analyze the effects of varying the number of hops  $k$  in toy graphs from the list [1,2,3,4,5] and the number of linked toy graphs topK from the list [1,5,10,15,30,50] to verify the sensitive:

Figure 3 (Left) illustrates relationships between accuracy and the toy graph hop  $k$ . We observe that as  $k$  increases, the volume of retrieved knowledge grows exponentially. However, an excessive accumulation of knowledge not only fails to enhance accuracy but also introduces increased irrelevant noise that burdens the GNNs. Notably, accuracy shows a trend of initial improvement followed by a decline as  $k$  is increased. This pattern suggests that at lower  $k$  values, the retrieved information tends to consist of isolated, less useful knowledge. In contrast, at higher  $k$  values, the GNNs struggle to process extensive reasoning chains, leading to the utilization of complex and abundant information that is less effective than even the baseline model's performance. Figure 3 (Right) shows effects on accuracy with different numbers of toy graphs topK. As with the previous figure, increasing topK demonstrates that an excessive amount of knowledge can hinder the GNNs' comprehension capabilities. Conversely, smaller topK results in insufficient knowledge to enhance performance on downstream tasks.

# 6 Conclusion

We introduced RAGRAP, a novel and general framework that enhances Graph Neural Networks (GNNs) by integrating Retrieval-Augmented Generation (RAG) techniques. This plug-and-play approach improves GNNs' ability to generalize to unseen data by retrieving relevant information. Experimental results show that RAGRAP outperforms state-of-the-art methods in various graph learning tasks, demonstrating its adaptability and robustness. While RAGRAP is currently limited to retrieving subgraphs, future research could explore using more graph-structured data such as nodes, edges, and trees to further enhance its capabilities. In general, our work provides valuable insights and serves as a reference for future Large Graph Models.

# Acknowledgments

This work is supported by the National Natural Science Foundation of China (No.U23A20468).

# References

[1] A. Asai, S. Min, Z. Zhong, and D. Chen. Retrieval-based language models and applications. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts), pages 41-46, 2023.  
[2] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In ICLR, 2024.  
[3] S.-V. Bogolin, I. Croitoru, H. Jin, Y. Liu, and S. Albanie. Cross modal retrieval with querybank normalisation. In CVPR, 2022.  
[4] K. M. Borgwardt, C. S. Ong, S. Schonauer, S. Vishwanathan, A. J. Smola, and H.-P. Kriegel. Protein function prediction via graph kernels. Bioinformatics, 21(suppl_1):i47-i56, 2005.  
[5] H. Cai, V. W. Zheng, and K. C.-C. Chang. A comprehensive survey of graph embedding: Problems, techniques, and applications. IEEE TKDE, 2018.  
[6] D. M. Chan, S. Ghosh, A. Rastrow, and B. Hoffmeister. Using external off-policy speech-to-text mappings in contextual end-to-end automated speech recognition, 2023.  
[7] M. Chen, W. Zhang, W. Zhang, Q. Chen, and H. Chen. Meta relational learning for few-shot link prediction in knowledge graphs. arXiv preprint arXiv:1909.01515, 2019.  
[8] W. Chen, H. Hu, X. Chen, P. Verga, and W. W. Cohen. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. In EMNLP, 2022.  
[9] X. Chen, S. Zhang, Y. Xiong, X. Wu, J. Zhang, X. Sun, Y. Zhang, Y. Zhao, and Y. Kang. Prompt learning on temporal interaction graphs. arXiv:2402.06326, 2024.  
[10] Z. Cheng, J. Zhang, X. Xu, G. Trajcevski, T. Zhong, and F. Zhou. Retrieval-augmented hypergraph for multimodal social media popularity prediction. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD '24, page 445-455, New York, NY, USA, 2024. Association for Computing Machinery.  
[11] X. Chu, Y. Jin, X. Wang, S. Zhang, Y. Wang, W. Zhu, and H. Mei. Wasserstein barycenter matching for graph size generalization of message passing neural networks. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org, 2023.  
[12] F. Cuonasu, G. Trappolini, F. Siciliano, S. Filice, C. Campagnano, Y. Maarek, N. Tonellotto, and F. Silvestri. The power of noise: Redefining retrieval for rag systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, volume 17 of SIGIR 2024, page 719-729. ACM, July 2024.  
[13] H. Cui, Z. Lu, P. Li, and C. Yang. On positional and structural node features for graph neural networks on non-attributed graphs, 2021.  
[14] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou. The faiss library, 2024.  
[15] Y. Duan, G. Zhang, S. Wang, X. Peng, Z. Wang, J. Mao, H. Wu, X. Jiang, and K. Wang. Cat-gnn: Enhancing credit card fraud detection via causal temporal graph neural networks. ArXiv, abs/2402.14708, 2024.  
[16] Y. Duan, J. Zhao, pengcheng, J. Mao, H. Wu, J. Xu, S. Wang, C. Ma, K. Wang, K. Wang, and X. Li. Causal deciphering and inpainting in spatio-temporal dynamics via diffusion model. 2024.  
[17] D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, and R. P. Adams. Convolutional networks on graphs for learning molecular fingerprints. In NeurIPS, 2015.  
[18] P. L. Eli Chien, Jianhao Peng and O. Milenkovic. Adaptive universal generalized pagerank graph neural network. In ICLR, 2021.

[19] F. Fang, Y. Bai, S. Ni, M. Yang, X. Chen, and R. Xu. Enhancing noise robustness of retrieval-augmented language models with adaptive adversarial training, 2024.  
[20] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen. Universal prompt tuning for graph neural networks. In NeurIPS, 2024.  
[21] Y. Fang, Y. Liang, B. Hui, Z. Shao, L. Deng, X. Liu, X. Jiang, and K. Zheng. Efficient large-scale traffic forecasting with transformers: A spatial data management perspective. In Inproceedings of SIGKDD, 2025.  
[22] Y. Fang, Y. Qin, H. Luo, F. Zhao, B. Xu, L. Zeng, and C. Wang. When spatio-temporal meet wavelets: Disentangled traffic forecasting via efficient spectral graph attention networks. In ICDE, 2023.  
[23] L. Gao, X. Ma, J. Lin, and J. Callan. Precise zero-shot dense retrieval without relevance labels, 2022.  
[24] X. Gao, H. Chen, and J. Haworth. A spatiotemporal analysis of the impact of lockdown and coronavirus on London's bicycle hire scheme: from response to recovery to a new normal. GIS, 2023.  
[25] X. Gao, J. Haworth, D. Zhuang, H. Chen, and X. Jiang. Uncertainty quantification in the road-level traffic risk prediction by spatial-temporal zero-inflated negative binomial graph neural network (stzinb-gnn). *GIScience* 2023, 2023.  
[26] X. Gao, X. Jiang, J. Haworth, D. Zhuang, S. Wang, H. Chen, and S. Law. Uncertainty-aware probabilistic graph neural networks for road-level traffic crash prediction. *Accident Analysis & Prevention*, 208:107801, 2024.  
[27] Z. Gao, X. Sun, Z. Liu, Y. Li, H. Cheng, and J. Li. Protein multimer structure prediction via PPI-guided prompt learning. In The Twelfth International Conference on Learning Representations (ICLR), 2024.  
[28] M. Gjoka, M. Kurant, C. T. Butts, and A. Markopoulou. Walking in facebook: A case study of unbiased sampling of osns. In 2010 Proceedings IEEE INFOCOM, pages 1-9, 2010.  
[29] A. Grover and J. Leskovec. node2vec: Scalable feature learning for networks. In SIGKDD, 2016.  
[30] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In NeurIPS, 2017.  
[31] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang. Lightgcn: Simplifying and powering graph convolution network for recommendation. In SIGIR, 2020.  
[32] Z. He, W. Hao, W.-T. Lu, C. Chen, K. Lerman, and X. Song. Alcap: Alignment-augmented music captioner. In ICASSP, 2023.  
[33] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models, 2021.  
[34] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec. Strategies for pre-training graph neural networks. arXiv preprint arXiv:1905.12265, 2019.  
[35] Z. Hu, Y. Dong, K. Wang, K.-W. Chang, and Y. Sun. Gpt-gnn: Generative pre-training of graph neural networks. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1857-1867, 2020.  
[36] J. Huang, G. Cai, J. Zhu, Z. Dong, R. Tang, W. Zhang, and Y. Yu. Recall-augmented ranking: Enhancing click-through rate prediction accuracy with cross-stage data. In WWW, 2024.  
[37] K. Huang and M. Zitnik. Graph meta learning via local subgraphs, 2020.  
[38] M. Huang, Y. Liu, X. Ao, K. Li, J. Chi, J. Feng, H. Yang, and Q. He. Auc-oriented graph neural network for fraud detection. In WWW, 2022.

[39] Q. Huang, H. Ren, P. Chen, G. Kržmanc, D. Zeng, P. Liang, and J. Leskovec. Prodigy: Enabling in-context learning over graphs, 2023.  
[40] Q. Huang, H. Ren, and J. Leskovec. Few-shot relational reasoning via connection subgraph pretraining. In NeurIPS, 2022.  
[41] T. Huang, Y. Dong, M. Ding, Z. Yang, W. Feng, X. Wang, and J. Tang. Mixgcf: An improved training method for graph neural network-based recommender systems. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 665-674, 2021.  
[42] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave. Atlas: Few-shot learning with retrieval augmented language models, 2022.  
[43] X. Jiang, Y. Fang, R. Qiu, H. Zhang, Y. Xu, H. Chen, W. Zhang, R. Zhang, Y. Fang, X. Chu, J. Zhao, and Y. Wang. Tc-rag:turing-complete rag's case study on medical llm systems. ArXiv, abs/2408.09199, 2024.  
[44] X. Jiang, Z. Qin, J. Xu, and X. Ao. Incomplete graph learning via attribute-structure decoupled variational auto-encoder. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, WSDM '24, page 304–312, New York, NY, USA, 2024. Association for Computing Machinery.  
[45] X. Jiang, R. Zhang, Y. Xu, R. Qiu, Y. Fang, Z. Wang, J. Tang, H. Ding, X. Chu, J. Zhao, and Y. Wang. Hykge: A hypothesis knowledge graph enhanced framework for accurate and reliable medical llms responses, 2024.  
[46] X. Jiang, D. Zhuang, X. Zhang, H. Chen, J. Luo, and X. Gao. Uncertainty quantification via spatial-temporal tweedie model for zero-inflated and long-tail travel demand prediction. In CIKM, 2023.  
[47] R. Y. Jiaxuan You and J. Leskovec. Position-aware graph neural networks. In ICML, 2019.  
[48] B. Jin, Y. Zhang, Q. Zhu, and J. Han. Heterformer: Transformer-based deep node representation learning on heterogeneous text-rich networks. In SIGKDD, pages 1020-1031, 2023.  
[49] J. Kim, S. M. Jaehyun Nam, J. Park, S.-W. Lee, M. Seo, J.-W. Ha, and J. Shin. Sure: Summarizing retrievals using answer candidates for open-domain qa of llms. In ICLR, 2024.  
[50] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2016.  
[51] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. tau Yih, T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021.  
[52] G. Li, M. Müller, A. Thabet, and B. Ghanem. Deep GCs: Can GCs go as deep as cnns? In ICCV, 2019.  
[53] H. Li, X. Wang, Z. Zhang, and W. Zhu. Out-of-distribution generalization on graphs: A survey, 2022.  
[54] K. Li, Y. Chen, Y. Liu, J. Wang, Q. He, M. Cheng, and X. Ao. Boosting the adversarial robustness of graph neural networks: An ood perspective. In International Conference on Learning Representations, 2024.  
[55] K. Li, Y. Liu, X. Ao, J. Chi, J. Feng, H. Yang, and Q. He. Reliable representations make a stronger defender: Unsupervised structure refinement for robust gnn. In SIGKDD, 2022.  
[56] R. Li, T. Zhong, X. Jiang, G. Trajcevski, J. Wu, and F. Zhou. Mining spatio-temporal relations via self-paced graph contrastive learning. In SIGKDD, 2022.  
[57] S. Li, R. Xie, Y. Zhu, X. Ao, F. Zhuang, and Q. He. User-centric conversational recommendation with multi-aspect user modeling. In SIGIR, 2022.

[58] X. Li, D. Lian, Z. Lu, J. Bai, Z. Chen, and X. Wang. Graphadapter: Tuning vision-language models with dual knowledge graph. In NeurIPS, volume 36, 2024.  
[59] X. Li, R. Zhao, Y. K. Chia, B. Ding, S. Joty, S. Poria, and L. Bing. Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous sources, 2023.  
[60] W. Lin and B. Byrne. Retrieval augmented visual question answering with outside knowledge. In EMNLP, 2022.  
[61] W. Lin, J. Mei, J. Chen, and B. Byrne. Preflmr: Scaling up fine-grained late-interaction multi-modal retrievers, 2024.  
[62] X. V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James, P. Rodriguez, J. Kahn, G. Szilvasy, M. Lewis, L. Zettlemoyer, and S. Yih. Ra-dit: Retrieval-augmented dual instruction tuning. In ICLR, 2024.  
[63] Y. Liu, X. Ao, F. Feng, and Q. He. Ud-gnn: Uncertainty-aware debiased training on semi-homophilous graphs. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1131-1140, 2022.  
[64] Y. Liu, X. Ao, F. Feng, Y. Ma, K. Li, T.-S. Chua, and Q. He. Flood: A flexible invariant learning framework for out-of-distribution generalization on graphs. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1548–1558, 2023.  
[65] Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, and Q. He. Pick and choose: a gnn-based imbalanced learning approach for fraud detection. In Proceedings of the Web Conference 2021, pages 3168-3177, 2021.  
[66] Y. Liu, K. Zhang, Y. Li, Z. Yan, C. Gao, R. Chen, Z. Yuan, Y. Huang, H. Sun, J. Gao, L. He, and L. Sun. Sora: A review on background, technology, limitations, and opportunities of large vision models, 2024.  
[67] Z. Liu, X. Yu, Y. Fang, and X. Zhang. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. In WWW, 2023.  
[68] S. Lu, N. Duan, H. Han, D. Guo, S. won Hwang, and A. Svyatkovskiy. Reacc: A retrieval-augmented code completion framework. In ACL, 2022.  
[69] H. Luo, Y.-S. Chuang, Y. Gong, T. Zhang, Y. Kim, X. Wu, D. Fox, H. Meng, and J. Glass. Sail: Search-augmented instruction learning, 2023.  
[70] J. Luo, W. Zhang, Y. Fang, X. Gao, D. Zhuang, H. Chen, and X. Jiang. Timeseries suppliers allocation risk optimization via deep black litterman model. ArXiv, abs/2401.17350, 2024.  
[71] K. Ma, H. Cheng, Y. Zhang, X. Liu, E. Nyberg, and J. Gao. Chain-of-skills: A configurable model for open-domain question answering, 2023.  
[72] H. Mao, X. Chen, Q. Fu, L. Du, S. Han, and D. Zhang. Neuron campaign for initialization guided by information bottleneck theory. In CIKM, 2021.  
[73] S. Maskey, R. Levie, Y. Lee, and G. Kutyniok. Generalization analysis of message passing neural networks on large random graphs. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 4805-4817. Curran Associates, Inc., 2022.  
[74] S. Matsugu, Y. Fujiwara, and H. Shiokawa. Uncovering the largest community in social networks at scale. In *IJCAI*, 2023.  
[75] OpenAI. Introducing chatgpt. https://openai.com/blog/chatgpt, 2022.  
[76] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.  
[77] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering, 2024.

[78] S. Parashar, Z. Lin, T. Liu, X. Dong, Y. Li, D. Ramanan, J. Caverlee, and S. Kong. The neglected tails of vision-language models. In CVPR, 2024.  
[79] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.  
[80] B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social representations. In SIGKDD, 2014.  
[81] Y. Qin, Y. Fang, H. Luo, F. Zhao, and C. Wang. Next point-of-interest recommendation with auto-correlation enhanced multi-modal transformer network. In SIGIR, 2022.  
[82] J. Qiu, Q. Chen, Y. Dong, J. Zhang, H. Yang, M. Ding, K. Wang, and J. Tang. GCC: Graph contrastive coding for graph neural network pre-training. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1150-1160, 2020.  
[83] Y. Qu, Y. Ding, J. Liu, K. Liu, R. Ren, W. X. Zhao, D. Dong, H. Wu, and H. Wang. Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering, 2021.  
[84] R. A. Rossi and N. K. Ahmed. The network data repository with interactive graph analytics and visualization. In AAAI Conference on Artificial Intelligence, pages 4292-4293, 2015.  
[85] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In ICLR, 2024.  
[86] S. Sharma and R. Sharma. Identifying possible rumor spreaders on twitter: A weak supervised learning approach. In 2021 International Joint Conference on Neural Networks (IJCNN), pages 1-8, 2021.  
[87] K. Soman, P. W. Rose, J. H. Morris, R. E. Akbas, B. Smith, B. Peetoom, C. Villouta-Reyes, G. Cerono, Y. Shi, A. Rizk-Jackson, S. Israni, C. A. Nelson, S. Huang, and S. E. Baranzini. Biomedical knowledge graph-enhanced prompt generation for large language models, 2023.  
[88] X. Su and T. M. Khoshgoftaar. A survey of collaborative filtering techniques. Adv. in Artif. Intell., 2009, Jan. 2009.  
[89] J. Sun, Y. Zhou, and C. Zong. One-shot relation learning for knowledge graphs via neighborhood aggregation and paths encoding. Transactions on Asian and Low-Resource Language Information Processing, 21(3):1–19, 2021.  
[90] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang. Gppt: Graph pre-training and prompt tuning to generalize graph neural networks. In SIGKDD, 2022.  
[91] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan. All in one: Multi-task prompting for graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (KDD'23), page 2120-2131, 2023.  
[92] X. Sun, J. Zhang, X. Wu, H. Cheng, Y. Xiong, and J. Li. Graph prompt learning: A comprehensive survey and beyond. arXiv:2311.16534, 2023.  
[93] Y. Tan, H. Lv, X. Huang, J. Zhang, S. Wang, and C. Yang. Musegraph: Graph-oriented instruction tuning of large language models for generic graph mining, 2024.  
[94] Y. Tan, Z. Zhou, H. Lv, W. Liu, and C. Yang. Walklm: A uniform language model fine-tuning framework for attributed graph embedding. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, NeurIPS, volume 36, pages 13308-13325. Curran Associates, Inc., 2023.  
[95] Z. Tan, R. Guo, K. Ding, and H. Liu. Virtual node tuning for few-shot node classification. In SIGKDD, 2023.

[96] J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line: Large-scale information network embedding. In WWW, 2015.  
[97] B. Teji, S. Roy, D. S. Dhami, D. Bhandari, and P. H. Guzzi. Graph embedding techniques for predicting missing links in biological networks: An empirical evaluation. IEEE Transactions on Emerging Topics in Computing, 12(1):190-201, 2024.  
[98] P. Velicković, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio. Graph attention networks. In ICLR, 2018.  
[99] S. Wang, K. Ding, C. Zhang, C. Chen, and J. Li. Task-adaptive few-shot node classification. Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022.  
[100] S. Wang, Y. Dong, X. Huang, C. Chen, and J. Li. FAITH: Few-shot graph classification with hierarchical task graphs. In International Joint Conference on Artificial Intelligence, 2022.  
[101] X. Wang, L. Zhu, and Y. Yang. T2vlad: Global-local sequence alignment for text-video retrieval. In CVPR, 2021.  
[102] Y. Wang, R. Ren, J. Li, W. X. Zhao, J. Liu, and J. Wen. Rear: A relevance aware retrieval augmented framework for open-domain question answering, 2024.  
[103] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.  
[104] J. Wu, X. Wang, F. Feng, X. He, L. Chen, J. Lian, and X. Xie. Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pages 726-735, 2021.  
[105] L. Xia, B. Kao, and C. Huang. Opengraph: Towards open graph foundation models, 2024.  
[106] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In ICLR, 2019.  
[107] P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, S. Subramanian, E. Bakhturina, M. Shoybi, and B. Catanzaro. Retrieval meets long context large language models. In ICLR, 2024.  
[108] Y. Xu, X. Chu, K. Yang, Z. Wang, P. Zou, H. Ding, J. Zhao, Y. Wang, and B. Xie. Seqcare: Sequential training with external medical knowledge graph for diagnosis prediction in healthcare data. In Proceedings of the ACM Web Conference 2023, pages 2819–2830, 2023.  
[109] Y. Xu, X. Jiang, X. Chu, Y. Xiao, C. Zhang, H. Ding, J. Zhao, Y. Wang, and B. Xie. *Protomix: Augmenting health status representation learning via prototype-based mixup*. In *Knowledge Discovery and Data Mining*, 2024.  
[110] Y. Xu, K. Yang, C. Zhang, P. Zou, Z. Wang, H. Ding, J. Zhao, Y. Wang, and B. Xie. Vecocare: Visit sequences-clinical notes joint learning for diagnosis prediction in healthcare data. In *IJCAI*, volume 23, pages 4921–4929, 2023.  
[111] Y. Xu*, R. Zhang*, X. Jiang*, Y. Feng, Y. Xiao, X. Ma, R. Zhu, X. Chu, J. Zhao, and Y. Wang. Parenting: Optimizing knowledge selection of retrieval-augmented language models with parameter decoupling and tailored tuning. arXiv preprint arXiv:2410.10360, 2024.  
[112] K. Yang, Y. Xu, P. Zou, H. Ding, J. Zhao, Y. Wang, and B. Xie. Kerprint: local-global knowledge graph enhanced diagnosis prediction for retrospective and prospective interpretations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 5357-5365, 2023.  
[113] S. Yang, X. Jiang, H. Zhao, W. Zeng, H. Liu, and Y. Jia. Faima: Feature-aware in-context learning for multi-domain aspect-based sentiment analysis. In COLING, 2024.  
[114] Y. Yang, L. Xia, D. Luo, K. Lin, and C. Huang. Graphpro: Graph pre-training and prompt learning for recommendation. In WWW, 2024.

[115] J. Yoo, N. Ahn, and K.-A. Sohn. Rethinking data augmentation for image super-resolution: A comprehensive analysis and a new strategy. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8372-8381, 2020.  
[116] O. Yoran, T. Wolfson, O. Ram, and J. Berant. Making retrieval-augmented language models robust to irrelevant context. In ICLR, 2024.  
[117] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen. Graph contrastive learning with augmentations. Advances in neural information processing systems, 33:5812-5823, 2020.  
[118] J. Yu, H. Yin, X. Xia, T. Chen, L. Cui, and Q. V. H. Nguyen. Are graph augmentations necessary? simple graph contrastive learning for recommendation. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 1294–1303, 2022.  
[119] W. Yu, H. Zhang, X. Pan, K. Ma, H. Wang, and D. Yu. Chain-of-note: Enhancing robustness in retrieval-augmented language models, 2023.  
[120] X. Yu, Y. Fang, Z. Liu, Y. Wu, Z. Wen, J. Bo, X. Zhang, and S. C. Hoi. Few-shot learning on graphs: from meta-learning to pre-training and prompting. arXiv preprint arXiv:2402.01440, 2024.  
[121] X. Yu, Z. Liu, Y. Fang, Z. Liu, S. Chen, and X. Zhang. Generalized graph prompt: Toward a unification of pre-training and downstream tasks on graphs. IEEE Transactions on Knowledge and Data Engineering, 2024.  
[122] X. Yu, Z. Liu, Y. Fang, and X. Zhang. Dygprompt: Learning feature and time prompts on dynamic graphs. arXiv preprint arXiv:2405.13937, 2024.  
[123] X. Yu, J. Zhang, Y. Fang, and R. Jiang. Non-homophilic graph pre-training and prompt learning. arXiv preprint arXiv:2408.12594, 2024.  
[124] X. Yu, C. Zhou, Y. Fang, and X. Zhang. Multigprompt for multi-task pre-training and prompting on graphs. In WWW, 2024.  
[125] Z. Yuan, Q. Jin, C. Tan, Z. Zhao, H. Yuan, F. Huang, and S. Huang. Ramm: Retrieval-augmented biomedical visual question answering with multi-modal pre-training. In CVPR, 2023.  
[126] C. Zhang, H. Yao, C. Huang, M. Jiang, Z. Li, and N. V. Chawla. Few-shot knowledge graph completion. In AAAI, 2020.  
[127] R. Zhang, X. Jiang, Y. Fang, J. Luo, Y. Xu, Y. Zhu, X. Chu, J. Zhao, and Y. Wang. Infinite-horizon graph filters: Leveraging power series to enhance sparse information aggregation. *ArXiv*, abs/2401.09943, 2024.  
[128] R. Zhang, Y. Xu, Y. Xiao, R. Zhu, X. Jiang, X. Chu, J. Zhao, and Y. Wang. Kapo: Knowledge-aware preference optimization for controllable knowledge selection in retrieval-augmented language models. arXiv preprint arXiv:2408.03297, 2024.  
[129] Y. Zhang, Z. Chen, Y. Fang, L. Cheng, Y. Lu, F. Li, W. Zhang, and H. Chen. Knowledgeable preference alignment for llms in domain-specific question answering, 2023.  
[130] Z. Zhang, H. Li, Z. Zhang, Y. Qin, X. Wang, and W. Zhu. Graph meets llms: Towards large graph models, 2023.  
[131] H. Zhao, A. Chen, X. Sun, H. Cheng, and J. Li. All in one and one for all: A simple yet effective method towards cross-domain graph pretraining, 2024.  
[132] R. Zhao, H. Chen, W. Wang, F. Jiao, X. L. Do, C. Qin, B. Ding, X. Guo, M. Li, X. Li, and S. Joty. Retrieving multimodal information for augmented generation: A survey, 2023.  
[133] J. Zhou, C. Xie, Z. Wen, X. Zhao, and Q. Xuan. Data augmentation on graphs: A technical survey, 2023.

[134] S. Zhou, U. Alon, F. F. Xu, Z. Wang, Z. Jiang, and G. Neubig. Docprompting: Generating code by retrieving the docs. In ICLR, 2023.  
[135] X. Zhou, R. Lumbantoruan, Y. Ren, L. Chen, X. Yang, and J. Shao. Dynamic bi-layer graph learning for context-aware sequential recommendation. ACM Trans. Recomm. Syst., 2(2), apr 2024.  
[136] F. Zhu, W. Lei, C. Wang, J. Zheng, S. Poria, and T.-S. Chua. Retrieving and reading: A comprehensive survey on open-domain question answering, 2021.  
[137] K. Zhu, X. Feng, X. Du, Y. Gu, W. Yu, H. Wang, Q. Chen, Z. Chu, J. Chen, and B. Qin. An information bottleneck perspective for effective noise filtering on retrieval-augmented generation. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1044–1069, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.  
[138] Y. Zhu, J. Guo, and S. Tang. Sgl-pt: A strong graph learner with graph prompt tuning, 2023.  
[139] Y. Zhu, Z. Ou, X. Mou, and J. Tang. Retrieval-augmented embodied agents. In CVPR, 2024.

# A Notations

The notations in this paper are summarized in Table 3.

Table 3: Notations Tables in RAGRAPH  

<table><tr><td>Notation</td><td>Definition</td></tr><tr><td>G/V/ε</td><td>The dynamic graph / node / edge set with Gt/Vt/Et as its entry</td></tr><tr><td>G^R</td><td>The resource graph with G^R as its entry</td></tr><tr><td>G^Q</td><td>The query graph with vc as its center node</td></tr><tr><td>G^T</td><td>The toy graph database</td></tr><tr><td>GTopK</td><td>The topK retrieved toy graphs</td></tr><tr><td>G^T</td><td>The toy graph</td></tr><tr><td>Ge(νm)</td><td>k-hop ego net for node vm</td></tr><tr><td>Xt/A_t/Y_t</td><td>The feature / edge weight / label matrix at time t</td></tr><tr><td>C</td><td>The set of label classes</td></tr><tr><td>N(v)</td><td>The neighbors of node v</td></tr><tr><td>H/O</td><td>The hidden embeddings / task-specific output vector with hi / oi as its vi vector</td></tr><tr><td>D</td><td>The κ-shot labeled set</td></tr><tr><td>t/τ</td><td>Timestamp</td></tr><tr><td>n</td><td>Number of nodes</td></tr><tr><td>d</td><td>The dimension of node feature</td></tr><tr><td>vi</td><td>The i-th node</td></tr><tr><td>vm/vc</td><td>The master node of toy graph / The center node of toy graph</td></tr><tr><td>f1/f2</td><td>The dimension of hidden embedding / task-specific output vector</td></tr><tr><td>ε</td><td>The minimum value</td></tr><tr><td>k</td><td>k hop</td></tr><tr><td>I(v)</td><td>The node importance for node v ∈ G^R</td></tr><tr><td>PR(·)/DC(·)</td><td>The PageRank / Degree Centrality value</td></tr><tr><td>pi</td><td>The sampling probability of node vi</td></tr><tr><td>K</td><td>The scaling constant</td></tr><tr><td>naug(G^T)</td><td>The number of augmentations of toy graph GT</td></tr><tr><td>S(vc,vm)</td><td>The weighted similarity between query node vc and master node vm</td></tr><tr><td>l</td><td>The layer of a GNN</td></tr><tr><td>α</td><td>The balance weight with α ∈ (0,1)</td></tr><tr><td>λ</td><td>The weight of mixup</td></tr><tr><td>wi</td><td>The weights of the time, structure, semantic, and environment similarities</td></tr><tr><td>γ</td><td>The reweight hyper-parameter</td></tr></table>

# B More Motivation Details

# B.1 Why Toy Graph Augmentation is needed

The reasons for toy graph augmentation:

- Expanding toy graph base, enriching the scale of the knowledge repository [115].  
- Simulating Real-World Scenarios: Real-world graphs often encounter challenges such as missing nodes [44], noisy attributes [54], and unexplored connections [97]. We introduce node dropout, noise injection, and edge removal to simulate these scenarios accurately.  
- Addressing Graph Domain Shift: To mitigate domain shift between the graph knowledge base and testing graphs, our augmentations employ Mixup techniques such as Node Interpolation and Edge Rewiring. These techniques interpolate between training samples to generate synthetic samples, effectively smoothing decision boundaries in embedding and reducing the model's sensitivity to minor variations in input data, thereby stabilizing predictions on domain shift testing samples [109].

# B.2 Why Noise-based Graph Prompt Tuning is needed

To address inherent challenges in toy graph quality, we introduce Noise-based Graph Prompting Tuning (c.f. Section 4.3.3). This method involves fine-tuning the model with artificially introduced noisy toy graphs (Inner-Toy-Graph Noise & Toy-Graph Noise), inspired by noise-tuning techniques in NLP [19, 12, 116]. Our approach enhances the model's robustness against real-world retrieval noise, as evidenced by superior performance compared to traditional tuning methods (in Main Text Tables 1 and 2). This approach reduces the stringent requirement for an exceptionally high-quality graph vector base, thereby ensuring robust performance across various tasks within our RAGRAPH, and significantly mitigating data quality impacts.

# B.3 Difficulty to construct and maintain high-quality and diverse graph vector base

In RAGRAPH, the toy graph base largely leverages significant prior research datasets in pre-trained GNNs [67, 105, 39, 124], which are trained on meticulously curated graph datasets and cover diverse domains, such as biology, chemistry, medicine recommendation tasks, etc. For example, the PROTEINS dataset [4], derived from cryo-electron microscopy and X-ray crystallography, and the ENZYMES dataset [100], based on EC enzyme classification, are meticulously annotated by medical experts.

# B.4 Why Inverse Importance Sampling Strategy is needed

The adoption of the Inverse Importance Sampling strategy is crucial. In RAGRAPH, subgraphs are sampled as toy graphs, where nodes with higher degrees (non-long-tail knowledge, extensively learned and embedded into GNN parameters) are more frequently included in subgraphs due to their extensive connections with neighbors, resulting in higher frequency in toy graph base [28]. Conversely, nodes with low degrees (long-tail knowledge), are more important but ignored. To mitigate this issue, we propose this by prioritizing nodes with lower degrees to capture long-tail knowledge when sampling.

# B.5 Why Four Similarities are needed

In practical applications, the four similarities all contribute to performance improvement and we state the significance as follows:

- Time information is crucial to predict future states or trends [114] via node history, i.e. in social networks, analyzing historical user interaction aids in predicting future behaviors.  
- Structure pertains to how nodes are interconnected and overall graph topology, vital for capturing similar graph structure patterns [13, 44, 113]. In transportation networks, factories are always located on the outer ring of the city, sharing similar structural connectivity, aiding in the discovery of spatiotemporal patterns [56, 16].  
- Sharing similar neighborhoods is essential for evaluating node similarity and correlation. In recommendations, shared purchase histories between users and products indicate potential interests, akin to collaborative filtering [88].  
- Semantic information measures similarity based on features [39]. In knowledge graphs, identifying relevant subgraphs to query nodes enhances retrieval accuracy based on semantic similarity.

# B.6 Why Knowledge Fusion is needed

Fusion and decoder here represent one of the core contributions of RAGRAPH:

- Overall Task Perspective: For the same tasks, the decoder can be directly employed to obtain outputs. For different tasks, the decoder can be masked and utilize pre-computed embeddings without training or be tuned to better adapt. This underscores our primary contribution, where the decoder functions as a versatile "plug-and-play" and "tune-free" component.  
- Integral Fusion Strategy: Fusion Strategy facilitates concurrent information propagation from toy graphs X (hidden embeddings) and Y (task-specific output vector) to query graph, aligning with our secondary contribution.

# B.7 How RAGRAPH works on par with RAG in NLP and CV

In NLP, RAG enhances the generation of LLM by retrieving relevant information via prompts. Similarly, in RAGraph, we enhance downstream graph learning by integrating information from retrieved toy graphs. Using these toy graphs with shared patterns assists the model inference. In our framework, the "generation" involves the retrieval-enhanced Graph Prompt: Toy Graph Intra Propagate & Query-Toy-Graph Inter Propagate to propagate retrieved knowledge (X and Y) into the query graph. To illustrate, we analyze this from both experiment and theory.

1. Experiment 1: We perform a case study to illustrate how "generation" works by displaying specific instances of node vectors in Appendix D.6.  
2. Experiment 2: In traditional GNN tasks, GCN, GAT, and GIN typically expand their receptive fields through stacked message-passing layers or neighborhood subgraph sampling for inference. Patterns learned in these contexts are often localized within the constrained receptive field. In contrast, in RAGRAPH, we observe that subgraphs sharing similar patterns often exhibit properties more aligned with downstream tasks. These subgraphs provide richer information for inference compared to simply enlarging receptive fields. As shown in Main Text Tables 1 and 2, Figure 3, RAGRAPH's strategy of incorporating toy graphs significantly outperforms baselines.  
3. **Theory 1:** Furthermore, we provide a theoretical justification of retrieval augmentation in GNNs (see Appendix B.4). From an information-theoretic perspective, introducing RAG knowledge into GNNs enhances the mutual information between input features  $X$  and output labels  $Y$ , such that:

$$
I (X, R A G; Y) \geq I (X; Y),
$$

thereby improving the performance of downstream tasks. This is aligned with the information theory of RAG in NLP [73].

4. Theory 2: Recent studies [11, 137] also suggest the generalization error diminishes with an increase in the node number of the graph in Theorem 1.1 [137]: the generalization error between the expected loss  $R_{\mathrm{exp}}(\Theta) = \mathbb{E}_{(x,y)\sim \mu_G}[\mathcal{L}(\Theta (x),y)]$  and empirical loss  $R_{\mathrm{emp}}(\Theta) = \frac{1}{m}\sum_{i = 1}^{m}\mathcal{L}(\Theta (x^i),y^i)$  are super bounded:

$$
\left| R _ {\exp} (\Theta) - R _ {\mathrm {e m p}} (\Theta) \right| \leq \sqrt {\frac {C}{m} q (n)},
$$

where  $C$  represents the model complexity (e.g., parameters),  $m$  denotes the training set size, and  $q(n) = \mathbb{E}_{n\sim \nu_G}[n^{-\frac{1}{D + 1}}]$  depends on the average graph size (node number) with  $\nu$  as the graph size distribution and  $D$  is the metric-measure space dimension. In RAGRAPH, retrieving similar toy graphs significantly increases the number of graph nodes (via Query-Toy-Graph Inter Propagate, linking toy graph nodes to query graph), significantly augmenting  $n$  while reducing  $q(n)$ . Consequently, the upper bound of generalization error decreases, promoting smoother graph learning convergence and enhancing pattern learning.

# C Further Methods Details

# C.1 Revisiting Graph Neural Networks

The goal of a GNN is to learn node embeddings based on an iterative aggregation of messages from the local network neighborhood. We use embedding matrix  $\{\mathbf{z}_v^{(L)}\}_{v\in \mathcal{V}}$  to denote the embedding for all the nodes after applying an  $L$ -layer GNN. The  $l$ -th layer of a GNN,  $\{\mathbf{z}_v^{(L)}\} = \mathrm{GNN}^{(l)}(\{\mathbf{z}_v^{(l - 1)}\})$ , can be written as:

$$
\mathbf {m} _ {u \rightarrow v} ^ {(l)} = \operatorname {M S G} ^ {(l)} \left(\mathbf {z} _ {u} ^ {(l - 1)}, \mathbf {z} _ {v} ^ {(l - 1)}\right),
$$

$$
\mathbf {z} _ {v} ^ {(l)} = \operatorname {A G G} ^ {(l)} \left(\left\{\mathbf {m} _ {u \rightarrow v} ^ {(l)} \mid u \in \mathcal {N} (v) \right\}, \mathbf {z} _ {v} ^ {(l - 1)}\right), \tag {7}
$$

where  $\mathbf{z}_v^{(l)}$  is the embedding for  $v\in V$  after passing through  $l$  layers,  $\mathbf{z}_v^{(0)} = x_v$  or  $h_v$  or  $o_v$ ,  $\mathbf{m}_{u\to v}^{(l)}$  is the message embedding, and  $\mathcal{N}(v)$  is the set of direct neighbors of  $v$ . Different GNNs can have various definitions of message-passing functions  $\mathrm{MSG}^{(l)}(\cdot)$  and aggregation functions  $\mathrm{AGG}^{(l)}(\cdot)$  and these two functions could be parameter-free.

# C.2 Key Construction of Position-aware Code

Given a randomly selected node anchor set  $\mathcal{V}_S\subset \mathcal{V}$ , we calculate the minimal distances,  $a.k.a.$  hops between the two node sets. Suppose  $v_{u}\in \mathcal{V},v_{w}\in \mathcal{V}_{S}$ , the distance similarity between node  $v_{u}$  and  $v_{w}$  can be depicted as  $dis(v_u,v_w)$ . By normalizing the similarity to [0, 1], distance-to-centroid  $d2c(v_u,v_w)$ :

$$
d 2 c \left(v _ {u}, v _ {w}\right) = \left\{ \begin{array}{l l} \frac {1}{d i s \left(v _ {u} , v _ {w}\right) + 1}, & \text {i f} d i s \left(v _ {u}, v _ {w}\right) <   d i s _ {q} \\ 0, & \text {o t h e r w i s e} \end{array} , \right. \tag {8}
$$

here hyperparameter  $dis_q$  is the maximum hops, the distance beyond this boundary is considered invalid. The structure feature of node  $v_{u}$  is  $d2c(v_u,\mathcal{V}_S)$ . By collecting all distances with anchor-set  $\nu_{S}$ , the structure  $S\in R^{n\times |\mathcal{V}_S|}$  is written as follows:

$$
\begin{array}{l} d 2 c \left(v _ {u}, \mathcal {V} _ {\mathcal {S}}\right) = \left[ d 2 c \left(v _ {u}, v _ {w}\right) \mid v _ {w} \in \mathcal {V} _ {\mathcal {S}} \subset \mathcal {V} \right], \\ C _ {u} = \left[ \left. v _ {u}, \mathcal {V} _ {\mathcal {S}}\right) \right| _ {v _ {w}} \subset \mathcal {V} \end{array} \tag {9}
$$

$$
S = \left[ d 2 c \left(v _ {u}, \mathcal {V} _ {\mathcal {S}}\right) \mid v _ {u} \in \mathcal {V} \right],
$$

where  $\left[\cdot\right]$  means the concatenation operation. The distance-to-centroid feature converts the non-Euclidean structure to the Euclidean structure.  $d2c$  dramatically reduces the size of the matrix and meanwhile contains more structure information instead of identifier information, the size of the anchor set is  $\log_2n$  follows P-GNNs[47, 44].

# C.3 Similarity Functions

For the history key, we adopt an exponential decay function to measure the time similarity values. We smooth the impact of time differences and provide a controlled decay coefficient  $\eta > 0$ . The time similarity,  $S_{\mathrm{time}}$ , between the same node  $v_{c}$  and  $v_{m}$  with different timestamp  $t(v_{m}), t(v_{c})$ , is defined as:

$$
S _ {\text {i m e}} \left(v _ {c}, v _ {m}\right) = e ^ {- \eta \left| t \left(v _ {c}\right) - t \left(v _ {m}\right) \right|}, \tag {10}
$$

where  $\eta$  is a positive parameter that controls the rate of exponential decay.

For the environment key, we match the environment of node  $v$  using Jaccard similarity to compare the sets of neighbors  $\mathcal{N}(v_c)$  in the query graph and  $\mathcal{N}(v_m)$  in the toy graph:

$$
S _ {\text {e n v i r o n m e n t}} \left(v _ {c}, v _ {m}\right) = \frac {\left| \mathcal {N} \left(v _ {c}\right) \cap \mathcal {N} \left(v _ {m}\right) \right|}{\left| \mathcal {N} \left(v _ {c}\right) \cup \mathcal {N} \left(v _ {m}\right) \right|}. \tag {11}
$$

For the hidden embedding key, we input the query graph into pre-trained GNNs to obtain the hidden embedding for the query node, with the similarity defined as:

$$
h _ {c} = \operatorname {G N N} \left(X _ {G ^ {\mathcal {Q}}}\right), \quad S _ {\text {s e m a n t i c}} \left(v _ {c}, v _ {m}\right) = \operatorname {c o s i n e} \left(h _ {c}, h _ {m}\right). \tag {12}
$$

For the position-aware code, we denote  $s_c, s_m$  as the position-aware code of node  $v_c, v_m$ , and utilize cosine similarity as before, defined as  $S_{\text{structure}}(v_c, v_m) = \cosine(s_c, s_m)$ .

# C.4 Proof of the Effectiveness of RAG

In this section, we will theoretically prove that introducing RAG knowledge can significantly improve the predictive performance of the model.

Assume  $X$  represents the input features,  $Y$  represents the target output labels, and  $RAG$  represents external knowledge related to the input features (or even the output labels). We analyze from the mutual information view, where  $I(X;Y)$  quantifies the dependency between  $X$  and  $Y$ , which reflects the performance of the model, the larger the value, the better the performance of the model [56, 72]. By introducing RAG knowledge  $RAG$  into GNNs, we can effectively increase the mutual information between the input features  $X$  and the output labels  $Y$  as  $I(X,RAG;Y) \geq I(X;Y)$ , thereby improve

the model's downstream task performance. The derivation is as follows:

$$
\begin{array}{l} I (X, R A G; \mathbf {Y}) - I (X; \mathbf {Y}) \\ = \sum_ {X, R A G, \mathbf {Y}} p (X, R A G, \mathbf {Y}) \log \frac {p (X , R A G , \mathbf {Y})}{p (X , R A G) p (\mathbf {Y})} - \sum_ {X, \mathbf {Y}} p (X, \mathbf {Y}) \log \frac {p (X , \mathbf {Y})}{p (X) p (\mathbf {Y})} \\ = \sum_ {X, R A G, \mathbf {Y}} p (X, R A G, \mathbf {Y}) \log \frac {p (X , R A G , \mathbf {Y})}{p (X , R A G) p (\mathbf {Y})} - \sum_ {X, R A G, \mathbf {Y}} p (X, R A G, \mathbf {Y}) \log \frac {p (X , \mathbf {Y})}{p (X) p (\mathbf {Y})} \\ = \sum_ {X, R A G, \mathbf {Y}} p (X, R A G, \mathbf {Y}) \log \left(\frac {p (X , R A G , \mathbf {Y})}{p (X , R A G)} \cdot \frac {p (X)}{p (X , \mathbf {Y})}\right) \\ = \sum_ {X, R A G, \mathbf {Y}} p (X, R A G, \mathbf {Y}) \log \frac {p (X , R A G , \mathbf {Y})}{p (R A G | X) p (X , \mathbf {Y})} \\ = \sum_ {X, R A G, \mathbf {Y}} p (R A G, \mathbf {Y} \mid X) p (X) \log \frac {p (R A G , \mathbf {Y} \mid X)}{p (R A G \mid X) p (\mathbf {Y} \mid X)} \\ = I (R A G; \mathbf {Y} \mid X) \geq 0, \tag {13} \\ \end{array}
$$

where  $\sum_{X,RAG,\mathbf{Y}} = \sum_X\sum_{RAG}\sum_{\mathbf{Y}}$ . Moreover,  $I(RAG;\mathbf{Y} \mid X)$  measures that how much additional knowledge  $RAG$  provides to assist in predicting  $Y$  based on  $X$ , this term will approach zero when the  $RAG$  is noise to the prediction task. In summary, the integration of RAG knowledge can enhance the mutual information between  $X$  and  $Y$ , thereby improving the performance and accuracy of downstream tasks.

# C.5 Algorithms

In this section, we will provide a detailed description of the algorithms of Toy Graph Construction (Algorithm 1) and Training and Inference with Toy Graphs Retrieval (Algorithm 2).

Algorithm 1. We outline the process for constructing toy graphs in Algorithm 1. In line 1, the toy graph database  $\mathcal{G}^{\mathcal{T}}$  is initialized.

Lines 2-15 describe the steps to construct toy graphs by iterating through each snapshot of the dynamic resource graph  $\mathcal{G}^{\mathcal{R}}$ .

In more detail, lines 3-5 details calculate the importance and reverse importance of each node within the snapshot. Following this, lines 6-7 involve normalizing the sampling probabilities according to the reverse importance values. The selection of a master node and the generation of its  $k$ -hop ego network are carried out in lines 8-9. Subsequently, line 10 involves augmenting the toy graph through specific data augmentation techniques.

Lines 11-16 detail the generation of key-value pairs for each toy graph. This includes saving the timestamp as the history key, the neighbors of the master node as the environment key, the structural encoding as the structure key, the hidden embedding as the semantic key, and the task-specific output vectors as the value. Each toy graph is then stored in the database  $\mathcal{G}^{\mathcal{T}}$ .

Ultimately, in line 18, the algorithm returns the toy graph database  $\mathcal{G}^{\mathcal{T}}$ .

Algorithm 2. We introduce the algorithm for training and inference with toy graph retrieval in Algorithm 2. Initially, in line 1, we define the required inputs, including the testing graph  $\mathcal{G}_{\mathrm{test}}$ , the toy graph database  $\mathcal{G}^{\mathcal{T}}$ , and other relevant parameters. The final output is the aggregated result  $\tilde{o}_c$ .

The RETRIEVETOYGRAPHS function, detailed in lines 3-11, initializes an empty similarity list and iterates over each toy graph in the database. Lines 5-6 compute various similarity metrics, and the overall similarity is determined in line 7. This similarity score is then added to the list. After sorting by similarity, the top  $K$  toy graphs are retrieved and returned in line 11.

Within the PROPAGATION function (lines 12-17), each retrieved toy graph undergoes intra-propagation in line 14. The intra-propagation step follows in line 16, ultimately returning the propagated results  $\mathbf{z}_c$ .

The KNOWLEDGEFUSION function, found in lines 18-21, combines the outputs from previous steps. The final combined outputs  $\tilde{o}_c$  are generated by the decoder and returned in line 21.

# Algorithm 1 Toy Graph Construction

Require: Dynamic Resource Graph  $\mathcal{G}^{\mathcal{R}}$  , node importance balance weight  $\alpha$  , toy graph scaling constant  $K$  , maximum hop  $k$    
Ensure: Toy graph embedding key-value database  $\mathcal{G}^T$    
1: Initialize toy graph database  $\mathcal{G}^T\gets \emptyset$    
2: for each snapshot  $G_{\tau}^{\mathcal{R}}\in \mathcal{G}^{\mathcal{R}}$  do  $\triangleright$  Construct Toy Graphs   
3: for each node  $v\in G_{\tau}^{\mathcal{R}}$  do   
4: Calculate importance  $I(v)\leftarrow \alpha \mathrm{PR}(v) + (1 - \alpha)\mathrm{DC}(v)$    
5: Reverse node importance  $I^{\prime}(v)\leftarrow \frac{1}{I(v) + \epsilon}$    
6: end for   
7: for each node  $v\in G_{\tau}^{\mathcal{R}}$  do   
8:Normalize sampling probabilities  $p_i\gets \frac{I'(v_i)}{\sum_{j = 1}^{n}I'(v_j)}$    
9: end for   
10: Sample master node  $v_m\gets$  WEIGHTEDSAMPLING  $(G_{\tau}^{\mathcal{R}},p_i)$  based on probability  $p_i$    
11: Generate  $k$  -hop ego net  $G_{\tau}^{e}(v_{m})$  for node  $v_{m}$    
12: Augment toy graph  $\{G^T\} \gets$  DATA AUGMENTATION  $(G_{\tau}^{e}(v_{m}),n_{\mathrm{aug}})$  with  $n_{\mathrm{aug}}(G_{\tau}^{e}(v_{m}))=$ $\lfloor K\cdot \bar{I} '(G_\tau^e (v_m))\rfloor$    
13: for each  $G^T\in \{G^T\}$  do  $\triangleright$  Generate keys-values pair   
14: Save timestamp  $\tau$  as history key   
15: Save neighbors  $\mathcal{N}(v_m^\tau)$  of master node  $v_{m}$  as environment key   
16: Save structural encoding  $s_m^\tau$  of node  $v_{m}$  via Eq. (9) as structure key   
17: Save hidden embedding  $h_m^\tau$  by feeding  $G^T$  into pre-trained GNNs as semantic key   
18: Save the hidden embedding  $\{h_i^\tau |v_i\in G^T\}$  as value   
19: Save task-specific output vectors  $\{o_i^\tau |v_i\in G^T\}$  by feeding  $\{h_i^\tau |v_i\in G^T\}$  into decoder as value   
20: Store toy graph  $G^T$  into database  $\mathcal{G}^T$    
21: end for   
22: end for   
23: return Toy graph database  $\mathcal{G}^T$

The main algorithm begins in line 22. If fine-tuning is enabled and the prompt loss has not converged, lines 23-34 detail the process of toy graph retrieval and propagation for each query graph. This includes the optional addition of noise in lines 26-29. The hidden embedding and task-specific output vectors are propagated in lines 30-31, and the aggregated outputs are fused in line 32. The prompt loss is computed, and fine-tuned parameters are updated in lines 33-34.

If fine-tuning is not required, lines 35-39 describe a similar process of toy graph retrieval and propagation, without the fine-tuning steps. The aggregated outputs are computed directly.

# D Further Experiment Details

# D.1 Datasets Statics

We employ eight benchmark datasets for evaluation including four public static classification datasets for node- and graph-level tasks.

(1) PROTEINS [4] is a collection of protein graphs, including the amino acid sequence, conformation, structure, and features such as active sites of the proteins. Each node represents a secondary structure, while each edge illustrates the neighboring relationship either within the amino acid sequence or in 3D space. The nodes are divided into three categories, and the graphs are classified into two distinct classes. This dataset is used for node and graph classification tasks, containing 1,113 graphs with an average of 39.06 nodes and 72.82 edges per graph, with a density of 4.8e-2.  
(2) COX2 [84] is a dataset of molecular structures, including 467 cyclooxygenase-2 inhibitors. Each node represents an atom, and each edge signifies a chemical bond between atoms, such as single, double, triple, or aromatic bonds. All the molecules belong to two categories. This dataset is used for

Algorithm 2 Training and Inference with Toy Graphs Retrieval  
Require: Testing graph  $\mathcal{G}_{\mathrm{test}}$  , toy graph database  $\mathcal{G}^T$  , pre-trained GNN model  $\mathrm{GNN}_{\Theta_0}(\cdot)$  , number of TopK toy graphs to retrieve, similarity weights  $w_{1},w_{2},w_{3},w_{4}$  , fine-tuning flag fine_tune, noise prompt-tuning flag add_noise   
Ensure: Aggregated output  $\tilde{o}_c$    
1: function RETRIEVETOYGRAPHS  $(G^{\mathcal{Q}},\mathcal{G}^{\mathcal{T}},TopK)$    
2: Initialize similarity list  $\{S\} \leftarrow \emptyset$    
3: for each toy graph  $G^T\in \mathcal{G}^T$  do   
4: Calculate time similarity  $S_{\mathrm{time}}$  , environment similarity  $S_{\mathrm{environment}}$  , structure similarity  $S_{\mathrm{structure}}$  and semantic similarity  $S_{\mathrm{semantic}}$    
5: Compute similarity  $S\gets w_{1}\cdot S_{\mathrm{time}} + w_{2}\cdot S_{\mathrm{structure}} + w_{3}\cdot S_{\mathrm{environment}} + w_{4}\cdot S_{\mathrm{semantic}}$    
6:Add  $(G^T,S)$  to  $\{S\}$    
7: end for   
8: Sort  $\{S\}$  by similarity in descending order   
9: Retrieve topK toy graphs  $G_{\mathrm{TopK}}^{T}\gets \{G^{T}\in \{S\}$  with topK similarities}   
10: return Retrieved toy graphs  $G_{\mathrm{TopK}}^{T}$    
11: end function   
12: function PROPAGATION  $(G^{\mathcal{Q}},G_{\mathrm{TopK}}^{\mathcal{T}})$    
13: for each toy graph  $G^T\in G_{\mathrm{TopK}}^T$  do   
14: Perform Intra Propagation  $\mathbf{z}_m\gets$  INTRAPROPAGATION  $(G^T)$    
15: end for   
16:  $\mathbf{z}_c\gets$  INTERPROPAGATION  $(G^{\mathcal{Q}},G_{\mathrm{TopK}}^{\mathcal{T}})$    
17: return  $\mathbf{z}_c$    
18: end function   
19: function KNOWLEDGEFUSION(hc,oc)   
20: Combined output  $\hat{\sigma}_c\gets \gamma o_c + (1 - \gamma)\mathrm{DECODER}(h_c)$    
21: return Combined outputs  $\tilde{o}_c$    
22: end function   
23: if fine_tune &  $\mathcal{L}_{\mathrm{prompt}}$  not converged then   
24: for each query graph  $G^{\mathcal{Q}}\in \mathcal{G}_{\mathrm{test}}$  do ▷ Toy Graph Retrieval and Propagation   
25:  $G_{\mathrm{TopK}}^T\gets$  RETRIEVETOYGRAPHS(GQ,G,T,TopK)   
26: if add_noise then   
27: for each toy graph  $G^T\in G_{\mathrm{TopK}}^T$  do   
28: Introduce noise  $G^T\gets$  ADDNOISE  $(G^T)$  ▷ Inner-Toy-Graph Noise   
29: end for   
30: Add bottom K toy graphs to  $G_{\mathrm{topK}}^T$  ▷ Toy-Graph Noise   
31: end if   
32: hc  $\leftarrow$  PROPAGATION(GQ,GT,TopK) ▷ Propagate hidden embedding   
33: oc  $\leftarrow$  PROPAGATION(GQ,GT,TopK) ▷ Propagate task-specific output vector   
34: Aggregated outputs  $\hat{o}_c\gets$  KNOWLEDGEFUSION(hc,oc)   
35: Compute loss Lprompt via  $\tilde{o}_c$  and  $\hat{o}_c$  ▷ Based on task-specific loss function   
36: Update fine-tuned parameters  $\Theta$  by minimizing Lprompt   
37: end for   
38: else   
39: for each query graph  $G^{\mathcal{Q}}\in \mathcal{G}_{\mathrm{test}}$  do ▷ Toy Graph Retrieval and Propagation   
40:  $G_{\mathrm{topK}}^T\gets$  RETRIEVETOYGRAPHS(GQ,G,T,TopK)   
41: hc  $\leftarrow$  PROPAGATION(GQ,GT,TopK) ▷ Propagate hidden embedding   
42: oc  $\leftarrow$  PROPAGATION(GQ,GT,TopK) ▷ Propagate task-specific output vector   
43: Aggregated outputs  $\hat{o}_c\gets$  KNOWLEDGEFUSION(hc,oc)   
44: end for   
45: end if   
46: return Aggregated outputs  $\tilde{o}_c$

Table 4: Statistics of the experimental datasets and summary of datasets.  

<table><tr><td>Statistics</td><td>TAOBAO</td><td>KOUBEI</td><td>AMAZON</td><td>PROTEINS</td><td>COX2</td><td>ENZYMES</td><td>BZR</td></tr><tr><td># Nodes per Graph</td><td>204,168</td><td>221,366</td><td>238,735</td><td>39.06</td><td>41.22</td><td>32.63</td><td>35.75</td></tr><tr><td># Edges per Graph</td><td>8,795,404</td><td>3,986,609</td><td>876,237</td><td>72.82</td><td>43.45</td><td>62.14</td><td>38.36</td></tr><tr><td># Density</td><td>8.6e-4</td><td>3.3e-4</td><td>6.2e-5</td><td>4.8e-2</td><td>2.6e-2</td><td>5.9e-2</td><td>3.0e-2</td></tr><tr><td># Graphs</td><td>1</td><td>1</td><td>1</td><td>1,113</td><td>467</td><td>600</td><td>405</td></tr><tr><td># Graph Classes</td><td>/</td><td>/</td><td>/</td><td>2</td><td>2</td><td>6</td><td>2</td></tr><tr><td># Node Features</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td><td>18</td><td>3</td></tr><tr><td># Node Classes</td><td>/</td><td>/</td><td>/</td><td>3</td><td>/</td><td>3</td><td>/</td></tr><tr><td>Snapshot Granularity</td><td>daily</td><td>weekly</td><td>weekly</td><td>/</td><td>/</td><td>/</td><td>/</td></tr><tr><td>Task</td><td>Edge</td><td>Edge</td><td>Edge</td><td>Node, Graph</td><td>Graph</td><td>Node, Graph</td><td>Graph</td></tr><tr><td>Type</td><td>Dynamic</td><td>Dynamic</td><td>Dynamic</td><td>Static</td><td>Static</td><td>Static</td><td>Static</td></tr><tr><td>Dataset Partition</td><td>Snapshot</td><td>Snapshot</td><td>Snapshot</td><td>Node, Graph</td><td>Graph</td><td>Node, Graph</td><td>Graph</td></tr></table>

graph classification tasks, with each graph having an average of 41.22 nodes and 43.45 edges and a density of 2.6e-2.

(3) ENZYMES [100] is a dataset of 600 enzymes collected from the BRENDA enzyme database. These enzymes are labeled into 6 categories according to their top-level EC enzyme classification. This dataset is used for node and graph classification tasks, with each graph having an average of 32.63 nodes and 62.14 edges and a density of  $5.9\mathrm{e} - 2$  
(4) BZR [84] is a collection of 405 ligands for the benzodiazepine receptor. Each ligand is represented by a graph, and all ligands are categorized into two groups. This dataset is used for graph classification tasks, with each graph having an average of 35.75 nodes and 38.36 edges and a density of 3.0e-2.

Additionally, we leverage three publicly available datasets encompassing a wide array of real-world scenarios in dynamic recommendation (link prediction):

(5) The TAOBAO dataset captures implicit feedback data from Taobao.com, a prominent Chinese e-commerce platform, collected over a span of 10 days. This dataset is used for edge classification tasks, containing 204,168 nodes and 8,795,404 edges, with a density of 8.6e-4.  
(6) The KOUBEI dataset records 9 weeks of user interactions with nearby stores on Koubei, a platform integrated within Alipay. This dataset is used for edge classification tasks, containing 221,366 nodes and 3,986,609 edges, with a density of 3.3e-4.  
(7) The AMAZON dataset comprises a collection of product reviews sourced from Amazon, spanning a duration of 13 weeks. This dataset is used for edge classification tasks, containing 238,735 nodes and 876,237 edges, with a density of 6.2e-5.

These datasets' detailed statistics are summarized in Table 4. The "Task" column provides information about the type of downstream task conducted on each dataset: "Node" denotes node classification tasks, "Graph" signifies graph classification tasks, and "Edge" indicates tasks related to link prediction. The "Type" column indicates the type of graph dataset: "Dynamic" for dynamic dataset  $t \geq 1$ , and "Static" for static dataset  $t = 1$ . For dynamic datasets, the "Snapshot Granularity" denotes the time granularity for each dataset. In our experimental setup, for dataset partition, dynamic graphs are partitioned according to snapshots, while static graphs are partitioned either by node or by the entire graph.

# D.2 Evaluation Matrices

Node and Graph classification evaluation. For the node classification, we use the prediction accuracy to measure the model.

Link prediction evaluation. For the link prediction, we evaluate the recall and ranking quality of the effects of recommendation following previous studies [118, 31]. We use Recall@k and NDCG@k as metrics. Note that this task should be a binary task. We denote the topk largest value as  $rel_{ij}, j \in [1,k]$  for node  $v_i$ .

Recall@k measures the ratio of true positive links contained in the top  $k$  predicted links for each node:

$$
\operatorname {R e c a l l} @ k = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \frac {\operatorname {r e l} _ {i j}}{\sum \mathbb {I} (A [ i : ] > 0)}, \tag {14}
$$

where  $rel_{ij} = 1$  if the  $j$ -th predicted link for node  $v_i$  exists, otherwise  $0$ .  $\mathbb{I}(\cdot)$  is the indicator function, and if  $A[i] > 0$  then  $\mathbb{I}(A[i] > 0) = 1$ .

NDCG@k (Normalized Discounted Cumulative Gain) is computed by normalizing DCG@k (Discounted Cumulative Gain) which accounts for the position of correctly predicted links. DCG@k is defined as:

$$
\mathrm {D C G} @ k = \frac {1}{n} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \frac {\operatorname {r e l} _ {i j}}{\log_ {2} (j + 1)}. \tag {15}
$$

# D.3 Baseline Details

In this section, we present the details of baselines.

Table 5: Baseline Code URLs of Github Repository  

<table><tr><td>Baseline</td><td>Type</td><td>Code Repo URL</td></tr><tr><td>GCN</td><td>Static</td><td>https://github.com/tkipf/gcn</td></tr><tr><td>GraphSAGE</td><td>Static</td><td>https://github.com/williamleif/GraphSAGE</td></tr><tr><td>GAT</td><td>Static</td><td>https://github.com/PeterV-/GAT</td></tr><tr><td>GIN</td><td>Static</td><td>https://github.com/weihua916/powerful-gnns</td></tr><tr><td>LightGCN</td><td>Dynamic</td><td>https://github.com/kuandeng/LightGCN</td></tr><tr><td>SGL</td><td>Dynamic</td><td>https://github.com/wujcan/SGL-Torch</td></tr><tr><td>MixGCF</td><td>Dynamic</td><td>https://github.com/Wu-Xi/SimGCL-MixGCF</td></tr><tr><td>SimGCL</td><td>Dynamic</td><td>https://github.com/Wu-Xi/SimGCL-MixGCF</td></tr><tr><td>GraphPro</td><td>Dynamic</td><td>https://github.com/HKUDS/GraphPro</td></tr><tr><td>GraphPrompt</td><td>Dynamic, Static</td><td>https://github.com/Starlien95/GraphPrompt</td></tr><tr><td>PRODIGY</td><td>Dynamic, Static</td><td>https://github.com/snap-stanford/prodigy</td></tr></table>

- GCN [50]: GCN is an end-to-end learning framework for graph-structured data. It utilizes neighborhood aggregation to integrate structural information, which is particularly effective in node classification and graph classification tasks.  
- GraphSAGE [30]: GraphSAGE, is a general and inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data.  
- GAT [98]: GAT is a spatial domain method, which aggregates information through the attention-learned edge weights.  
- GIN [106]: GIN utilizes a multi-layer perceptron to sum the results of GNN and learns a parameter to control residual connection.  
- LightGCN [31]: LightGCN learns user and item embeddings by linearly propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings learned at all layers as the final embedding.  
- SGL [104]: SGL is to supplement the classical supervised task of recommendation with an auxiliary self-supervised task, which reinforces node representation learning via self-discrimination.  
- MixGCF [41]: MixGCF generates the synthetic negative by aggregating embeddings from different layers of raw negatives' neighborhoods to perform collaborative filtering.

- SimGCL [118]: SimGCL applies unsupervised contrastive learning to enhance representation learning, making it suitable for link prediction tasks. It is applied to dynamic graphs to test its adaptability and performance.  
- GraphPro [114]: GraphPro extends GraphPrompt by introducing spatial and temporal prompts tailored for dynamic graph learning, enhancing the ability to capture both structural and temporal relationships within graph data.  
- GraphPrompt [67]: GraphPrompt integrates pre-training and downstream tasks using a unified template approach and employs task-specific prompts to enhance sub-task learning, applicable to both dynamic and static graph contexts.  
- PRODIGY [39]: PRODIGY focuses on facilitating downstream tasks through in-context examples and learning from the  $X \to Y$  paradigm. It is implemented to enhance learning in both dynamic and static graphs by leveraging contextual learning strategies.

# D.4 Implementation Details

Implementations are done using the PyTorch 2.3.0 framework [79] in Python 3.11, on an Ubuntu server equipped with 1 V100 GPU and an Intel(R) Xeon(R) CPU.

In node and graph classification tasks: For baseline GCN [50], we employ a 2-layer architecture and set the hidden dimension as 256. For GraphSAGE [30], we utilize the mean aggregator and employ a 2-layer architecture. The hidden dimension is also set to 256. For GAT [98], we employ a 2-layer architecture and set the hidden dimension as 256. Besides, we apply 8 attention heads in the first GAT layer. Similarly, for GIN [106], we also employ a 2-layer architecture and set the hidden dimension as 256. For GraphPrompt, we follow [67] to employ a 2-layer GCN as the backbone and set the hidden dimensions as 256.

In the link prediction task: For LightGCN, SGL, MixGCF, SimGCL and GraphPro, we employ a 3-layer GNN architecture and set the hidden dimension as 64 with Low-Rank Adaptation (LoRA) [33] rank equals to 16. For GraphPro, the backbone graph encoder is SimGCL.

Moreover, for all three tasks, the hyper-parameters of baselines are based on the recommended values provided in the paper. In PRODIGY and RAGRAPH,  $k$  is set to 2, top  $K$  is set to 5,  $\gamma$  is set to 0.8 for PROTEINS and 0.5 for ENZYMES in node level,  $\gamma$  is set to 0.5 for PROTEINS, 0.6 for COX2, 0.8 for ENZYMES and 0.5 for BZR in graph level,  $\alpha = \lambda = 0.5, K = 3, w_{1} = w_{2} = w_{3} = 0.05, w_{4} = 0.85$ .

# D.5 Resource Graph Scalability Study

(a) Node Classification on ENZYMES dataset

(b) Node Classification on PROTEINS dataset  
Figure 4: Performance comparisons of RAGRAPH and several baselines with different proportions of training and resource data.

We assess the impact of varying amounts of training and resource data on model performance. As illustrated in Figure 4, we vary the proportion of train and resource graph size from  $10\%$  to  $80\%$ , with increments of  $10\%$ , and conduct experiments on node classification tasks using the ENZYMES and PROTEINS datasets, respectively. For comparative analysis, we select GIN, GraphPrompt, and PRODIGY as baseline models. To ensure fairness in our experiments, we maintain a consistent ratio of train to resource data at 3:5 during fine-tuning, utilizing the sum of these as a retrieval database.

As shown in Figure 4, there is a clear trend where the accuracy of the model improves as the proportion of the dataset increases. However, the rate of accuracy improvement starts to plateau once a certain dataset proportion is reached (i.e.,  $30\%$  in PRODIGY and RAGRAPH,  $40\%$  in PROTEINS for GraphPrompt). Among the evaluated models, GIN and GraphPrompt show the slowest convergence rates, whereas PRODIGY converges at a moderate pace, and RAGRAPH converges the fastest. This rapid convergence in PRODIGY and RAGRAPH is attributed to its ability to engage in effective knowledge retrieval, significantly enhancing the model's comprehension abilities. Remarkably, both PRODIGY and RAGRAPH can achieve commendable results in downstream tasks even with a small proportion of the dataset. Compared to PRODIGY, RAGRAPH exhibits superior performance because while PRODIGY primarily learns a mapping from  $X$  to  $Y$ , RAGRAPH not only learns this mapping but also integrates additional knowledge into GNNs more effectively. This integration becomes increasingly beneficial as the dataset proportion grows, allowing RAGRAPH to outperform other models, particularly at higher data volumes where it can better leverage its knowledge integration capabilities.

# D.6 Qualitative Analyses of Toy Graphs Retrieving

In this section, we conduct qualitative analyses of the toy graphs retrieving experiment. For the sake of understanding, we conduct experiments under normal settings where the dimensionality of the task-specific output vector is equal to the number of classes.

On the ENZYMES dataset, for a 3-class node classification task, regarding node "13984", which belongs to class 3, if we only use the GraphPrompt Backbone, the resulting one-hot encoding is [0.28,0.34,0.38].

However, since the node is of class 3, we expect the one-hot encoding to be as close as possible to  $[0,0,1]$ . In RAGRAP retrieval, taking the top 3 retrieved graphs as examples, the connection weights for these 3 toy graphs to query graphs are 0.5, 0.7, and 0.1, respectively, and their corresponding label one-hot encodings are  $[0,0,1]$ ,  $[0,0,1]$ , and  $[0,1,0]$ . Therefore, the result obtained by propagating the task-specific output vector through toy graphs is:  $[0,0.1,1.2]$ , and after normalization, the result is  $[0,0.08,0.92]$ .

Meanwhile, the vector obtained by propagating toy graphs hidden embedding and via decoder is [0.37,0.32,0.66]. The retrieval of toy graphs notably enhances performance at both the task-specific output vector and hidden embedding levels. The final vector is obtained through a weighted sum with  $\gamma = 0.5$  in Eq(6) is [0.185,0.20,0.79], after normalization the result is [0.157,0.170,0.673], which greatly enhances the model's discriminative ability compared to GraphPrompt [0.28,0.34,0.38].

Figure 5: Qualitative analyses of toy graphs retrieving - how "generation" works.

Figure 6: Difference Illustration between PRODIGY and RAGRAPH.

# E Difference between ICL (PRODIGY) and RAG (RAGRAPH)

In this section, we explore the distinctions between PRODIGY and RAGRAPH from several critical perspectives, as illustrated in Figure 6:

- PRODIGY: This approach utilizes fixed examples as rules, which may not be optimal for dynamic and evolving scenarios. PRODIGY primarily focuses on learning direct mappings from  $X$  to  $Y$  through in-context learning. However, it encounters challenges in integrating external information that is more pertinent to the query node. This is particularly problematic when the distribution of each node belonging to the same label class varies, and simply learning the mapping based on the prototype node will somehow be misleading.  
- RAGRAPH: In contrast, RAGRAPH is designed to handle non-static, streaming knowledge, making it well-suited to dynamic graph structures and evolving tasks. It actively retrieves relevant knowledge on-demand, effectively incorporating information about both  $X$  and  $Y$  from external sources into GNNs. Moreover, RAGRAPH can operate without the need for model fine-tuning, providing substantial flexibility. This adaptability enables RAGRAPH to excel in tasks that require continuous adaptation to changing conditions and the integration of external, relevant information.

In summary, we argue that a qualified Retrieval-Augmented Generation (RAG) system for Graph Learning should fulfill several essential criteria to effectively support complex reasoning tasks: 1) It should retrieve ample feature and task-related label information, analogous to how attributes are gathered in the NLP domain to stimulate the reasoning capabilities of LLMs; 2) The system should adapt to new tasks or unseen datasets without requiring fine-tuning of model parameters; 3) Knowledge within the system must be dynamically updated and stored, ensuring current and relevant data utilization.

# F Broader Impacts

Our work builds on the widespread application of Retrieval-Augmented Generation (RAG) in large language models (LLMs) and aims to extend its success to graph data, thereby constructing graph foundation models. This approach allows models to transfer rapidly without requiring learnable parameters, avoiding potential performance degradation from fine-tuning pre-trained models. As a result, RAG is particularly effective in domains with scarce and long-tail data, such as network anomaly detection, rare disease diagnosis/treatment, supply chain disruption, and new user recommendations.

Additionally, our model establishes an excellent paradigm by incorporating retrieved features and label information into the learning process, significantly enhancing the model's understanding capabilities. Our work provides valuable insights and serves as a reference for future Large Graph Models.

# G Data Ethics Statement

To evaluate the efficacy of this work, we conducted experiments that only use publicly available datasets, namely, PROTEINS, COX2, ENZYMES, BZR $^3$ , TAOBAO, KOUBEI and AMAZON in accordance to their usage terms and conditions if any. We further declare that no personally identifiable information was used, and no human or animal subject was involved in this research.

# Footnotes:

Page 0: *Indicates equal contribution. † Yasha Wang and Junfeng Zhao are corresponding authors. 
Page 30: <sup>3</sup>https://chrsmrrs.github.io/datasets/ 
