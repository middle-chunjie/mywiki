# Constructing Tree-based Index for Efficient and Effective Dense Retrieval

Haitao Li

DCST, Tsinghua University

Zhongguancun Laboratory

liht22@mails.tsinghua.edu.cn

Jiaxin Mao

GSAI, Renmin University of China  
maojiaxin@gmail.com

Qingyao Ai

DCST, Tsinghua University

Zhongguancun Laboratory

aiqy@tsinghua.edu.cn

Yiqun Liu*

DCST, Tsinghua University

Zhongguancun Laboratory

yiqunliu@tsinghua.edu.cn

Zhao Cao

Huawei Poisson Lab

caozhao1@huawei.com

Jingtao Zhan

DCST, Tsinghua University

Zhongguancun Laboratory

chenjia0831@gmail.com

Zheng Liu

Huawei Poisson Lab

liuzheng107@huawei.com

# ABSTRACT

Recent studies have shown that Dense Retrieval (DR) techniques can significantly improve the performance of first-stage retrieval in IR systems. Despite its empirical effectiveness, the application of DR is still limited. In contrast to statistic retrieval models that rely on highly efficient inverted index solutions, DR models build dense embeddings that are difficult to be pre-processed with most existing search indexing systems. To avoid the expensive cost of brute-force search, the Approximate Nearest Neighbor (ANN) algorithm and corresponding indexes are widely applied to speed up the inference process of DR models. Unfortunately, while ANN can improve the efficiency of DR models, it usually comes with a significant price on retrieval performance.

To solve this issue, we propose JTR, which stands for Joint optimization of TRee-based index and query encoding. Specifically, we design a new unified contrastive learning loss to train tree-based index and query encoder in an end-to-end manner. The tree-based negative sampling strategy is applied to make the tree have the maximum heap property, which supports the effectiveness of beam search well. Moreover, we treat the cluster assignment as an optimization problem to update the tree-based index that allows overlapped clustering. We evaluate JTR on numerous popular retrieval benchmarks. Experimental results show that JTR achieves better retrieval performance while retaining high system efficiency compared with widely-adopted baselines. It provides a potential solution to balance efficiency and effectiveness in neural retrieval system designs.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGIR '23, July 23-27, 2023, Taipei, Taiwan.

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 978-1-4503-9408-6/23/07...$15.00

https://doi.org/10.1145/nnnnnnn.nnnnnnn

# CCS CONCEPTS

- Information systems  $\rightarrow$  Information retrieval query processing; Retrieval models and ranking.

# KEYWORDS

Information Retrieval, Approximate Nearest Neighbor, Tree-based Index

# ACM Reference Format:

Haitao Li, Qingyao Ai, Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Zheng Liu, and Zhao Cao. 2023. Constructing Tree-based Index for Efficient and Effective Dense Retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23), July 23-27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

# 1 INTRODUCTION

Information Retrieval (IR) system has become one of the most important tools for people to find useful information online. In practice, an industry IR system usually needs to retrieve a small set of relevant documents from millions or even billions of documents. The computational cost of brute-force search is usually unacceptable due to the system latency requirements. Therefore, indexes are often employed to achieve a fast response. One of the most well-known examples is the inverted index [24]. Unfortunately, such traditional indexing techniques cannot be applied to recently proposed Dense Retrieval (DR) models. As DR models have shown promising performance in multi-stage retrieval systems, especially in the first stage retrieval [16, 17, 20, 29, 33, 39], how to develop effective and efficient indexing technique for dense retrieval has become an important question for the research community.

Efficient DR solutions such as Approximate Nearest Neighbor (ANN) search algorithms have been widely applied in the first stage retrieval process. Popular methods include tree-based indexes [2], locality sensitive hashing (LSH) [15], product quantization (PQ) [13, 14], hierarchical navigable small-world network (HNSW) [23], etc. Since tree-based indexes can achieve sub-linear time complexity by pruning low-quality candidates, it has drawn significant attention.

Despite success in improving retrieval efficiency, most existing tree-based indexes often come with the degradation of retrieval performance. There are two main reasons: (1) The majority of existing indexes cannot benefit from supervised data because they use task-independent reconstruction error as the loss function. (2) The training objectives of the index structure and the encoder are inconsistent. In general, the optimization goal of index structure is to minimize the approximation error, while the optimization goal of the encoder is to get better retrieval performance. This inconsistency may lead to sub-optimal results.

To balance the effectiveness and efficiency of the tree-based indexes, we propose JTR, which stands for Joint optimization of TRee-based index and query encoding. To jointly optimize index structure and query encoder in an end-to-end manner, JTR drops the original "encoding-indexing" training paradigm and designs a unified contrastive learning loss. However, training tree-based indexes using contrastive learning loss is non-trivial due to the problem of differentiability. To overcome this obstacle, the tree-based index is divided into two parts: cluster node embeddings and cluster assignment. For differentiable cluster node embeddings, which are small but very critical, we design tree-based negative sampling to optimize them. For cluster assignment, an overlapped cluster method is applied to iteratively optimize it.

To verify the effectiveness of our method, we compared JTR with a wide range of ANN methods on two large publicly available datasets. The empirical experimental results show that JTR achieves better performance compared to baselines while keeping high efficiency. Through the ablation study, we have confirmed the effectiveness of the proposed strategies<sup>1</sup>.

In short, our main contributions are as follows:

(1) We propose a novel tree-based index for Dense Retrieval. Benefiting from the tree structure, it achieves sub-linear time complexity while still yielding promising results.  
(2) We propose a new joint optimization framework to learn both the tree-based index and query encoder. To the best of our knowledge, JTR is the first joint-optimized retrieval approach with tree-based index. This framework improves end-to-end retrieval performance through unified contrastive learning loss and tree-based negative sampling.  
(3) We relax the constraint that documents are mutually exclusive between clusters and propose an efficient clustering solution in which a document can be assigned into multiple clusters to optimize the cluster assignment. We demonstrate that overlapped cluster is beneficial for improving retrieval performance.

# 2 RELATED WORK

# 2.1 Dual Encoders

With the development of Pre-trained Language Models (PLMs), Dense Retrieval (DR) has developed rapidly in the last decade. DR typically uses dual encoders to obtain embeddings of queries and documents and utilizes inner products as similarity factors. For a query-document pair  $< q, d >$ , the primary goal is to learn a presentation function  $f(\cdot)$  that maps text to higher-dimensional

embeddings. The general form of a dual encoder is as follows:

$$
s (q, d) = <   f (q), f (d) > \tag {1}
$$

Where,  $<, >$  refers to the inner product.  $s(q,d)$  is the relevance score. To further improve the performance of dual encoders, many researchers have improved the loss function, sampling method, and so on to make them more suitable for information retrieval tasks [3, 8, 12, 18, 19, 34, 35, 37]. For example, Lee et al. [18] suggest the use of random negative sampling in large batches, which increases the difficulty of training. Zhan et al. [37] propose the dynamic hard negative sampling method, which effectively improves the performance of the DR model.

# 2.2 Approximate Nearest Neighbor Search

When queries and documents are encoded into embeddings, the retrieval problem can be considered as a Nearest Neighbor (NN) search problem. The simplest nearest neighbor search method is brute-force search, which becomes impractical when the corpus size explodes. Therefore, most studies use Approximate Nearest Neighbor (ANN) search. In general, there are four common methods, including tree-based methods [2], hashing [15, 28], quantization-based methods [13, 14], and graph-based methods [23]. Tree-based and graph-based methods divide embeddings into different spaces and query embeddings are retrieved only in similar spaces. Hashing and quantization-based methods compress vector representations in different ways. Both are important and are often used in combination in practice.

# 2.3 Joint Optimization

Existing DR systems follow the "encoding-indexing" paradigm, Inconsistent optimization objectives of the two steps during training will lead to sub-optimal performance. Previous studies have shown that the combination of optimized index construction and retrieval model training can make the index directly benefit from the annotation information and achieve better retrieval performance [9-11, 36, 38, 41]. Zhan et al [36] explore joint optimization of query encoding and product quantization, achieving state-of-the-art results. Furthermore, RepCONC [38] treats quantization as a constrained clustering process, requiring vectors to be clustered around quantized centroids. These approaches achieve good vector compression results, but the retrieval time complexity is still linear with respect to the corpus size. Tree-based methods have shown promising performance in recommender systems [40-42]. For example, TDM [41] proposes a tree-based deep recommendation model, which can be incorporated into any high-level model for retrieval. However, these methods are designed for recommendation systems and cannot be directly applied to retrieval systems.

Recently, Tay et al. [30] regarded transformer memory as the Differentiable Search Index (DSI). DSI maps documents into document IDs when indexing, and generates the corresponding docid for query when retrieving, so as to unify the training and indexing process. On this basis, Wang et al. [31] propose the Neural Corpus Indexer (NCI), which supports end-to-end document retrieval through a sequence-to-sequence network. These mode-based indexes provide a new paradigm for learning end-to-end retrieval systems. However, they are not suitable for larger-scale web searches.

Table 1: Important notations present in this paper.  

<table><tr><td colspan="2">Common Notation</td></tr><tr><td>q</td><td>a specific query</td></tr><tr><td>d</td><td>a specific document</td></tr><tr><td>n</td><td>a specific cluster node</td></tr><tr><td>D</td><td>the document corpus, D = {d1, d2, ..., dn}</td></tr><tr><td>D</td><td>dimension of dense embeddings</td></tr><tr><td>εc</td><td>embedding of cluster nodes</td></tr><tr><td>ed</td><td>embedding of documents</td></tr><tr><td>si</td><td>documents set assigned to leaf node i</td></tr><tr><td>Φ(·)</td><td>the query encoder</td></tr><tr><td colspan="2">Tree Parameters</td></tr><tr><td>β</td><td>branch balance factor, representing the number of child nodes of non-leaf nodes</td></tr><tr><td>γ</td><td>leaf balance factor, representing the maximum number of documents contained by one leaf node</td></tr><tr><td>b</td><td>beam size, indicating that the top b leaf nodes will be retrieved</td></tr><tr><td colspan="2">Overlapped Cluster Parameters</td></tr><tr><td>L</td><td>the number of queries in the training set</td></tr><tr><td>K</td><td>the number of leaf nodes</td></tr><tr><td>N</td><td>the number of documents</td></tr><tr><td>M</td><td>matrix M ∈ {0,1}L×K, which represents the relationship between the training queries and the leaf nodes</td></tr><tr><td>Y</td><td>matrix Y ∈ {0,1}L×N, which indicates the relevance of the queries and documents</td></tr><tr><td>C</td><td>matrix C ∈ {0,1}N×K, which represents the relationship between documents and leaf nodes</td></tr><tr><td>λ</td><td>number of repeated documents, which means that one document appears at most λ times in the leaf nodes after overlapped cluster</td></tr></table>

And when adding or deleting documents to the system, it is difficult to update the index. Compared with them, this paper focuses on the joint optimization of index and encoders, which preserves the advantages of the original index structure.

# 3 THE JTR MODEL

In this section, we first introduce the preliminary and tree structure in JTR. Secondly, the end-to-end joint optimization process is described in detail. Finally, we show how to optimize the cluster assignments of documents. Table 1 shows the important notations that are present in this paper.

# 3.1 Preliminary

Dense Retrieval can be formalized as follows: given a query  $q$  and the corpus  $D = \{d_1, d_2, \dots, d_n\}$ , the model needs to retrieve the top  $k$  most relevant documents from  $D$  with  $q$ . The training set is given in the form of  $\{(q_i, d_i) \dots\}$ , which means that  $q_i$  and  $d_i$  are relevant.

Tree-based index clusters all documents and prunes irrelevant cluster nodes at retrieval time to improve retrieval efficiency. As depicted in Figure 1, components in JTR are as follows:

- Deep Encoder  $\Phi$ , which encodes the query  $q$  into a  $\mathbb{D}$ -dimensional dense embedding. Following the previous work, BERT [7] is employed as the encoder.

Figure 1: Illustration of the JTR tree structures. The integer represents the sequence number of the node. In this case, The tree has a depth of 3, number of clusters 4, branch balance factor  $\beta = 2$ , and leaf balance factor  $\gamma = 4$ . The beam size  $b$  is set to 2.

Figure 2: Initialization of the tree structure. The integer in nodes indicates the number of documents the node contains. In this case, the total number of documents is 10, the branch balance factor  $\beta = 2$ , and the leaf balance factor  $\gamma = 5$ . If the node contains more documents than  $\gamma$ , then k-means will be performed on the document embedding in the node until all nodes contain less than  $\gamma$  documents. The embedding of each node is initialized as the cluster centroid embedding.

- Cluster Nodes, which consist of root node, intermediate nodes, and leaf nodes. The leaf nodes correspond to the fine-grained clusters, while the intermediate nodes correspond to the coarse-grained clusters. All clustering nodes are represented as trainable dense embeddings  $\widetilde{e}_{c_k} \in \mathbb{R}^{\mathbb{D}}$ , where  $k$  denotes the  $k$ -th clustering node. More specifically, given the dense embedding of a query  $\Phi(q) \in \mathbb{R}^{\mathbb{D}}$ , the relevance scores of the query and the clustered nodes are calculated by  $s = \widetilde{e}_c^T \cdot \Phi(q)$  at each level of the tree. Based on these scores, top  $b$  leaf nodes are returned and the documents within them are further ranked. The parameter  $b$  represents the beam size. Cluster node embeddings are crucial for a tree. We will describe how to optimize them in the following section.  
- Cluster Assignment, which represents the distribution of documents to leaf nodes. Assume there are  $\mathrm{K}$  leaf nodes, we use  $S^i = \{d_1^i, d_2^i, \ldots\}$  to denote the documents assigned to leaf node  $i$ . The initial cluster assignment is constructed by the k-means algorithm on document embeddings  $\widetilde{e}_{d_k} \in \mathbb{R}^{\mathbb{D}}$ , where  $k$  denotes the

$k$ -th document. The score  $s = \tilde{e}_d^T \cdot \Phi(q)$  is applied to indicate the relevance of the query to the document. As mentioned above, we only calculate the relevance score for the documents in the top  $b$  leaf nodes. Since k-means is an unsupervised clustering method, which divides documents into mutually exclusive clusters where documents in each cluster share the same semantics. However, this does not correspond to reality. Therefore, we relax this constraint and apply an overlapped cluster approach to optimize cluster assignment. The details will be described in section 3.3.

In this paper, we define  $\beta$  as the branch balance factor, representing the number of child nodes of non-leaf nodes, and  $\gamma$  as the leaf balance factor, representing the maximum number of documents contained by one leaf node.

JTR builds the tree index by recursively using the k-means algorithm. Specifically, given the corpus to be indexed, all documents are encoded into embeddings with the trained document encoder. Then, all embeddings are clustered into  $\beta$  clusters by the k-means algorithm. For each node that contains more than  $\gamma$  documents, the k-means is applied recursively until all nodes contain less than  $\gamma$  documents. The embedding of each node is initialized as the cluster centroid embedding. Figure 2 illustrates the initialization of the tree structure. We can observe that the tree index has the following properties: (1) Each non-leaf node of the tree has  $\beta$  child nodes. The depth of the tree is influenced by branch balance factor  $\beta$  and leaf balance factor  $\gamma$ . (2) The tree may be unbalanced. In the large corpus, this unbalance is usually insignificant and does not significantly affect the effectiveness of retrieval.

# 3.2 End-to-End Optimization

In this section, we introduce the end-to-end joint optimization process. In Figure 3, we compare the workflow of JTR and existing works. As shown in Figure 3(a), the existing methods follow a two-step "encoding-indexing" process. They first train the query encoder and document encoder with the ranking loss. After that, the well-trained document embeddings are used to train the tree-based index under the guidance of MSE loss. The training process of tree-based index is independent and cannot benefit from supervised data. In contrast, Figure 3(b) shows the optimization process of JTR. The joint optimization is mainly implemented by two strategies: unified contrastive learning loss and tree-based negative sampling. Next, we describe our motivation and the specific design in detail.

3.2.1 Motivation. For tree-based indexes, ensuring retrieval effectiveness and efficiency relies on two main components: pruning low-quality nodes and cluster assignment. For pruning low-quality nodes, tree-based indexes usually utilize beam search [25] to achieve efficient top  $b$  leaf nodes. Pruning the correct node at the upper levels can lead to a serious accumulation of errors. From this perspective, we argue that tree-based indexes should have the maximum heap property, which can well support the effectiveness of beam search. More specifically, the maximum heap property can be expressed as follows:

$$
p ^ {j} (n | q) = \frac {\underset {n _ {c} \in \{n ^ {\prime} s c h i l d r e n n o d e i n l e v e l j + 1 \}} {\max } p ^ {j + 1} \left(n _ {c} | q\right)}{\alpha^ {(j)}} \tag {2}
$$

(a) Workflow of existing tree-based methods

(b) Workflow of JTR  
Figure 3: Comparison of the workflow of JTR and existing methods. The solid arrows indicate that the gradient propagates backward, while the dashed arrows indicate that the gradient does not propagate.

Where  $p^j (n|q)$  represents the relevant probability between query  $q$  and cluster node  $n$  in level  $j$  and  $\alpha^{(j)}$  is the normalized term in level  $j$ . This formula indicates that the relevant probability of a parent node relies on the maximum relevant probability of its children nodes. In other words, given a specific query  $q$ , the parent of the optimal top node also belongs to the top node of the upper level. The maximum heap property is the basis of beam search [25].

In practice, we do not need to know the exact relevant probabilities of the cluster nodes. The relevance rank order of the nodes at each level is sufficient for accurate beam search. Therefore, we design a new contrastive learning loss to optimize cluster nodes. Moreover, the tree-based negative sample sampling technique is applied to improve the performance of JTR.

3.2.2 Unified Contrastive Learning Loss. Existing DR models usually use the contrastive learning loss function. Specifically, given a query  $q$ , let  $d^{+}$  and  $d^{-}$  be relevant documents and negative documents. The loss function is formulated as follows:

$$
L \left(q, d ^ {+}, d _ {1} ^ {-}, \dots \dots , d _ {n} ^ {-}\right) = - \log \frac {\exp \left(s \left(q , d ^ {+}\right)\right)}{\exp \left(s \left(q , d ^ {+}\right)\right) + \sum_ {j = 1} ^ {n} \exp \left(s \left(q , d _ {j} ^ {-}\right)\right)} \tag {3}
$$

Where  $s(,)$  is the relevance score. The purpose of contrastive learning loss is to make the query closer to related documents in the embedding space compared to the irrelevant ones. However, this loss function optimizes the embeddings of queries and documents and is not applicable to existing ANN methods. In JTR, we adapt this

Figure 4: The sampling process of JTR. The integer represents the sequence number of the node. We select the brother nodes of positive samples as negative samples.

loss to optimize the tree-based index. Specifically, given a training data  $(q_{k},d_{k})$ ,  $n_k$  is the leaf node that contains  $d_{k}$ . Therefore,  $n_k$  is the positive sample of the current level, and the leaf nodes not containing  $d_{k}$  are negative samples. On this basis, the ancestor nodes of  $n_k$  are also treated as positive samples of the level in which they are located. To make the tree-based index meet the properties of the maximum heap, we optimize it with negative sampling at each level. Let  $n^+$  denotes the node embedding of positive samples and  $n^{-}$  denotes the node embedding of negative samples. The unified contrastive learning loss is formalized as follows:

$$
L \left(q, n ^ {+}, n _ {1} ^ {-}, \dots \dots , n _ {n} ^ {-}\right) = - \log \frac {\exp \left(s \left(q , n ^ {+}\right)\right)}{\exp \left(s \left(q , n ^ {+}\right)\right) + \sum_ {j = 1} ^ {n} \exp \left(s \left(q , n _ {j} ^ {-}\right)\right)} \tag {4}
$$

where  $s(,)$  is the inner product. The unified contrastive learning loss function jointly optimizes the query encoder parameters and the embedding of clustering nodes. Since the ancestor node of the positive sample remains the positive sample during training, the tree-based index can meet the maximum heap property under the guidance of this loss function.

Training the tree-based index with the contrastive learning loss is non-trivial because the clustering assignment is not differentiable. To solve this problem, we initialize the cluster assignments using Figure 3(a) and only train the cluster node embeddings, which can benefit from supervised data directly.

3.2.3 Tree-based Negative Sampling. To further improve the performance of JTR, we design the tree-based negative sampling strategy. As shown in Figure 4, given a specific query, the leaf node with number three is a positive sample since it contains the relevant document. As mentioned above, the leaf node with number one is a positive sample at level<sub>1</sub> because it serves as the father node of the positive sample. Since the tree is constructed with the k-means algorithm, the brother nodes of each positive sample are considered to be closer to the positive sample in the embedding space. Hence, we select them as negative samples for training. As shown in Figure 3(b), the query embedding is fed into the "Search Negatives" module and brother nodes of the positive sample are returned under the current parameter.

# Algorithm 1: Tree Retrieval algorithm

Input: the trained tree  $\mathcal{T}$ , query  $q$ , beam size  $b$ , query encoder  $\Phi(\cdot)$

Output:  $k$  nearest approximate candidates

1 Result set  $A = \emptyset$ , candidate set  $Z = \{\text{the root node } n_1\}$ .  
2 while  $Z\neq \emptyset$  do

3 Remove all leaf nodes from  $Z$  and insert them into  $A$

4 Calculate  $s = \widetilde{e}_{c_n}^T\cdot \Phi (q)$  for each remaining node  $n\in Z$ $e_{c_n}$  is the embedding of current node.  
5 According to the  $s$ , top  $(b - len(A))$  nodes in  $Z$  are selected to form the set  $I$ . There are no leaf nodes in  $I$ .  
6 Update the  $Z\colon Z = \{\text{children nodes of } n \mid n \in I\}$ .

7 end

8 Compute  $s = \widetilde{e}_d^T \cdot \Phi(q)$  for documents  $d$  contained by nodes in  $A$  to get top  $k$  candidates.

# 3.3 Optimized Overlapped Cluster

As mentioned above, cluster assignment plays an essential role in the tree-based index. However, k-means is an unsupervised clustering method, which divides documents into mutually exclusive clusters, and documents in each cluster should share the same semantics. We observe in practice that the semantics of a document are complex and multi-topic. Putting each document in a cluster can limit the performance of the tree-based index. To solve this problem, we propose overlapped clusters to further optimize cluster assignment.

The overlapped cluster has been studied in the field of unsupervised learning [4, 21, 32]. Inspired by Liu et al. [21], we formulate the cluster assignment problem in information retrieval as an optimization problem. Suppose that there are L queries and N documents in the training corpus and all documents have been clustered and put into the K leaf nodes of the tree index.

First, we define the ground truth matrix  $\mathbb{Y} \in \{0,1\}^{\mathrm{L} \times \mathrm{N}}$  as:

$$
\mathbb {Y} _ {i, j} = \left\{ \begin{array}{l l} 1 & \text {i f d o c} _ {j} \text {i s r e l e v a n t t o q u e r y} _ {i} \\ 0 & \text {o t h e r w i s e} \end{array} \right. \tag {5}
$$

$\mathbb{Y}$  describes the relevance of the training queries and the documents and is the best result the model is trying to achieve.

Then, we input the training queries into the tree-based index to get the leaf nodes associated with them. We define the relationship between queries and leaf nodes as matrix  $\mathbb{M} \in \{0,1\}^{\mathrm{L} \times \mathrm{K}}$  and the original cluster assignment as matrix  $\mathbb{C} \in \{0,1\}^{\mathrm{N} \times \mathrm{K}}$ . It is worth noting that the initialized cluster assignment is obtained from Figure 3(a).

$$
\mathbb {M} _ {i, j} = \left\{ \begin{array}{c c} 1 & \text {i f q u e r y} _ {i} \text {m a t c h e s l e a f} _ {j} \\ 0 & \text {o t h e r w i s e} \end{array} \right. \tag {6}
$$

$$
\mathbb {C} _ {i, j} = \left\{ \begin{array}{c c} 1 & \text {i f d o c} _ {i} \text {m a t c h e s l e a f} _ {j} \\ 0 & \text {o t h e r w i s e} \end{array} \right. \tag {7}
$$

Then we can represent the relationship between queries and documents of JTR as:

$$
\hat {\mathbb {Y}} = \operatorname {B i n a r y} (\mathbb {M} \times \mathbb {C} ^ {\mathrm {T}}) \tag {8}
$$

where  $\text{Binary}(A) = I_A$  is the element-wise indicator function. When the element  $a > 0$ ,  $I_a = 1$ , which ensures  $\hat{\mathbb{Y}} \in \{0,1\}^{\mathrm{L} \times \mathrm{N}}$ .  $\times$  stands

Figure 5: Illustration of the optimized overlapped cluster. In this case, there are 3 queries, 4 leaf nodes, and 5 documents. The  $q_{i} \backslash l_{i} \backslash d_{i}$  represent the i-th query\leaf node\document respectively. We set the number of overlapped clustering  $\lambda = 2$ . The values in the red boxes are identified by the  $Proj(\cdot)$  function. In practice, if a document has the same value for two leaves in  $C^{*}$ , the  $Proj(\cdot)$  function prefers to keep the document in its original leaf.

for matrix cross product. Then, the performance of JTR can be expressed as the intersection of  $\mathbb{Y}$  and  $\hat{\mathbb{Y}}$ . The formula is as follows:

$$
\text {R e c a l l} = \left| \hat {\mathbb {Y}} \cap \mathbb {Y} \right| \tag {9}
$$

where  $|\cdot|$  returns the number of non-zero elements in the matrix. Since  $\mathbb{Y}$  and  $\mathbb{Y}$  are binary matrices, We can obtain the following formula:

$$
\operatorname {R e c a l l} = | \hat {\mathbb {Y}} \cap \mathbb {Y} | = T r (\mathbb {Y} ^ {\mathrm {T}} \times \hat {\mathbb {Y}}) \tag {10}
$$

where  $Tr(\cdot)$  returns the trace of the matrix. It is found that the performance of JTR is proportional to the trace of the matrix. In order to maximize the recall rate, we formulate the cluster assignment problem as follows:

$$
\underset {S _ {1}, S _ {2}, \dots \dots , S _ {k}} {\text {m a x i m i z e}} T r (\mathbb {Y} ^ {\mathrm {T}} \times \text {B i n a r y} (\mathbb {M} \times \mathbb {C} ^ {\mathrm {T}}) \tag {11}
$$

$$
s. t. \sum_ {i = 1} ^ {K} I _ {d \in S _ {i}} \leq \lambda , \forall d \in \{d _ {1}, \dots \dots . d _ {N} \} \tag {12}
$$

$$
\bigcup_ {i = 1} ^ {K} S _ {i} = \left\{d _ {1}, \dots \dots d _ {N} \right\} \tag {13}
$$

where  $S_{i}$  represents the  $i$ th cluster assignment.  $I$  is the indicator function. Equation 12 guarantees that any document  $d$  appears at most  $\lambda$  times in all clusters. Equation 13 ensures that the clusters include all documents.

This optimization problem is an NP-complete problem. Following Liu et al. [21], we approximate the objective function with a continuous, RelU-like function.

$$
\operatorname {B i n a r y} (\mathbb {M} \times \mathbb {C} ^ {\mathrm {T}}) \approx \max  (\mathbb {M} \times \mathbb {C} ^ {\mathrm {T}}, 0) = \mathbb {M} \times \mathbb {C} ^ {\mathrm {T}} \tag {14}
$$

In the optimization function, the cluster information is hidden in the matrix  $\mathbb{C}$ .  $Tr(\mathbb{Y}^{\mathrm{T}}\times \mathbb{M}\times \mathbb{C}^{\mathrm{T}})$  is linear with matrix  $\mathbb{C}$ , so the optimal solution of  $\mathbb{C}$  is the projection of  $\mathbb{Y}^{\mathrm{T}}\times \mathbb{M}$  onto the constraint set. Therefore, the optimization problem has a closed-form solution:

$$
C ^ {*} = \operatorname {P r o j} \left(\mathbb {Y} ^ {\mathrm {T}} \times \mathbb {M}\right) \tag {15}
$$

where the  $Proj(\cdot)$  operator selects the top  $\lambda$  elements for each line of the matrix. In the field of Information Retrieval, there exists the problem of data sparsity. Only a small set of the documents have the corresponding training queries, resulting in a lot of rows in  $\mathbb{Y}^{\mathrm{T}}$  with all entities being zeros. To solve this problem, we use the trained DR model to retrieve the top- $k$  documents of each training

query to construct matrix  $\bar{\mathbb{Y}}\in \{0,1\}^{\mathrm{L}\times \mathrm{N}}$ . In other words, if  $doc_{j}$  is the top  $k$  document recalled by query  $i$  using the DR model, then  $\bar{\mathbb{Y}}_{i,j} = 1$ . Hence, every line of  $\bar{\mathbb{Y}}$  has  $k$  nonzero entries. In short, the final solution is:

$$
C ^ {*} = \operatorname {P r o j} \left(\bar {\mathbb {Y}} ^ {\mathrm {T}} \times \mathbb {M}\right) \tag {16}
$$

The proposed solution is based on the intuition that two documents are more likely to cluster into the same class if they are related to the same query. It is worth noting that only the cluster assignment is changed in this process, the structure of the tree and the cluster node embeddings do not change.

After the  $\mathbb{C}^*$  is determined, We re-optimize the JTR as described in Section 3.2 to accommodate the new cluster assignment. Figure 5 shows the process of optimized overlapped cluster. We acknowledge that there exists a few documents that are not relevant to any training query. These documents are retained in the original clusters.

# 3.4 Tree Retrival

For a specific query, the tree retrieval process is described in algorithm 1. At each level of the tree, we select the top nodes and their child nodes as candidates for the next level. It saves retrieval time by pruning less relevant nodes. In the tree structure of JTR, the leaf nodes are not always at the same level. Therefore, the  $A$  set is used to preserve the encountered leaf nodes. In line 5 of Algorithm 1, we select top  $b - len(A)$  nodes in  $Z$  each time to create  $I$ . Since the leaf nodes in  $Z$  are removed and inserted into  $A$  in line 3, there are no leaf nodes in  $I$ .

In the JTR, the retrieval process is hierarchical and top-to-down. If the tree has K leaf nodes, N documents and beam size is  $b$  and branch balance factor is  $\beta$ , the time complexity of retrieving leaf nodes is  $O(\beta * b * \log K)$ . For retrieval in set  $A$ , the maximum number of documents in  $A$  is  $b * \gamma$ , where  $\gamma$  is leaf balance factor. Since  $\gamma$  can be approximated as  $N / K$ , the time complexity of this part is  $O(b * N / K)$ . In summary, the retrieval time complexity of JTR is  $O(\beta * b * \log K) + O(b * N / K)$ . The overall time complexity is below the linear time complexity, which greatly improves the retrieval efficiency.

Table 2: Results on the MS MARCO dataset. AQT stands for Average Query processing Time, which is measured by averaging time over each query of the MS MARCO Dev set with a single thread and a single batch on the CPU.  ${}^{ * }/{}^{* * }$  denotes that JTR performs significantly better than baselines at  $p < {0.05}/{0.01}$  level using the two-tailed pairwise t-test. The best method in each column is marked in bold.  

<table><tr><td rowspan="2">Model</td><td colspan="2">MARCO Passage</td><td>DL19 Passage</td><td>DL20 Passage</td><td rowspan="2">AQT ms</td><td colspan="2">MARCO Doc</td><td>DL19 Doc</td><td>DL20 Doc</td><td rowspan="2">AQT ms</td></tr><tr><td>MRR@100</td><td>R@100</td><td>NDCG@10</td><td>NDCG@10</td><td>MRR@100</td><td>R@100</td><td>NDCG@10</td><td>NDCG@10</td></tr><tr><td>IVFFlat</td><td>0.311*</td><td>0.778</td><td>0.580*</td><td>0.615</td><td>75</td><td>0.349**</td><td>0.839</td><td>0.572</td><td>0.527*</td><td>63</td></tr><tr><td>PQ</td><td>0.289**</td><td>0.717**</td><td>0.448**</td><td>0.546**</td><td>149</td><td>0.290**</td><td>0.788**</td><td>0.498**</td><td>0.458**</td><td>58</td></tr><tr><td>IVFPQ</td><td>0.252**</td><td>0.653**</td><td>0.532**</td><td>0.540**</td><td>15</td><td>0.279**</td><td>0.695**</td><td>0.465**</td><td>0.423**</td><td>10</td></tr><tr><td>JPQ</td><td>0.306**</td><td>0.832</td><td>0.611</td><td>0.607</td><td>152</td><td>0.347**</td><td>0.889</td><td>0.575</td><td>0.536</td><td>55</td></tr><tr><td>Annoy</td><td>0.144**</td><td>0.263**</td><td>0.361**</td><td>0.385**</td><td>132</td><td>0.148**</td><td>0.253**</td><td>0.504**</td><td>0.463**</td><td>57</td></tr><tr><td>FALCONN</td><td>0.295**</td><td>0.719**</td><td>0.554**</td><td>0.532**</td><td>23</td><td>0.321**</td><td>0.756**</td><td>0.496**</td><td>0.460**</td><td>9</td></tr><tr><td>FLANN</td><td>0.271**</td><td>0.629**</td><td>0.551**</td><td>0.578**</td><td>18</td><td>0.294**</td><td>0.649**</td><td>0.418**</td><td>0.368**</td><td>5</td></tr><tr><td>IMI</td><td>0.314</td><td>0.697**</td><td>0.542**</td><td>0.565**</td><td>37</td><td>0.348**</td><td>0.788**</td><td>0.535*</td><td>0.468**</td><td>26</td></tr><tr><td>HNSW</td><td>0.289**</td><td>0.732**</td><td>0.546**</td><td>0.559**</td><td>11</td><td>0.334**</td><td>0.783**</td><td>0.503**</td><td>0.507**</td><td>5</td></tr><tr><td>JTR</td><td>0.318</td><td>0.778</td><td>0.610</td><td>0.632</td><td>30</td><td>0.364</td><td>0.848</td><td>0.590</td><td>0.565</td><td>18</td></tr></table>

# 4 EXPERIMENT SETTINGS

In this section, we introduce our experimental settings, including implementation details, datasets and metrics, baselines.

# 4.1 Datasets and Metrics

The experiments are conducted on the dataset MS MARCO [27], which is a large-scale ad-hoc retrieval benchmark. It contains two large-scale tasks: Document Retrieval and Passage Retrieval. Passage Retrieval has a corpus of  $8.8M$  passages,  $0.5M$  training queries, and  $7k$  development queries. Document Retrieval has a corpus of  $3.3M$  documents,  $0.4M$  training queries, and  $5k$  development queries.

We conduct our experiments on two tasks from MS MARCO Dev, TREC2019 DL [5] and TREC2020 DL [6]. The MS MARCO Dev is extracted from Bing's search logs, and each query is marked as relevant to a few documents. TREC2019 DL and TREC2020 DL are collections that contain extensively annotated documents for each query, i.e., using four-level annotation criteria. We use  $R@100$  to evaluate the recall performance of different methods.  $MRR@100$  and  $NDCG@10$  are applied to measure ranking performance.

# 4.2 Baselines

We select several state-of-the-art ANN indexes as our baselines. Here are more details:

IVFFlat [24]: IVFFlat is the classic inverted index. When applied to dense retrieval, IVF defines nlist clusters in the vector space, and only the top nprobe clusters close to the query embedding are retrieved. We set nprobe to be the same as beam size  $b$ .

PQ [14]: PQ is an ANN index based on product quantization. We set the number of embedding segments to 32 and the number of coding bits to 8.

IVFPQ [14]: IVFPQ is one of the fastest indexes available, which combines "inverted index + product quantization". The parameter settings are the same as IVFFlat and PQ.

JPQ [36]: JPQ implements the joint optimization of query encoder and product quantization. We set the number of segments into which each embedding will be divided as 24.

Annoy [2]: Annoy improves retrieval efficiency by building a binary tree. We set the number of trees to 100.

FALCONN [28]: FALCONN is an optimized LSH method, which supports hyperplane LSH and cross polyhedron LSH. We use the recommended parameter settings as adopted in the corresponding paper.

FLANN [26]: FLANN is one of the most complete ANN open-source libraries, including liner, kdtree, kmeans tree, and other index methods. We use automatic tuning to get the best parameters.

IMI [1]: IMI is a multilevel inverted index, which combines product quantization and inverted index to bring very good search performance. The number of bits is set to 12.

HNSW [23]: HNSW is a typical and widely-used graph-based index. We set the number of links to 8 and ef-construction to 100.

For a fair comparison, all ANN indexes are operated on the document embeddings formed by STAR [37]. For IVFFlat, PQ, IVFPQ, HNSW and IMI, we implement them with the Faiss  $^{2}$ . For Annoy  $^{3}$ , FALCONN  $^{4}$  and FLANN  $^{5}$ , we use the official library to implement them.

# 4.3 Implementation Details

We implemented JTR using PyTorch and Transformers. Initial embeddings were obtained using STAR [37] for all documents. We also load the checkpoint of STAR to warm up the query encoder. In our experiment, the default  $\gamma$  is 1000 and  $\beta$  is 10. The dimension of embeddings is 768. For the training setup, we use the AdamW [22] optimizer with a batch size of 32. The learning rate is set to  $5 \times 10^{-6}$ . In the overlapped cluster of the tree, we use the top 100 documents obtained from ADORE-STAR [37] to build matrix  $\bar{\mathbb{Y}}$ . All experiments are evaluated on the workstation with Xeon Gold 5218 CPUs and RTX 3090 GPUs.

# 5 EXPERIMENT RESULTS

# 5.1 Comparison with ANN Methods

The performance comparisons between SAILER and baselines are shown in Table 2. We derive the following observations from the experiment results.

(a) MRR curves

(b) Recall curves

- PQ and JPQ are ANN methods based on product quantization, which have a linear time complexity with respect to the corpus size. As expected, their average query processing time is the longest among all baselines. IVFPQ is an inverted index based on product quantization, which improves retrieval efficiency but damages retrieval effectiveness.  
- Benefiting from a different index structure, the average query processing time of IVFPQ, FALCONN, FLANN, and HNSW is lower than that of JTR. However, they suffered a loss in effectiveness.  
- Annoy and FLANN are existing tree-based indexes. Compared with them, JTR achieved the best results. To the best of our knowledge, JTR is the best tree-based index available, which shows the great potential of tree-bases indexes for dense retrieval.  
- Overall, JTR performs significantly better than the baseline method on most measures. In terms of effectiveness, JTR achieves the best MRR/NDCG and the second best recall performance (i.e., R@100) on all datasets. Compared to the baseline with the best recall (i.e., JPQ), JTR achieves 3 to 5 times latency speedup. We can conclude that JTR has a very competitive effectiveness compared to other ANN indexes.

To further analyze the ability of different indexes to balance efficiency and effectiveness, we plot AQT-MRR curves with varying parameters. As shown in Figure 6, we have the following findings:

- IVFPQ and FALCONN are limited by resources, which makes it difficult to achieve high retrieval performance.  
- When our requirement on recall is extremely high (e.g., larger than 0.8), brute-force search could be more efficient than any indexing technique, which makes brute-force search-based algorithms like JPQ more efficient than JTR. After all, any indexing process would add more time complexity if we eventually need to


Figure 7:  $\mathbf{R}@\mathbf{100}$  different  $\lambda$  ranging from 1 to 5 and  $b$  ranging from 10 to 50.

Figure 6: Trade-off curves for different ANN methods. AQT stands for Average Query processing Time. Bottom and right is better.  
Figure 8: MRR@100 versus different  $\lambda$  ranging from 1 to 5 and  $b$  ranging from 10 to 50.  
Figure 9: AQT versus different  $\lambda$  ranging from 1 to 5 and  $b$  ranging from 10 to 50. AQT stands for Average Query processing Time.

check all documents in the corpus. However, such high requirements on recall are not common in web search or other similar retrieval tasks.

- To sum up, JTR has a very outstanding effectiveness-efficiency trade-off ability, which can always achieve better results with a short retrieval latency. JTR provides a potential solution to balance efficiency and effectiveness for dense retrieval.

# 5.2 Hyperparameter Sensitivity

5.2.1 Hyperparameters for Tree Retrieval. In this section, we investigate the impact of the overlap number  $\lambda$  and the beam size  $b$ . All experiments were conducted on the MS MARCO Doc Dev dataset.

We set  $\lambda = 1,2,3,4,5$  and  $b = 10,20,30,40,50$  to observe the retrieval performance of JTR. Figures 7, 8 and 9 show the changes in retrieval quality and retrieval efficiency with hyperparameters. We can get the following observations:

- Both retrieval quality and latency increase with  $\lambda$ . When  $\lambda$  increases, the leaf node contains more documents. The number of candidate documents goes up which leads to better results and higher latency.

Table 3: Ablation study on MS MARCO Dev Doc dataset.  

<table><tr><td rowspan="2">Model</td><td colspan="3">MARCO Dev Doc</td></tr><tr><td>MRR@100</td><td>R@100</td><td>AQT(ms)</td></tr><tr><td>IVFFlat</td><td>0.310</td><td>0.714</td><td>24</td></tr><tr><td>Tree</td><td>0.256</td><td>0.556</td><td>5</td></tr><tr><td>+Joint Optimization</td><td>0.296</td><td>0.640</td><td>5</td></tr><tr><td>+Reorganize clusters</td><td>0.303</td><td>0.678</td><td>5</td></tr><tr><td>+Overlapped clustering</td><td>0.327</td><td>0.743</td><td>8</td></tr></table>

- The slope of MRR and R curve decreases with the increase of  $\lambda$ , but the slope of AQT remains unchanged. That means the gain from overlapped cluster is getting smaller.  
- When we fix  $\lambda$ , it is found that the gains of MRR and R decreased with the increase of  $b$ . The reason may be that the most relevant documents have already clustered in the top leaf nodes.  
- In Figure 9, we note that the gap between curves increases when  $\lambda$  is larger, which shows that a large  $\lambda$  could lead to larger computation costs on each leaf node.  
- Overall, the increase of  $b$  and  $\lambda$  can lead to better performance and longer latency. Therefore, we recommend the moderate value for  $\lambda$  and  $b$  to maximize the trade-off between effectiveness and efficiency.

5.2.2 Hyperparameter for Overlapped Cluster. In this section, we fix the model as ADORE-STAR and change  $k$  from 25 to 150, where  $k$  denotes the number of documents recalled with the dense retrieval model. Figure 10 presents the retrieval performance on each  $k$  and  $b$  value. Obviously, the larger  $k$ , the more information contained in  $\bar{\mathbb{Y}}$ , and the better the final performance. When  $k$  is larger than 100, the gain becomes smaller. Therefore, we set  $k = 100$  in our experiments, which provides sufficient information.

(a) MRR curves

(b) Recall curves  
Figure 10: MRR@100 and R@100 versus different  $k$  ranging from 25 to 150 and  $b$  ranging from 10 to 50.

# 5.3 Ablation Study

In this section, we conduct ablation studies on JTR to explore the importance of different components. We use IVFFlat as the baseline.

IVFFlat forms as many clusters as the leaf nodes of the tree. The nprobe of IVFFlat and the beam size of JTR are both set to 10. We use the following four model variants:

- Tree: The STAR model is used to obtain the document embeddings, and then we build the tree index with MSE loss.  
- +Joint Optimization: Joint optimization of query encoders and tree-based indexes using the unified contrastive learning loss function.  
+Reorganize Clusters: Update the clustering of documents, which means the number of overlapped clustering  $\lambda = 1$  
- +Overlapped Clustering: Documents can be re-occurring in the clustering nodes. For convenience, the number of overlapped clustering  $\lambda = 2$ .

Table 3 shows the MRR@100 and R@100 on the development set of the MS MARCO Document Ranking task. As the results show, the tree structure significantly reduces the retrieval latency. All of Joint Optimization, Reorganize Clusters and Overlapped Clustering improve the performance of JTR. Tree-based indexes benefit from supervised data directly by Joint Optimization. Reorganize clusters makes the distribution of embeddings more reasonable. Overlapped cluster mines different semantic information in documents. This result demonstrates the effectiveness of our method.

# 6 CONCLUSION

To improve the efficiency of the DR models while ensuring the effectiveness, we propose JTR, which jointly optimizes tree-based index and query encoder in an end-to-end manner. To achieve this goal, we carefully design a unified contrastive learning loss and tree-based negative sampling strategy. Through the above strategies, the constructed index tree possess the maximum heap property which easily supports beam search. Moreover, for cluster assignment, which is not differentiable w.r.t. contrastive learning loss, we introduce overlapped cluster optimization. We further conducted extensive experiments on several popular retrieval benchmarks. Experimental results show that our approach achieves competitive results compared with widely-adopted baselines. Ablation studies demonstrate the effectiveness of our strategies. Unfortunately, since different indexes have varying degrees of code optimization, we do not report the memory case in our paper. In the future, we will try to jointly optimize PQ and tree-based index to achieve the "effectiveness-efficiency-memory" tradeoff.

# ACKNOWLEDGMENTS

This work is supported by the Natural Science Foundation of China (62002194), Tsinghua University Guoqiang Research Institute, Tsinghua-Tencent Tiangong Institute for Intelligent Computing and the Quan Cheng Laboratory.

# REFERENCES

[1] Artem Babenko and Victor Lempitsky. 2014. The inverted multi-index. IEEE transactions on pattern analysis and machine intelligence 37, 6 (2014), 1247-1260.  
[2] Erik Bernhardsson. 2017. Annoy: approximate nearest neighbors in  $\mathrm{C + + }$  /Python optimized for memory usage and loading/saving to disk. GitHub https://github. com/spotify/annoy (2017).  
[3] Jia Chen, Haitao Li, Weihang Su, Qingyao Ai, and Yiqun Liu. [n.d.]. THUIR at WSDM Cup 2023 Task 1: Unbiased Learning to Rank. ([n.d.]).  
[4] Guillaume Cleuziou. 2008. An extended version of the k-means method for overlapping clustering. In 2008 19th International Conference on Pattern Recognition. IEEE, 1-4.  
[5] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M Voorhees. 2020. Overview of the TREC 2019 deep learning track. arXiv preprint arXiv:2003.07820 (2020).  
[6] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Fernando Campos, and Ellen M. Voorhees. 2020. Overview of the TREC 2020 Deep Learning Track. ArXiv abs/2003.07820 (2020).  
[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).  
[8] Qian Dong, Yiding Liu, Suqi Cheng, Shuaiqiang Wang, Zhicong Cheng, Shuzi Niu, and Dawei Yin. 2022. Incorporating Explicit Knowledge in Pre-trained Language Models for Passage Re-ranking. arXiv preprint arXiv:2204.11673 (2022).  
[9] Yan Fang, Jingtao Zhan, Yiqun Liu, Jiaxin Mao, Min Zhang, and Shaoping Ma. 2022. Joint Optimization of Multi-vector Representation with Product Quantization. In Natural Language Processing and Chinese Computing: 11th CCF International Conference, NLPCC 2022, Guilin, China, September 24-25, 2022, Proceedings, Part I. Springer, 631-642.  
[10] Chao Feng, Wuchoo Li, Defu Lian, Zheng Liu, and Enhong Chen. 2022. Recommender Forest for Efficient Retrieval. Advances in Neural Information Processing Systems 35 (2022), 38912-38924.  
[11] Chao Feng, Defu Lian, Zheng Liu, Xing Xie, Le Wu, and Enhong Chen. 2022. Forest-based Deep Recommender. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 523-532.  
[12] Luyu Gao, Zhuyun Dai, Tongfei Chen, Zhen Fan, Benjamin Van Durme, and Jamie Callan. 2021. Complement lexical retrieval model with semantic residual embeddings. In European Conference on Information Retrieval. Springer, 146-160.  
[13] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence 36, 4 (2013), 744-755.  
[14] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117-128.  
[15] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data 7, 3 (2019), 535-547.  
[16] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906 (2020).  
[17] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 39-48.  
[18] Jinhyuk Lee, Minjoon Seo, Hannaneh Hajishirzi, and Jaewoo Kang. 2019. Contextualized sparse representations for real-time open-domain question answering. arXiv preprint arXiv:1911.02896 (2019).  
[19] Haitao Li, Jia Chen, Weihang Su, Qingyao Ai, and Yiqun Liu. 2023. Towards Better Web Search Performance: Pre-training, Fine-tuning and Learning to Rank. arXiv preprint arXiv:2303.04710 (2023).  
[20] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2020. Distilling dense representations for ranking using tightly-coupled teachers. arXiv preprint arXiv:2010.11386 (2020).  
[21] Xuanqing Liu, Wei-Cheng Chang, Hsiang-Fu Yu, Cho-Jui Hsieh, and Inderjit Dhillon. 2021. Label disentanglement in partition-based extreme multilabel classification. Advances in Neural Information Processing Systems 34 (2021), 15359-15369.  
[22] Ilya Loshchilov and Frank Hutter. 2018. Fixing weight decay regularization in adam. (2018).

[23] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824-836.  
[24] Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1723-1727.  
[25] Clara Meister, Tim Vieira, and Ryan Cotterell. 2020. Best-first beam search. Transactions of the Association for Computational Linguistics 8 (2020), 795-809.  
[26] Marius Muja and David Lowe. 2009. Flann-fast library for approximate nearest neighbors user manual. Computer Science Department, University of British Columbia, Vancouver, BC, Canada 5 (2009).  
[27] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human generated machine reading comprehension dataset. In CoCo@ NIPs.  
[28] Ninh Pham and Tao Liu. 2022. Falcon++: A Locality-sensitive Filtering Approach for Approximate Nearest Neighbor Search. arXiv preprint arXiv:2206.01382 (2022).  
[29] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2020. RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2010.08191 (2020).  
[30] Yi Tay, Vinh Q Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a differentiable search index. arXiv preprint arXiv:2202.06991 (2022).  
[31] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, et al. 2022. A Neural Corpus Indexer for Document Retrieval. arXiv preprint arXiv:2206.02743 (2022).  
[32] Joyce Jyoung Whang, Inderjit S Dhillon, and David F Gleich. 2015. Nonexhaustive, overlapping k-means. In Proceedings of the 2015 SIAM international conference on data mining. SIAM, 936-944.  
[33] Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, et al. 2023. T2Ranking: A large-scale Chinese Benchmark for Passage Ranking. arXiv preprint arXiv:2304.03679 (2023).  
[34] Shenghao Yang, Haitao Li, Zhumin Chu, Jingtao Zhan, Yiqun Liu, Min Zhang, and Shaoping Ma. 2022. THUIR at the NTCIR-16 WWW-4 Task. Proceedings of NTCIR-16, to appear (2022).  
[35] Shenghao Yang, Yiqun Liu, Xiaohui Xie, Min Zhang, and Shaoping Ma. 2023. Enhance Performance of Ad-hoc Search via Prompt Learning. In Information Retrieval: 28th China Conference, CCIR 2022, Chongqing, China, September 16–18, 2022, Revised Selected Papers. Springer, 28–39.  
[36] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Jointly optimizing query encoder and product quantization to improve retrieval performance. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2487-2496.  
[37] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1503-1512.  
[38] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2022. Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 1328-1336.  
[39] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. RepBERT: Contextualized text embeddings for first-stage retrieval. arXiv preprint arXiv:2006.15498 (2020).  
[40] Han Zhu, Daqing Chang, Ziru Xu, Pengye Zhang, Xiang Li, Jie He, Han Li, Jian Xu, and Kun Gai. 2019. Joint optimization of tree-based index and deep model for recommender systems. Advances in Neural Information Processing Systems 32 (2019).  
[41] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1079-1088.  
[42] Jingwei Zhuo, Ziru Xu, Wei Dai, Han Zhu, Han Li, Jian Xu, and Kun Gai. 2020. Learning optimal tree models under beam search. In International Conference on Machine Learning. PMLR, 11650-11659.

# Footnotes:

Page 0: *Corresponding author 
Page 1: $^{1}$ Code are available at https://github.com/CSHaitao/JTR. 
Page 6: $^{2}$ https://github.com/facebookresearch/faiss <sup>3</sup>https://github.com/spotify/annoy $^{4}$ https://github.com/FALCONN-LIB/FALCONN 5 https://github.com/flann-lib/flann <sup>6</sup>https://huggingface.co/docs/transformers/index 
