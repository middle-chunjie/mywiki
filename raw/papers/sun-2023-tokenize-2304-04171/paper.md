# Learning to Tokenize for Generative Retrieval

Weiwei Sun $^{1}$ , Lingyong Yan $^{2}$ , Zheng Chen $^{1}$ , Shuaiqiang Wang $^{2}$ , Haichao Zhu $^{2}$ , Pengjie Ren $^{1}$ , Zhumin Chen $^{1}$ , Dawei Yin $^{2}$ , Maarten de Rijke $^{3}$ , Zhaochun Ren $^{1}$

$^{1}$ Shandong University  $^{2}$ Baidu Inc  $^{3}$ University of Amsterdam

{sunnweiwei,lingyongy,shqiang.wang}@gmail.com,{202000130223,renpengjie,chenzhumin,zhaochun. ren}@sdu.edu.cn

hczhu@ir.hit.edu.cn, yindawei@acm.org, m.derijke@uva.nl

# ABSTRACT

Conventional document retrieval techniques are mainly based on the index-retrieve paradigm. It is challenging to optimize pipelines based on this paradigm in an end-to-end manner. As an alternative, generative retrieval represents documents as identifiers (docid) and retrieves documents by generating docids, enabling end-to-end modeling of document retrieval tasks. However, it is an open question how one should define the document identifiers. Current approaches to the task of defining document identifiers rely on fixed rule-based docids, such as the title of a document or the result of clustering BERT embeddings, which often fail to capture the complete semantic information of a document.

We propose GENRET, a document tokenization learning method to address the challenge of defining document identifiers for generative retrieval. GENRET learns to tokenize documents into short discrete representations (i.e., docids) via a discrete auto-encoding approach. Three components are included in GENRET: (i) a tokenization model that produces docids for documents; (ii) a reconstruction model that learns to reconstruct a document based on a docid; and (iii) a sequence-to-sequence retrieval model that generates relevant document identifiers directly for a designated query. By using an auto-encoding framework, GENRET learns semantic docids in a fully end-to-end manner, where the produced docids can be reconstructed back to the original documents to ensure their semantics. We also develop a progressive training scheme to capture the autoregressive nature of docids and to stabilize training.

We conduct experiments on the NQ320K, MS MARCO, and BEIR datasets to assess the effectiveness of GENRET. GENRET establishes the new state-of-the-art on the NQ320K dataset. Especially, compared to generative retrieval baselines, GENRET can achieve significant improvements on the unseen documents (e.g., at least  $+14\%$  relative improvements in terms of R@1). Furthermore, GenRet can better represent and retrieve documents that have not been seen during the training phase compared to previous rule-based tokenization methods. GENRET also outperforms comparable baselines on MS MARCO and BEIR, demonstrating the method's generalizability. $^{1}$

# 1 INTRODUCTION

Document retrieval plays an essential role in web search applications and various downstream knowledge-intensive tasks, such as question-answering and dialogue systems as it is aimed on identifying relevant documents to satisfy users' queries. Most traditional document retrieval approaches apply sparse retrieval methods, which rely on building an inverted index with term matching metrics such as TF-IDF [38], query likelihood [22], or BM25 [39].

The term matching metrics, however, often suffer from a lexical mismatch [24].

Major progress has recently been made in dense retrieval (DR) models due to advances in pre-trained language models (LMs) [13, 19, 31, 47]. As illustrated in Figure 1 (a), DR methods learn dense representations of both queries and documents using dual encoders, and subsequently retrieve relevant documents using maximal inner product search (MIPS) [18, 19]. DR methods are able to address the lexical mismatch issue with state-of-the-art performance on various retrieval tasks [25, 29].

Despite their success, DR approaches face two main limitations [5, 28]: (i) DR models employ an index-retrieval pipeline with a fixed search procedure (MIPS), making it difficult to jointly optimize all modules in an end-to-end way; and (ii) The learning strategies (e.g., contrastive learning [19]) are usually not consistent with the pretraining objectives, such as the next token prediction [3], which makes it hard to leverage knowledge in pre-trained LMs [1].

(a) Dense retrieval

(b) Generative retrieval  
Figure 1: Two types of document retrieval models: (a) Dense retrieval encodes queries and documents to dense vectors and retrieves documents by MIPS; (b) Generative retrieval tokenizes documents as docids and autoregressively generates docids as retrieval results.

Generative retrieval. Recently, generative retrieval has emerged as a new paradigm for document retrieval [1, 5, 42, 46, 52, 53]. As illustrated in Figure 1 (b), generative retrieval models directly generate a ranked list of document identifiers (docids) for a given query using generative LMs. Specifically, there are two main steps involved in generative retrieval models: (i) Document tokenization, where each document in the corpus is tokenized as a sequence of discrete characters, i.e., docids, and (ii) Generation as retrieval, where the docids of relevant documents are output by autoregressively decoding for a given query. Unlike DR, the generative paradigm presents an end-to-end solution for document retrieval tasks [42]. It also offers a promising approach to better exploit the capabilities of recent large LMs [1, 46].

Learning to tokenize documents. Document tokenization plays a crucial role in generative retrieval, as it defines how the document is distributed in the semantic space [42]. And it is still an open problem how to define the document identifiers. Most previous

generative methods tend to employ rule-based document tokenizers, such as generating titles or URLs [5, 52], or clustering results from off-the-shelf document embeddings [42, 46]. However, such rule-based methods are usually ad-hoc and do not generalize well. In particular, the tokenization results potentially perform well on retrieving documents that have been seen during training, but generalize poorly to new or out-of-distribution documents [23, 27].

The proposed method. To address the above problem, this paper proposes GENRET, a document tokenization learning framework that learns to tokenize a document into semantic docids in a discrete auto-encoding scheme. Specifically, GENRET consists of a shared sequence-to-sequence-based document tokenization model, a retrieval model, and a document reconstruction model. In the proposed auto-encoding learning scheme, the tokenization model learns to convert documents to discrete docids, which are subsequently utilized by the reconstruction model to reconstruct the original document. The generative retrieval model is trained to generate docids in an autoregressive manner for a given query. The above three models are optimized in an end-to-end fashion to achieve seamless integration.

We further identify two challenges when using auto-encoding to optimize a generative retrieval model: (i) docids with an autoregressive nature, and (ii) docids with diversity. To address the first challenge and also to stabilize the training of GENRET, we devise a progressive training scheme. This training scheme allows for a stable training of the model by fixing optimized prefix docids  $z_{<t}$ . To optimize the docids at each step, three proposed losses are utilized: (i) a reconstruction loss for predicting the document using the generated docid, (ii) a commitment loss for committing the docid and to avoid forgetting, and (iii) a retrieval loss for optimizing the retrieval performance end-to-end. To address the second challenge, we propose a parameter initialization strategy and a re-assignment of the docid based on a diverse clustering technique to increase the diversity of the generated docids.

Experiments. We conduct experiments on three well-known document retrieval benchmark datasets: (i) NQ320K, with a subset of Wikipedia [21, 42]; (ii) MS MARCO, with web pages relevant to a set of search queries [4, 52]; and (iii) BEIR, with heterogeneous retrieval tasks for out-of-distribution evaluation [43]. Our experimental results demonstrate that GENRET attains superior retrieval performance against state-of-the-art dense or generative retrieval models. Experiments on NQ320K show that GENRET establishes the new state-of-the-art on this dataset, achieving  $+14\%$  relative improvements on the unseen test set compared to the best baseline method. Experiments on MS MARCO and six BEIR datasets also shows that GENRET significantly outperforms existing generative methods and achieves competitive results compared to the best dense retrieval model. Experiments on retrieving new documents, analytical experiments, and efficiency analysis confirm the effectiveness of the proposed model.

Contributions. In this paper we make the following contributions: (i) We propose GENRET, a generative retrieval model that represents documents as semantic discrete docids. To the best of our knowledge, this is the first tokenization learning method for document retrieval. (ii) We propose an auto-encoding approach, where the docids generated by our tokenization model are reconstruct by a

reconstruction model to ensure the docids capture the semantic information of the document. (iii) We devise a progressive training scheme to model the autoregressive nature of docids and stabilize the training process. (iv) Experimental results demonstrate that GENRET achieves significant improvements, especially on unseen documents, compared to generative retrieval baselines.

# 2 PRELIMINARIES

The document retrieval task can be formalized as the process of retrieving a relevant document  $d$  for a search query  $q$  from a collection of documents  $\mathcal{D}$ . Each document,  $d\in \mathcal{D}$ , is a plain text document consisting of a sequence of tokens, denoted as  $d = \{d_{1},\dots ,d_{|d|}\}$  where  $|d|$  represents the total number of tokens in the document.

Unlike dense retrieval methods, which return the most relevant documents based on the relevance score of each document with respect to a given query  $q$ , generative retrieval models aim to directly generate documents for a given query  $q$  using a generative model.

Document tokenization. For generative retrieval models, it is usually challenging and computationally inefficient to directly generate original documents of typically long length. Therefore, most existing approaches rely on the technique named document tokenization, which represents a document  $d = \{d_1, \ldots, d_{|d|}\}$  as a shorter sequence of discrete tokens (docid)  $z = \{z_1, \ldots, z_t, \ldots, z_M\}$ , where each token  $z_t$  is as a  $K$ -way categorical variable, with  $z_t \in 1, 2, \ldots, K$ .

As an alternative sequence of the original document, the tokenized docid  $z$  should satisfy the following two properties: (i) different documents have short but different docids; (ii) docids capture the semantics of their associated documents as much as possible [42]. Because  $z$  is a sequence of a fixed length and usually shorter than the original document  $d$ , the model's training and inference can be simplified and more efficient.

As mentioned above, this paper employs a tokenization model  $Q \colon d \to z$  to map  $d$  to docid  $z$ . More details about  $Q$  are provided in Section 3.1.

Generation as retrieval. After tokenizing each document to docid  $z$ , a generative retrieval model  $P \colon q \to z$  learns to retrieve relevant documents by generating a query  $q$  to a docid  $z$  autoregressively:

$$
z = \prod_ {t = 1} ^ {M} P \left(z _ {t} \mid z _ {<   t}, q\right), \tag {1}
$$

where  $z_{<t}$  denotes the prefix of  $z$  up to time step  $t$ . The model employs a constrained decoding technique to ensure that the generated docid  $z$  exists in the corpus  $\mathcal{D}$  [5]. This is achieved by constructing a prefix tree based on the valid docids in  $\mathcal{D}$  and truncating the generation probability of invalid docids to 0.0 during the decoding process. The model retrieves multiple documents using beam search.

# 3 GENRET

Conventionally, document tokenization is done by a fixed preprocessing step, such as using the title of a document or the results of hierarchical clustering obtained from BERT [5, 42]. However, it has been observed that such ad-hoc document tokenization methods often fail to capture the complete semantics of a document. For

Figure 2: An overview of the proposed method. The proposed method utilizes a document tokenization model to convert a given document into a sequence of discrete tokens, referred to as a docid. This tokenization process allows for the reconstruction of the original document through a reconstruction model. Subsequently, an autoregressive generation model is employed to retrieve documents through the generation of their respective docids.

example, the title of a web page often does not exist or has low relevance to the content of the web page, and the use of clustering-based docids arbitrarily defines the document in discrete space.

In this paper, we propose GENRET, a novel tokenization learning method based on discrete auto-encoding, to learn semantic docid in a fully end-to-end manner. Figure 2 gives an overview of the proposed method. The proposed GENRET comprises three main components: (i) a sequence-to-sequence based retrieval model  $(P(z\mid q))$ , (ii) a document tokenization model  $(Q(z\mid d))$ , and (iii) a reconstruction model  $(R(d\mid z))$ . The document tokenization model tokenizes a document  $d$  into unique discrete variables  $z$ , and the retrieval model is trained to generate the latent variables  $z$  for a given query  $q$ . In addition, the reconstruction model is used to re-generate the original document from  $z$  to ensure  $z$  captures the semantics of the original document as much as possible.

We detail the model architecture of the document tokenization and document retrieval model in Section 3.1, the devised reconstruction model in Section 3.2, and the model optimization method in Section 3.3.

# 3.1 Document tokenization and retrieval model

Since document tokenization and generative retrieval both aim to map the input text to a discrete docid, we use a shared T5 Transformer architecture for document tokenization and generative retrieval models. Specifically, given an input text  $d^2$ , the T5-based tokenization model encodes  $d$  and a prefix of docid  $z_{<t}$  and continuously produces latent representation  $\mathbf{d}_t$  of  $d$  at time step  $t$ :

$$
\mathbf {d} _ {t} = \operatorname {D e c o d e r} (\operatorname {E n c o d e r} (d), z _ {<   t}) \in \mathbb {R} ^ {D}, \tag {2}
$$

where  $D$  denotes the hidden size of the model,  $\mathrm{Encoder}(d)$  denotes the output of the Encoder.

Then, the tokenization model generates a token for each document based on  $\mathbf{d}_t$ . At each timestep  $t$ , we define an external embedding matrix named codebook  $\mathbf{E}_t \in \mathbb{R}^{K \times D}$ , where  $K$  is the size of the discrete latent space. There are  $K$  embedding vectors  $\mathbf{e}_{t,j} \in \mathbb{R}^D, j \in [K]$ , and each vector  $\mathbf{e}_{t,j}$  can be regarded as the centroid of a segmentation.

Based on the codebook  $\mathbf{E}_t$ , the discrete latent variable  $z_{t}$  at timestep  $t$  is calculated by a dot-product look-up using the codebook

$\mathbf{E}_t$

$$
Q \left(z _ {t} = j \mid z _ {<   t}, d\right) = \operatorname {S o f t m a x} _ {j} \left(\mathbf {d} _ {t} \cdot \mathbf {E} _ {t} ^ {\top}\right), \tag {3}
$$

where  $Q(z_{t} = j \mid z_{<t}, d)$  denotes the probability of tokenizing  $d$  to a particular value  $j \in [K]$  at timestep  $t$ , Softmax $_{j}$  is a softmax function to output the probability of axis  $j$ .

Finally, the tokenization model selects the docid that achieves the maximum probability to define the docid  $z_{t}$ :

$$
z _ {t} = \underset {j} {\arg \max } Q \left(z _ {t} = j \mid z _ {<   t}, d\right). \tag {4}
$$

in which the model selects the id  $j$  corresponding to the embedding vector  $\mathbf{e}_{t,j}$  with the maximum inner-product with  $\mathbf{d}_t$  as the docid  $z_{t}$  at timestep  $t$ .

The generative retrieval model  $P(z \mid q)$  shares the same architecture as  $Q(z \mid d)$ , while generating  $z$  using the input query  $q$ , as formulated in Eq. 1.

# 3.2 Document reconstruction model

The docid generated by the tokenization model  $Q$  is required to capture the semantic information of the document. To this end, we propose an auto-encoding training scheme, where a reconstruction model  $R \colon z \to d$  that predicts  $d$  using  $z$  is designed to force the tokenization model  $Q \colon d \to z$  to reproduce a docid  $z$  that can be reconstructed back-to-the original document.

The input of the reconstruction model is docid  $z$ , and the output is its associated document  $d$ . We first embed  $z$  into representation matrix  $\mathbf{z} = \{\mathbf{z}_1, \dots, \mathbf{z}_M\} \in \mathbb{R}^{M \times D}$  using the codebook of the tokenization model:

$$
\mathbf {z} = \left\{\mathbf {e} _ {1, z _ {1}}, \mathbf {e} _ {2, z _ {2}}, \dots , \mathbf {e} _ {M, z _ {M}} \right\} \in \mathbb {R} ^ {M \times D}, \tag {5}
$$

where each  $t \in [M]$ ,  $\mathbf{z}_t = \mathbf{e}_{t,z_t} \in \mathbb{R}^D$  is the embedding vector of  $z_t$  in the  $t$ -step codebook  $\mathbf{E}_t$ .

We then devise a retrieval-based reconstruction model that predicts the target document  $d$  by retrieving it from document collection  $\mathcal{D}$ , based on the inputs  $\mathbf{z}$ . The relevance score between the input docid  $z$  and the target document  $d$  is defined as follows:

$$
R (d \mid \mathbf {z}) = \prod_ {t = 1} ^ {M} \frac {\exp \left(\mathbf {z} _ {t} \cdot \operatorname {s g} \left(\mathbf {d} _ {t} ^ {\top}\right)\right)}{\sum d ^ {*} \in S \left(z _ {<   t}\right) \exp \left(\mathbf {z} _ {t} \cdot \operatorname {s g} \left(\mathbf {d} _ {t} ^ {* \top}\right)\right)}, \tag {6}
$$

where  $S(z_{<t})$  is a sub-collection of  $\mathcal{D}$  consisting of documents that have a docid prefix that is the same as  $z_{<t} \cdot d^{*} \in S(z_{<t})$  represents a document from the sub-collection  $S(z_{<t})$ .  $\mathbf{d}_t$  and  $\mathbf{d}^{*}_{t}$  are continuous

# Progressive training

Figure 3: Progressive training scheme.  $z_{t}$  (docid at timestep  $t$ ) is optimized at the  $t$ -th training step, while  $z_{<t}$  (docids before timestep  $t$ ) are kept fixed.

representation of documents  $d$  and  $d^{*}$ , respectively, as defined in Eq. 2. The operator  $\mathrm{sg}(\cdot)$  is the stop gradient operator defined as follows:

$$
\operatorname {s g} (x) = \left\{ \begin{array}{l l} x, & \text {f o r w a r d p a s s} \\ 0, & \text {b a c k w a r d p a s s .} \end{array} \right. \tag {7}
$$

Intuitively,  $R(d \mid \mathbf{z})$  is designed to retrieve a specific document  $d$  from a set of documents  $S(z_{<t})$  at each timestep  $t$ . The set  $S(z_{<t})$  only includes those documents that are assigned the same docid prefix  $z_{<t}$  as the target document  $d$ . By utilizing this loss function, at each step  $t$ , the model is facilitated to learn the residual semantics of the documents not captured by the previous docid  $z_{<t}$ .

# 3.3 Model optimization

For the document tokenization model  $Q(z \mid d)$ , generative retrieval model  $P(z \mid q)$  and reconstruction model  $R(d \mid z)$ , jointly optimizing these three models using auto-encoding is challenging for the following two reasons:

- Learning docids in an autoregressive fashion. That is: (i) The prediction of the  $z_{t}$  at time  $t$  relies on previously predicted docids  $z_{<t}$ , which is often under-optimized at the beginning and rapidly changes during training, making it difficult to reach convergence. (ii) Simultaneously optimizing  $z$  makes it challenging to guarantee a unique docid assignment. To stabilize the training of GENRET, we devise a progressive training scheme (see Section 3.3.1).  
- Generating docids with diversity. Optimizing the model using auto-encoding often leads to unbalanced docid assignment: a few major docids are assigned to a large number of documents and most other docids are rarely assigned. Such a sub-optimal distribution of docids affects the model distinguishability, which in turns triggers length increments of docids in order to distinguish conflicting documents. We introduce two diverse clustering techniques to ensure docid diversity (see Section 3.3.2).

3.3.1 Progressive training scheme. To optimize each of the three models listed above in an autoregressive manner, we propose a progressive auto-encoding learning scheme, as illustrated in Figure 3. The whole learning scheme contains  $M$  learning steps with respect to the final docid in  $M$ -token. And the docid  $z_{T}$  at step  $T \in [M]$  is learned and optimized at the corresponding learning step. Besides, at each step  $T \in [M]$ , the docid  $z_{T}$  and the model parameters associated with  $z_{T}$  generation are updated, while previously produced docids  $z_{<T}$  and other parameters are kept fixed. By progressively performing the above process, we can finally optimize and learn our models.

At each optimization step, say the  $T$ -step, we devise the learning objective for document tokenization consisting of three loss functions detailed below.

Reconstruction loss. We utilize the reconstruction model  $R(d \mid z)$  as an auxiliary model to learn to optimize the docid generation, whose main goal is capturing as much semantics in the docid as possible. Therefore, we define a reconstruction loss function of step  $T$  as follows:

$$
\mathcal {L} _ {\operatorname {R e c}} = - \log R (d \mid \hat {\mathbf {z}} _ {\leq T})
$$

$$
\hat {\mathbf {z}} _ {\leq T} = \left\{\operatorname {s g} \left(\mathbf {z} _ {1}\right), \dots , \operatorname {s g} \left(\mathbf {z} _ {T - 1}\right), \mathbf {z} _ {T} \right\} \in \mathbb {R} ^ {T \times D} \tag {8}
$$

$$
\forall t \in [ T ]: \mathbf {z} _ {t} = \mathbf {e} _ {t, j ^ {*}} \in \mathbb {R} ^ {D},   j ^ {*} = \underset {j} {\arg \max } Q (z _ {t} = j \mid z _ {<   t}, d),
$$

where  $\hat{z}_{\leq T}$  is the first  $T$  representations of the  $z$ , and only the variable  $\mathbf{z}_T$  is optimized in step  $T$ .  $Q(z_{t} = j \mid z_{< t}, d)$  is defined in Eq. 3. And the document tokenization model  $Q$  can therefore be optimized when minimizing  $\mathcal{L}_{\mathrm{Rec}}$ .

Of note, since the computation involves a non-differentiable operation - arg max(·), we apply straight-through gradient estimation to back-propagate the gradient from reconstruction loss [44, 49]. Specifically, the gradients to document representation  $\mathbf{d}_T$  are defined as  $\frac{\partial\mathcal{L}_{\mathrm{Rec}}}{\partial\mathbf{d}_T} \coloneqq \frac{\partial\mathcal{L}_{\mathrm{Rec}}}{\partial\mathbf{z}_T}$ . And the gradients to the codebook embedding  $\mathbf{e}_{T,j}$  are defined as  $\frac{\partial\mathcal{L}_{\mathrm{Rec}}}{\partial\mathbf{e}_{T,j}} \coloneqq 1_{z_T = j}\frac{\partial\mathcal{L}_{\mathrm{Rec}}}{\partial\mathbf{z}_T}$ .

Commitment loss. In addition, to make sure the predicted docid commits to an embedding and to avoid models forgetting previous docid  $z_{<t}$ , we add a commitment loss as follows:

$$
\mathcal {L} _ {\mathrm {C o m}} = - \sum_ {t = 1} ^ {T} \log Q \left(z _ {t} \mid z _ {<   t}, d\right). \tag {9}
$$

Retrieval loss. For the generative retrieval model  $P$ , we jointly learn it together with the document tokenization model  $Q$ , where  $P$  learns to generate the docids of relevant documents  $d$  given a query  $q$ . Specifically, suppose  $(q, d)$  are a query and relevant document pair; we define the learning objective of retrieval model  $P$  as:

$$
\mathcal {L} _ {\text {R e t}} = - \log \frac {\exp \left(\mathbf {q} _ {T} \cdot \mathbf {d} _ {T}\right)}{\sum_ {d ^ {-} \sim B} \exp \left(\mathbf {q} _ {T} \cdot \mathbf {d} ^ {-} _ {T}\right)} - \sum_ {t = 1} ^ {T} \log P \left(z _ {t} \mid z _ {<   t}, q\right), \tag {10}
$$

where the first term is a ranking-oriented loss enhancing the model using  $(q,d)$  pair;  $d^{-}$  is an in-batch negative document from the same training mini-batch  $B$ ;  $\mathbf{q}_T$  and  $\mathbf{d}_T$  denote the representation of  $q$  and  $d$  at timestep  $T$ . The second term is the cross-entropy loss for generating docid  $z$  based on query  $q$ .

The final loss we use at step-  $T$  is the sum of reconstruction loss, commitment loss, and retrieval loss:

$$
\mathcal {L} = \mathcal {L} _ {\text {R e c}} + \mathcal {L} _ {\text {C o m}} + \mathcal {L} _ {\text {R e t}}. \tag {11}
$$

3.3.2 Diverse clustering technique. To ensure diversity of generated docids, we adopt two diverse clustering techniques—codebook initialization and docid re-assignment at each progressive training step, where codebook initialization mainly aims to increase the balance of semantic space segmentation, and the docid re-assignment mainly aims to increase the balance of docid assignments.

Codebook initialization. In order to initialize the codebook for our model, we first warm-up the model by passing the continuous representation  $\mathbf{d}_T$  to the reconstruction model instead of the

docid representation  $\mathbf{z}_T$  as defined in Eq. 5. During this warm-up phase, we optimize the model using the reconstruction loss  $\mathcal{L}_{\mathrm{Rec}}$  and commitment loss  $\mathcal{L}_{\mathrm{Com}}$ . Next, we collect the continuous representations  $\mathbf{d}_T$  of all documents in  $\mathcal{D}$ , and cluster them into  $K$  groups. The centroids of these clusters are then used as the initialized codebook  $\mathbf{E}_T$ . To balance the initialized docid distribution, we utilize a diverse constrained clustering algorithm, Constrained  $K$ -Means, which modifies the cluster assignment step (E in EM) by formulating it as a minimum cost flow (MCF) linear network optimization problem [2].

Docid re-assignment. In order to assign docids to a batch of documents, we modify the dot-product look-up results in Eq. 3 by ensuring that the docid for different documents in the batch are distinct [6, 49]. Specifically, let  $\mathbf{D}_t = \{\mathbf{d}_t^{(1)},\dots,\mathbf{d}_t^{(B)}\} \in \mathbb{R}^{B\times D}$  denote the continuous representation of a batch of documents with batch size of  $B$ . The dot-product results are represented by  $\mathbf{H} = \mathbf{D}_t\cdot \mathbf{E}_t^\top \in \mathbb{R}^{B\times K}$ . To obtain distinct docids, we calculate an alternative  $\mathbf{H}^* = \mathrm{Diag}(\mathbf{u})\exp (\frac{\mathbf{H}}{\epsilon})\mathrm{Diag}(\mathbf{v})$ , where  $\mathbf{u}$  and  $\mathbf{v}$  are re-normalization vectors in  $\mathbb{R}^K$  and  $\mathbb{R}^B$ , respectively. The re-normalization vectors are computed via the iterative Sinkhorn-Knopp algorithm [9]. Finally,  $\mathbf{H}^*$  is used instead of  $\mathbf{H}$  in the Softmax (Eq. 3) and arg max (Eq. 4) operations to obtain the docid  $z_{t}$ .

# 4 EXPERIMENTAL SETUP

# 4.1 Datasets

We conduct experiments on three well-known document retrieval datasets: NQ [21], MS MARCO [4], and BEIR [43].

NQ320K. NQ320K is a popular dataset for evaluating generative retrieval models [42, 46]. It is based on the Natural Questions (NQ) dataset proposed by Google [21]. NQ320k consists of 320k query-document pairs, where the documents are gathered from Wikipedia pages, and the queries are natural language questions. We follow the evaluation setup in NCI [46] and further split the test set into two subsets: seen test, in which the annotated target documents of the queries are included in the training set; and unseen test, in which no labeled document is included in the training set.

MS MARCO. MS MARCO is a collection of queries and web pages from Bing search. Akin to NQ320k and following [52], we sample a subset of documents from the labeled documents, and use their corresponding queries for training. We evaluate the models on the queries of the MS MARCO dev set and retrieval on the sampled document subset.

BEIR. BEIR is a collection of datasets for heterogeneous retrieval tasks. In this paper, we evaluate the models on 6 BEIR datasets, which include distinct retrieval tasks and document collections from NQ and MS MARCO: (i) BEIR-Arg retrieves a counterargument to an argument; (ii) BEIR-Covid retrieves scientific articles about the COVID-19 pandemic; (iii) BEIR-NFC retrieves medical documents from PubMed; (iv) BEIR-SciFact retrieves scientific papers for fact-checking; (v) BEIR-SciDocs retrieves citations for scientific papers; (vi) BEIR-FiQA retrieves financial documents.

We summarize the statistics of above datasets in Table 1.

Table 1: Statistics of datasets used in our experiments. The three values split by / on # Test queries denote the number of queries in the full, seen subset, and unseen subset, respectively. In BEIR, all queries in the test set are unseen.  

<table><tr><td>Dataset</td><td>#Docs</td><td>#Test queries</td><td>#Train pairs</td></tr><tr><td>NQ320K</td><td>109,739</td><td>7,830 / 6,075 / 1,755</td><td>307,373</td></tr><tr><td>MS MARCO</td><td>323,569</td><td>5,187 / 807 / 4,380</td><td>366,235</td></tr><tr><td>BEIR-Arg</td><td>8,674</td><td>1,406</td><td>-</td></tr><tr><td>BEIR-Covid</td><td>171,332</td><td>50</td><td>-</td></tr><tr><td>BEIR-NFC</td><td>3,633</td><td>323</td><td>-</td></tr><tr><td>BEIR-SciFact</td><td>5,183</td><td>300</td><td>-</td></tr><tr><td>BEIR-SciDocs</td><td>25,657</td><td>1,000</td><td>-</td></tr><tr><td>BEIR-FiQA</td><td>57,638</td><td>648</td><td>-</td></tr></table>

# 4.2 Evaluation metrics

On NQ320K, we use Recall@{1,10,100} and Mean Reciprocal Rank (MRR)@100 as evaluation metrics, following [46]. On MS MARCO, we use Recall@{1, 10, 100} and MRR@10 as evaluation metrics, following [52]. On BEIR, we use nDCG@10 as the main metrics and calculate the average nDCG@10 values across multiple downstream sub-datasets as overall metrics.

# 4.3Baselines

We consider three types of baselines: sparse retrieval methods, dense retrieval methods, and generative retrieval methods.

The sparse retrieval baselines are as follows:  $\bullet$  BM25, uses the tfidf feature to measure term weights; we use the implementation from http://pyserini.io/.  $\bullet$  DocT5Query, expands a document with possible queries predicted by a finetuned T5 with this document as the input.

The dense retrieval baselines are as follows: DPR [19], a dual-encoder model using the representation of the [CLS] token of BERT. ANCE [47], an asynchronously updated ANN indexer is utilized to mine hard negatives for training a RoBERTa-based dual-encoder model. Sentence-T5 [30], a dual-encoder model that uses T5 to produce continuous sentence embeddings. We reproduce Sentence-T5 (ST5 for short) on our datasets, the model is based on T5-Base EncDec model and is trained with in-batch negatives. GTR [31], a state-of-the-art dense retrieval model that pre-trains sentence-T5 on billions of paired data using contrastive learning. Contriever [16], a dual-encoder model pre-trained using unsupervised contrastive learning with independent cropping and inverse cloze task.

And the generative retrieval baselines are as follows:  $\bullet$  GENRE [5] an autoregressive retrieval model that generates the document's title. The original GENRE is trained on the KILT dataset [33] using BART, and we reproduce GENRE on our datasets using T5 for a fair comparison. For datasets without title, we use the first 32 tokens of the document as pseudo-title.  $\bullet$  DSI [42], which represents documents using hierarchical K-means clustering results, and indexes documents using the first 32 tokens as pseudo-queries. As the original code is not open source, we reproduce DSI using T5-base and the docids of NCI [46].  $\bullet$  SEAL [1] uses arbitrary n-grams in documents as docids, and retrieves documents under the constraint

of a pre-built FM-indexer. We refer to the results reported by Wang et al. [46]. CGR-Contra [23], a title generation model with a contextualized vocabulary embedding and a contrastive learning loss. DSI-QG [53], uses a query generation model to augment the document collection. We reproduce the DSI-QG results using T5 and our dataset. NCI [46], uses a prefix-aware weight-adaptive decoder and various query generation strategies, including DocAs-Query and DocT5Query. In particular, NCI augments training data by generating 15 queries for each document. Ultron [52], uses a three-stage training pipeline and represents the document as three types of identifiers, including URL, PQ, and Atomic.

We highlight three of our reproduced baselines that constitute a fair comparison with the proposed method, all of which use the T5 model and experimental setup, but they differ model outputs: (i) Sentence-T5 outputs continuous vectors, (ii) GENRE outputs document titles, (iii) DSI-QG outputs clustering ID, while GENRET outputs docids learned using the proposed tokenization method.

# 4.4 Implementation details

Hyper-parameters. In our experiments, we utilize the T5-Base model [35] as the base Transformer and initialize a new codebook embedding  $\mathbf{E}_t$  for each time step. We set the number of clusters to be  $K = 512$  for all datasets, with the length of the docid  $M$  being dependent on the number of documents present. For datasets containing a larger number of candidate documents, a larger value of  $M$  is set to ensure that all documents are assigned unique document ids. In the docid re-assignment, the hyper-parameter  $\epsilon$  is set to 1.0, and the Sinkhorn-Knopp algorithm is executed for 100 iterations.

Indexing with query generation. Following previous work [45, 46, 53], we use query generation models to generate synthetic (query, document) pairs for data augmentation. Specifically, we use the pre-trained query generation model from DocT5Query [8] to augment the NQ and MS MARCO datasets. In query generation, we use nucleus sampling with parameters  $p = 0.8$ ,  $t = 0.8$  and generate five queries for each document in the collection. For the BEIR datasets, we use the queries generated by GPL [45], which can be downloaded from their website. GPL uses a DocT5Query [8] generator trained on MS MARCO to generate about 250K queries for each BEIR dataset.

Training and inference. The proposed models and the reproduced baselines are implemented with PyTorch 1.7.1 and Hugging-Face transformers 4.22.2. We optimize the model using AdamW and set the learning rate to  $5e - 4$ . The batch size is 256, and the model is optimized for up to 500k steps for each timestep. In progressive training, we first warm up the model for 5K steps and then initialize the codebook using the clustering centroids as mentioned in Section 3.3.1. We use constrained clustering<sup>4</sup> to obtain diverse clustering results. During inference, we use beam search with constrained decoding [5] and a beam size of 100.

# 5 EXPERIMENTAL RESULTS

# 5.1 Main results

Results on NQ320K. In Table 2, we list the results on NQ320K. GENRET outperforms both the strong pre-trained dense retrieval model, GTR, and the previous best generative retrieval method, NCI, thereby establishing a new state-of-the-art on the NQ320K dataset. Furthermore, our results reveal that existing generative retrieval methods perform well on the seen test but lag behind dense retrieval methods on the unseen test. For example, NCI obtains an MRR@100 of 76.8 on the seen test, which is higher than the MRR@100 of 65.3 obtained by GTR-Base. However, on unseen test data, NCI performs worse than GTR-Base. In contrast, GENRET performs well on both seen and unseen test data. This result highlights the ability of GENRET to combine the advantages of both dense and generative retrieval by learning discrete docids with semantics through end-to-end optimization.

Results on MS MARCO. Table 3 presents the results on the MS MARCO dataset. GENRET significantly outperforms previous generative retrieval methods and achieves comparable results with the state-of-the-art dense retrieval method GTR. Furthermore, previous generative retrieval methods (e.g., GENRE, Ultron) utilizing metadata such as the title and URL, while exhibiting decent performance on the NQ320K dataset, underperform in comparison to previous-best dense retrieval (GTR) and sparse retrieval (DocT5Query) methods on the MS MARCO dataset. This can likely because that the NQ320K dataset retrieves Wikipedia documents, where metadata like the title effectively capture the semantics of the document. In the case of the MS MARCO dataset, which is a web search dataset, the metadata often does not adequately characterize the documents, resulting in a decline in performance of the generative retrieval model. In contrast, GENRET learns to generate semantic docids that effectively enhance the generative retrieval model.

Results on BEIR. Table 4 lists the results of the baselines and GENRET on six datasets of BEIR. These datasets represent a diverse range of information retrieval scenarios. On average, GENRET outperforms strong baselines including BM25 and GTR-Base, and achieves competitive results compared to state-of-the-art sparse and dense retrieval methods. In comparison to the ST5 GPL method that utilizes the same training data and backbone T5 model, GENRET achieves better results. Additionally, GENRET demonstrates a significant improvement over the previous generative retrieval model GENRE that utilizes titles as docids. Furthermore, GENRE performs poorly on some datasets, such as BEIR-Covid and BEIRSciDocs. This may be because the titles of the documents in these datasets do not adequately capture their semantic content.

# 5.2 Performance on retrieving new documents

In this experiment, we investigate the impact of various document tokenization techniques on the ability of generative retrieval models to retrieve new documents. The generative models with different tokenization methods are trained on NQ320K data, excluding unseen documents, and are evaluated on NQ320K Unseen test set and BEIR-{Arg, NFC, SciDocs} datasets. For the baseline methods, which use rule-based document tokenization methods, the docids are generated for the target document collection using their respective tokenization techniques. In contrast, our proposed method

Table 2: Results on Natural Questions (NQ320K). The results of the methods marked with  $\dagger$  are from our own re-implementation, others are from their official implementation. Methods with  $\clubsuit$  use additional annotated document retrieval data during training.  $*$  and \*\* indicate significant improvements over previous-best generative retrieval baselines with p-value  $< 0.05$  and p-value  $< 0.01$ , respectively.  $\sharp$  and  $\sharp$  indicate significant improvements over previous-best dense retrieval baselines with p-value  $< 0.05$  and p-value  $< 0.01$ , respectively. The best results for each metric are indicated in boldface.  

<table><tr><td rowspan="2">Method</td><td colspan="4">Full test</td><td colspan="4">Seen test</td><td colspan="4">Unseen test</td></tr><tr><td>R@1</td><td>R@10</td><td>R@100</td><td>MRR@100</td><td>R@1</td><td>R@10</td><td>R@100</td><td>MRR@100</td><td>R@1</td><td>R@10</td><td>R@100</td><td>MRR@100</td></tr><tr><td colspan="13">Sparse retrieval</td></tr><tr><td>BM25 [39]</td><td>29.7</td><td>60.3</td><td>82.1</td><td>40.2</td><td>29.1</td><td>59.8</td><td>82.4</td><td>39.5</td><td>32.3</td><td>61.9</td><td>81.2</td><td>42.7</td></tr><tr><td>DocT5Query [8]</td><td>38.0</td><td>69.3</td><td>86.1</td><td>48.9</td><td>35.1</td><td>68.3</td><td>86.4</td><td>46.7</td><td>48.5</td><td>72.9</td><td>85.0</td><td>57.0</td></tr><tr><td colspan="13">Dense retrieval</td></tr><tr><td>DPR [19]</td><td>50.2</td><td>77.7</td><td>90.9</td><td>59.9</td><td>50.2</td><td>78.7</td><td>91.6</td><td>60.2</td><td>50.0</td><td>74.2</td><td>88.7</td><td>58.8</td></tr><tr><td>ANCE [47]</td><td>50.2</td><td>78.5</td><td>91.4</td><td>60.2</td><td>49.7</td><td>79.2</td><td>92.3</td><td>60.1</td><td>52.0</td><td>75.9</td><td>88.0</td><td>60.5</td></tr><tr><td>Sentence-T5†[30]</td><td>53.6</td><td>83.0</td><td>93.8</td><td>64.1</td><td>53.4</td><td>83.9</td><td>94.7</td><td>63.8</td><td>56.5</td><td>79.5</td><td>90.7</td><td>64.9</td></tr><tr><td>GTR-Base* [31]</td><td>56.0</td><td>84.4</td><td>93.7</td><td>66.2</td><td>54.4</td><td>84.7</td><td>94.2</td><td>65.3</td><td>61.9</td><td>83.2</td><td>92.1</td><td>69.6</td></tr><tr><td colspan="13">Generative retrieval</td></tr><tr><td>GENRE†[5]</td><td>55.2</td><td>67.3</td><td>75.4</td><td>59.9</td><td>69.5</td><td>83.7</td><td>90.4</td><td>75.0</td><td>6.0</td><td>10.4</td><td>23.4</td><td>7.8</td></tr><tr><td>DSI†[42]</td><td>55.2</td><td>67.4</td><td>78.0</td><td>59.6</td><td>69.7</td><td>83.6</td><td>90.5</td><td>74.7</td><td>1.3</td><td>7.2</td><td>31.5</td><td>3.5</td></tr><tr><td>SEAL [1]</td><td>59.9</td><td>81.2</td><td>90.9</td><td>67.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CGR-Contra [23]</td><td>63.4</td><td>81.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DSI-QG†[53]</td><td>63.1</td><td>80.7</td><td>88.0</td><td>69.5</td><td>68.0</td><td>85.0</td><td>91.4</td><td>74.3</td><td>45.9</td><td>65.8</td><td>76.3</td><td>52.8</td></tr><tr><td>NCI [46]</td><td>66.4</td><td>85.7</td><td>92.4</td><td>73.6</td><td>69.8</td><td>88.5</td><td>94.6</td><td>76.8</td><td>54.5</td><td>75.9</td><td>84.8</td><td>62.4</td></tr><tr><td>Ours</td><td>68.1*#</td><td>88.8*#</td><td>95.2*</td><td>75.9*#</td><td>70.2#</td><td>90.3#</td><td>96.0#</td><td>77.7#</td><td>62.5**</td><td>83.6**</td><td>92.5**</td><td>70.4**</td></tr></table>

Table 3: Results on MS MARCO. The results of the methods marked with  $\dagger$  are from our own re-implementation. Methods with  $\clubsuit$  use additional annotated retrieval data for training.  $^{*} / ^{**}$  indicates significant improvements over previous generative retrieval baselines with p-value  $< 0.05 / 0.01$ . The best results for each metric are indicated in boldface.  

<table><tr><td>Method</td><td>R@1</td><td>R@10</td><td>R@100</td><td>MRR@10</td></tr><tr><td colspan="5">Sparse retrieval</td></tr><tr><td>BM25 [39]</td><td>39.1</td><td>69.1</td><td>86.2</td><td>48.6</td></tr><tr><td>DocT5Query [8]</td><td>46.7</td><td>76.5</td><td>90.4</td><td>56.2</td></tr><tr><td colspan="5">Dense retrieval</td></tr><tr><td>ANCE [47]</td><td>45.6</td><td>75.7</td><td>89.6</td><td>55.6</td></tr><tr><td>Sentence-T5†[30]</td><td>41.8</td><td>75.4</td><td>91.2</td><td>52.8</td></tr><tr><td>GTR-Base* [31]</td><td>46.2</td><td>79.3</td><td>93.8</td><td>57.6</td></tr><tr><td colspan="5">Generative retrieval</td></tr><tr><td>GENRE†[5]</td><td>35.6</td><td>57.6</td><td>79.1</td><td>42.3</td></tr><tr><td>Ultron-URL [52]</td><td>29.6</td><td>67.8</td><td>-</td><td>40.0</td></tr><tr><td>Ultron-PQ [52]</td><td>31.6</td><td>73.1</td><td>-</td><td>45.4</td></tr><tr><td>Ultron-Atomic [52]</td><td>32.8</td><td>74.1</td><td>-</td><td>46.9</td></tr><tr><td>Ours</td><td>47.9**</td><td>79.8**</td><td>91.6**</td><td>58.1**</td></tr></table>

uses a tokenization model to tokenize the documents in the target collection, producing the docids. However, our method may result in duplicate docids. In such cases, all corresponding documents are retrieved and shuffled in an arbitrary order. The results of this evaluation are summarized in Table 5.

Document tokenization methods that do not consider the semantic information of the documents, such as Naive String and Atomic, are ineffective in retrieving new documents without model

Table 4: Results on BEIR. The metric is nDCG@10. The results of the methods marked with  $\dagger$  are from our own reimplementation. ST5 GPL denotes Sentence-T5 trained on GPL datasets [45].  

<table><tr><td>Method</td><td>Arg</td><td>Covid</td><td>NFC</td><td>SciFact</td><td>SciDocs</td><td>FiQA</td><td>Avg.</td></tr><tr><td colspan="8">Sparse retrieval</td></tr><tr><td>BM25 [39]</td><td>29.1</td><td>58.9</td><td>33.5</td><td>67.4</td><td>14.8</td><td>23.6</td><td>37.8</td></tr><tr><td>DocT5Query [8]</td><td>34.9</td><td>71.3</td><td>32.8</td><td>67.5</td><td>16.2</td><td>29.1</td><td>41.9</td></tr><tr><td colspan="8">Dense retrieval</td></tr><tr><td>ANCE [47]</td><td>31.4</td><td>73.3</td><td>23.1</td><td>50.8</td><td>12.2</td><td>29.5</td><td>36.7</td></tr><tr><td>ST5 GPL†[30]</td><td>32.1</td><td>74.4</td><td>30.1</td><td>58.6</td><td>12.7</td><td>26.0</td><td>39.0</td></tr><tr><td>GTR-Base [31]</td><td>37.3</td><td>61.2</td><td>30.0</td><td>58.4</td><td>14.0</td><td>35.1</td><td>39.3</td></tr><tr><td>Contriever [16]</td><td>40.0</td><td>68.8</td><td>33.5</td><td>61.4</td><td>16.3</td><td>30.7</td><td>41.8</td></tr><tr><td colspan="8">Generative retrieval</td></tr><tr><td>GENRE†[53]</td><td>42.5</td><td>14.7</td><td>20.0</td><td>42.3</td><td>6.8</td><td>11.6</td><td>30.0</td></tr><tr><td>Ours</td><td>34.3</td><td>71.8</td><td>31.6</td><td>63.9</td><td>14.9</td><td>30.2</td><td>41.1</td></tr></table>

updating. Methods that consider the semantic information of the documents, such as those based on title or BERT clustering, show some improvement. Our proposed document tokenization method significantly improves over these existing rule-based document tokenization methods. For instance, when the model trained on NQ - a factoid QA data based on Wikipedia documents - is applied to a distinct retrieval task on a different document collection, BEIRSciDocs, a citation retrieval task on a collection of scientific articles, our proposed document tokenization model still showed promising results with an nDCG@10 of 12.3, which is comparable to those models trained on the target document collection. This suggests

Table 5: Zero-shot evaluation on retrieving new documents with different document tokenization methods. The second column indicates the type of docid, where BERT-HC denotes BERT-Hierarchical-Clustering [42], Prefix-HC denotes Prefix-aware BERT-Hierarchical-Clustering [46], and dAE denotes discrete auto-encoding.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Docid</td><td rowspan="2">NQ (R@1) Unseen</td><td colspan="3">BEIR (nDCG@10)</td></tr><tr><td>Arg</td><td>NFC</td><td>SciDocs</td></tr><tr><td>DSI-Naive† [42]</td><td>Naive String</td><td>0.0</td><td>0.1</td><td>1.0</td><td>0.1</td></tr><tr><td>DSI-Atomic† [42]</td><td>Atomic</td><td>0.0</td><td>0.2</td><td>0.8</td><td>0.1</td></tr><tr><td>GENRE† [5]</td><td>Title</td><td>6.0</td><td>0.0</td><td>2.4</td><td>0.6</td></tr><tr><td>DSI† [42]</td><td>BERT-HC</td><td>1.3</td><td>1.8</td><td>11.1</td><td>5.9</td></tr><tr><td>NCI [46]</td><td>Prefix-HC</td><td>15.5</td><td>0.9</td><td>4.3</td><td>1.2</td></tr><tr><td>Ours</td><td>dAE</td><td>34.2</td><td>12.1</td><td>12.1</td><td>12.3</td></tr></table>

Figure 4: Left: Docid distribution on NQ320K. The id are sorted by the assigned frequency. Right: Ablation study on NQ320K.


that our proposed method effectively encodes the semantic information of documents in the docid and leads to a better fit between the docid and the generative retrieval model.

# 5.3 Analytical experiments

We further conduct analytical experiments to study the effectiveness of the proposed method.

In Figure 4 (left), we plot the frequencies of docids at the first timestep of various learning methods. We label each method using a box with a docid and a diversity metric  $d$ , which is calculated by:  $d = 1 - \frac{1}{2n}\sum_{j=1}^{K}|n_j - n_u|$ , where  $|\cdot|$  represents the absolute value,  $n$  denotes the total number of documents,  $n_j$  denotes the number of documents that have a docid  $= j$ , and  $n_u = \frac{n}{K}$  is the expected number of documents per docid under the uniform distribution.

The results demonstrate the superiority of GENRET (represented by the yellow line) in terms of distribution uniformity. It uses all the potential docid  $k = 512$  and achieves the highest diversity metric with a value of  $d = 0.90$ . The method without docid reassignment also yields a relatively balanced distribution, with a diversity metric of  $d = 0.77$ . However, the distribution of the method without diverse codebook initialization is highly uneven, which can be attributed to the fact that most of the randomly initialized codebook embeddings are not selected by the model during the initial training phase, leading to their lack of update and further selection in subsequent training. Additionally, the models without diverse clustering tend

Table 6: Efficiency analysis.  

<table><tr><td>Method</td><td>Memory</td><td>Time (Offline)</td><td>Top-K</td><td>Time (Online)</td></tr><tr><td>ANCE</td><td>1160MB</td><td>145min</td><td>100</td><td>0.69s</td></tr><tr><td>GTR-Base</td><td>1430MB</td><td>140min</td><td>100</td><td>1.97s</td></tr><tr><td rowspan="2">GENRE</td><td rowspan="2">851MB</td><td rowspan="2">0min</td><td>100</td><td>1.41s</td></tr><tr><td>10</td><td>0.69s</td></tr><tr><td rowspan="2">DSI</td><td rowspan="2">851MB</td><td rowspan="2">310min</td><td>100</td><td>0.32s</td></tr><tr><td>10</td><td>0.21s</td></tr><tr><td rowspan="2">Ours</td><td rowspan="2">860MB</td><td rowspan="2">220min</td><td>100</td><td>0.16s</td></tr><tr><td>10</td><td>0.10s</td></tr></table>

to converge to a trivial solution where all documents are assigned the same docid.

In Figure 4 (right), the results of two ablated variants are presented. First, GENRET w/o learning is a generative model that has been trained directly using the final output docid from GENRET, without utilizing the proposed learning scheme. Its retrieval performance is comparable to that of GENRET on seen test data; however, it is significantly lower on unseen test data. The proposed progressive auto-encoding scheme is crucial for the model to capture the semantic information of documents, rather than just the well-defined discrete docid. Secondly, GENRET w/ T5-Small uses a small model, and its performance is inferior to that of GENRET using T5-Base. However, the gap between the performance on seen and unseen test data is smaller, which could be attributed to the limited fitting capacity of the small model.

# 5.4 Efficiency analysis

In Table 6, we compare GENRET with baseline models on MS MARCO (323,569 documents) in terms of memory footprint, offline indexing time (not including the time for neural network training), and online retrieval latency for different Top-K values. We have four observations: (i) The memory footprint of generative retrieval models (GENRE, DSI-QG, and the proposed model) is smaller than of dense and sparse retrieval methods. The memory footprint of generative retrieval models is only dependent on the model parameters, whereas dense and sparse retrieval methods require additional storage space for document embeddings, which increases linearly with the size of the document collection. (ii) DSI and GENRET take a longer time for offline indexing, as DSI involves encoding and clustering documents using BERT, while GENRET requires tokenizing documents using a tokenization model. Dense retrieval's offline time consumption comes from document encoding; GENRE uses titles hence no offline computation. (iii) The online retrieval latency of the generative retrieval model is associated with the beam size (i.e., Top-K) and the length of the docid. GENRET utilizes diverse clustering to generate a shorter docid, resulting in improved online retrieval speed compared to DSI and GENRE.

# 5.5 Case study

Table 7 shows an example of outputs of GENRE, NCI, and GENRET for the query "what state courts can order a new trial" and its corresponding document in NQ320K. The results show that GENRET, unlike the baselines, successfully returns the docid of the target

Table 7: Models outputs on NQ320K. The yellow and gray backgrounds denote the words with higher attention at step  $t = 1$  or  $t = 2$  of GENRET. Docid-D denotes tokenized docid of document D; Docid-Q denotes generated docid for query Q.  

<table><tr><td colspan="3">Test query (Q): what state courts can order a new trial
Target document (D): United States appellate procedure involves the rules and regulations for filing appeals in state courts and federal courts. The nature of an appeal can vary greatly depending on the type of case [...]</td></tr><tr><td>- GENRE:</td><td colspan="2">Docid-D: Appellate procedure in the United States; 
Docid-Q: Admission to the Union (X)</td></tr><tr><td>- NCI:</td><td>Docid-D: 22-18-10-1;</td><td>Docid-Q: 14-10-0-4 (X)</td></tr><tr><td>- GenRet:</td><td>Docid-D: 95-375-59;</td><td>Docid-Q: 95-375-59 (✓)</td></tr></table>

document. We highlight words in the target document based on their attention activation in GENRET at different time steps  $t$ . The yellow color indicates words that received higher attention at  $t = 1$ , while gray indicates words that received higher attention at  $t = 2$ . The example shows that the model focuses on different words at different time steps. GENRET gives more attention to words related to the topic, such as Appellate, in  $t = 1$ , and more attention to words related to the country, such as United States, in  $t = 2$ .

# 6 RELATED WORK

Sparse retrieval. Traditional sparse retrieval calculates the document score using term matching metrics such as TF-IDF [38], query likelihood [22] or BM25 [39]. It is widely used in practice due to its outstanding trade-off between accuracy and efficiency. Some methods adaptively assign the term importance using deep neural network [12, 14, 51]. With the recent development of pre-trained LMs, DeepCT [10] and HDCT [11] calculate term importance using contextualized text representation from BERT. Doc2Query [32] and DocT5Query [8] predict relevant queries to augment documents before building the BM25 index using a generative model like T5. Sparse retrieval often suffers from the lexical mismatches [24].

Dense retrieval. Dense retrieval (DR) presents queries and documents in dense vectors and models their similarities with the inner product or cosine similarity [19]. Compared with sparse retrieval, dense retrieval relieves the lexical mismatch problem. Various techniques have been proposed to improve DR models, such as hard negative mining [34, 47], late interaction [20, 41], and knowledge distillation [15, 26]. Recent studies have shown the effectiveness of pre-training DR models using contrastive learning on large-scale corpora [16, 31, 37]. Despite their success, DR approaches have several limitations [5, 28]: (i) DR models employ an index-retrieval pipeline with a fixed search procedure (MIPS), making it difficult to optimize the model end-to-end [42, 46]. (ii) Training DR models relies on contrastive learning [19] to distinguish positives from negatives, which is inconsistent with large LMs training objectives [3] and fails to fully utilize the capabilities of pre-trained LMs [1].

Generative retrieval. Generative retrieval is increasing gaining attention. It retrieves documents by generating their docid using a generative model like T5. Generative retrieval presents an end-to-end solution for document retrieval tasks [28, 42] and allows for better exploitation of the capabilities of large generative LMs [1]. Cao et al. [5] first propose an autoregressive entity retrieval model

to retrieve documents by generating titles. Tay et al. [42] propose a differentiable search index (DSI) and represent the document as atomic id, naive string, or semantic string. Bevilacqua et al. [1] suggest using arbitrary spans of a document as docids. Additionally, multiple-stage pre-training [7, 52], query generation [46, 52, 53], contextualized embedding [23], and continual learning [27], have been explored in recent studies. However, existing generative retrieval models have a limitation in that they rely on fixed document tokenization to produce docids, which often fails to capture the semantic information of a document [42]. It is an open question of how one should define the docids. To further capture document semantics in docid, we propose document tokenization learning methods. The semantic docid is automatically generated by the proposed discrete auto-encoding learning scheme in an end-to-end manner.

Discrete representation learning. Learning discrete representations using neural networks is an important research area in machine learning. For images, Rolfe [40] proposes the discrete variational autoencoder, and VQ-VAE [44] learns quantized representations via vector quantization. Dall-E [36] uses an autoregressive model to generate discrete image representation for text-to-image generation. Recently, representation learning has attracted considerable attention in NLP tasks, for tasks such as machine translation [54], dialogue generation [50], and text classification [17, 48]. For document retrieval, RepCONC [49] uses a discrete representation learning method based on constrained clustering for vector compression. We propose a document tokenization learning method for generative retrieval, which captures the autoregressive nature of docids by progressive training and enhances the diversity of docids by diverse clustering techniques.

# 7 CONCLUSIONS

This paper has proposed a document tokenization learning method for generative retrieval, named GENRET. The proposed method learns to tokenize documents into short discrete representations (i.e., docids) via a discrete auto-encoding approach, which ensures the semantics of the generated docids. A progressive training method and two diverse clustering techniques have been proposed to enhance the training of the model. Empirical results on various document retrieval datasets have demonstrated the effectiveness of the proposed method. Especially, GENRET achieves outperformance on unseen documents and can be well generalized to multiple retrieval tasks. In future work, we would like to extend the approach to large document collections. We also plan to explore generative pre-training for document tokenization using large-scale language models. Additionally, we intend to investigate the dynamic adaptation of docid prefixes for progressive training.

# REFERENCES

[1] Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wen tau Yih, Sebastian Riedel, and Fabio Petroni. 2022. Autoregressive Search Engines: Generating Substrings as Document Identifiers. In NeurIPS.  
[2] Paul S. Bradley, Kristin P. Bennett, and Ayhan Demiriz. 2000. Constrained KMeans Clustering. Microsoft Research (2000).  
[3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Praffulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack

Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In NeurIPS.  
[4] Daniel Fernando Campos, Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng, and Bhaskar Mitra. 2016. MS MARCO: A Human Generated MAchine Reading COprehension Dataset. ArXiv abs/1611.09268 (2016).  
[5] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Autoregressive Entity Retrieval. In ICLR.  
[6] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. 2020. Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. *ArXiv* abs/2006.09882 (2020).  
[7] Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Y. Liu, Yixing Fan, and Xueqi Cheng. 2022. CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks. In CIKM.  
[8] David R. Cheriton. 2019. From Doc2query to DocTTTTTquery. Online preprint.  
[9] Marco Cuturi. 2013. Sinkhorn Distances: Lightspeed Computation of Optimal Transport. In NIPS.  
[10] Zhuyun Dai and Jamie Callan. 2019. Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval. ArXiv abs/1910.10687 (2019).  
[11] Zhuyun Dai and Jamie Callan. 2020. Context-Aware Document Term Weighting for Ad-Hoc Search. In WWW.  
[12] Mostafa Dehghani, Hamed Zamani, Aliaksei Severyn, Jaap Kamps, and W. Bruce Croft. 2017. Neural Ranking Models with Weak Supervision. In SIGIR.  
[13] Daniel Gillick, Alessandro Presta, and Gaurav Singh Tomar. 2018. End-to-End Retrieval in Continuous Space. ArXiv abs/1811.08008 (2018).  
[14] Jiafeng Guo, Yixing Fan, Qingyao Ai, and W. Bruce Croft. 2016. A Deep Relevance Matching Model for Ad-hoc Retrieval. In CIKM.  
[15] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy J. Lin, and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. In SIGIR.  
[16] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Information Retrieval with Contrastive Learning. In TMLR.  
[17] Shuning Jin, Sam Wiseman, Karl Stratos, and Karen Livescu. 2020. Discrete Latent Variable Representations for Low-Resource Text Classification. In ACL.  
[18] Jeff Johnson, Matthijs Douze, and Herve Jegou. 2017. Billion-Scale Similarity Search with GPUs. IEEE Transactions on Big Data 7 (2017), 535-547.  
[19] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Yu Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In EMNLP.  
[20] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In SIGIR.  
[21] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc V. Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. TACL 7 (2019), 453-466.  
[22] John Lafferty and ChengXiang Zhai. 2001. Document Language Models, Query Models, and Risk Minimization for Information Retrieval. In SIGIR.  
[23] Hyunjil Lee, Jaeyoung Kim, Hoyeon Chang, Hanseok Oh, Sohee Yang, Vladimir Karpukhin, Yi Lu, and Minjoon Seo. 2022. Contextualized Generative Retrieval. ArXiv abs/2210.02068 (2022).  
[24] Jimmy Lin, Rodrigo Nogueira, and Andrew Yates. 2022. Pretrained Transformers for Text Ranking: BERT and Beyond. Springer Nature (2022).  
[25] Yiding Liu, Guan Huang, Jiaxiang Liu, Weixue Lu, Suqi Cheng, Yukun Li, Daiting Shi, Shuaiqiang Wang, Zhicong Cheng, and Dawei Yin. 2021. Pre-trained Language Model for Web-scale Retrieval in Baidu Search. In SIGKDD.  
[26] Yuxiang Lu, Yiding Liu, Jiaxiang Liu, Yunsheng Shi, Zhengjie Huang, Shi Feng, Yu Sun, Hao Tian, Hua Wu, Shuaiqiang Wang, Dawei Yin, and Haifeng Wang. 2022. ERNIE-Search: Bridging Cross-Encoder with Dual-Encoder via Self On-the-fly Distillation for Dense Passage Retrieval. ArXiv abs/2205.09153 (2022).  
[27] Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Quang Tran, Jinfeng Rao, Marc-Alexander Najork, Emma Strubell, and Donald Metzler. 2022.  $\mathrm{DSI} + +$ : Updating Transformer Memory with New Documents. ArXiv abs/2212.09744 (2022).  
[28] Donald Metzler, Yi Tay, and Dara Bahri. 2021. Rethinking Search: Making Domain Experts out of Dilettantes. In SIGIR.  
[29] Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas A. Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David P. Schnurr, Felipe Petroski Such, Kenny Sai-Kin Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, and Lilian Weng. 2022. Text and Code Embeddings by Contrastive Pre-Training. ArXiv abs/2201.10005 (2022).  
[30] Jianmo Ni, Gustavo Hern'andez 'Abrego, Noah Constant, Ji Ma, Keith B. Hall, Daniel Matthew Cer, and Yinfei Yang. 2022. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models. In Findings of ACL.

[31] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hern'andez 'Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. 2021. Large Dual Encoders Are Generalizable Retrievers. ArXiv abs/2112.07899 (2021).  
[32] Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document Expansion by Query Prediction. ArXiv abs/1904.08375 (2019).  
[33] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yaccine Jernite, Vassilis Plachouras, Tim Rocktaschel, and Sebastian Riedel. 2021. KILT: A Benchmark for Knowledge Intensive Language Tasks. In NAACL.  
[34] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In NAACL.  
[35] Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR (2020).  
[36] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2020. Zero-Shot Text-to-Image Generation. In ICML.  
[37] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In EMNLP.  
[38] Stephen E. Robertson and Steve Walker. 1997. On Relevance Weights with Little Relevance Information. In SIGIR.  
[39] Stephen E. Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. (2009).  
[40] Jason Tyler Rolfe. 2017. Discrete Variational Autoencoders. In ICLR.  
[41] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In NAACL.  
[42] Yi Tay, Vinh Quang Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer Memory as a Differentiable Search Index. In NeurIPS.  
[43] Nandan Thakur, Nils Reimers, Andreas Ruckl'e, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In NeurIPS Datasets and Benchmarks Track (Round 2).  
[44] Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 2017. Neural Discrete Representation Learning. In NIPS.  
[45] Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. 2022. GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval. In NAACL.  
[46] Yujing Wang, Ying Hou, Hong Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, and Mao Yang. 2022. A Neural Corpus Indexer for Document Retrieval. In NeurIPS.  
[47] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In ICLR.  
[48] Erxin Yu, Lan Du, Yuan Jin, Zhepei Wei, and Yi Chang. 2022. Learning Semantic Textual Similarity via Topic-informed Discrete Latent Variables. ArXiv abs/2211.03616 (2022).  
[49] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, M. Zhang, and Shaoping Ma. 2022. Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval. In WSDM.  
[50] Tiancheng Zhao, Kyusong Lee, and Maxine Eskenazi. 2018. Unsupervised Discrete Sentence Representation Learning for Interpretable Neural Dialog Generation. In ACL.  
[51] Guoqing Zheng and Jamie Callan. 2015. Learning to Reweight Terms with Distributed Representations. In SIGIR.  
[52] Yujia Zhou, Jing Yao, Zhicheng Dou, Ledell Yu Wu, Peitian Zhang, and Ji rong Wen. 2022. Ultron: An Ultimate Retriever on Corpus with a Model-based Indexer. ArXiv abs/2208.09257 (2022).  
[53] Shengyao Zhuang, Houxing Ren, Linjun Shou, Jian Pei, Ming Gong, G. Zuccon, and Daxin Jiang. 2022. Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation. *ArXiv* abs/2206.10128 (2022).  
[54] Lukasz Kaiser, Aurko Roy, Ashish Vaswani, Niki Parmar, Samy Bengio, Jakob Uszkoreit, and Noam M. Shazeer. 2018. Fast Decoding in Sequence Models using Discrete Latent Variables. In ICML.

# Footnotes:

Page 0: 1Preprint. Work in progress. 
Page 2: 2We use document  $d$  here for the denotation, noting that the computation is the same when  $q$  is input. 
Page 5: <sup>3</sup>https://public.ukp.informatik.tu-darmstadt.de/kwang/gplgenerated-data/beir/ $^{4}$ https://github.com/joshlk/k-means-constrained 
