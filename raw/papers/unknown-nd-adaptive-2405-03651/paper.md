Adaptive Retrieval and Scalable Indexing for $k$-NN Search with Cross-Encoders
================================================================================

Nishant Yadav1, Nicholas Monath2, Manzil Zaheer2, Rob Fergus2, Andrew McCallum1  
 1 University of Massachusetts Amherst, 2 Google DeepMindNow at Google DeepMind

###### Abstract

Cross-encoder (CE) models which compute similarity
by jointly encoding a query-item pair perform better than
using dot-product with embedding-based models (dual-encoders)
at estimating query-item relevance.
Existing approaches perform $k$-NN search with cross-encoders by
approximating the CE similarity with a vector embedding space fit either with
dual-encoders (DE) or CUR matrix factorization.
DE-based retrieve-and-rerank approaches suffer from poor
recall as DE generalizes poorly to new domains and
the test-time retrieval with DE is decoupled
from the CE.
While CUR-based approaches can be more accurate than
the DE-based retrieve-and-rerank approach, such approaches
require a prohibitively large number of CE calls
to compute item embeddings, thus making it impractical
for deployment at scale.
In this paper, we address these shortcomings with our proposed sparse-matrix factorization based method
that efficiently computes latent query and item representations to
approximate CE scores and performs $k$-NN search with the approximate CE similarity.
In an offline indexing stage, we compute item embeddings by
factorizing a sparse matrix containing query-item CE scores
for a set of train queries.
Our method produces a high-quality approximation while
requiring only a fraction of CE similarity calls
as compared to CUR-based methods, and allows for leveraging
DE models to initialize the embedding space while
avoiding compute- and resource-intensive
finetuning of DE via distillation.
At test time, we keep item embeddings fixed and perform retrieval over multiple rounds,
alternating between a) estimating the test query embedding by
minimizing error in approximating CE scores of items retrieved thus far,
and b) using the updated test query embedding for retrieving more items in the next round.
Our proposed $k$-NN search method can achieve up to 5% and 54% improvement
in $k$-NN recall for $k\=1$ and 100 respectively over the widely-used DE-based retrieve-and-rerank approach.
Furthermore, our proposed approach to index the items by aligning item embeddings with the CE
achieves up to 100$\times$ and 5$\times$ speedup over CUR-based and
dual-encoder distillation based approaches respectively while matching or improving
$k$-NN search recall over baselines.

1 Introduction
--------------

Efficient and accurate nearest neighbor search is paramount for
retrieval*(Menon et al., [2022]; Rosa et al., [2022]; Qu et al., [2021])*,
classification in large output spaces
(e.g., entity linking*(Ayoola et al., [2022]; Logeswaran et al., [2019]; Wu et al., [2020])*),
non-parametric models*(Das et al., [2022]; Wang et al., [2022])*,
and many other such applications in machine learning*(Goyal et al., [2022]; Izacard et al., [2023]; Bahri et al., [2020])*.
The accuracy and efficiency of nearest neighbor search depends on a combination of factors
(1) the computational cost of pairwise distance comparisons between datapoints,
(2) preprocessing time for constructing a nearest neighbor index
(e.g., dimensionality reduction *(Indyk, [2000])*,
quantization*(Ge et al., [2013]; Guo et al., [2020])*,
data structure construction*(Beygelzimer et al., [2006]; Malkov \& Yashunin, [2018]; Zaheer et al., [2019])*),
and (3) the time taken to query the index to retrieve the nearest neighbor(s).

Similarity functions such as cross-encoders which take a pair of
data points as inputs and directly output a scalar score, have achieved
state-of-the-art results on numerous tasks (e.g., QA *(Qu et al., [2021]; Thakur et al., [2021b])*,
entity linking *(Logeswaran et al., [2019])*).
However, these models are exceptionally computationally expensive since these are typically parameterized by several layers of neural models such as transformers*(Vaswani et al., [2017])*,
and scoring each item for a given query requires a forward pass of the large parametric model,
making them impractical similarity functions to use directly in nearest neighbor indices*(Yadav et al., [2022])*.
Initial work has approximated search with cross-encoders (CE) for a given test query
using a heuristic retrieve-and-rerank approach that uses a separate model
to retrieve a subset of items followed by re-ranking using the CE.
Prior work performs the initial retrieval using dot-product of sparse query/item embedding
from models such as BM25, or
dense query/item embeddings from models such as dual-encoders (DE) which are typically
trained on the same task and data as the CE.
To support search with CE, recent
work*(Yadav et al., [2022]; [2023])* improves upon heuristic retrieve-and-rerank approaches, by directly learning an embedding space that approximates the CE score function. These approaches use CUR decomposition*(Mahoney \& Drineas, [2009])* to compute (relatively) low-dimensional embeddings for queries and items. The item embeddings are computed by
scoring each item against a set of *anchor/train* queries. At test-time, the test query embedding is computed by
using CE scores of the test query against a set of (adaptively-chosen) *anchor* items.

Both DE-based retrieve-and-rerank and CUR-based
methods are not well suited for a typical application setting
in $k$-NN search – building an index on a new set of
targets with a given (trained) similarity function.
The DE-based approach has several disadvantages in this setting.
DE models show poor generalization to new domains and thus require
additional fine-tuning on the target domain to improve performance*Yadav et al. ([2022]); Thakur et al. ([2021a])*.
This can be both resource-intensive as well time-consuming.
Furthermore, it requires access to the parameters (not just embedding outputs)
of the DE, which might not be possible if the DE is provided by an API service.
On the other hand, while CUR-based approaches outperform retrieve-and-rerank
approaches without additional fine-tuning of DE, they require computing a dense
score matrix by scoring each item against a set of anchor/train queries.
This does not scale well with the number of items.
For instance, for a domain with 500 anchor/train queries and 10K items,
it takes around 10 hours111On an Nvidia 2080ti GPU with 12 GB memory using batch size\=50 to compute the dense query-item score matrix with a CE parameterized using bert-base *(Yadav et al., [2022])*. By simple extrapolation,
indexing 5 million items using 500 queries would take around 5000 GPU hours.

In this paper, we propose a sparse-matrix factorization-based
approach to improve the efficiency of fitting an embedding space to approximate the cross-encoder for $k$-NN search.
Our proposed approach significantly reduces the offline
indexing cost as compared to existing approaches by constructing
a sparse matrix containing cross-encoder scores between
a set of training queries $(\mathcal{Q}_{\textrm{train}})$
and all the items $(\mathcal{I})$,
and using efficient matrix factorization methods to
produce a set of item embeddings that are aligned with
the cross-encoder.
At test-time, our proposed approach, Axn,
computes a test query embedding to approximate
cross-encoder scores between the test query and items,
and performs retrieval using approximate cross-encoder scores. Axn performs retrieval over multiple rounds while
keeping the item embedding fixed and incrementally refining the test query embedding using
cross-encoder scores of the items retrieved in previous rounds.
In the first round, the cross-encoder is used to score the test query
against a small number of items chosen uniformly at random
or baseline retrieval methods such as dual-encoder or BM25.
In each subsequent round, Axn alternates between
(a) updating the test query embedding to improve
the approximation of the cross-encoder score of items retrieved so far, and
(b) retrieving additional items using the improved approximation of the cross-encoder,
and computing the exact cross-encoder scores for the retrieved items.
Finally, the retrieved items are ranked based on exact cross-encoder scores
and the top-$k$ items returned as the $k$-nearest neighbors for the given test query.

We perform an empirical evaluation of our method using cross-encoder models
trained for the task of entity linking and
information retrieval on ZeShEL *(Logeswaran et al., [2019])* and BeIR *(Thakur et al., [2021b])* benchmark respectively.
Our proposed $k$-NN search method can be used together with dense item
embeddings produced by any method such as baseline dual-encoder models
and still yield up to 5% and 54% improvement in $k$-NN recall for $k\=$1 and 100 respectively
over retrieve-and-rerank style inference with the same dual-encoder.
Furthermore, our proposed approach to align item embeddings with the cross-encoder
achieves up to 100$\times$ and 5$\times$ speedup over CUR-based approaches and
training dual-encoders via distillation-based respectively while matching or improving
test-time $k$-NN search recall over baseline approaches.

2 Proposed Approach
-------------------

##### Task Description

A cross-encoder model $f:\mathcal{Q}\times\mathcal{I}\rightarrow\mathbb{R}$
maps a query-item pair $(q,i)\in\mathcal{Q}\times\mathcal{I}$ to a scalar similarity.
We consider the task of similarity search with the cross-encoder, in
particular finding the $k$-nearest neighbors items for a given query $q$ from a fixed set of items $\mathcal{I}$:

|  | $\mathscr{N}(q)\triangleq\mathop{\mathrm{\arg top}k}_{i\in\mathcal{I}}f(q,i)% \vspace{-0.2cm}$ |  | (1) |
| --- | --- | --- | --- |

where $\mathop{\mathrm{\arg top}k}$ returns the indices of the top $k$ scoring items of the function.
Exact $k$-NN search with a cross-encoder would require $\mathcal{O}(\lvert\mathcal{I}\rvert)$
cross-encoder calls as an item needs to be jointly encoded with the test query
in order to compute its score.
Since cross-encoders are typically parameterized using deep neural
models such as transformers*(Vaswani et al., [2017])*,
$\mathcal{O}(\lvert\mathcal{I}\rvert)$ calls to the cross-encoder
model can be prohibitively expensive at test time.
Therefore, we tackle the task of
approximate $k$-NN search with cross-encoder models.
Let $\hat{f}(\cdot,\cdot)$ denote the approximation to the cross-encoder
that is learned using exact cross-encoder scores
for a sample of query-item pairs.
We refer to the approximate $k$-nearest neighbors as
$\hat{\mathscr{N}}(q)\triangleq\mathop{\mathrm{\arg top}k}_{i\in\mathcal{I}}%
\hat{f}(q,i)$
and measure the quality of the approximation using nearest neighbor recall:
$\frac{\lvert\hat{\mathscr{N}}(q)\cap\mathscr{N}(q)\rvert}{\lvert\mathscr{N}(q)\rvert}$

In this work, we assume black-box access to the cross-encoder222Approximating a neural scoring function by compressing,
approximating, quantizing the scoring function is widely studied but outside
the scope of this paper.,
access to the set of items and train queries from the target domain,
and a base dual-encoder ($\textsc{DE}_{\textsc{src}}$) trained on the same
task and source data as the cross-encoder.
In §[2.1], we first present our proposed sparse-matrix factorization
based method to compute item embeddings in an offline step.
In §[2.2], we present
an online approach to compute a test query embedding
to approximate the cross-encoder scores and perform
$k$-NN search using the approximate cross-encoder scores.

### 2.1 Proposed Offline Indexing of Items

In this section, we describe our proposed
approach to efficiently align the item embeddings with the cross-encoder
where efficiency is measured in terms of the number of
training samples (query-item pairs) required to be gathered and scored
using the cross-encoder and wall-clock time to fit an approximation of
the cross-encoder model.
We consider an approximation of the cross-encoder
with an inner-product space where a query ($q$) and an
item ($i$) are represented with $d$-dimensional
vectors $\mathbf{u}_{q}\in\mathbb{R}^{d}$ and
$\mathbf{v}_{i}\in\mathbb{R}^{d}$ respectively.
$k$-NN search using this approximation corresponds to solving
the following vector-based nearest neighbor search:

|  | $\hat{\mathscr{N}}(q)\triangleq\mathop{\mathrm{\arg top}k}_{i\in\mathcal{I}}% \mathbf{u}_{q}\mathbf{v}_{i}^{\intercal}.$ |  | (2) |
| --- | --- | --- | --- |

This vector-based $k$-nearest neighbor search can potentially be made more
efficient using data structures such as cover trees *(Beygelzimer et al., [2006])*, HNSW*(Malkov \& Yashunin, [2018])*,
or any of the many other highly effective vector nearest neighbor search
indexes *(Guo et al., [2020]; Johnson et al., [2019])*.
The focus of our work is not on a new way to make the vector nearest
neighbor search more efficient, but rather to develop
efficient and accurate methods of fitting the embedded representations of
$\mathbf{u}_{q}$ and $\mathbf{v}_{i}^{\intercal}$ to approximate the cross-encoder scores.

Let $G\in\mathbb{R}^{|\mathcal{Q}_{\textrm{train}}|\times|\mathcal{I}|}$
denote the pairwise similarity matrix containing the exact cross-encoder
over the pairs of training queries ($\mathcal{Q}_{\textrm{train}}$) and items ($\mathcal{I}$).
We assume that $G$ is *partially observed* or incomplete, that is only a very small subset of the query-item pairs ($\mathscr{P}_{\textrm{train}}$) are observed in $G$.
Let $U\in\mathbb{R}^{|\mathcal{Q}_{\textrm{train}}|\times d}$
and $V\in\mathbb{R}^{|\mathcal{I}|\times d}$ be
matrices such that each row corresponds to the embedding of a
query $q\in\mathcal{Q}_{\textrm{train}}$ and an item $i\in\mathcal{I}$ respectively.
We optimize the following widely-used objective for matrix completion to estimate $U$ and $V$ via stochastic gradient descent:

|  | $\min_{\begin{subarray}{c}U\in\mathbb{R}^{|\mathcal{Q}_{\textrm{train}}|\times d% },V\in\mathbb{R}^{\lvert\mathcal{I}\rvert\times d}\end{subarray}}\|(G-UV^{% \intercal})_{\mathscr{P}_{\textrm{train}}}\|_{2}$ |  | (3) |
| --- | --- | --- | --- |

where $(\cdot)_{\mathscr{P}_{\textrm{train}}}$ denotes projection on the set of
observed entries in $G$.
There are two important considerations:
(1) how to select with values of $G$ to observe
(and incur the cost of running the cross-encoder model), and
(2) how to compute/parameterize the matrices $U$ and $V$.

##### Constructing Sparse Matrix $G$

Given a set of items ($\mathcal{I}$) and train queries ($\mathcal{Q}_{\text{train}}$),
we construct the sparse matrix $G$ by selecting $k_{d}$ items $\mathcal{I}_{q}\subset\mathcal{I}$
for each query $q\in\mathcal{Q}_{\textrm{train}}$ either
uniformly at random or using top-$k_{d}$ items from a baseline
retrieval method such as the base dual-encoder ($\textsc{DE}_{\textsc{src}}$).
This approach requires $k_{d}\lvert\mathcal{Q}_{\textrm{train}}\rvert$ calls to the cross-encoder.
We also experiment with an approach that selects
$k_{d}$ queries $\mathcal{Q}_{i}\subset\mathcal{Q}_{\textrm{train}}$
for each item $i\in\mathcal{I}$,
and thus requires $k_{d}\lvert\mathcal{I}\rvert$ calls to the cross-encoder.

##### Parameterizing and Training $U$ and $V$

* •

    Transductive(MFTrns): In this setting, $U$ and
    $V$ are trainable parameters and are learned by optimizing the
    objective in Eq.[4].
    $U$ and $V$ can be optionally initialized using query and item
    embeddings from the base dual-encoder ($\textsc{DE}_{\textsc{src}}$).
    Note that this parameterization requires scoring each item against at least
    a small number of queries to update the embedding of an item from its initialized value,
    thus requiring scoring of $\mathcal{O}(\lvert\mathcal{I}\rvert)$ query-item pairs to construct the sparse matrix $G$.
    Such an approach may not scale well with the number of items as
    the number of cross-encoder calls to construct $G$ and the number of trainable parameters are both linear in the number of items.
    For instance, when $\lvert\mathcal{I}\rvert$ \= 5 million, $d\=1000$, $V$ would contain 5 billion trainable parameters.

* •

    Inductive(MFInd): In this setting, we train parametric models
    to produce query and item embeddings $U$ and $V$ from (raw)
    query and item features such as textual descriptions of queries and items.
    Unlike transductive approaches, inductive matrix factorization approaches can
    produce embeddings for unseen queries and items, and thus can be used to
    produce embeddings for items not scored against any train query in matrix $G$ as well as embeddings for test queries $q_{\texttt{test}}\notin\mathcal{Q}_{\textrm{train}}$.
    Prior work typically uses $\textsc{DE}_{\textsc{src}}$ (a DE trained on the same task and source domains
    as the CE) and finetunes $\textsc{DE}_{\textsc{src}}$ on the target domain via distillation using the CE.
    However, training all parameters of such parametric encoding models via
    distillation can be compute- and resource-intensive as these models
    are built using several layers of neural models such as transformers.
    Recall that our goal is to efficiently build an accurate approximation of the
    CE on a given target domain.
    Thus, to improve the efficiency of fitting the approximation of the CE,
    we propose to train a shallow MLP model (using data from the target domain) that takes query/item embeddings from $\textsc{DE}_{\textsc{src}}$ as input and outputs
    updated embeddings while keeping $\textsc{DE}_{\textsc{src}}$ parameters frozen.

### 2.2 Proposed Test-Time $k$-NN Search Method: Axn

At test-time, we need to perform $k$-NN search for a test query
$q_{\texttt{test}}\notin\mathcal{Q}_{\textrm{train}}$, and thus need to
compute an embedding for the test query in order to approximate
cross-encoder scores and perform retrieval with the approximate scores.
Note that computing the test query embedding by factorizing the matrix $G$ at *test-time* while including the test query $q_{\texttt{test}}$ is computationally infeasible.
Thus, an ideal solution would be to compute item representations
in an offline indexing step, and
compute the test query embedding *on-the-fly* while keeping
item embeddings fixed.
A potential solution is to use a parametric model such as $\textsc{DE}_{\textsc{src}}$ or MFInd to compute test query embedding, perform retrieval using inner-product scores
between test query and item embeddings, and finally, re-rank the retrieved items
using the cross-encoder.
While such a retrieve-and-rerank approach can work, the retrieval
step on such an approach is decoupled from the re-ranking model,
and thus may result in poor recall.

In this work, we propose an adaptive approach Axn, which
stands for ”Adaptive Cross-Encoder Nearest Neighbor Search”.
As described in Algorithm[1], Axn performs retrieval over $\mathscr{R}$ rounds while incrementally
refining the cross-encoder approximation for $q_{\texttt{test}}$ by
updating $\mathbf{u}_{q_{\texttt{test}}}$, the embedding for $q_{\texttt{test}}$.
The test-time inference latency (and throughput) depends largely on
the number of cross-encoder calls made at test time as each cross-encoder call
requires a forward pass through a large neural model.
Thus, we operate under a fixed computational budget which allows for up to
$\mathcal{B}_{\textsc{ce}}$ cross-encoder calls at test-time.

*Algorithm 1  Axn - Test-time $k$-NN Search Inference*

1: Input: $q$: Test query, $V\in\mathbb{R}^{\lvert\mathcal{I}\rvert\times d}$ Item Embeddings, $\mathscr{R}$: Number of iterative search rounds, $k_{s}$: Number of items to retrieve in each round, $f_{\theta}$: Cross-Encoder (CE) model

2: Output: $\hat{S}$: Approximate scores of $q$ with all items,
$\mathscr{A}_{\mathscr{R}}$: Retrieved items with CE scores in $\mathbf{a}_{\mathscr{R}}$.

3:$\mathscr{A}_{1}\leftarrow\textsc{Init}(\mathcal{I},k_{s})$ $\rhd$ Initial set of items

4:$\mathbf{a}_{1}\leftarrow[f_{\theta}(q,i)]_{i\in\mathscr{A}_{1}}$ $\rhd$ CE scores of $q$ with items in $\mathscr{A}_{1}$

5:$\mathbf{u}_{q}\leftarrow\text{Solve-Linear-Regression}(V,\mathscr{A}_{1},%
\mathbf{a}_{1})$ $\rhd$ Compute query embedding by solving Eq.[4]

6: for$r\leftarrow 2\text{ to }\mathscr{R}$do

7:$\hat{S}^{(r)}\leftarrow\mathbf{u}_{q}\times V^{\intercal}$ $\rhd$ Update approx. scores

8:$\mathscr{A}_{r}\leftarrow\mathscr{A}_{r-1}\cup\mathop{\mathrm{\arg top}k}_{i%
\in\mathcal{I}\setminus\mathscr{A}_{r-1},k\=k_{s}}\hat{S}^{(r)}_{i}$ $\rhd$ Retrieve $k_{s}$ new items

9:$\mathbf{a}_{r}\leftarrow\mathbf{a}_{r-1}\oplus[f_{\theta}(q,i)]_{i\in\mathscr{%
A}_{r}\setminus\mathscr{A}_{r-1}}$ $\rhd$ Compute CE scores of new items

10:$\mathbf{u}_{q}\leftarrow\text{Solve-Linear-Regression}(V,\mathscr{A}_{r},%
\mathbf{a}_{r})$ $\rhd$ Compute query embedding by solving Eq.[4]

11:$\hat{S}\leftarrow\mathbf{u}_{q}\times V^{\intercal}$ $\rhd$ Compute approx. scores

12: return $\hat{S},\mathscr{A}_{\mathscr{R}},\mathbf{a}_{\mathscr{R}}$

Let $\mathscr{A}_{r}$ be the cumulative set of items chosen up to round $r$.
In the first round ($r\=1$), we select $\mathcal{B}_{\textsc{ce}}/\mathscr{R}$ items either uniformly at random
or using separate retrieval models such as dual-encoders or BM25
and compute the exact cross-encoder scores of these items for the given test query.
We compute the test query embedding $\mathbf{u}_{q_{\texttt{test}}}$ by solving the following
system of linear equations

|  | $V_{\mathscr{A}_{r}}\mathbf{u}_{q_{\texttt{test}}}\=\mathbf{a}_{r}$ |  | (4) |
| --- | --- | --- | --- |

where $V_{\mathscr{A}_{r}}\in\mathbb{R}^{\lvert\mathscr{A}_{r}\rvert\times d}$
contains embeddings for items in $\mathscr{A}_{r}$, and $\mathbf{a}_{r}$
contains cross-encoder scores for $q_{\texttt{test}}$ paired with items in $\mathscr{A}_{r}$.
In round $r>1$, we select additional $\mathcal{B}_{\textsc{ce}}/\mathscr{R}$ items from
$\mathcal{I}\setminus\mathscr{A}_{r-1}$
using inner-product of test query embedding $\mathbf{u}_{q_{\texttt{test}}}$ and item
embeddings $\mathbf{v}_{i}$ (line[8] in Alg.[1]).

|  | $\mathscr{A}_{r}\=\mathscr{A}_{r-1}\cup\mathop{\mathrm{\arg top}k}_{i\in\mathcal% {I}\setminus\mathscr{A}_{r-1},k\=\mathcal{B}_{\textsc{ce}}/\mathscr{R}}\mathbf{% u}_{q_{\texttt{test}}}\mathbf{v}_{i}^{\intercal}$ |  | (5) |
| --- | --- | --- | --- |

After computing $\mathscr{A}_{r}$, we compute CE scores for new items
chosen in round $r$, and we update the test query embedding
$\mathbf{u}_{q_{\texttt{test}}}$ by solving Eq.[4] with the latest set of items $\mathscr{A}_{r}$ which includes additional
items selected in round $r$.
Note that solving for $\mathbf{u}_{q_{\texttt{test}}}$ in Eq[4] is akin to solving a linear regression problem with embeddings of items
in $\mathscr{A}_{r}$ as features and cross-encoder scores of the items
as regression targets.
We solve Eq.[4] analytically to get
$\mathbf{u}_{q_{\texttt{test}}}~{}\=~{}(V_{\mathscr{A}_{r}}^{\intercal}V_{%
\mathscr{A}_{r}})^{\dagger}V_{\mathscr{A}_{r}}^{\intercal}\mathbf{a}_{r}$ where $M^{\dagger}$ denotes pseudo-inverse of a matrix $M$.

At the end of $\mathscr{R}$ rounds, we obtain $\mathscr{A}_{\mathscr{R}}$ containing $\mathcal{B}_{\textsc{ce}}$ items,
all of which have been scored using the cross-encoder model. We return
top-$k$ items from this set sorted based on exact cross-encoder scores as the set of approximate $k$-NN for given test query $q_{\texttt{test}}$

|  | $\hat{\mathscr{N}}(q_{\texttt{test}})\=\mathop{\mathrm{\arg top}k}_{i\in\mathscr% {A}_{\mathscr{R}}}f(q_{\texttt{test}},i)$ |  | (6) |
| --- | --- | --- | --- |

##### Regularizing Test Query Embedding

The system of equation in Eq[4] in round $r$ contains $\lvert\mathscr{A}_{r}\rvert$ equations with $d$ variables and is an under-determined system when $\lvert\mathscr{A}_{r}\rvert<d$.
In such a case, there exist infinitely many solutions to
Eq[4] and the test query
embedding $\mathbf{u}_{q_{\texttt{test}}}$ can achieve zero approximation
error on items in $\mathscr{A}_{r}$, and may show poor generalization
when estimating cross-encoder scores for items in $\mathcal{I}\setminus\mathscr{A}_{r}$.
Since the approximate scores are used to select the additional set
of items in round $r+1$ (line[8] in Alg.[1]),
such poor approximation affects the additional set of items chosen, and
subsequently, it may affect the overall retrieval quality in certain settings.
To avoid such overfitting, we compute the final test query embedding as:

|  | $\mathbf{u}_{q_{\texttt{test}}}\=(1-\lambda)\mathbf{u}_{q_{\texttt{test}}}^{% \text{(LinReg)}}+\lambda\mathbf{u}^{\text{(param)}}_{q_{\texttt{test}}}$ |  | (7) |
| --- | --- | --- | --- |

where $\mathbf{u}_{q_{\texttt{test}}}^{\text{(LinReg)}}$ is the analytical solution
to the linear system in Eq.[4] and $\mathbf{u}^{\text{(param)}}_{q_{\texttt{test}}}$ is the test query
embedding obtained from a parametric model such as a dual-encoder
or an inductive matrix factorization model. We tune the
weight parameter $\lambda\in[0,1]$ on the dev set.

3 Experiments
-------------

In our experiments, we evaluate proposed approaches
and baselines on the task of finding $k$-nearest neighbors
for cross-encoder (CE) models as well as on downstream tasks.
We use cross-encoders trained for the downstream task of zero-shot entity
linking and zero-shot information retrieval and
present extensive analysis of the effect of various design choices
on the offline indexing latency and the test-time retrieval recall.

##### Experimental Setup

We run experiments on two datasets/benchmarks – ZeShEL *(Logeswaran et al., [2019])*,
a zero-shot entity linking benchmark, and BeIR benchmark*(Thakur et al., [2021b])*,
a collection of information retrieval datasets for evaluating
zero-shot performance of IR models.
We use separate CE models for ZeShEL and BeIR datasets,
trained using ground-truth labeled data from the corresponding dataset.
For evaluation, we use two test domains from ZeShEL dataset –YuGiOh and Star Trek with 10K and 34K items (entities) respectively, and
we use SciDocs and Hotpot-QA datasets from BeIR with 25K and 5M items (documents) respectively.
These domains were *not* part of the data used to
train the corresponding cross-encoder models.
Following the precedent set by previous
work*(Yadav et al., [2022]; [2023])*, we create a
train/test split uniformly at random for each ZeShEL domain.
For datasets from BeIR, we use pseudo-queries
released as part of the benchmark as train queries
and test on queries in the official test split in BeIR benchmark.
We use queries in the train split to
train proposed matrix factorization models or
baseline DE models via distillation,
and we evaluate on the corresponding domain’s test split.
We refer interested readers to Appendix[A] for more details about datasets, cross-encoder training, and model architecture.

##### Baselines

We compare with the following retrieve-and-rerank baselines,
denoted by RnRX, where top-scoring items wrt baseline scoring method $X$ are retrieved
and then re-ranked using the CE.

* •

    tf-idf: It computes the similarity score for a query-item pair using the dot-product of sparse query/item vectors containing TF-IDF weights.

* •

    Dual-Encoders (DE):
    It computes query-item scores using the dot-product of dense embeddings
    produced by encoding queries and items separately.
    We experiment with two DE models.

    + –
            $\textsc{DE}_{\textsc{src}}$: DE trained on the same *source* data and downstream task as the cross-encoder model. This model is *not* trained or finetuned on the target domains
            used for evaluation in this work.

        + –
            $\textsc{DE}_{\textsc{dstl}}$: This corresponds to $\textsc{DE}_{\textsc{src}}$ further finetuned via distillation using the
            cross-encoder model on the *target* domain i.e. the domain used for evaluation.

We also compare with adaCUR *(Yadav et al., [2023])*,
a CUR-based approach that computes a dense matrix with CE
scores between training queries and all items to index the items, and performs adaptive retrieval at test time.
We use $\textsc{adaCUR}_{X}$ to denote inference with adaCUR method when items in the
first round are chosen using method $X\in{\textsc{DE}_{\textsc{src}},\textsc{tf-idf}}$.
We refer readers to Appendix[A] for implementation details for all baselines and proposed approaches.

##### Proposed Approach

We construct the sparse matrix $G$ on the target domain
by selecting top-scoring items wrt $\textsc{DE}_{\textsc{src}}$ for each
query in $\mathcal{Q}_{\text{train}}$ followed by
computing the CE scores for observed query-item pairs in $G$.
We use $\textsc{DE}_{\textsc{src}}$ to initialize embeddings for train queries
and all items, followed by inductive (MFInd)
or transductive (MFTrns) matrix factorization
while minimizing the objective function in[3].
We use the same sparse matrix $G$ when training DE
via distillation ($\textsc{DE}_{\textsc{dstl}}$) on the target domain.
We use AxnX,Y to denote the proposed $k$-NN search
method (§[2.2]) when using
method $X$ to compute item embeddings and method $Y$
to retrieve items in the first round.

##### Evaluation Metrics

Following the precedent set by previous work*(Yadav et al., [2022]; [2023])*,
we use Top-$k$-Recall@$m$ for test queries as the evaluation metric which measures
the fraction of $k$-nearest neighbors as per the CE which
are present in the set of $m$ retrieved items.
For each method, we retrieve $m$ items and re-rank them using
exact CE scores.
We also evaluate the quality of the retrieved $k$-NN items wrt the CE
on the downstream task. We use entity linking accuracy for ZeShEL,
and we use downstream task specific nDCG@10 and recall for BeIR domains.

For each approach, we calculate the time taken for indexing a given set of
items from the target domain which involves some or all of the
following steps: a) computing query/item embeddings using $\textsc{DE}_{\textsc{src}}$,
b) computing (dense or sparse) query-item score matrix $G$ for $\mathcal{Q}_{\text{train}}$,
c) gradient-based training using $G$ to estimate item embeddings for MFTrns or parameters of models such as $\textsc{DE}_{\textsc{dstl}}$ and MFInd, and
d) for $\textsc{DE}_{\textsc{dstl}}$ and MFInd, computing updated item embeddings after training.

### 3.1 Results

Figure[1] vs Retrieve-and-Rerank (RnR) ‣ 3.1 Results ‣ 3 Experiments ‣ Adaptive Retrieval and Scalable Indexing for 𝑘-NN Search with Cross-Encoders") shows Top-1-Recall@Inference-Cost\=100 and Top-100-Recall@Inference-Cost\=500
versus the total wall-clock time taken to index the items for various approaches on YuGiOh and Hotpot-QA. adaCUR can control the indexing time by
varying $\lvert\mathcal{Q}_{\text{train}}\rvert$, the number of train queries,
while MF and distillation-based methods can control
the indexing time by varying $\lvert\mathcal{Q}_{\text{train}}\rvert$ and
the number of items scored per train query ($k_{d}$).
For YuGiOh, we use $\lvert\mathcal{Q}_{\text{train}}\rvert\leq$ 500 for all methods, and for Hotpot-QA, we use $\lvert\mathcal{Q}_{\text{train}}\rvert\leq$ 1K for adaCUR and $\lvert\mathcal{Q}_{\text{train}}\rvert\leq$ 50K with other methods.

##### Proposed Inference (Axn) vs Retrieve-and-Rerank (RnR)

Axn consistently provides
improvement over the corresponding retrieve-and-rerank (RnR) baseline.
For instance, Axn$\textsc{DE}_{\textsc{src}}$,$\textsc{DE}_{\textsc{src}}$ provides
an improvement of 5.2% for $k$\=1 and 54% for $k$\=100 over RnR$\textsc{DE}_{\textsc{src}}$ for domain\=YuGiOh.
Note that this performance improvement comes at *no additional* offline indexing cost and with negligible test-time overhead333We refer readers to §[B.1] for analysis of overhead incurred by Axn. RnRtf-idf performs poorly on YuGiOh while it serves
as a strong baseline for Hotpot-QA, potentially due to
differences in task, data, and CE model.
On Hotpot-QA, Top-$k$-Recall for Axn can be further
improved by sampling items in the first round using tf-idf (AxnZ,tf-idf) instead of
$\textsc{DE}_{\textsc{src}}$ (AxnZ,$\textsc{DE}_{\textsc{src}}$) for all indexing
methods $\text{Z}\in{\textsc{DE}_{\textsc{src}},\textsc{DE}_{\textsc{dstl}},\textsc{MF%
}\textsubscript{}}$.

![Refer to caption]()

![Refer to caption]()

*(a) Domain \= YuGiOh, 10K items*

![Refer to caption]()

*(b) Domain \= Hotpot-QA, 5 million items*

*Figure 1:  Top-1-Recall and Top-100-Recall at inference cost budget ($m$) of 100 and 500 CE calls respectively
versus indexing time for various approaches. Matrix factorization approaches (MF) can
be significantly faster than adaCUR and training DE via distillation ($\textsc{DE}_{\textsc{dstl}}$).
The proposed adaptive $k$-NN search method (Axn)
provides consistent improvement over corresponding retrieve-and-rerank style inference (RnR).*

![Refer to caption]()![Refer to caption]()

![Refer to caption]()\phantomcaption

*Figure 2: Breakdown of indexing latency of MF and $\textsc{DE}_{\textsc{dstl}}$ into various steps with training time shown on the right of each bar for different values of
$\lvert\mathcal{Q}_{\text{train}}\rvert$ and no. of items scored per query ($k_{d}$).*

##### Matrix Factorization vs $\textsc{DE}_{\textsc{dstl}}$

Unsurprisingly, performance on the target domain can be further improved
by using data from the target domain to fit an embedding space to approximate the CE.
As shown in Figure[1] vs Retrieve-and-Rerank (RnR) ‣ 3.1 Results ‣ 3 Experiments ‣ Adaptive Retrieval and Scalable Indexing for 𝑘-NN Search with Cross-Encoders"), our proposed matrix factorization based approaches (MF)
can be significantly more efficient than the distillation-based ($\textsc{DE}_{\textsc{dstl}}$) approaches
while matching or outperforming $\textsc{DE}_{\textsc{dstl}}$ in terms of
$k$-NN search recall in the majority of the cases.
Figure[2] vs Retrieve-and-Rerank (RnR) ‣ 3.1 Results ‣ 3 Experiments ‣ Adaptive Retrieval and Scalable Indexing for 𝑘-NN Search with Cross-Encoders") shows the breakdown of total indexing time of $\textsc{DE}_{\textsc{dstl}}$ and MF for different numbers of training queries ($\lvert\mathcal{Q}_{\text{train}}\rvert$) and
number of items scored per query ($k_{d}$) using the CE
in the sparse matrix $G$.
As expected, both the time taken to compute $G$ and
the training time increases with the number of queries
and the number of items scored per query.
The training time does not increase proportionally after 10K queries
as we allocated a maximum training time of 24 hours for all methods.
For MF, the majority of the time is spent either
in computing sparse matrix $G$ or the initial item embeddings.
While we report total GPU hours taken for CE calls to compute $G$
and initial item embeddings, these steps can be easily
parallelized across multiple GPUs without any communication overhead.
Since $\textsc{DE}_{\textsc{dstl}}$ trains all parameters of a
large parametric neural model, it requires large amounts
of GPU memory and takes up to several hours444We trained dual-encoders on an Nvidia RTX8000 GPU with 48 GB memory for a maximum of 24 hours..
In contrast, MF-approaches
require significantly less memory555We used an Nvidia 2080ti with 12 GB memory for MF-based methods. and training time as these approaches train the item embeddings
as free parameters (MFTrns) or train a shallow neural network on top
of fixed embeddings (MFInd) from an existing DE.
We report results for MFTrns on small-scale domains (e.g. YuGiOh with 10K items)
and for MFInd on large-scale domain Hotpot-QA (5 million items).
We refer interested readers to Appendix[B.3] for comparison of MFTrns and MFInd on small- and large-scale datasets.

##### Proposed Approaches vs adaCUR

Our proposed inference method (Axn) in combination with MF or DE can outperform or closely match the performance
of adaCUR while requiring orders of magnitude less compute for the offline
indexing stage, on both small- and large-scale datasets.
For instance, $\textsc{adaCUR}_{\textsc{DE}_{\textsc{src}}}$ requires 1000+ GPU hours
for embedding 5 million items in Hotpot-QA, and
achieves Top-1-Recall@100 \= 75.9 and Top-100-Recall@500 \= 44.8.
In contrast, MFInd with $\lvert\mathcal{Q}_{\text{train}}\rvert$\=10K and 100 items per query
takes less than three hours to fit
item embeddings, and AxnMFInd,$\textsc{DE}_{\textsc{src}}$ achieves Top-1-Recall@100 \= 80.5 and Top-100-Recall@500 \= 42.6.

![Refer to caption]()

![Refer to caption]()

*(a) YuGiOh*

![Refer to caption]()

*(b) Star Trek*

![Refer to caption]()

*(c) SciDocs*

![Refer to caption]()

*(d) Hotpot-QA*

*Figure 3: Downstream task performance versus indexing time for proposed and baseline approaches on different domains. All methods use a fixed inference cost budget of 100 cross-encoder calls.*

##### Downstream Task Performance

Figure[3] shows
downstream task performance on proposed and baseline approaches
including Exact which performs exact brute-force search using CE at test-time.
For Hotpot-QA, we observe that improvement in $k$-NN search accuracy
results in improvement in downstream task performance with Exact brute-force performing the best.
We observe a different trend on SciDocs, YuGiOh, and Star Trek where Exact search results in suboptimal performance as compared to RnR$\textsc{DE}_{\textsc{src}}$.
For instance, RnR$\textsc{DE}_{\textsc{src}}$ achieves accuracy of 50.6
while the accuracy of Exact is 49.8 on the downstream task of entity linking
on YuGiOh.
We believe that this difference in trends in $k$-NN search
performance and downstream task performance could be due to differences
in the training setup of the corresponding CE (i.e.
the loss function and negatives used during training,
see Appendix[A.1] for details)
as well as the nature of the task and data.
While beyond the scope of this paper, it would be interesting
to explore different loss functions and training strategies such as using negative items mined using $k$-NN search strategies
proposed in this work
to improve the robustness and generalization capabilities of
cross-encoders and minimize such discrepancies in
$k$-NN search and downstream task performance.

We refer readers to Appendix[B] for
an analysis of the overhead incurred by Axn(§[B.1]),
a comparison of Axn with pseudo-relevance feedback based approaches (§[B.2]),
an analysis of design choices for our proposed approach (§[B.3],[B.4]), and
results on other downstream evaluation metrics for BeIR.

4 Related Work
--------------

##### Approximating Similarity Function

Matrix factorization methods have been widely used
for computing low-rank approximation of
dense distance and kernel matrices*(Musco \& Woodruff, [2017]; Bakshi \& Woodruff, [2018]; Indyk et al., [2019])*,
non-PSD matrices*(Ray et al., [2022])* as well as for estimating missing entries in sparse matrices*(Koren et al., [2009]; Luo et al., [2014]; Yu et al., [2014]; Mehta \& Rana, [2017]; Xue et al., [2017])*.
In this work, we focus on methods for factorizing sparse matrices
instead of dense matrices
as computing each entry in the matrix (i.e. CE score for a query-item pair)
requires a forward-pass through an expensive neural model.
An essential assumption for matrix completion methods
is that the underlying matrix $M$ is low-rank, thus enabling
recovery of the missing entries while only observing a small
fraction of entries in $M$*(Candes \& Recht, [2012]; Nguyen et al., [2019])*.
Theoretically, such matrix completion methods require $\Omega(nr)$
samples to recover an $m\times n$ matrix of rank $r$ with
$m\leq n$*(Krishnamurthy \& Singh, [2013]; Xu et al., [2015])*. The sample complexity can be
improved in the presence of features describing rows and columns of the
matrix, often referred to as side information*(Jain \& Dhillon, [2013]; Xu et al., [2013]; Zhong et al., [2019])*.
Inductive matrix completion (MFInd) approaches leverage such query and item
features to improve the sample complexity and also enable generalization to unseen queries (rows) and items (columns).
Training dual-encoder (DE) models via distillation using a cross-encoder (CE),
where the DE consumes raw query and item features
(such as query/item description) and produces query/item
embeddings, can be seen as solving an inductive matrix factorization problem.
A typical training objective for training DE involves
minimizing the discrepancy between CE (teacher model)
and DE (student model) scores on observed entries in the sparse
matrix*(Hofstätter et al., [2020]; Reddi et al., [2021]; Thakur et al., [2021a])*.
Recent work has explored different strategies for
distillation-based training of DE such as curriculum learning
based methods*(Zeng et al., [2022])*, joint training of CE and DE
to mutually improve the performance of both models*(Liu et al., [2022]; Ren et al., [2021])*.
Inductive MF methods (MFInd) used in this work also share similar motivations to adapters*(Houlsby et al., [2019])* which
introduce a small number of trainable parameters
between layers of the model, and may reduce
training time and memory requirements
in certain settings*(Rücklé et al., [2021])*. MFInd used in this work only
trains a shallow MLP on top of query/item embeddings from DE while keeping DE parameters frozen,
and does not introduce
any parameters in the DE.

##### Nearest Neighbor Search

$k$-NN search has been widely studied in applications where
the inputs are described as vectors in
$\mathbb{R}^{d}$*(Clarkson et al., [2006]; Li et al., [2019])*,
and the similarity is computed using simple (dis-)similarity
functions such as inner-product*(Johnson et al., [2019]; Guo et al., [2020])* and
$\ell_{2}$-distance*(Kleinberg, [1997]; Chávez et al., [2001]; Hjaltason \& Samet, [2003])*.
These approaches typically work by speeding up
each distance/similarity
computation*(Jegou et al., [2010]; Hwang et al., [2012]; Zhang et al., [2014]; Yu et al., [2017]; Bagaria et al., [2021])* as well as constructing tree-based*(Beygelzimer et al., [2006]; Dong et al., [2020])* or graph-based data structures*(Malkov \& Yashunin, [2018]; Wang et al., [2021a]; Groh et al., [2022])* over the given item set to efficiently
navigate and prune the search space to find (approximate)
$k$-NN items for a given query.
Recent work also explores such graph-based*(Boytsov \& Nyberg, [2019a]; Tan et al., [2020]; [2021]; MacAvaney et al., [2022])*, or
tree-based*(Boytsov \& Nyberg, [2019b])* data structures for non-metric
and parametric similarity functions.
Another line of work explores model quantization*(Nayak et al., [2019]; Liu et al., [2021])* and early-exit strategies*(Xin et al., [2020a]; [b])* to approximate the neural model while speeding up each forward pass through
the model and reducing its memory footprint.
It would be interesting to study if such data structures
and approaches to speed up cross-encoder score computation can be
combined with matrix factorization based approaches proposed in this work
to further improve recall-vs-cost trade-offs for $k$-NN search with cross-encoders.

##### Pseudo-Relevance Feedback (PRF)

Similar to PRF-based methods in information retrieval*(Rocchio Jr, [1971]; Lavrenko \& Croft, [2001])*, our proposed
$k$-NN search method Axn refines the test query representation
using model-based feedback. In our case, we use the cross-encoder scores
of items retrieved in the previous round as feedback to update the test query
representation. PRF-based approaches have been widely used in information
retrieval for retrieval with sparse*(Li et al., [2018]; Mao et al., [2020]; [2021])* and dense embeddings*(Yu et al., [2021]; Wang et al., [2021b])*.
We refer readers to Appendix §[B.2] for comparison with a recent PRF-based method*(Sung et al., [2023])*.

5 Conclusion
------------

In this paper, we present an approach to perform $k$-NN search
with cross-encoders by efficiently approximating the cross-encoder
scores using dot-product of learned test query and item embeddings.
In the offline indexing step, we compute item embeddings to index a given set of items
from a target domain by factorizing a sparse query-item score matrix, leveraging
existing dual-encoder models to initialize the item embeddings while avoiding
computationally-expensive distillation-based training
of dual-encoder models.
At test time, we compute the test query embedding to approximate
cross-encoder scores of the given test query for a small set of adaptively-chosen items, and perform retrieval with the approximate cross-encoder
scores.
We perform extensive empirical analysis on two zero-shot retrieval benchmarks
and show that our proposed approach provides significant improvement
in test-time $k$-NN search recall-vs-cost tradeoffs
while still requiring significantly less compute resources
for indexing items from a target domain as compared to previous approaches.

Acknowledgments
---------------

We thank members of UMass IESL for helpful discussions and feedback. This work was supported
in part by the Center for Data Science and the Center for Intelligent Information Retrieval, in part
by the National Science Foundation under Grant
No. NSF1763618, in part by the Chan Zuckerberg
Initiative under the project “Scientific Knowledge
Base Construction”, in part by International Business Machines Corporation Cognitive Horizons
Network agreement number W1668553, in part by
Amazon Digital Services, and in part using highperformance computing equipment obtained under
a grant from the Collaborative R\&D Fund managed
by the Massachusetts Technology Collaborative.
Any opinions, findings, conclusions, and recommendations expressed in this material are those of
the authors and do not necessarily reflect those of
the sponsor(s).

References
----------

* Ayoola et al. (2022)Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, and
Andrea Pierleoni.ReFinED: An efficient zero-shot-capable approach to end-to-end
entity linking.In *Proceedings of the 2022 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies: Industry Track*, pp. 209–220. Association for Computational
Linguistics, 2022.
* Bagaria et al. (2021)Vivek Bagaria, Tavor Z Baharav, Govinda M Kamath, and N Tse David.Bandit-based monte carlo optimization for nearest neighbors.*IEEE Journal on Selected Areas in Information Theory*,
2(2):599–610, 2021.
* Bahri et al. (2020)Dara Bahri, Heinrich Jiang, and Maya Gupta.Deep k-nn for noisy labels.In *International Conference on Machine Learning*, pp. 540–550. PMLR, 2020.
* Bakshi \& Woodruff (2018)Ainesh Bakshi and David Woodruff.Sublinear time low-rank approximation of distance matrices.*Advances in Neural Information Processing Systems*, 2018.
* Beygelzimer et al. (2006)Alina Beygelzimer, Sham Kakade, and John Langford.Cover trees for nearest neighbor.In *Proceedings of the 23rd international conference on Machine
learning*, pp. 97–104, 2006.
* Boytsov \& Nyberg (2019a)Leonid Boytsov and Eric Nyberg.Accurate and fast retrieval for complex non-metric data via
neighborhood graphs.In *International Conference on Similarity Search and
Applications*, pp. 128–142. Springer, 2019a.
* Boytsov \& Nyberg (2019b)Leonid Boytsov and Eric Nyberg.Pruning algorithms for low-dimensional non-metric k-nn search: a case
study.In *International Conference on Similarity Search and
Applications*, pp. 72–85. Springer, 2019b.
* Candes \& Recht (2012)Emmanuel Candes and Benjamin Recht.Exact matrix completion via convex optimization.*Communications of the ACM*, 55(6):111–119,
2012.
* Chávez et al. (2001)Edgar Chávez, Gonzalo Navarro, Ricardo Baeza-Yates, and José Luis
Marroquín.Searching in metric spaces.*ACM computing surveys (CSUR)*, 33(3):273–321, 2001.
* Clarkson et al. (2006)Kenneth L Clarkson et al.Nearest-neighbor searching and metric space dimensions.*Nearest-neighbor methods for learning and vision: theory and
practice*, pp. 15–59, 2006.
* Das et al. (2022)Rajarshi Das, Ameya Godbole, Ankita Naik, Elliot Tower, Manzil Zaheer, Hannaneh
Hajishirzi, Robin Jia, and Andrew McCallum.Knowledge base question answering by case-based reasoning over
subgraphs.In *International Conference on Machine Learning*. PMLR, 2022.
* Dong et al. (2020)Yihe Dong, Piotr Indyk, Ilya Razenshteyn, and Tal Wagner.Learning space partitions for nearest neighbor search.2020.
* Ge et al. (2013)Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun.Optimized product quantization.*IEEE transactions on pattern analysis and machine
intelligence*, 36(4):744–755, 2013.
* Goyal et al. (2022)Anirudh Goyal, Abram Friesen, Andrea Banino, Theophane Weber, Nan Rosemary Ke,
Adrià Puigdomènech Badia, Arthur Guez, Mehdi Mirza, Peter C
Humphreys, Ksenia Konyushova, Michal Valko, Simon Osindero, Timothy
Lillicrap, Nicolas Heess, and Charles Blundell.Retrieval-augmented reinforcement learning.In *Proceedings of the 39th International Conference on Machine
Learning*, 2022.
* Groh et al. (2022)Fabian Groh, Lukas Ruppert, Patrick Wieschollek, and Hendrik PA Lensch.Ggnn: Graph-based gpu nearest neighbor search.*IEEE Transactions on Big Data*, 9(1):267–279, 2022.
* Guo et al. (2020)Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and
Sanjiv Kumar.Accelerating large-scale inference with anisotropic vector
quantization.In *International Conference on Machine Learning*, pp. 3887–3896, 2020.
* Hjaltason \& Samet (2003)Gisli R Hjaltason and Hanan Samet.Index-driven similarity search in metric spaces (survey article).*ACM Transactions on Database Systems (TODS)*, 28(4):517–580, 2003.
* Hofstätter et al. (2020)Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan,
and Allan Hanbury.Improving efficient neural ranking models with cross-architecture
knowledge distillation.*ArXiv*, abs/2010.02666, 2020.
* Houlsby et al. (2019)Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin
De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly.Parameter-efficient transfer learning for nlp.In *International Conference on Machine Learning*, pp. 2790–2799. PMLR, 2019.
* Hwang et al. (2012)Yoonho Hwang, Bohyung Han, and Hee-Kap Ahn.A fast nearest neighbor search algorithm by nonlinear embedding.In *2012 IEEE Conference on Computer Vision and Pattern
Recognition*, pp. 3053–3060. IEEE, 2012.
* Indyk (2000)Piotr Indyk.Dimensionality reduction techniques for proximity problems.In *Proceedings of the eleventh annual ACM-SIAM symposium on
Discrete algorithms*, pp. 371–378, 2000.
* Indyk et al. (2019)Pitor Indyk, Ali Vakilian, Tal Wagner, and David P Woodruff.Sample-optimal low-rank approximation of distance matrices.In *Conference on Learning Theory*, pp. 1723–1751. PMLR,
2019.
* Izacard et al. (2023)Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave.Atlas: Few-shot learning with retrieval augmented language models.*Journal of Machine Learning Research*, 24, 2023.URL [http://jmlr.org/papers/v24/23-0037.html](http://jmlr.org/papers/v24/23-0037.html "").
* Jain \& Dhillon (2013)Prateek Jain and Inderjit S Dhillon.Provable inductive matrix completion.*arXiv preprint arXiv:1306.0626*, 2013.
* Jegou et al. (2010)Herve Jegou, Matthijs Douze, and Cordelia Schmid.Product quantization for nearest neighbor search.*IEEE Transactions on Pattern Analysis and Machine
Intelligence*, 33(1):117–128, 2010.
* Johnson et al. (2019)Jeff Johnson, Matthijs Douze, and Hervé Jégou.Billion-scale similarity search with gpus.*IEEE Transactions on Big Data*, 7(3):535–547, 2019.
* Kleinberg (1997)Jon M Kleinberg.Two algorithms for nearest-neighbor search in high dimensions.In *Proceedings of the twenty-ninth annual ACM symposium on
Theory of computing*, pp. 599–608, 1997.
* Koren et al. (2009)Yehuda Koren, Robert Bell, and Chris Volinsky.Matrix factorization techniques for recommender systems.*Computer*, 42(8):30–37, 2009.
* Krishnamurthy \& Singh (2013)Akshay Krishnamurthy and Aarti Singh.Low-rank matrix and tensor completion via adaptive sampling.*Advances in Neural Information Processing Systems*, 26, 2013.
* Lavrenko \& Croft (2001)Victor Lavrenko and W. Bruce Croft.Relevance based language models.SIGIR ’01, pp. 120–127. Association for Computing Machinery,
2001.ISBN 1581133316.doi: 10.1145/383952.383972.URL [https://doi.org/10.1145/383952.383972](https://doi.org/10.1145/383952.383972 "").
* Li et al. (2018)Canjia Li, Yingfei Sun, Ben He, Le Wang, Kai Hui, Andrew Yates, Le Sun, and
Jungang Xu.NPRF: A neural pseudo relevance feedback framework for ad-hoc
information retrieval.In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing*, pp. 4482–4491. Association for Computational
Linguistics, 2018.doi: 10.18653/v1/D18-1478.URL [https://aclanthology.org/D18-1478](https://aclanthology.org/D18-1478 "").
* Li et al. (2019)Wen Li, Ying Zhang, Yifang Sun, Wei Wang, Mingjie Li, Wenjie Zhang, and Xuemin
Lin.Approximate nearest neighbor search on high dimensional
data—experiments, analyses, and improvement.*IEEE Transactions on Knowledge and Data Engineering*,
32(8):1475–1488, 2019.
* Liu et al. (2021)Fangxin Liu, Wenbo Zhao, Zhezhi He, Yanzhi Wang, Zongwu Wang, Changzhi Dai,
Xiaoyao Liang, and Li Jiang.Improving neural network efficiency via post-training quantization
with adaptive floating-point.In *Proceedings of the IEEE/CVF International Conference on
Computer Vision*, pp. 5281–5290, 2021.
* Liu et al. (2022)Fangyu Liu, Yunlong Jiao, Jordan Massiah, Emine Yilmaz, and Serhii Havrylov.Trans-encoder: Unsupervised sentence-pair modelling through self-and
mutual-distillations.In *International Conference on Learning Representations,
ICLR*, 2022.
* Logeswaran et al. (2019)Lajanugen Logeswaran, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Jacob
Devlin, and Honglak Lee.Zero-shot entity linking by reading entity descriptions.In *Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics*, pp. 3449–3460. Association for
Computational Linguistics, 2019.
* Loshchilov \& Hutter (2019)Ilya Loshchilov and Frank Hutter.Decoupled weight decay regularization.In *International Conference on Learning Representations
(ICLR)*, 2019.URL [https://openreview.net/forum?id\=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7 "").
* Luo et al. (2014)Xin Luo, Mengchu Zhou, Yunni Xia, and Qingsheng Zhu.An efficient non-negative matrix-factorization-based approach to
collaborative filtering for recommender systems.*IEEE Transactions on Industrial Informatics*, 10(2):1273–1284, 2014.
* MacAvaney et al. (2022)Sean MacAvaney, Nicola Tonellotto, and Craig Macdonald.Adaptive re-ranking with a corpus graph.In *Proceedings of the 31st ACM International Conference on
Information \& Knowledge Management*, pp. 1491–1500, 2022.
* Mahoney \& Drineas (2009)Michael W Mahoney and Petros Drineas.Cur matrix decompositions for improved data analysis.*Proceedings of the National Academy of Sciences*, 106(3):697–702, 2009.
* Malkov \& Yashunin (2018)Yu A Malkov and Dmitry A Yashunin.Efficient and robust approximate nearest neighbor search using
hierarchical navigable small world graphs.*IEEE Transactions on Pattern Analysis and Machine
Intelligence*, 42(4):824–836, 2018.
* Mao et al. (2020)Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han,
and Weizhu Chen.Generation-augmented retrieval for open-domain question answering.*arXiv preprint arXiv:2009.08553*, 2020.
* Mao et al. (2021)Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han,
and Weizhu Chen.Generation-augmented retrieval for open-domain question answering.In *Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers)*, pp. 4089–4100.
Association for Computational Linguistics, 2021.doi: 10.18653/v1/2021.acl-long.316.URL [https://aclanthology.org/2021.acl-long.316](https://aclanthology.org/2021.acl-long.316 "").
* Mehta \& Rana (2017)Rachana Mehta and Keyur Rana.A review on matrix factorization techniques in recommender systems.In *2017 2nd International Conference on Communication Systems,
Computing and IT Applications (CSCITA)*, pp. 269–274. IEEE, 2017.
* Menon et al. (2022)Aditya Menon, Sadeep Jayasumana, Ankit Singh Rawat, Seungyeon Kim, Sashank
Reddi, and Sanjiv Kumar.In defense of dual-encoders for neural ranking.In *International Conference on Machine Learning*. PMLR, 2022.
* Musco \& Woodruff (2017)Cameron Musco and David P Woodruff.Sublinear time low-rank approximation of positive semidefinite
matrices.In *IEEE 58th Annual Symposium on Foundations of Computer
Science (FOCS)*, pp. 672–683. IEEE, 2017.
* Nayak et al. (2019)Prateeth Nayak, David Zhang, and Sek Chai.Bit efficient quantization for deep neural networks.In *2019 Fifth Workshop on Energy Efficient Machine Learning and
Cognitive Computing-NeurIPS Edition (EMC2-NIPS)*, pp. 52–56. IEEE, 2019.
* Nguyen et al. (2019)Luong Trung Nguyen, Junhan Kim, and Byonghyo Shim.Low-rank matrix completion: A contemporary survey.*IEEE Access*, 7:94215–94237, 2019.
* Pedregosa et al. (2011)F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.Scikit-learn: Machine learning in Python.*Journal of Machine Learning Research*, 12:2825–2830,
2011.
* Qu et al. (2021)Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang
Dong, Hua Wu, and Haifeng Wang.RocketQA: An optimized training approach to dense passage
retrieval for open-domain question answering.In *Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies*, pp. 5835–5847. Association for Computational Linguistics,
2021.
* Ray et al. (2022)Archan Ray, Nicholas Monath, Andrew McCallum, and Cameron Musco.Sublinear time approximation of text similarity matrices.*Proceedings of the AAAI Conference on Artificial Intelligence*,
36(7):8072–8080, 2022.
* Reddi et al. (2021)Sashank Reddi, Rama Kumar Pasumarthi, Aditya Menon, Ankit Singh Rawat, Felix
Yu, Seungyeon Kim, Andreas Veit, and Sanjiv Kumar.Rankdistil: Knowledge distillation for ranking.In *International Conference on Artificial Intelligence and
Statistics*, pp. 2368–2376. PMLR, 2021.
* Ren et al. (2021)Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, QiaoQiao She, Hua Wu, Haifeng
Wang, and Ji-Rong Wen.RocketQAv2: A joint training method for dense passage retrieval
and passage re-ranking.In *Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing*. Association for Computational Linguistics,
2021.URL [https://aclanthology.org/2021.emnlp-main.224](https://aclanthology.org/2021.emnlp-main.224 "").
* Rocchio Jr (1971)Joseph John Rocchio Jr.Relevance feedback in information retrieval.*The SMART retrieval system: experiments in automatic document
processing*, 1971.
* Rosa et al. (2022)Guilherme Rosa, Luiz Bonifacio, Vitor Jeronymo, Hugo Abonizio, Marzieh Fadaee,
Roberto Lotufo, and Rodrigo Nogueira.In defense of cross-encoders for zero-shot retrieval.*arXiv preprint arXiv:2212.06121*, 2022.
* Rücklé et al. (2021)Andreas Rücklé, Gregor Geigle, Max Glockner, Tilman Beck, Jonas
Pfeiffer, Nils Reimers, and Iryna Gurevych.AdapterDrop: On the efficiency of adapters in transformers.In *Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing*, pp. 7930–7946. Association for Computational
Linguistics, 2021.doi: 10.18653/v1/2021.emnlp-main.626.URL [https://aclanthology.org/2021.emnlp-main.626](https://aclanthology.org/2021.emnlp-main.626 "").
* Sanh et al. (2019)Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.Distilbert, a distilled version of bert: smaller, faster, cheaper and
lighter.*arXiv preprint arXiv:1910.01108*, 2019.
* Sung et al. (2023)Mujeen Sung, Jungsoo Park, Jaewoo Kang, Danqi Chen, and Jinhyuk Lee.Optimizing test-time query representations for dense retrieval.In *Findings of the Association for Computational Linguistics:
ACL 2023*, pp. 5731–5746. Association for Computational Linguistics, 2023.URL [https://aclanthology.org/2023.findings-acl.354](https://aclanthology.org/2023.findings-acl.354 "").
* Tan et al. (2020)Shulong Tan, Zhixin Zhou, Zhaozhuo Xu, and Ping Li.Fast item ranking under neural network based measures.In *International Conference on Web Search and Data Mining*,
pp. 591–599, 2020.
* Tan et al. (2021)Shulong Tan, Weijie Zhao, and Ping Li.Fast neural ranking on bipartite graph indices.*Proceedings of the VLDB Endowment*, 15(4):794–803, 2021.
* Thakur et al. (2021a)Nandan Thakur, Nils Reimers, Johannes Daxenberger, and Iryna Gurevych.Augmented SBERT: Data augmentation method for improving bi-encoders
for pairwise sentence scoring tasks.In *Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies*, pp. 296–310. Association for Computational Linguistics,
2021a.
* Thakur et al. (2021b)Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and
Iryna Gurevych.BEIR: A heterogeneous benchmark for zero-shot evaluation of
information retrieval models.In *Thirty-fifth Conference on Neural Information Processing
Systems Datasets and Benchmarks Track (Round 2)*, 2021b.URL [https://openreview.net/forum?id\=wCu6T5xFjeJ](https://openreview.net/forum?id=wCu6T5xFjeJ "").
* Vaswani et al. (2017)Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.Attention is all you need.*Advances in Neural Information Processing Systems*, 2017.
* Wang et al. (2021a)Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang.A comprehensive survey and experimental comparison of graph-based
approximate nearest neighbor search.*arXiv preprint arXiv:2101.12631*, 2021a.
* Wang et al. (2022)Shuohang Wang, Yichong Xu, Yuwei Fang, Yang Liu, Siqi Sun, Ruochen Xu,
Chenguang Zhu, and Michael Zeng.Training data is more valuable than you think: A simple and effective
method by retrieving from training data.In *Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, 2022.
* Wang et al. (2020)Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou.Minilm: Deep self-attention distillation for task-agnostic
compression of pre-trained transformers.*Advances in Neural Information Processing Systems*,
33:5776–5788, 2020.
* Wang et al. (2021b)Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis.Pseudo-relevance feedback for multiple representation dense
retrieval.In *Proceedings of the 2021 ACM SIGIR International Conference
on Theory of Information Retrieval*, pp. 297–306, 2021b.
* Wu et al. (2020)Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke
Zettlemoyer.Scalable zero-shot entity linking with dense entity retrieval.In *Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP)*, pp. 6397–6407. Association for
Computational Linguistics, 2020.
* Xin et al. (2020a)Ji Xin, Rodrigo Nogueira, Yaoliang Yu, and Jimmy Lin.Early exiting BERT for efficient document ranking.In *Proceedings of SustaiNLP: Workshop on Simple and Efficient
Natural Language Processing*, pp. 83–88. Association for Computational
Linguistics, 2020a.doi: 10.18653/v1/2020.sustainlp-1.11.URL [https://aclanthology.org/2020.sustainlp-1.11](https://aclanthology.org/2020.sustainlp-1.11 "").
* Xin et al. (2020b)Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin.DeeBERT: Dynamic early exiting for accelerating BERT inference.In *Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics*, pp. 2246–2251. Association for
Computational Linguistics, 2020b.doi: 10.18653/v1/2020.acl-main.204.URL [https://aclanthology.org/2020.acl-main.204](https://aclanthology.org/2020.acl-main.204 "").
* Xu et al. (2013)Miao Xu, Rong Jin, and Zhi-Hua Zhou.Speedup matrix completion with side information: Application to
multi-label learning.*Advances in neural information processing systems*, 26, 2013.
* Xu et al. (2015)Miao Xu, Rong Jin, and Zhi-Hua Zhou.Cur algorithm for partially observed matrices.In *International Conference on Machine Learning*, pp. 1412–1421. PMLR, 2015.
* Xue et al. (2017)Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen.Deep matrix factorization models for recommender systems.In *IJCAI*, volume 17, pp. 3203–3209. Melbourne, Australia,
2017.
* Yadav et al. (2022)Nishant Yadav, Nicholas Monath, Rico Angell, Manzil Zaheer, and Andrew
McCallum.Efficient Nearest Neighbor Search for Cross-encoder
Models using Matrix Factorization.In *Proceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing*, pp. 2171–2194, 2022.URL [https://aclanthology.org/2022.emnlp-main.140](https://aclanthology.org/2022.emnlp-main.140 "").
* Yadav et al. (2023)Nishant Yadav, Nicholas Monath, Manzil Zaheer, and Andrew McCallum.Efficient k-NN Search with Cross-encoders using Adaptive
Multi-Round CUR Decomposition.In *Findings of the Association for Computational Linguistics:
EMNLP*, 2023.
* Yu et al. (2021)HongChien Yu, Chenyan Xiong, and Jamie Callan.Improving query representations for dense retrieval with pseudo
relevance feedback.In *Proceedings of the 30th ACM International Conference on
Information \& Knowledge Management*, pp. 3592–3596, 2021.
* Yu et al. (2014)Hsiang-Fu Yu, Cho-Jui Hsieh, Si Si, and Inderjit S Dhillon.Parallel matrix factorization for recommender systems.*Knowledge and Information Systems*, 41:793–819,
2014.
* Yu et al. (2017)Hsiang-Fu Yu, Cho-Jui Hsieh, Qi Lei, and Inderjit S Dhillon.A greedy approach for budgeted maximum inner product search.*Advances in Neural Information Processing Systems*, 30, 2017.
* Zaheer et al. (2019)Manzil Zaheer, Guru Guruganesh, Golan Levin, and Alex Smola.Terrapattern: A nearest neighbor search service.*Pre-print*, 2019.
* Zeng et al. (2022)Hansi Zeng, Hamed Zamani, and Vishwa Vinay.Curriculum learning for dense retrieval distillation.In *Proceedings of the 45th International ACM SIGIR Conference
on Research and Development in Information Retrieval*, pp. 1979–1983,
2022.
* Zhang et al. (2014)Ting Zhang, Chao Du, and Jingdong Wang.Composite quantization for approximate nearest neighbor search.In *International Conference on Machine Learning*, pp. 838–846. PMLR, 2014.
* Zhong et al. (2019)Kai Zhong, Zhao Song, Prateek Jain, and Inderjit S Dhillon.Provable non-linear inductive matrix completion.*Advances in Neural Information Processing Systems*, 32, 2019.

Appendix A Training and Implementation Details
----------------------------------------------

| Dataset | Domain | $\lvert\mathcal{I}\rvert$ | $(\lvert\mathcal{Q}_{\text{train}}\rvert/\lvert\mathcal{Q}_{\text{test}}\rvert)$ Splits | Train Query ($\mathcal{Q}_{\text{train}}$) Type |
| --- | --- | --- | --- | --- |
| ZeShEL | YuGiOh | 10,031 | (100/3274), (500/2874), (2000/1374) | Real Queries |
| ZeShEL | Star Trek | 34,430 | (100/4127), (500/3727), (2000/2227) | Real Queries |
| BeIR | SciDocs | 25,657 | {1K, 10K, 50K}/1000 | Pseudo-Queries |
| BeIR | Hotpot-QA | 5,233,329 | {1K, 10K, 50K}/1000 | Pseudo-Queries |

*Table 1:  Statistics on number of items ($\mathcal{I}$), number of queries in train $(\mathcal{Q}_{\text{train}})$ and test $(\mathcal{Q}_{\text{test}})$ splits for each domain.
Following the precedent set by*Yadav et al. ([2022])*,
we create train/test split by
splitting the queries in each ZeShEL domain uniformly at random,
and experiment with three values of $\lvert\mathcal{Q}_{\text{train}}\rvert\in{100,500,2000}$.
For BeIR domains, we use pseudo-queries released as part of the benchmark as train queries ($\mathcal{Q}_{\text{train}}$) and run $k$-NN evaluation on test-queries from the official test-split (as per BeIR benchmark) of these domains.
For HotpotQA, we use the first 1K queries out of a total of 7K test queries and we use all 1K test queries for SciDocs.*

### A.1 Training Cross-Encoder Models

In our experiments, we use [emb]-CE, a cross-encoder model
variant proposed by*Yadav et al. ([2022])* that
jointly encodes a query-item pair and computes the final score
using dot-product of contextualized query and item embeddings
extracted after joint encoding.

##### ZeShEL Dataset

For ZeShEL, we use the cross-encoder model checkpoint666 <https://huggingface.co/nishantyadav/emb_crossenc_zeshel> released by*Yadav et al. ([2022])*.
We refer readers to *Yadav et al. ([2022])* for further details on parameterization and training of the cross-encoder.

##### BeIR Benchmark

For BeIR, we use the cross-encoder model checkpoint777 <https://huggingface.co/nishantyadav/emb_crossenc_msmarco_miniLM> trained on MS-MARCO dataset and released by*Yadav et al. ([2023])*.
The cross-encoder model is parameterized
using a 6-layer Mini-LM888[https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 "") model*(Wang et al., [2020])* and uses the dot-product based scoring mechanism
for cross-encoders proposed by*Yadav et al. ([2022])*.

### A.2 Training Dual-Encoder and Matrix Factorization Models

For BeIR datasets, we train matrix factorization models and $\textsc{DE}_{\textsc{dstl}}$ using sparse matrix $G$
containing number of train queries $\lvert\mathcal{Q}_{\text{train}}\rvert\in{\text{1K, 10K, 50K}}$ with
number of items per query $k_{d}\in{100,1000}$.
For ZeShEL datasets, we use $\lvert\mathcal{Q}_{\text{train}}\rvert\in{\text{100, 500, 2000}}$ with
the number of items per query $k_{d}\in{100,1000}$ for matrix factorization models
and $k_{d}\in{25,100}$ for training $\textsc{DE}_{\textsc{dstl}}$ model.
Table[1] shows train/test splits used for each domain.

#### A.2.1 Training Dual-Encoder Models

We train dual-encoder models on Nvidia RTX8000 GPUs with 48 GB GPU memory.

##### ZeShEL dataset

We report results for DE baselines as reported in*Yadav et al. ([2022])*.
The DE models were initialized using bert-base-uncased and
contain separate query and item encoders, thus resulting in a total of $2\times 110M$ parameters.
The DE models are trained using cross-entropy loss
to match the DE score distribution with the CE score distribution.
We refer readers to*Yadav et al. ([2022])* for details related to
training of DE models on ZeShEL dataset.

##### BeIR benchmark

For BeIR domains, we use a dual-encoder model checkpoint999[msmarco-distilroberta-base-v2: www.sbert.net/docs/pretrained-models/msmarco-v2.html](https://www.sbert.net/docs/pretrained-models/msmarco-v2.html "") released as part of sentence-transformer repository as $\textsc{DE}_{\textsc{src}}$, unless specified otherwise.
This DE model was initialized using distillbert-base *(Sanh et al., [2019])* model and trained on MS-MARCO dataset
which contains 40 million (query, positive document (item), negative document (item)) triplets
using triplet ranking loss.
This $\textsc{DE}_{\textsc{src}}$ is not trained on target domains SciDocs and Hotpot-QA used for running $k$-NN experiments in this paper.
We finetune $\textsc{DE}_{\textsc{src}}$ via distillation on the target domain to get the $\textsc{DE}_{\textsc{dstl}}$ model.
Given a set of training queries $\mathcal{Q}_{\text{train}}$ from the target domain,
we retrieve top-100 or top-1000 items for each query, score the items with the cross-encoder
model and train the dual-encoder by minimizing cross-entropy loss between
predicted query-item scores (using DE) and target query-item scores (obtained
using CE).
We train $\textsc{DE}_{\textsc{dstl}}$ using AdamW*(Loshchilov \& Hutter, [2019])* optimizer with learning rate 1e-5 and accumulating gradient over 4 steps. We trained for 10 epochs when using top-100 items per query and
for 4 epochs when using top-1000 items per query. We allocate a maximum time of 24 hours
for training.

#### A.2.2 Matrix-Factorization Models

We train both transductive (MFTrns) and inductive (MFInd)
matrix factorization models on Nvidia 2080ti GPUs with 12 GB GPU memory for all datasets
with the exception that we trained MFTrns for Hotpot-QA on Nvidia A100 GPUs with 80 GB GPU memory.
We use AdamW optimizer*(Loshchilov \& Hutter, [2019])* with learning rate and number of epochs as shown in Table[2].
Training MFTrns on Hotpot-QA required 80 GB GPU memory as
it involved training 768-dimensional embeddings for 5 million items which roughly translates
to around 4 billion trainable parameters, and we used AdamW optimizer with stores additional
memory for each trainable parameter.
For smaller datasets with the number of items of the order of 50K,
smaller GPUs with 12 GB memory sufficed.

For inductive matrix factorization (MFInd), we train a 2-layer MLP with skip-connection on top of query and item embeddings from $\textsc{DE}_{\textsc{src}}$.
For a given input embedding $x_{\text{in}}\in\mathbb{R}^{d}$, we compute the output embedding $x_{\text{out}}\in\mathbb{R}^{d}$ as

|  | $\displaystyle x^{\prime}_{\text{out}}$ | $\displaystyle\=b_{2}+W_{2}^{\intercal}\text{gelu}(b_{1}+W_{1}^{\intercal}x_{% \text{in}})$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle x_{\text{out}}$ | $\displaystyle\=\sigma(w_{\text{skip}})x^{\prime}_{\text{out}}+(1-\sigma(w_{% \text{skip}}))x$ |  |
| --- | --- | --- | --- |

where $W_{1}\in\mathbb{R}^{d\times 2d},b_{1}\in\mathbb{R}^{2d},W_{2}\in\mathbb{R}^{2d%
\times d},b_{2}\in\mathbb{R}^{d},w_{\text{skip}}\in\mathbb{R}$ are learnable parameters and $\sigma(.)$ is the sigmoid function.
We initialize $w_{\text{skip}}$ with -5 and use default PyTorch initialization
for other parameters.
We trained separate MLP models
for queries and items.
We would like to highlight that a simple 2-layer MLP *without* the skip connection i.e. using $x^{\prime}_{\text{out}}$ as the final output embedding
performed poorly in our experiments and it did not generalize well to unseen queries and items.

| Domain | MF Type | Learning Rate | Number of Epochs |
| --- | --- | --- | --- |
| SciDocs | MFTrns | 0.005 | 4 if $(\lvert\mathcal{Q}_{\text{train}}\rvert,k_{d})\in{\text{(10K,1K), (50K, 1K}}$ else 10 |
| SciDocs | MFInd | 0.005 | 10 if $(\lvert\mathcal{Q}_{\text{train}}\rvert,k_{d})\in{\text{(10K,1K), (50K, 1K}}$ else 20 |
| Hotpot-QA | MFTrns | 0.001 | 4 if $(\lvert\mathcal{Q}_{\text{train}}\rvert,k_{d})\in{\text{(10K,1K), (50K, 1K}}$ else 10 |
| Hotpot-QA | MFInd | 0.001 | 10 if $(\lvert\mathcal{Q}_{\text{train}}\rvert,k_{d})\in{\text{(10K,1K), (50K, 1K}}$ else 20 |
| YuGiOh | MFTrns | 0.001 | 20 |
| Star Trek | MFTrns | 0.001 | 20 |

*Table 2: Hyperparameters for transductive (MFTrns) and inductive (MFInd) matrix factorization models for different number of training queries ($\lvert\mathcal{Q}_{\text{train}}\rvert$) and number of items per train query ($k_{d}$) in sparse matrix $G$.*

### A.3 tf-idf

For BeIR datasets, we use bm25 with parameters as reported in*Thakur et al. ([2021b])* and for ZeShEL, we use tf-idf with default parameters from Scikit-learn*(Pedregosa et al., [2011])*, as reported in *Yadav et al. ([2022])*.

### A.4 Test-Time Inference with Axn, adaCUR , and RnR

For RnRX, we retrieve top-scoring items using dot-product of query and item embeddings
computed using baseline retrieval method $X$ and re-rank the retrieved items using the cross-encoder model.
For RnRMFTrns, we use dense query embedding from base dual-encoder
model $\textsc{DE}_{\textsc{src}}$ for test-queries $q_{\texttt{test}}\notin\mathcal{Q}_{\text{train}}$ along with item embeddings learnt using transductive matrix factorization to retrieve-and-rerank items for
the given test query.

For both adaCUR and Axn, we use $\mathscr{R}\=10$ for domains in BeIR and $\mathscr{R}\=5$ for domains
in ZeShEL unless stated otherwise.
For BeIR datasets, we tune Axn weight parameter $\lambda$ (in eq[7]) on the dev set. We refer interested
readers to §[B.2] for the effect of $\lambda$ on final performance.
For ZeShEL, we report results for $\lambda\=0$.
For Hotpot-QA, we restrict our $k$-NN search with AxnX,Y and $\textsc{adaCUR}_{Y}$ to top-10K items wrt method $Y$, $Y\in{\textsc{DE}_{\textsc{src}},\textsc{tf-idf}}$.
For other domains, we do not use any such heuristic and search over all items.

##### Cross-Encoder Score Normalization for Axn

Figure[4(a)] shows query-item score distribution for the cross-encoder model and $\textsc{DE}_{\textsc{src}}$ on SciDocs datasets from BeIR benchmark.
For cross-encoder models trained on BeIR dataset, we observe that the cross-encoder
and $\textsc{DE}_{\textsc{src}}$ model produce query-item scores in significantly different ranges.
Since $\textsc{DE}_{\textsc{src}}$ is used to initialize the embedding space for
matrix factorization approaches, this resulted in a mismatch in the range of the target score distribution from the cross-encoder in sparse matrix $G$ and the initial predicted score distribution from $\textsc{DE}_{\textsc{src}}$.
Consequently, using raw cross-encoder scores while training MF models and while computing test query embedding
by solving the linear regression problem in Eq[4] leads to a poor approximation of the cross-encoder.
To alleviate this issue, we normalize the cross-encoder scores to match the
score distribution from $\textsc{DE}_{\textsc{src}}$ model using two parameters $\alpha,\beta\in\mathbb{R}$.

|  | $s_{\text{final}}(q,i)\=\beta(s_{\text{init}}(q,i)-\alpha)$ |  |
| --- | --- | --- |

where $s_{\text{init}}(q,i)$ and $s_{\text{final}}(q,i)$ are initial and normalized cross-encoder scores, and $\alpha$ and $\beta$ are
estimated by re-normalizing cross-encoder distribution to match dual-encoder score distribution using 100 training queries.
Note that such score normalization does not affect the final ranking
of items.

We do *not* perform any such normalization for ZeShEL datasets the cross-encoder and
$\textsc{DE}_{\textsc{src}}$ model output scores in similar ranges as shown in Figure[4(b)].

![Refer to caption]()

*(a) SciDocs*

![Refer to caption]()

*(b) YuGiOh*

*Figure 4:  Score distribution for cross-encoder (CE) and dual-encoder (DE) models on SciDocs for BeIR and YuGiOh from ZeShEL. For each domain, we use cross-encoder and dual-encoder models trained on the corresponding task. See §[A.1] for details on cross-encoder training and §[A.2.1] for dual-encoder training.*

Appendix B Additional Results and Analysis
------------------------------------------

### B.1 Overhead of Adaptive Retrieval with Axn

![Refer to caption]()

*(a)  $\textsc{adaCUR}_{\textsc{DE}_{\textsc{src}}}$*

![Refer to caption]()

*(b) Axn$\textsc{DE}_{\textsc{src}}$,$\textsc{DE}_{\textsc{src}}$*

*Figure 5: Breakdown of inference latency for $\textsc{adaCUR}_{\textsc{DE}_{\textsc{src}}}$ and Axn$\textsc{DE}_{\textsc{src}}$,$\textsc{DE}_{\textsc{src}}$ under different test-time CE call budgets for domain\=Hotpot-QA. See §[B.1] for detailed discussion.*

![Refer to caption]()

*(a)  SciDocs*

![Refer to caption]()

*(b) Hotpot-QA*

*Figure 6: Top-$k$-Recall versus number of rounds for Axn$\textsc{DE}_{\textsc{src}}$,$\textsc{DE}_{\textsc{src}}$ under different test-time cross-encoder call budgets for domains Hotpot-QA and SciDocs. Number of rounds ($\mathscr{R}$) \= 1 corresponds to retrieve-and-rerank style inference with $\textsc{DE}_{\textsc{src}}$ i.e. RnR$\textsc{DE}_{\textsc{src}}$. Top-$k$-Recall generally improves with the number of rounds and saturates around 5 to 10 rounds.*

Figures[5(a)] and[5(b)] show total inference latency for adaCUR and Axn for varying number of rounds ($\mathscr{R}$) at different cross-encoder (CE) calls budgets.
The secondary y-axis in Figure[5] shows the breakdown of the
inference latency into three main steps in Algorithm[1] -
(a) CE Calls: computing CE scores for retrieved items (line[9]),
(b) solving linear regression problem to update test query embedding for Axn (line[10])
(c) Matrix Multiply: updating approximate scores for all items (line[7])
followed by retrieving items using approximate scores.
In case of adaCUR , computing query embedding in step (b)
involves computing the pseudo-inverse of a matrix instead of solving
a linear regression problem.

As shown in Figure[5], the overhead of adaptive retrieval is negligible
for $\mathscr{R}\=5$ to 10, and the overhead increases linearly with the number of rounds. Axn$\textsc{DE}_{\textsc{src}}$,$\textsc{DE}_{\textsc{src}}$ for $\mathscr{R}\=1$ corresponds to RnR$\textsc{DE}_{\textsc{src}}$, retrieve-and-rerank style inference using $\textsc{DE}_{\textsc{src}}$.
We observe that Axn incurs less overhead than adaCUR under the same test-time CE call budget.
Each CE call takes an amortized time of $\sim$2 ms101010On an Nvidia 2080ti GPU with 12 GB memory for a 6-layer Mini-LM*(Wang et al., [2020])* based model. when computing CE scores with a batch-size of up to 50 for domain\=Hotpot-QA.
While the time complexity of updating the approximate scores is linear in the number of items,
we observe that this step can be significantly sped up using GPUs/TPUs, and use of efficient
vector-based $k$-NN search methods.
In this work, to get an efficient implementation for large domains such as Hotpot-QA,
we first shortlist 10K items for the test query using
the baseline retrieval method (e.g. $\textsc{DE}_{\textsc{src}}$),
and only update the approximate scores for those 10K
during inference using brute-force computation of scores for all 10K items.
Further, note that the approximate scores are only used for retrieving
items (line[8] in Alg.[1]), and this operation
can also be implemented on CPUs using efficient vector-based $k$-NN search methods*(Malkov \& Yashunin, [2018]; Guo et al., [2020])* without the need for brute-force computation of approximate scores
for all items.

### B.2 Comparing different query embedding methods

![Refer to caption]()

![Refer to caption]()![Refer to caption]()

![Refer to caption]()

*(a) SciDocs*

![Refer to caption]()![Refer to caption]()

![Refer to caption]()

*(b) Hotpot-QA*

*Figure 7: Top-$k$-Recall versus inference cost for different test query embedding methods on domains SciDocs and Hotpot-QA. See §[B.2] for detailed discussion.*

![Refer to caption]()

*Figure 8: Top-$k$-Recall for Axn$\textsc{DE}_{\textsc{src}}$,$\textsc{DE}_{\textsc{src}}$ for different
values of $\lambda$ parameter in eq[7].
We use 200 queries from the validation set in Hotpot-QA and
the value in parentheses in the legend denotes average Top-1-Recall,
averaged over different test-time inference cost budgets.
For $k\=1$, using $\lambda\=0.8$ yields the best performance
and for $k\=100$, we use $\lambda\=0$ unless specified otherwise.*

Our proposed $k$-NN search method shares a similar motivation
to pseudo-relevance feedback (PRF) methods that aim to improve the
quality of retrieval by updating the initial query representation
using heuristic or model-based feedback on retrieved items.
We show results for TouR *(Sung et al., [2023])*, a recent PRF-based method that,
similar to our method, also optimizes the test query representations
using retrieval results while utilizing the CE call budget of $\mathcal{B}_{\textsc{ce}}$ CE calls over $\mathscr{R}$ rounds.
However, unlike Axn, TouR uses a single gradient-based update to query embedding to minimize
KL-Divergence (TouR-ce) or mean-squared error (TouR-mse) between approximate and exact scores for top-$\mathcal{B}_{\textsc{ce}}/\mathscr{R}$ items in each round.
In contrast, Axn computes the analytical solution to
the least-square problem in Eq.[4] in each round,
and optionally computes a weighted sum with the test query
embedding from a dense parametric model such as a dual-encoder using weight $\lambda\in[0,1]$ in Eq.[7].
For TouR-ce, we use learning rate \=0.1 (chosen from {0.1, 0.5, 1.0}) and for TouR-mse, we use learning rate \= 1e-3 (chosen from {1e-2, 1e-3, 1e-4}).

Figure[7] shows
Top-$k$-Recall and downstream task metrics versus test-time inference CE cost budget ($\mathcal{B}_{\textsc{ce}}$) for Axn$\textsc{DE}_{\textsc{src}}$,$\textsc{DE}_{\textsc{src}}$ under two settings of the weight parameter, $\lambda\=0$ and 0.8, and for $\textsc{DE}_{\textsc{src}}$ and TouR baselines.
For both SciDocs and Hotpot-QA, $\textsc{Axn}_{\lambda\=0.8}$ performs better than $\textsc{Axn}_{\lambda\=0}$ for $k$-NN search when $k\=1$ while $\lambda\=0$ works better for searching for $k$\=100 nearest neighbors. TouR and Axn achieve similar Top-1-Recall at smaller inference costs with Axn performing marginally better
than TouR at larger cost budgets. However, for $k\=100$,
$\textsc{Axn}_{\lambda\=0}$ achieves significantly better
recall than TouR.
We observe mixed trends for downstream task metrics. For instance,
$\textsc{Axn}_{\lambda\=0.8}$ and TouR baselines yield similar
performance for nDCG@10 on both SciDocs and Hotpot-QA and
for downstream task recall on Hotpot-QA while
$\textsc{Axn}_{\lambda\=0}$ performs better than all baselines
on downstream task recall for SciDocs.

### B.3 Transductive versus Inductive Matrix Factorization

![Refer to caption]()

![Refer to caption]()

*(a) SciDocs*

![Refer to caption]()

*(b) Hotpot-QA*

*Figure 9:  Top-$k$-Recall versus indexing time for transductive (MFTrns) and
inductive (MFInd) matrix factorization for SciDocs and Hotpot-QA.
We report Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 CE calls respectively. See §[B.3] for detailed discussion.*

Figure[9] shows
Top-$k$-Recall versus indexing time for $\textsc{DE}_{\textsc{src}}$, and transductive (MFTrns) and inductive (MFInd)
matrix factorization in combination with two test-time inference methods: proposed inference method (Axn) and retrieve-and-rerank style (RnR) inference.
We construct the sparse matrix $G$ by selecting
top-$k_{d}$ items for each train query using $\textsc{DE}_{\textsc{src}}$,
and report results for $\lvert\mathcal{Q}_{\text{train}}\rvert\in{1K,10K,50K}$ and
$k_{d}\in{100,1000}$. We use $\textsc{DE}_{\textsc{src}}$ to initialize
the query and item embeddings for MF methods.

Recall that MFTrns trains item embeddings as free-parameters,
and thus requires scoring an item against a small number of
train queries in order to update the item embedding.
For this reason, MFTrns performs marginally better
than or at par with MFInd on small-scale data SciDocs with 25K items,
as selecting even for $\lvert\mathcal{Q}_{\text{train}}\rvert\=1000,k_{d}\=100$, results
in each item being scored with four queries on average.
However, MFTrns performs poorly for large-scale
data Hotpot-QA (with 5 million items) due to the increased sparsity of matrix $G$, providing marginal to no improvement over $\textsc{DE}_{\textsc{src}}$.
In contrast, MFInd provides consistent improvement over
$\textsc{DE}_{\textsc{src}}$ on Hotpot-QA.

### B.4 Effect of Sparse Matrix Construction Strategy

|  |  | Hotpot-QA | | | SciDocs | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sparse Matrix | $\lvert\mathcal{Q}_{\text{train}}\rvert$, $k_{d}$ | Time to | Train-Time | | Time to | Train-Time | |
| Construction Strategy |  | compute $G$ | MFInd | MFTrns | compute $G$ | MFInd | MFTrns |
| $k_{d}$ items per query | 1K, 100 | 3 mins | 5 mins (20) | - | 10 mins | 5 mins (10) | 1.5 mins (10) |
| | 1K, 1000 | 31 mins | 20 mins (20) | - | 1.6 hrs | 20 mins (20) | 7 mins (10) |
| 10K, 100 | 30 mins | 20 mins (20) | 1.2 hrs (10) | 1.6 hrs | 20 mins (20) | 7.5 mins (10) |
| 10K, 1000 | 5.2 hrs | 3 hrs (20) | 3.2 hrs ( 4 ) | 16.7 hrs | 3.2 hrs (20) | 1.1 hrs ( 4) |
| 50K, 100 | 2.6 hrs | 1.2 hrs (20) | 4.1 hrs (10) | 8.3 hrs | 1.3 hrs (20) | 0.6 hrs (10) |
| 50K, 1000 | 26.3 hrs | 9 hrs (10) | 16 hrs ( 4) | 82 hrs | 14 hrs (10) | 3.7 hrs ( 4) |
| $k_{d}$ queries per item | 50K, 2 | 5.8 hrs | 3hrs (20) | 7.5 hrs (10) | 5 mins | 3 mins (20) | 6.5 mins (20) |
| | 50K, 5 | 12.7 hrs | 8hrs (20) | 8.5 hrs ( 4) | 14 mins | 5 mins (20) | 9 mins (20) |
| 50K, 10 | 23 hrs | 9hrs (10) | 16 hrs ( 4) | 26 mins | 6 mins (20) | 10 mins (20) |

*Table 3: Breakdown of indexing latency for transductiveMFTrns and inductiveMFInd matrix factorization methods on SciDocs and Hotpot-QA. For each setting, we show the number of epochs for training the model in parentheses.
Total indexing time also includes the time taken to compute initial query and item embeddings using $\textsc{DE}_{\textsc{src}}$. Computing item embeddings takes 90 seconds for SciDocs (with 25K items) and $\sim$2 hours for Hotpot-QA (with 5 million items) on an Nvidia 2080ti GPU with 12 GB GPU memory.*

Figure[10] shows
Top-$k$-Recall versus indexing time for
and MF with two different strategies to construct
sparse matrix $G$ and Table[3] shows the time taken
to construct the sparse matrix $G$ and the time taken to train the matrix factorization model.
$\mathcal{Q}-*$ indicates that $G$ is constructed by selecting a fixed number of $k_{d}$ items per *query* in $\mathcal{Q}_{\text{train}}$, and $\mathcal{I}-*$ indicates that $G$ is constructed by selecting fixed number of $k_{d}$ queries per *item* in $\mathcal{I}$.
When selecting a fixed number of items per query, we experiment with $\lvert\mathcal{Q}_{\text{train}}\rvert\in$ {1K, 10K, 50K } and $k_{d}\in{100,1000}$.
When selecting a fixed number of queries per item, we first create a pool of 50K queries and then select $k_{d}$ queries per item for $k_{d}\in{2,5,10}$.

Transductive Matrix Factorization For MFTrns, both $\mathcal{Q}-*$ and $\mathcal{I}-*$ strategies yield similar
Top-$k$-Recall at a given indexing cost on SciDocs as
both strategies result in each item being scored with
at least a few queries.
However, on Hotpot-QA, selecting a fixed number of items
per query may not result in each item being scored against
some queries, and thus $\mathcal{Q}-*$ variants
yield marginal (if any) improvement over $\textsc{DE}_{\textsc{src}}$.
$\mathcal{I}-*$ variants perform better than $\textsc{DE}_{\textsc{src}}$ and corresponding $\mathcal{Q}-*$ variants as each
item is scored against a fixed number of queries.
Note that this performance improvement comes
at the cost of an increase in time required to
compute sparse matrix $G$, as shown in Table[3].

Inductive Matrix Factorization For MFInd, we observe that $\mathcal{Q}-*$ variants
consistently provide better recall-vs-indexing time trade-offs
as compared to corresponding $\mathcal{I}-*$ variants on both SciDocs and Hotpot-QA.

![Refer to caption]()

![Refer to caption]()

*a-1 SciDocs*

![Refer to caption]()

*a-2 Hotpot-QA*

*(a) Transductive Matrix Factorization (MFTrns)*

![Refer to caption]()

*b-1 SciDocs*

![Refer to caption]()

*b-2 Hotpot-QA*

*(b) Inductive Matrix Factorization (MFInd)*

*Figure 10:  Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 cross-encoder calls respectively versus indexing time (in hours) for different strategies of constructing sparse matrix $G$. $\mathcal{Q}-*$ indicates that $G$ is constructed by selecting a fixed number of items per *query* in $\mathcal{Q}_{\text{train}}$, and $\mathcal{I}-*$ indicates that $G$ is constructed by selecting fixed number of queries per *item* in $\mathcal{I}$. For $\mathcal{Q}-*$ approaches, the text annotations indicate
($\lvert\mathcal{Q}_{\text{train}}\rvert,k_{d}$) pairs where $\lvert\mathcal{Q}_{\text{train}}\rvert$ is the number of anchor/train queries and $k_{d}$ is the number of items per query in the sparse matrix $G$. For $\mathcal{I}-*$ approaches, the text annotations indicate the number of queries per item in the sparse matrix $G$.
See §[B.4] for detailed discussion.*

![Refer to caption]()

![Refer to caption]()![Refer to caption]()

*(a) $\lvert\mathcal{Q}_{\text{train}}\rvert\=100,\lvert\mathcal{Q}_{\text{test}}%
\rvert\=3274$*

![Refer to caption]()![Refer to caption]()

*(b) $\lvert\mathcal{Q}_{\text{train}}\rvert\=500,\lvert\mathcal{Q}_{\text{test}}%
\rvert\=2874$*

![Refer to caption]()![Refer to caption]()

*(c) $\lvert\mathcal{Q}_{\text{train}}\rvert\=2000,\lvert\mathcal{Q}_{\text{test}}%
\rvert\=1374$*

*Figure 11:  Top-$k$-Recall and downstream task accuracy versus indexing time for various approaches on domain\=YuGiOh.
We report Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 CE calls respectively, and downstream task accuracy for fixed inference cost of 100 CE calls.
Each subfigure shows results for different train/test splits.*

![Refer to caption]()

![Refer to caption]()![Refer to caption]()

*(a) $\lvert\mathcal{Q}_{\text{train}}\rvert\=100,\lvert\mathcal{Q}_{\text{test}}%
\rvert\=4127$*

![Refer to caption]()![Refer to caption]()

*(b) $\lvert\mathcal{Q}_{\text{train}}\rvert\=500,\lvert\mathcal{Q}_{\text{test}}%
\rvert\=3727$*

![Refer to caption]()![Refer to caption]()

*(c) $\lvert\mathcal{Q}_{\text{train}}\rvert\=2000,\lvert\mathcal{Q}_{\text{test}}%
\rvert\=2227$*

*Figure 12:  Top-$k$-Recall and downstream task accuracy versus indexing time for various approaches on domain\=Star Trek.
We report Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 CE calls respectively, and downstream task accuracy for fixed inference cost of 100 CE calls.
Each subfigure shows results for different train/test splits.*

![Refer to caption]()

![Refer to caption]()![Refer to caption]()

![Refer to caption]()

*Figure 13:  Top-$k$-Recall and downstream task performance metrics versus indexing time for various approaches on domain\=SciDocs.
We report Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 cross-encoder (CE) calls respectively, and downstream task metrics for fixed inference cost of 100 cross-encoder calls. We report results for transductive matrix factorization (MFTrns) in these plots. The base dual-encoder ($\textsc{DE}_{\textsc{src}}$) in these plots is a 6-layer distilbert model finetuned on MS-MARCO dataset. The $\textsc{DE}_{\textsc{src}}$ model is available at [https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2 "").*

![Refer to caption]()

![Refer to caption]()![Refer to caption]()

![Refer to caption]()

*Figure 14:  Top-$k$-Recall and downstream task performance metrics versus indexing time for various approaches on domain\=Hotpot-QA.
We report Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 cross-encoder (CE) calls respectively, and downstream task metrics for fixed inference cost of 100 cross-encoder calls. We report results for inductive matrix factorization (MFInd) in these plots. The base dual-encoder ($\textsc{DE}_{\textsc{src}}$) in these plots is a 6-layer distilbert model finetuned on MS-MARCO dataset. The $\textsc{DE}_{\textsc{src}}$ model is available at [https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2 "").*

![Refer to caption]()

![Refer to caption]()![Refer to caption]()

![Refer to caption]()

*Figure 15:  Top-$k$-Recall and downstream task performance metrics versus indexing time for various approaches on domain\=SciDocs.
We report Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 cross-encoder (CE) calls respectively, and downstream task metrics for fixed inference cost of 100 cross-encoder calls. We report results for transductive matrix factorization (MFTrns) in these plots. The base dual-encoder ($\textsc{DE}_{\textsc{src}}$) in these plots is a 12-layer bert-base model finetuned on MS-MARCO dataset. The model is available at [https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5 "").*

![Refer to caption]()

![Refer to caption]()![Refer to caption]()

![Refer to caption]()

*Figure 16:  Top-$k$-Recall and downstream task performance metrics versus indexing time for various approaches on domain\=Hotpot-QA.
We report Top-1-Recall and Top-100-Recall at fixed inference cost budget ($m$) of 100 and 500 cross-encoder (CE) calls respectively, and downstream task metrics for fixed inference cost of 100 cross-encoder calls. We report results for inductive matrix factorization (MFInd) in these plots. The base dual-encoder ($\textsc{DE}_{\textsc{src}}$) in these plots is a 12-layer bert-base model finetuned on MS-MARCO dataset. This $\textsc{DE}_{\textsc{src}}$ model is available at [https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5 "").*
