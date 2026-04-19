# The Vendi Score: A Diversity Evaluation Metric for Machine Learning

Dan Friedman<sup>1</sup> and Adji Bousso Dieng<sup>1, 2,*</sup>

$^{1}$ Department of Computer Science, Princeton University  $^{2}$ Vertaix

*Published in Transactions on Machine Learning Research (07/2023), Reviewed on OpenReview

https://openreview.net/forum?id=g97OHbQyk1

July 4, 2023

# Abstract

Diversity is an important criterion for many areas of machine learning (ml), including generative modeling and dataset curation. However, existing metrics for measuring diversity are often domain-specific and limited in flexibility. In this paper, we address the diversity evaluation problem by proposing the Vendi Score, which connects and extends ideas from ecology and quantum statistical mechanics to ml. The Vendi Score is defined as the exponential of the Shannon entropy of the eigenvalues of a similarity matrix. This matrix is induced by a user-defined similarity function applied to the sample to be evaluated for diversity. In taking a similarity function as input, the Vendi Score enables its user to specify any desired form of diversity. Importantly, unlike many existing metrics in ml, the Vendi Score does not require a reference dataset or distribution over samples or labels, it is therefore general and applicable to any generative model, decoding algorithm, and dataset from any domain where similarity can be defined. We showcase the Vendi Score on molecular generative modeling where we found it addresses shortcomings of the current diversity metric of choice in that domain. We also applied the Vendi Score to generative models of images and decoding algorithms of text where we found it confirms known results about diversity in those domains. Furthermore, we used the Vendi Score to measure mode collapse, a known shortcoming of generative adversarial networks (gans). In particular, the Vendi Score revealed that even gans that capture all the modes of a labelled dataset can be less diverse than the original dataset. Finally, the interpretability of the Vendi Score allowed us to diagnose several benchmark ml datasets for diversity, opening the door for diversity-informed data augmentation<sup>1</sup>.

Keywords: diversity, evaluation, entropy, ecology, quantum statistical mechanics, machine learning

(a)



(b)



(c)  
Figure 1: (a) The Vendi Score, vs in the figure, can be interpreted as the effective number of unique elements in a sample. It increases linearly with the number of modes in the dataset. IntDiv, the expected dissimilarity, becomes less sensitive as the number of modes increases, converging to 1. (b) Combining distinct similarity functions can increase the Vendi Score, as should be expected of a diversity metric, while leaving IntDiv unchanged. (c) IntDiv does not take into account correlations between features, but the Vendi Score does. The Vendi Score is highest when the items in the sample differ in many attributes, and the attributes are not correlated with each other.



# 1 Introduction

Diversity is a criterion that is sought after in many areas of machine learning (ml), from dataset curation and generative modeling to reinforcement learning, active learning, and decoding algorithms. A lack of diversity in datasets and models can hinder the usefulness of ml in many critical applications, e.g. scientific discovery. It is therefore important to be able to measure diversity.

Many diversity metrics have been proposed in ML, but these metrics are often domain-specific and limited in flexibility. These include metrics that define diversity in terms of a reference dataset (Heusel et al., 2017; Sajjadi et al., 2018), a pre

trained classifier (Salimans et al., 2016; Srivastava et al., 2017), or discrete features, like n-grams (Li et al., 2016). In this paper, we propose a general, reference-free approach that defines diversity in terms of a user-specified similarity function.

Our approach is based on work in ecology, where biological diversity has been defined as the exponential of the entropy of the distribution of species within a population (Hill, 1973; Jost, 2006; Leinster, 2021). This value can be interpreted as the effective number of species in the population. To adapt this approach to ML, we define the diversity of a collection of elements  $x_{1}, \ldots, x_{n}$  as the exponential of the entropy of the eigenvalues of the  $n \times n$  similarity matrix  $K$ , whose entries are equal to the similarity scores between each pair of elements. This entropy can be seen as the von Neumann entropy associated with  $K$  (Bengtsson and Žyczkowski, 2017), so we call our metric the Vendi Score, for the von Neumann diversity.

Contributions. We summarize our contributions as follows:

- We extend ecological diversity to ML, and propose the Vendi Score, a metric for evaluating diversity in ML. We study the properties of the Vendi Score, which provides us with a more formal understanding of desiderata for diversity.  
- We showcase the flexibility and wide applicability of the Vendi Score, characteristics that stem from its sole reliance on the sample to be evaluated for diversity and a user-defined similarity function, and highlight the shortcomings of existing metrics used to measure diversity in different domains.

# 2 Are We Measuring Diversity Correctly in ML?

Several existing metrics for diversity rely on a reference distribution or dataset. These reference-based metrics define diversity in terms of coverage of the reference. They assume access to an embedding function—such as a pretrained Inception model (Szegedy et al., 2016)—that maps samples to real-valued vectors. One example of a reference-based metric is Fréchet Inception distance (fid) (Heusel et al., 2017), which measures the Wasserstein-2 distance between two Gaussian distributions, one Gaussian fit to the embeddings of the reference sample and another one fit to the embeddings of the sample to be evaluated for diversity. fid was originally proposed for evaluating image generative adversarial networks (gans) but has since been applied to text (Cifka et al., 2018) and molecules (Preuer et al., 2018) using domain-specific neural network encoders. Sajjadi et al. (2018) proposed a two-metric evaluation paradigm using precision and recall, with precision measuring quality and recall measuring diversity in terms of coverage of the reference distribution. Several other variations of precision and recall have been proposed (Kynkänniemi et al., 2019; Simon et al., 2019; Naeem et al., 2020). Compared to these approaches, the Vendi Score is a reference-free metric, measuring the intrinsic diversity of a set rather than the relationship to a reference distribution. This means that the Vendi Score should be used along side a quality metric, but can be applied in settings where there is no reference distribution.

Some other existing metrics evaluate diversity using a pre-trained classifier, therefore requiring labeled datasets. For example, the Inception score (is) (Salimans et al., 2016), which is mainly used to evaluate the perceptual quality of image generative

models, evaluates diversity using the entropy of the marginal distribution of class labels predicted by an ImageNet classifier. Another example is number of modes (nom) (Srivastava et al., 2017), a metric used to evaluate the diversity of gans. nom is calculated by using a classifier trained on a labeled dataset and then counting the number of unique labels predicted by the classifier when using samples from a gan as input. Both is and nom define diversity in terms of predefined labels, and therefore require knowledge of the ground truth labels and a separate classifier.

In some discrete domains, diversity is often evaluated in terms of the distribution of unique features. For example in natural language processing (nlp), a standard metric is n-gram diversity, which is defined as the number of distinct n-grams divided by the total number of n-grams (e.g. Li et al., 2016). These metrics require an explicit, discrete feature representation.

There are proposed metrics that use similarity scores to define diversity. The most widely used metric of this form is the average pairwise similarity score or the complement, the average dissimilarity. In text, variants of this metric include pairwise-bleu (Shen et al., 2019) and d-lex-sim (Fomicheva et al., 2020), in which the similarity function is an n-gram overlap metric such as bleu (Papineni et al., 2002). In biology, average dissimilarity is known as IntDiv (Benhenda, 2017), with similarity defined as the Jaccard (Tanimoto) similarity between molecular fingerprints. Average similarity has some shortcomings, which we highlight in 1. The figure shows the similarity matrices induced by a shape similarity function and/or a color similarity function. Each of the similarity functions is 1 when the index of the column and the index of the row have the same shape or color and 0 otherwise. As shown in 1, the average similarity–here measured by IntDiv–becomes less sensitive as diversity increases and does not account for correlations between features. This is not the case for the Vendi Score, which accounts for correlations between features and is able to capture the increased diversity resulting from composing distinct similarity functions. Related to the metric we propose here is a similarity-sensitive diversity metric proposed in ecology by Leinster and Cobbold (2012), and which was introduced in the context of ml by Posada et al. (2020). This metric is based on a notion of entropy defined in terms of a similarity profile, a vector whose entries are equal to the expected similarity scores of each element. Like IntDiv, it does not account for correlations between features.

Some other diversity metrics in the ml literature fall outside of these categories. The Birthday Paradox Test (Arora and Zhang, 2018) aims to estimate the size of the support of a generative model, but requires some manual inspection of samples. gilbo (Alemi and Fischer, 2018) is a reference-free metric but is only applicable to latent variable generative models. Kviman et al. (2022) measure the diversity of ensembles of variational approximations using the Jensen-Shannon Divergence (jsd); this metric is only applicable to sets of probability distributions. Mitchell et al. (2020) introduce metrics for diversity and inclusion, defining diversity in terms of the representation of socially relevant attributes like gender and race, and using the term heterogeneity to refer to variety in arbitrary attributes; in this paper, we use the term diversity to have the same sense as heterogeneity, meaning variety in arbitrary (user-specified) attributes. In the context of drug exploration, Xie et al. (2022) propose a metric based on the size of the largest subset of elements such that

the similarity between any pair of elements is below some threshold, but this metric requires setting a threshold. Similarly, in the field of evolutionary computation, quality diversity (qd) algorithms (Pugh et al., 2015), have assessed diversity by discretizing the feature space into grid of bins and counting the number of covered bins, but this approach requires picking a bin size.

As discussed above, several attempts have been made to measure diversity in ml. However, the proposed metrics can be limited in their applicability in that they require a reference dataset or predefined labels, or are domain-specific and applicable to one class of models. The existing metrics that do not have those applicability limitations have shortcomings when it comes to capturing diversity that we have illustrated in 1.

# 3 Measuring Diversity with the Vendi Score

We now define the Vendi Score, state its properties, and study its computational complexity. (We relegate all proofs of lemmas and theorems to the appendix.)

# 3.1 Defining the Vendi Score

To define a diversity metric in ml we look to ecology, the field that centers diversity in its work. In ecology, one main way diversity is defined is as the exponential of the entropy of the distribution of the species under study (Jost, 2006; Leinster, 2021). This is a reasonable index for diversity. Consider a population with a uniform distribution over  $n$  species, with entropy  $\log(n)$ . This population has maximal ecological diversity  $n$ , the same diversity as a population with  $n$  members, each belonging to a different species. The ecological diversity decreases as the distribution over the species becomes less uniform, and is minimized and equal to one when all members of the population belong to the same species. For a more extensive mathematical discussion of entropy and diversity in the context of biodiversity, we refer readers to Leinster (2021).

How can we extend this way of thinking about diversity to ml? One naive approach is to define diversity as the exponential of the Shannon entropy of the probability distribution defined by a machine learning model or dataset. However, this approach is limiting in that it requires a probability distribution for which entropy is tractable, which is not possible in many ml settings. We would like to define a diversity metric that only relies on the samples being evaluated for diversity. And we would like for such a metric to achieve its maximum value when all samples are dissimilar and its minimum value when all samples are the same. This implies the need to define a similarity function over the samples. Endowed with such a similarity function, we can define a form of entropy that only relies on the samples to be evaluated for diversity. This leads us to the Vendi Score:

Definition 3.1 (Vendi Score). Let  $x_{1},\ldots ,x_{n}\in \mathcal{X}$  denote a collection of samples, let  $k:\mathcal{X}\times \mathcal{X}\to \mathbb{R}$  be a positive semidefinite similarity function, with  $k(x,x) = 1$  for all  $x$ , and let  $K\in \mathbb{R}^{n\times n}$  denote the kernel matrix with entry  $K_{i,j} = k(x_i,x_j)$ . Denote by  $\lambda_1,\dots ,\lambda_n$  the eigenvalues of  $K / n$ . The Vendi Score (VS) is defined as the exponential

of the Shannon entropy of the eigenvalues of  $K / n$ :

$$
V S _ {k} \left(x _ {1}, \dots , x _ {n}\right) = \exp \left(- \sum_ {i = 1} ^ {n} \lambda_ {i} \log \lambda_ {i}\right), \tag {1}
$$

where we use the convention  $0 \log 0 = 0$ .

To understand the validity of the Vendi Score as a mathematical object, note that the eigenvalues of  $K / n$  are nonnegative (because  $k$  is positive semidefinite) and sum to one (because the diagonal entries of  $K / n$  are equal to  $1 / n$ ). The Shannon entropy is therefore well-defined and the Vendi Score is well-defined. In this form, the Vendi Score can also be seen as the effective rank of the kernel matrix  $K$ . Effective rank was introduced by Roy and Vetterli (2007) in the context of signal processing; the effective rank of a matrix is defined as the exponential of the entropy of the normalized singular values. Effective rank has also been used in machine learning, for example, to evaluate word embeddings (Torregrossa et al., 2020) and to study the implicit bias of gradient descent for low-rank solutions (Arora et al., 2019).

The Vendi Score can be expressed directly as a function of the kernel similarity matrix  $K$ :

Lemma 3.1. Consider the same setting as Definition 3.1. Then

$$
V S _ {k} \left(x _ {1}, \dots , x _ {n}\right) = \exp \left(- \operatorname {t r} \left(\frac {\boldsymbol {K}}{n} \log \frac {\boldsymbol {K}}{n}\right)\right). \tag {2}
$$

The lemma makes explicit the connection of the Vendi Score to quantum statistical mechanics: the Vendi Score is equal to the exponential of the von Neumann entropy associated with  $K / n$  (Bengtsson and Žyczkowski, 2017; Bach, 2022). In quantum statistical mechanics, the state of a quantum system is described by a density matrix, often denoted  $\rho$ . The von Neumann entropy of  $\rho$  quantifies the uncertainty in the state of the system (Wilde, 2013). The normalized similarity matrix  $K / n$  here plays the role of the density matrix.

Our formulation of the Vendi Score assumes that  $x_{1}, \ldots, x_{n}$  were sampled independently, and so  $p(x_{i}) \approx \frac{1}{n}$  for all  $i$ . This is the usual setting in ML and the setting we study in our experiments. However, we can generalize the Vendi Score to a setting in which we have an explicit probability distribution over the sample space  $\mathcal{X}$  (see Definition 7.1 in the appendix).

# 3.2 Understanding the Vendi Score

Figure 1 illustrates the behavior of the Vendi Score on simple toy datasets in which each element is defined by a shape and a color, and similarity is defined to be 1 if elements share both shape and color, 0.5 if they share either shape or color, and 0 otherwise.

First, Figure 1a illustrates that the Vendi Score is an effective number, and can be understood as the effective number of dissimilar elements in a sample. The value of measuring diversity with effective numbers has been argued in ecology (e.g. Hill, 1973; Patil and Taillie, 1982; Jost, 2006) and economics (Adelman, 1969). Effective

numbers provide a consistent basis for interpreting diversity scores, and make it possible to compare diversity scores using ratios and percentages. For example, in Figure 1a, when the number of modes doubles from two to four, the Vendi Score doubles as well. If we doubled the number of modes from four to eight, the Vendi Score would double once again.

Figures 1b and 1c illustrate another strength of the Vendi Score, which is that it accounts for correlations between features. Given distinct similarity functions  $k$  and  $k'$ , the Vendi Score calculated using the combined similarity function  $\frac{1}{2} k(x) + \frac{1}{2} k'(x)$  can be greater than the average of the individual Vendi Scores if the two similarity functions describe distinct dimensions of variation. Furthermore, the Vendi Score increases when the items in the sample differ in more attributes, and the attributes become less correlated with each other.

The Vendi Score has several desirable properties as a diversity metric. We summarize them in the following theorem.

Theorem 3.1 (Properties of the Vendi Score). Consider the same definitions in 3.1 and 7.1.

1. Effective number. If  $k(x_{i},x_{j}) = 0$  for all  $i\neq j$ , then  $VS_{k}(x_{1},\ldots ,x_{n})$  is maximized and equal to  $n$ . If  $k(x_{i},x_{j}) = 1$  for all  $i,j$ , then  $VS_{k}(x_{1},\ldots ,x_{n})$  is minimized and equal to 1.  
2. Identical elements. Suppose  $k(x_{i},x_{j}) = 1$  for some  $i\neq j$ . Let  $\pmb{p}^{\prime}$  denote the probability distribution created by combining  $i$  and  $j$ , i.e.  $p_i' = p_i + p_j$  and  $p_j' = 0$ . Then the Vendi Score is unchanged,

$$
V S _ {k} \left(x _ {1}, \dots , x _ {n}, \boldsymbol {p}\right) = V S _ {k} \left(x _ {1}, \dots , x _ {n}, \boldsymbol {p} ^ {\prime}\right).
$$

3. Partitioning. Suppose  $S_{1}, \ldots, S_{m}$  are collections of samples such that, for any  $i \neq j$ , for all  $x \in S_{i}, x' \in S_{j}$ ,  $k(x, x') = 0$ . Then the diversity of the combined samples depends only on the diversities of  $S_{1}, \ldots, S_{m}$  and their relative sizes. In particular, if  $p_{i} = |S_{i}| / \sum_{j} |S_{j}|$  is the relative size of  $S_{i}$  and  $H(p_{1}, \ldots, p_{m})$  denotes the Shannon entropy, then the Vendi Score is the geometric mean,

$$
V S _ {k} \left(S _ {1}, \dots , S _ {m}\right) = \exp \left(H \left(p _ {1}, \dots , p _ {m}\right)\right) \prod_ {i = 1} ^ {m} V S _ {k} \left(S _ {i}\right) ^ {p _ {i}}.
$$

4. Symmetry. If  $\pi_1, \ldots, \pi_n$  is a permutation of  $1, \ldots, n$ , then

$$
V S _ {k} (x _ {1}, \dots , x _ {n}) = V S _ {k} (x _ {\pi_ {1}}, \dots , x _ {\pi_ {n}}).
$$

The effective number property provides a consistent frame of reference for interpreting the Vendi Score: a sample with a Vendi Score of  $m$  can be understood to be as diverse as a sample consisting of  $m$  completely dissimilar elements. The identical elements property provides some justification for our use of a sampling approximation: for example, calculating the empirical Vendi Score of a sample of 90 blue diamonds and 10 yellow squares is equivalent to calculating the probability-weighted Vendi Score of a sample of one blue spade and one yellow square, with  $p = (0.9, 0.1)$ . The partitioning property is analogous to the partitioning property of the Shannon entropy and means that if two samples are completely dissimilar we can calculate the diversity of the union of the samples using only the diversity of each sample independently and their relative sizes. The symmetry property means that the Vendi

Score will be the same regardless of how we order the rows and columns in the similarity matrix.

# 3.3 Calculating the Vendi Score

Calculating the Vendi Score for a sample of  $n$  elements requires finding the eigenvalues of an  $n \times n$  matrix, which has a time complexity of  $O(n^3)$ . However, when embeddings of the observations (or feature vectors) are available, which is the case in many ml settings and in many of the applications we consider in this paper, one can use similarity functions defined as inner products between the embeddings  $\phi(x) \in \mathbb{R}^d$ , with  $d \ll n$ . That is, we can use the similarity matrix  $K = X^\top X$ , where  $X \in \mathbb{R}^{n \times d}$  is the embedding/feature matrix with row  $X_{i,:} = \phi(x_i)$ . The eigenvalues of  $K / n$  are the same as the eigenvalues of the covariance matrix  $XX^\top / n$ , therefore we can calculate the Vendi Score exactly in a time of  $O(d^2 n + d^3) = O(d^2 n)$ . This is the same complexity as existing metrics such as fid (Heusel et al., 2017), which require calculating the covariance matrix of Inception embeddings.

When embeddings aren't available, the Vendi Score can be approximated using column sampling methods (i.e. the Nyström method; Williams and Seeger, 2000).

Sample complexity. The Vendi Score is the exponential of the kernel entropy,  $H(K) = -\mathrm{tr}\left(\frac{K}{n}\log \frac{K}{n}\right)$ . Bach (2022) proves that empirical estimator of the kernel entropy has a convergence rate proportional to  $1 / \sqrt{n}$ , where  $n$  is the number of samples (Appendix 7.5).

# 3.4 Connections to Other Areas in ML

Here we remark on the connections between the Vendi Score and other commonly studied objects in ml that make use of the eigenvalues of a similarity matrix.

Determinantal Point Processes. The Vendi Score bears a relationship to Determinantal Point Processes (dpps), which have been used in machine learning for diverse subset selection (Kulesza et al., 2012). A dpp is a probability distribution over subsets of a ground set  $\mathcal{X}$  parameterized by a positive semidefinite kernel matrix  $K$ . The likelihood of drawing any subset  $X \subseteq \mathcal{X}$  is defined as proportional to  $|K_X|$ , the determinant of the similarity matrix restricted to elements in  $X$ :  $p(X) \propto |K_X| = \prod_i \lambda_i$ , where  $\lambda_i$  are the eigenvalues of  $K_X$ . The likelihood function has a geometric interpretation, as the square of the volume spanned by the elements of  $X$  in an implicit feature space. However, the dpp likelihood is not commonly used for evaluating diversity, and has some limitations. For example, it is always equal to 0 if the sample contains any duplicates, and the geometric meaning is arguably less straightforward to interpret than the Vendi Score, which can be understood in terms of the effective number of dissimilar elements.

Spectral Clustering. The eigenvalues of the similarity matrix are also related to spectral clustering algorithms (Von Luxburg, 2007), which use a matrix known as the graph Laplacian, defined  $L = D - K$ , where  $K$  is a symmetric, weighted adjacency matrix with non-negative entries, and  $D$  is a diagonal matrix with  $D_{i,i} = \sum_{j}K_{i,j}$ . The eigenvalues of  $L$  can be used to characterize different properties of the graph—for

example, the multiplicity of the eigenvalue 0 is equal to the number of connected components. As a metric for diversity, the Vendi Score is somewhat more general than the number of connected components: it provides a meaningful measure even for fully connected graphs, and captures within-component diversity.

# 4 Experiments

We illustrate the Vendi Score, which we now denote by vs for the rest of this section, on synthetic data to illustrate that it captures intuitive notions of diversity, and then apply it to a variety of setting in ml. We used vs to evaluate the diversity of generative models of molecules, an application where diversity plays an important role in enabling discovery. We compare vs to IntDiv, a function of the average similarity:

$$
\mathrm {I n t D i v} (x _ {1}, \ldots , x _ {n}) = 1 - \frac {1}{n ^ {2}} \sum_ {i, j} k (x _ {i}, x _ {j}).
$$

We found that vs identifies some model weaknesses that are not detected by IntDiv. We also applied vs to generative models of images, and decoding algorithms of text, where we found it confirms what we know about diversity in those applications. We also used vs to measure mode collapse in gans and datasets and show that it reveals finer-grained distinctions in diversity than current metrics for measuring mode collapse. Finally, we used vs to analyze the diversity of several image, text, and molecule datasets, gaining insights into the diversity profile of those datasets. (Implementation details are provided in Appendix 8.)

Figure 2: VS increases proportionally with diversity in three sets of synthetic datasets. In each row, we sample datasets from univariate mixture-of-normal distributions, varying either the number of components, the mixture proportions, or the per-component variance. The datasets are depicted in the left, as histograms, and the diversity scores are plotted on the right.

Figure 3: The kernel matrices for 250 molecules sampled from the hmm, aaa, and the original dataset, sorted lexicographically by smiles string representation. The samples have similar IntDiv scores, but the hmm samples score much lower on vs. The figure shows that the hmm generates a number of exact duplicates. vs is able to capture the hmm's lack of diversity while IntDiv cannot.




# 4.1 Synthetic experiments

To illustrate the behavior of the Vendi Score, we calculate the diversity of simple datasets drawn from a mixture of univariate normal distributions, varying either the number of components, the mixture proportions, or the per-component variance. We measure similarity using the RBF kernel:  $k(x,x^{\prime}) = \exp (\| x - x^{\prime}\|^{2} / 2\sigma^{2})$ . The results are illustrated in Figure 2. VS behaves consistently and intuitively in all three settings: in each case, VS can be interpreted as the effective number of modes, ranging between one and five in the first two rows and increasing from five to seven in the third row as we increase within-mode variance. On the other hand, the behavior of IntDiv is different in each settings: for example, IntDiv is relatively insensitive to within-mode variance, and additional modes bring diminishing returns.

In Appendix 9.1, we also validate that vs captures mode dropping in a simulated setting, using image and text classification datasets, where we have information about the ground truth class distribution. In both cases, vs has a stronger correlation with the true number of modes compared to IntDiv.

# 4.2 Evaluating molecular generative models for diversity

Next, we evaluate the diversity of samples from generative models of molecules. For generative models to be useful for the discovery of novel molecules, they ought to be diverse. The standard diversity metric in this setting is IntDiv. We evaluate samples from generative models provided in the moses benchmark (Polykovskiy et al., 2020), using the first 2500 valid molecules in each sample. Following prior work, our similarity function is the Morgan fingerprint similarity (radius 2), implemented in RDKit. In Figure 3, we highlight an instance where vs and IntDiv disagree: IntDiv ranks the hmm among the most diverse models, while vs ranks it as the least diverse (the complete results are in Appendix Table 4). The hmm has a high IntDiv score

because, on average, the hmm molecules have low pairwise similarity scores, but there are a number of clusters of identical or nearly identical molecules.

# 4.3 Assessing mode collapse in GANs

Mode collapse is a failure mode of gans that has received a lot of attention from the ml community (Metz et al., 2017; Dieng et al., 2019). The main metric for measuring mode collapse, called number of modes(nom), can only be used to assess mode collapse for gans trained on a labelled dataset. nom is computed by training a classifier on the labeled training data and counting the number of unique classes that are predicted by the trained classifier for the generated samples. In Table 1, we evaluate two models that were trained on the Stackedmnist dataset, a standard setting for evaluating mode collapse in gans. Stackedmnist is created by stacking three mnist images along the color channel, creating 1000 classes corresponding to 1000 number of modes.

<table><tr><td>Model</td><td>nom</td><td>Mode Div.</td><td>vs</td></tr><tr><td>Self-cond. gan</td><td>1000</td><td>921.0</td><td>746.7</td></tr><tr><td>Presgan</td><td>1000</td><td>948.7</td><td>866.6</td></tr><tr><td>Original</td><td>1000</td><td>950.8</td><td>943.7</td></tr></table>

Table 1: vs captures a more fine-grained notion of diversity than number of modes(nom). Although Presgan and Self-cond.gan both capture all the 1000 modes, vs reveals that Presgan is more diverse than Self-cond.gan and that they both are less diverse than the original dataset.

In prior work, mode collapse is evaluated by training an mnist classifier and counting the number of unique classes that are predicted for the generated samples. We adapt this approach and we calculate vs using the probability product kernel (Jebara et al., 2004):  $k(x,x^{\prime}) = \sum_{y}p(y\mid x)^{\frac{1}{2}}p(y\mid x^{\prime})^{\frac{1}{2}}$ , where the class likelihoods are given by the classifier. We compare Presgan (Dieng et al., 2019) and Self-conditioned gan (Liu et al., 2020), two gans that are known to capture all the modes. Table 1 shows that Presgan and Self-conditioned gan have the same diversity according to number of modes, they capture all 1000 modes. However, vs reveals a more fine-grained notion of diversity, indicating that Presgan is more diverse than Self-conditioned gan and that both are less diverse than the original dataset. One possibility is that vs is capturing imbalances in the mode distribution. To see whether this is the case, we also calculate what we call Mode Diversity, the exponential entropy of the predicted mode distribution:  $\exp H(\hat{p} (y))$ , where  $\hat{p} (y) = \frac{1}{n}\sum_{i = 1}^{n}p(y\mid x_i)$ . The generative models score lower on vs than Mode Diversity, indicating that low scores cannot be entirely attributed to imbalances in the mode distribution. Therefore vs captures more aspects of diversity, even when we are using the same representations as existing methods.

# 4.4 Evaluating image generative models for diversity

We now evaluate several recent models for unconditional image generation, comparing the diversity scores with standard evaluation metrics, is (Salimans et al., 2016),

<table><tr><td>Model</td><td>is↑</td><td>fid↓</td><td>Prec↑</td><td>Rec↑</td><td>vs↑</td><td>is↑</td><td>fid↓</td><td>Prec↑</td><td>Rec↑</td><td>vs↑</td></tr><tr><td colspan="6">cifar-10</td><td colspan="5">ImageNet 64×64</td></tr><tr><td>Original</td><td></td><td></td><td></td><td></td><td>19.50</td><td></td><td></td><td></td><td></td><td>43.93</td></tr><tr><td>vdvae</td><td>5.82</td><td>40.05</td><td>0.63</td><td>0.35</td><td>12.87</td><td>9.68</td><td>57.57</td><td>0.47</td><td>0.37</td><td>18.04</td></tr><tr><td>DenseFlow</td><td>6.01</td><td>34.54</td><td>0.62</td><td>0.38</td><td>13.55</td><td>5.62</td><td>102.90</td><td>0.36</td><td>0.17</td><td>12.71</td></tr><tr><td>iddpm</td><td>9.24</td><td>4.39</td><td>0.66</td><td>0.60</td><td>16.86</td><td>15.59</td><td>19.24</td><td>0.59</td><td>0.58</td><td>24.28</td></tr><tr><td colspan="6">LSUN Cat 256×256</td><td colspan="5">LSUN Bedroom 256×256</td></tr><tr><td>Original</td><td></td><td></td><td></td><td></td><td>15.12</td><td></td><td></td><td></td><td></td><td>8.99</td></tr><tr><td>Stylegan2</td><td>4.84</td><td>7.25</td><td>0.58</td><td>0.43</td><td>13.55</td><td>2.55</td><td>2.35</td><td>0.59</td><td>0.48</td><td>8.76</td></tr><tr><td>adm</td><td>5.19</td><td>5.57</td><td>0.63</td><td>0.52</td><td>13.09</td><td>2.38</td><td>1.90</td><td>0.66</td><td>0.51</td><td>7.97</td></tr><tr><td>rq-vt</td><td>5.76</td><td>10.69</td><td>0.53</td><td>0.48</td><td>14.91</td><td>2.56</td><td>3.16</td><td>0.60</td><td>0.50</td><td>8.48</td></tr></table>

Table 2: vs generally agrees with the existing metrics. On low-resolution datasets (top left and top right) the diffusion model performs better on all of the metrics. On the lsun datasets (bottom left and bottom right), the diffusion model gets the highest quality scores as measured by is, but scores lower on vs. No model matches the diversity score of the original dataset they were trained on.

fid (Heusel et al., 2017), Precision (Sajjadi et al., 2018), and Recall (Sajjadi et al., 2018). The models we evaluate represent popular classes of generative models, including a variational autoencoder (vdvae; Child, 2020), a flow model (Dense-Flow; Grecic et al., 2021), diffusion models (iddpm, Nichol and Dhariwal, 2021; adm Dhariwal and Nichol, 2021), gan-based models (Karras et al., 2019, 2020), and an auto-regressive model (rq-vt; Lee et al., 2022). The models are trained on CIFar-10 (Krizhevsky, 2009), ImageNet (Russakovsky et al., 2015), or two categories from the lsun dataset (Yu et al., 2015). We either select models that provide precomputed samples, or download publicly available model checkpoints and sample new images using the default hyperparameters. (More details are in Appendix 8.)

The standard metrics in this setting use a pre-trained Inception ImageNet classifier to map images to real vectors. Therefore, we calculate vs using the cosine similarity between Inception embeddings, using the same 2048-dimensional representations used for evaluating fid and Precision/Recall. As a result, the highest possible similarity score is 2048. The baseline metrics are reference-based, with the exception of is. fid and is capture diversity implicitly. Recall was introduced to capture diversity explicitly, with diversity defined as coverage of the reference distribution.

The results of this comparison are in Table 2. On the lower resolution datasets (top left and top right), vs generally agrees with the existing metrics. On those datasets the diffusion model performs better on all of the metrics. On the lsun datasets (bottom left and bottom right), the diffusion model gets the highest quality scores as measured by precision and recall, but scores lower on vs. In these cases, vs can be interpreted as complementing the existing metrics. For example, on lsun Cat, the ADM model achieves a precision score of 0.63 and recall of 0.52, implying that  $63\%$  of generated images look like reference images, and that the generated images cover  $52\%$  of the reference distribution; however, the low vs suggests that the remaining images have low internal diversity—for example, the model may generate many

near-duplicates. No model matches the diversity score of the original dataset they were trained on. In addition to comparing the diversity of the models, we can also compare the diversity scores between datasets: as a function of Inception similarity, the most diverse dataset is ImageNet  $64 \times 64$ , followed by CIFar-10, followed by lsun Cat, and then lsun Bedroom. Cat (all cats, but coming in different species), followed by lsun Bedrooms.

vs should be understood as the diversity with respect to a specific similarity function, in this case, the Inception ImageNet similarity. We illustrate this point in in the appendix (Figure 6) by comparing the top eigenvalues of the kernel matrices corresponding to the cosine similarity between Inception embeddings and pixel vectors. Inception similarity captures a form of semantic similarity, with components corresponding to particular cat breeds, while the pixel kernel provides a simple form of visual similarity, with components corresponding to broad differences in lightness, darkness, and color.

# 4.5 Evaluating decoding algorithms for text for diversity

<table><tr><td>Source</td><td>BLEU</td><td>N-gram diversity (↑)</td><td>D1K(↑)</td></tr><tr><td>Human</td><td></td><td>0.82</td><td>3.12</td></tr><tr><td>Beam Search</td><td>0.27</td><td>0.42</td><td>2.44</td></tr><tr><td>DBS γ = 0.2</td><td>0.25</td><td>0.49</td><td>2.49</td></tr><tr><td>DBS γ = 0.5</td><td>0.22</td><td>0.63</td><td>2.87</td></tr><tr><td>DBS γ = 0.5</td><td>0.21</td><td>0.68</td><td>2.95</td></tr></table>

Table 3: Quality and diversity scores for an image captioning model using different decoding algorithms. BS: Beam search. DBS: Diverse beam search (Vijayakumar et al., 2018), varying the diversity penalty  $\gamma$ . BLEU measures n-gram overlap with the human-written reference captions, a proxy for quality. VS is calculated using a BLEU score kernel. Using Diverse Beam Search with higher diversity penalties leads to higher diversity scores, according to both metrics, but a lower quality score. The underlying model is a GPT-2-based model trained on MS COCO.

We evaluate diversity on the ms coco image-captioning dataset (Lin et al., 2014), following prior work on diverse text generation (Vijayakumar et al., 2018). In this setting, the subjects of evaluation are diverse decoding algorithms rather than parametric models. Given a fixed conditional model of text  $p(x \mid c)$ , where  $c$  is some conditioning context, the aim is to identify a "Diverse N-Bet List", a list of sentences that have high likelihood but are mutually distinct. The baseline metric we compare to is n-gram diversity (Li et al., 2016), which is the proportion of unique n-grams divided by the total number of n-grams. We define similarity using the n-gram overlap kernel: for a given  $n$ , the n-gram kernel  $k_{n}$  is the cosine similarity between bag-of-n-gram feature vectors. We use the average of  $k_{1}, \ldots, k_{4}$ . This ensures that vs and n-gram diversity are calculated using the same feature representation. Each image in the validation split has five captions written by different human annotators, and we compare these with captions generated by a publicly available captioning

Figure 4: The categories in CIFar-100 with the lowest and highest vs, defining similarity as the cosine similarity between either Inception embeddings or pixel vectors. We show 100 examples from each category, in decreasing order of average similarity, with the image at the top left having the highest average similarity scores according to the corresponding kernel.

model trained on this dataset  $^{3}$ . For each image, we generate five captions using either beam search or diverse beam search (dbs) (Vijayakumar et al., 2018). dbs takes a parameter,  $\gamma$ , called the diversity penalty, and we vary this between 0.2, 0.6, and 0.8.

Table 3 shows that all diversity metrics increase as expected, ranking beam search the lowest, the human captions the highest, and dbs in between, increasing with the diversity penalty. The human diversity score of 4.88 can be interpreted as meaning that, on average, all five human-written captions are almost completely dissimilar from each other, while beam search effectively returns only three distinct responses for every five that it generates.

# 4.6 Diagnosing datasets for diversity

In Figure 4, we calculate vs for samples from different categories in cifar-100, using the cosine similarity between either Inception embeddings or pixel vectors. The pixel diversity is highest for categories like "aquarium fish", which vary in color, brightness, and orientation, and lowest for categories like "cockroach" in which images have similar regions of high pixel intensity (like white backgrounds). The Inception diversity is less straightforward to interpret, but might correspond to some form of semantic diversity—for example, the Inception diversity might be lower for classes like "castle," that correspond to distinct ImageNet categories, and higher for categories like "clock" and "keyboard" that are more difficult to classify.

In Appendix 9.5, we show additional examples from text, molecules, and other image datasets.

# 5 Limitations

Here, we discuss several things to consider when interpreting vs scores. First, vs is a reference-free metric, meaning that it measures the internal diversity of a set and not how it relates to a reference distribution. While this makes vs useful in settings where there is no reference distribution, it also means that it is possible to get a high diversity score by, for example, sampling random noise. This is also true of other reference-free metrics, like IntDiv and n-gram diversity. Therefore, vs should be used alongside a quality metric. Second, like other similarity-based metrics, vs is dependent on the choice of similarity function. If the similarity function is too sensitive, all sets will appear very diverse, while if it is not sensitive enough, all sets will have low diversity. Additionally, the wrong choice of similarity function can introduce biases that lead to skewed diversity scores. Therefore, care should be taken when choosing a similarity function to ensure that it is appropriate for the specific application. Finally, the computational cost of calculating vs can be high when the similarity function is not associated with low-dimensional embeddings.

# 6 Discussion

We introduced the Vendi Score, a metric for evaluating diversity in ml. The Vendi Score is defined as a function of the pairwise similarity scores between elements of a sample and can be interpreted as the effective number of unique elements in the sample. The Vendi Score is interpretable, general, and applicable to any domain where similarity can be defined. It is unsupervised, in that it does not require labels or a reference probability distribution or dataset. Importantly, the Vendi Score allows its user to specify the form of diversity they want to measure via the similarity function. We showed the Vendi Score can be computed efficiently exactly and showcased its usefulness in several ml applications, different datasets, and different domains. In future work, we will leverage the Vendi Score to improve data augmentation, an important ml approach in settings with limited data.

# Acknowledgements

Adji Bousso Dieng is supported by the National Science Foundation, Office of Advanced Cyberinfrastructure (OAC): #2118201. We thank Sadhika Malladi for pointing us to the effective rank. Adji Bousso Dieng would like to dedicate this paper to her PhD advisors, David Blei and John Paisley.

# References

Adelman, M. A. (1969). Comment on the "H" concentration measure as a number-equivalent. The Review of economics and statistics, pages 99-101.  
Alemi, A. A. and Fischer, I. (2018). GILBO: one metric to measure them all. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 7037-7046.  
Arora, S., Cohen, N., Hu, W., and Luo, Y. (2019). Implicit regularization in deep matrix factorization. In Advances in Neural Information Processing Systems.  
Arora, S. and Zhang, Y. (2018). Do GANs actually learn the distribution? some theory and empirics. In International Conference on Learning Representations.  
Bach, F. (2022). Information theory with kernel methods. arXiv preprint arXiv:2202.08545.  
Bengtsson, I. and Žyczkowski, K. (2017). Geometry of quantum states: an introduction to quantum entanglement. Cambridge university press.  
Benhenda, M. (2017). ChemGAN challenge for drug discovery: can AI reproduce natural chemical diversity? arXiv preprint arXiv:1708.08227.  
Bird, S. (2006). Nltk: The natural language toolkit. In Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions, pages 69-72.  
Child, R. (2020). Very deep VAEs generalize autoregressive models and can outperform them on images. arXiv preprint arXiv:2011.10650.  
Cífka, O., Severyn, A., Alfonseca, E., and Filippova, K. (2018). Eval all, trust a few, do wrong to none: Comparing sentence generation models. arXiv preprint arXiv:1804.07972.  
Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.  
Dhariwal, P. and Nichol, A. (2021). Diffusion models beat GANs on image synthesis. Advances in Neural Information Processing Systems, 34:8780-8794.  
Dieng, A. B., Ruiz, F. J., Blei, D. M., and Titsias, M. K. (2019). Prescribed generative adversarial networks. arXiv preprint arXiv:1910.04302.  
Fomicheva, M., Sun, S., Yankovskaya, L., Blain, F., Guzmán, F., Fishel, M., Aletras, N., Chaudhary, V., and Specia, L. (2020). Unsupervised quality estimation for neural machine translation. Transactions of the Association for Computational Linguistics, 8:539-555.  
Gao, T., Yao, X., and Chen, D. (2021). SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894-6910.  
Grcic, M., Grubišić, I., and Šegvic, S. (2021). Densely connected normalizing flows. Advances in Neural Information Processing Systems, 34:23968-23982.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. (2017). Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.  
Hill, M. O. (1973). Diversity and Evenness: A Unifying Notation and Its Consequences. Ecology, 54(2):427-432.  
Jebara, T., Kondor, R., and Howard, A. (2004). Probability product kernels. The Journal of Machine Learning Research, 5:819-844.  
Jost, L. (2006). Entropy and Diversity. Oikos, 113(2):363-375.  
Karras, T., Laine, S., and Aila, T. (2019). A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401-4410.  
Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., and Aila, T. (2020). Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110-8119.  
Keung, P., Lu, Y., Szarvas, G., and Smith, N. A. (2020). The multilingual Amazon reviews corpus. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing.  
Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report.  
Kulesza, A., Taskar, B., et al. (2012). Determinantal point processes for machine learning. Foundations and Trends® in Machine Learning, 5(2-3):123-286.  
Kviman, O., Melin, H., Koptagel, H., Elvira, V., and Lagergren, J. (2022). Multiple importance sampling elbo and deep ensembles of variational approximations. In International Conference on Artificial Intelligence and Statistics, pages 10687-10702. PMLR.  
Kynkänniemi, T., Karras, T., Laine, S., Lehtinen, J., and Aila, T. (2019). Improved precision and recall metric for assessing generative models. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pages 3927-3936.  
Lee, D., Kim, C., Kim, S., Cho, M., and Han, W.-S. (2022). Autoregressive Image Generation using Residual Quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523-11532.  
Leinster, T. (2021). Entropy and Diversity: The Axiomatic Approach. Cambridge University Press.  
Leinster, T. and Cobbold, C. A. (2012). Measuring Diversity: The Importance of Species Similarity. Ecology, 93(3):477-489.  
Li, J., Galley, M., Brockett, C., Gao, J., and Dolan, W. B. (2016). A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 110-119.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In European conference on computer vision, pages 740-755. Springer.  
Liu, S., Wang, T., Bau, D., Zhu, J.-Y., and Torralba, A. (2020). Diverse image generation via self-conditioned gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).  
Liu, Z., Luo, P., Wang, X., and Tang, X. (2015). Deep learning face attributes in the wild. In International Conference on Computer Vision.  
Metz, L., Poole, B., Pfau, D., and Sohl-Dickstein, J. (2017). Unrolled generative adversarial networks. In International Conference on Learning Representations.  
Mitchell, M., Baker, D., Moorosi, N., Denton, E., Hutchinson, B., Hanna, A., Gebru, T., and Morgenstern, J. (2020). Diversity and inclusion metrics in subset selection. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, pages 117-123.  
Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., and Yoo, J. (2020). Reliable fidelity and diversity metrics for generative models. In International Conference on Machine Learning, pages 7176-7185. PMLR.  
Nichol, A. Q. and Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162-8171. PMLR.  
Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311-318.  
Parmar, G., Zhang, R., and Zhu, J.-Y. (2022). On aliased resizing and surprising subtleties in gan evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11410-11420.  
Patil, G. and Taillie, C. (1982). Diversity as a Concept and Its Measurement. Journal of the American Statistical Association, 77(379):548-561.  
Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin, A., Johansson, S., Chen, H., Nikolenko, S., Aspuru-Guzik, A., and Zhavoronkov, A. (2020). Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models. Frontiers in Pharmacology.  
Posada, J. G., Vani, A., Schwarzer, M., and Lacoste-Julien, S. (2020). Gait: A geometric approach to information theory. In International Conference on Artificial Intelligence and Statistics, pages 2601-2611. PMLR.  
Preuer, K., Renz, P., Unterthiner, T., Hochreiter, S., and Klambauer, G. (2018). Fréchet ChemNet distance: a metric for generative models for molecules in drug discovery. Journal of chemical information and modeling, 58(9):1736-1741.  
Pugh, J. K., Soros, L. B., Szerlip, P. A., and Stanley, K. O. (2015). Confronting the challenge of quality diversity. In Proceedings of the 2015 Annual Conference on Genetic and Evolutionary Computation, pages 967-974.

Radford, A., Metz, L., and Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. In arXiv:1511.06434.  
Roy, O. and Vetterli, M. (2007). The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pages 606-610. IEEE.  
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. (2015). ImageNet large scale visual recognition challenge. International journal of computer vision, 115(3):211-252.  
Sajjadi, M. S., Bachem, O., Lucic, M., Bousquet, O., and Gelly, S. (2018). Assessing generative models via precision and recall. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 5234-5243.  
Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X., and Chen, X. (2016). Improved Techniques for Training GANs. In Lee, D. D., Sugiyama, M., Luxburg, U. V., Guyon, I., and Garnett, R., editors, Advances in Neural Information Processing Systems 29, pages 2234-2242. Curran Associates, Inc.  
Sanchez-Lengeling, B., Wei, J. N., Lee, B. K., Gerkin, R. C., Aspuru-Guzik, A., and Wiltschko, A. B. (2019). Machine learning for scent: Learning generalizable perceptual representations of small molecules. arXiv preprint arXiv:1910.10685.  
Shen, T., Ott, M., Auli, M., and Ranzato, M. (2019). Mixture models for diverse machine translation: Tricks of the trade. In International conference on machine learning, pages 5719-5728. PMLR.  
Simon, L., Webster, R., and Rabin, J. (2019). Revisiting precision and recall definition for generative model evaluation. In International Conference on Machine Learning (ICML).  
Song, J., Meng, C., and Ermon, S. (2021). Denoising diffusion implicit models. In International Conference on Learning Representations.  
Srivastava, A., Valkov, L., Russell, C., Gutmann, M. U., and Sutton, C. (2017). VEEGAN: reducing mode collapse in GANs using implicit variational learning. In Advances in Neural Information Processing Systems.  
Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818-2826.  
Torregrossa, F., Claveau, V., Kooli, N., Gravier, G., and Allesiardo, R. (2020). On the correlation of word embedding evaluation metrics. In Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020), pages 4789-4797.  
Vijayakumar, A., Cogswell, M., Selvaraju, R., Sun, Q., Lee, S., Crandall, D., and Batra, D. (2018). Diverse beam search for improved description of complex scenes. In Proceedings of the AAAI Conference on Artificial Intelligence.  
Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17(4):395-416.  
Wilde, M. M. (2013). Quantum information theory. Cambridge University Press.

Williams, A., Nangia, N., and Bowman, S. (2018). A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112-1122.  
Williams, C. and Seeger, M. (2000). Using the nystrom method to speed up kernel machines. Advances in Neural Information Processing Systems, 13.  
Wolf, T., Debut, L., Sanh, V., Chaumont, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. (2019). Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771.  
Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. In arXiv:1708.07747.  
Xie, Y., Xu, Z., Ma, J., and Mei, Q. (2022). How much of the chemical space has been explored? selecting the right exploration measure for drug discovery. In ICML 2022 2nd AI for Science Workshop.  
Yu, F., Seff, A., Zhang, Y., Song, S., Funkhouser, T., and Xiao, J. (2015). LSUN: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365.

# 7 Proofs

# 7.1 Probability-weighted Vendi Score

Definition 7.1 (Probability-Weighted Vendi Score). Let  $\pmb{p} \in \Delta_n$  denote a probability distribution on a discrete space  $\mathcal{X} = \{x_1, \ldots, x_n\}$ , where  $\Delta_n$  denotes the  $(n - 1)$ -dimensional simplex, let  $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$  be a positive semidefinite similarity function, with  $k(x, x) = 1$  for all  $x$ , and let  $K \in \mathbb{R}^{n \times n}$  denote the kernel matrix with  $K_{i,j} = k(x_i, x_j)$ . Let  $\tilde{K}_p = \mathrm{diag}(\sqrt{p}) K \mathrm{diag}(\sqrt{p})$  denote the probability-weighted kernel matrix. Let  $\lambda_1, \dots, \lambda_n$  denote the eigenvalues of  $\tilde{K}_p$ . The Vendi Score (VS) is defined as the exponential of the Shannon entropy of the eigenvalues of  $\tilde{K}_p$ :

$$
V S _ {k} (x _ {1}, \ldots , x _ {n}, \pmb {p}) = \exp \left(- \sum_ {i = 1} ^ {S} \lambda_ {i} \log \lambda_ {i}\right). \qquad \qquad (3)
$$

When all elements in the sample are completely dissimilar, the probability-weighted Vendi Score defined in 7.1 reduces to the exponential of the Shannon entropy of the weighting distribution:

Lemma 7.1. Let  $\pmb{p} \in \Delta_{n}$  be a probability distribution over  $x_{1},\ldots ,x_{n}$  and suppose  $k(x_{i},x_{j}) = 0$  for all  $i \neq j$ . Then  $VS_{k}(x_{1},\dots,x_{n},\pmb {p}) = \exp H(\pmb {p})$ , the exponential of the Shannon entropy of  $\pmb{p}$ .

# 7.2 Proof of 3.1

Lemma. Consider the same setting as Definition 3.1. Then

$$
\mathrm {V S} _ {k} (x _ {1}, \ldots , x _ {n}) = \exp \left(- \operatorname {t r} \left(\frac {K}{n} \log \frac {K}{n}\right)\right). \tag {4}
$$

Proof. For any square matrix  $X \in \mathbb{R}^{n \times n}$ , if  $X$  has an eigendecomposition  $X = U\Lambda U^{-1}$ , then  $\log X = U(\log \Lambda)U^{-1}$ , where  $\log \Lambda = \mathrm{diag}(\log \lambda_1, \dots, \log \lambda_n)$  is a diagonal matrix whose diagonal entries are the logarithms of the eigenvalues of  $X$ . Also,  $\operatorname{tr}(X) = \operatorname{tr}\left(U\Lambda U^{-1}\right) = \operatorname{tr}(\Lambda)$ , because the trace is similarity-invariant.  $K / n$  is diagonalizable because it is positive semidefinite, so let  $K / n = U\Lambda U^{-1}$  denote the eigendecomposition. Then

$$
\begin{array}{l} \operatorname {t r} \left(\boldsymbol {K} / n \log \boldsymbol {K} / n\right) = \operatorname {t r} \left(\boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {- 1} \log \left(\boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {- 1}\right)\right) \\ = \operatorname {t r} \left(U \Lambda U ^ {- 1} U (\log \Lambda) U ^ {- 1}\right) \\ = \operatorname {t r} (\Lambda \log \Lambda) \\ = \sum_ {i = 1} ^ {n} \lambda_ {i} \log \lambda_ {i}. \\ \end{array}
$$

Therefore

$$
\mathrm {V S} _ {k} (x _ {1}, \ldots , x _ {n}) = \exp \left(- \sum_ {i = 1} ^ {n} \lambda_ {i} \log \lambda_ {i}\right) = \exp \left(- \operatorname {t r} \left(\frac {\boldsymbol {K}}{n} \log \frac {\boldsymbol {K}}{n}\right)\right).
$$


# 7.3 Proof of 7.1

Lemma. Let  $\pmb{p} \in \Delta_{n}$  be a probability distribution over  $x_{1},\ldots ,x_{n}$  and suppose  $k(x_{i},x_{j}) = 0$  for all  $i \neq j$ . Then  $\mathrm{VS}_k(x_1,\dots,x_n,\pmb {p}) = \exp H(\pmb {p})$ , the exponential of the Shannon entropy of  $\pmb{p}$ .

Proof. If all elements in  $\pmb{p}$  are completely dissimilar, then  $\tilde{\kappa}_p$  is a diagonal matrix, and the eigenvalues  $\lambda_1, \ldots, \lambda_S$  are the diagonal entries, which are the entries of  $\pmb{p}$ . So the von Neumann entropy of  $\tilde{\kappa}_p$  is identical to the Shannon entropy of  $\pmb{p}$ , and the exponential is the Vendi Score.

# 7.4 Proof of Theorem 3.1

Proof. (a) Effective number: If  $\pmb{p}$  is the uniform distribution over  $N$  completely dissimilar elements, then  $\tilde{\pmb{K}}_p$  is a diagonal matrix with each diagonal entry equal to  $1 / N$ . The eigenvalues of a diagonal matrix are the diagonal entries, so  $\mathrm{VS}_K(\pmb{p}) = \exp H(1 / N, \dots, 1 / N) = \exp \log N = N$ . On the other hand, if all elements are completely similar to each other, then  $\tilde{\pmb{K}}_p$  has rank one and so the Vendi Score is equal to one.

(b) Identical elements: The eigenvalues of  $\tilde{K}_p$  are the same as the eigenvalues of the covariance matrix of the corresponding feature space:

$$
\tilde {\Sigma} _ {p} = \sum_ {i = 1} ^ {N} p (x _ {i}) \phi (x _ {i}) \phi (x _ {i}) ^ {\top}.
$$

Suppose elements  $i$  and  $j$  are identical, and let  $\pmb{p}^{\prime}$  denote the probability distribution created by combining  $i$  and  $j$ , i.e.  $p_i' = p_i + p_j$  and  $p_j' = 0$ . Clearly,  $\tilde{\Sigma}_p = \tilde{\Sigma}_{p'}$ , and so  $\mathrm{VS}_k(x_1,\ldots,x_n,p) = \mathrm{VS}_k(x_1,\ldots,x_n,p')$ .

(c) Partitioning: Suppose  $N$  samples are partitioned into  $M$  groups  $\mathcal{S}_1, \ldots, \mathcal{S}_M$  such that, for any  $i \neq j$ , for all  $x \in S_i, x' \in S_j$ ,  $k(x, x') = 0$ . Let  $p_i = |S_i| / \sum_j |S_j|$  denote the relative size of group  $i$ , and let  $\pmb{K}$  denote kernel matrix of  $\cup_{i} S_{i}$ , sorted in order of group index, and let  $\pmb{K}_{S_i}$  denote the restriction of  $\pmb{K}$  to elements in  $S_i$ . Then  $K/N$  is a block diagonal matrix, with each block  $i$  equal to  $p_i K_{S_i}$ . The eigenvalues of a block diagonal matrix are the combined eigenvalues of each block, and the partitioning property then follows from the partitioning property of the Shannon entropy.

(e) Symmetry: The eigenvalues of a matrix are unchanged by orthonormal transformation, and the Shannon entropy is symmetric in its arguments, so the Vendi Score is symmetric.

# 7.5 Sample Complexity

The Vendi Score is the exponential of the kernel entropy,  $H(K) = -\mathrm{tr}\left(\frac{K}{n}\log \frac{K}{n}\right)$ . Bach (2022) proves that empirical estimator of the kernel entropy,  $\hat{H}$ , has a convergence rate proportional to  $1 / \sqrt{n}$ , where  $n$  is the number of samples. Additionally,

by Jensen's inequality, the  $\mathbb{E}[\hat{H}]$  is no greater than  $H$ . Therefore:

$$
\begin{array}{l} \exp (H) - \exp (\hat {H}) \leq \exp (H) - \exp \left(H - \frac {1}{\sqrt {n}}\right) \\ = \exp (H) - \exp (H) / \exp \left(\frac {1}{\sqrt {n}}\right) \\ = \exp (H) \left(1 - \frac {1}{\sqrt {n}}\right). \\ \end{array}
$$

The empirical estimator of the Vendi Score therefore also has a convergence rate proportional to  $1 / \sqrt{n}$ , with a constant term depending on the true entropy.

# 8 Implementation Details

# 8.1 Images

Stacked MNIST We train GANs on Stacked MNIST using the publicly available code for PresGANs  $^{4}$  and self-conditioned GANs  $^{5}$ . The models share the same DCGAN (Radford et al., 2015) architecture and are trained on the same dataset of 60,000 Stacked MNIST images, rescaled to  $32 \times 32$  pixels, and other hyperparameters are set according to the descriptions in the papers. The models are trained for 50 epochs and the diversity scores are evaluated every five epochs by taking 10,000 samples. For both models, we report the scores from the epoch corresponding to the highest VS score. As in prior work (Metz et al., 2017), we classify Stacked MNIST digits by applying a pretrained MNIST classifier to each color channel independently. The 1000-dimensional Stacked MNIST probability vector is then the tensor product of the three 10-dimensional probability vectors predicted for the three channels.

Obtaining Image Samples In Section 4.4, we calculate the diversity scores of several recent generative models of images. We select models that represent a range of families of generative models and and provide publicly available samples or model checkpoints for common image datasets. On the low-resolution datasets, we generate 50,000 samples from each model using the official code for VDVAE, $^{6}$  DenseFlow, $^{7}$ , and IDDPM, $^{8}$ , each of which provides a checkpoint for unconditional image generation models on CIFAR-10 and ImageNet-64. For IDDPM, we sample using DDIM (Song et al., 2021) for 250 steps, and otherwise use the default sampling parameters. For the higher-resolution datasets, we use the 50,000 precomputed samples provided by Dhariwal and Nichol (2021) $^{9}$  for ADM and StyleGAN models. We obtain 50,000 samples from the RQ-VAE/Transformer model using the code and checkpoints provided by the authors, $^{10}$  with the default sampling parameters.

Calculating Image Metrics In Table 2, we calculate standard image quality and diversity metrics, which are based on Inception embeddings. These Inception-based metrics are sensitive to a number of implementation details (Parmar et al., 2022) and in general cannot be compared directly between papers. For a consistent comparison, we calculate all scores using the evaluation code provided by Dhariwal and Nichol (2021). We also calculated FID and Precision/Recall using the provided reference images and statistics, with the exception of CIFAR-10, for which we use the training set as the reference. (The diversity scores of the Original datasets in Table 2 are calculated using these reference images.) As a result, the numbers in this table may not be directly comparable to results reported in prior work.

# 8.2 Text

Obtaining Image Captions In Section 4.5, we sample image captions from a pretrained image-captioning model, $^{11}$  which is publicly available in Hugging Face (Wolf et al., 2019), and we use the Hugging Face implementation of beam search and diverse beam search. For beam search we use a beam size of 5. For diverse beam search, we use a beam size of 10, a beam group size of 10, and set the number of return sequences to 5.

Calculating Text Metrics The text metrics we use are calculated in terms of word n-grams, and therefore depend on how sentences are tokenized into words. We calculate all text metrics using the pre-trained wordpiece tokenizer used by the captioning models. We use the implementation of the BLEU score in NLTK (Bird, 2006).

# 9 Additional Results

# 9.1 Assessing Mode Dropping in Datasets

In Figure 5, we examine whether VS captures mode dropping in a controlled setting, where we have information about the ground truth class distribution. We simulate mode dropping by sampling equal-sized subsets of two classification datasets, with each subset  $\mathcal{S}_i$  containing examples sampled uniformly from the first  $i$  categories. We perform this experiment one image dataset (mnist) and one text dataset (multinli; Williams et al., 2018), using simple similarity functions. We compare vs to the Internal Diversity (IntDiv), defined as above.

mnist consists of  $28 \times 28$ -pixel images of hand-written digits, divided into ten classes. The similarity score we use is the cosine similarity between pixel vectors:  $k(x,x^{\prime}) = \langle x,x^{\prime}\rangle /\| x\| \| x^{\prime}\|$ , where  $x,x^{\prime}$  are  $28^{2}$ -dimensional vectors with entries specifying pixel intensities between 0 and 1. multinli is a multi-genre sentence-pair classification dataset. We use the premise sentences from the validation split (mismatched), which are drawn from one of ten genres. We define similarity using the n-gram overlap kernel: for a given  $n$ , the n-gram kernel  $k_{n}$  can be expressed as the cosine similarity between feature vectors  $\phi^n (x)$ , where

Figure 5: Detecting mode dropping in image and text datasets. We evaluate vs and IntDiv on datasets containing 500 examples drawn uniformly from between one and ten classes: digits in mnist and sentences genres in Multinli. Compared to IntDiv, vs increases more consistently with the number of classes.


$\phi_i^n (x)$  is equal to the number of times n-gram  $i$  appears in  $x$ . We use the average of  $k(x,x^{\prime}) = \frac{1}{4}\sum_{n = 1}^{4}k_{n}(x,x^{\prime})$ .

The results (Figure 5) show that vs generally increases with the number of classes, even using these simple similarity scores. In mnist (left), vs increases roughly linearly for the first six digits (0-5) and then fluctuates. This could occur if the new modes are similar to the other modes in the sample, or have low internal diversity. In multinli (right), vs increases monotonically with the number of genres represented in the sample. In both cases, vs has a stronger correlation with the number of modes compared to IntDiv.

# 9.2 Evaluating molecular generative models for diversity

We evaluate samples from generative models provided in the moses benchmark (Polykovskiy et al., 2020), using the first 2,500 valid molecules in each sample. Following prior work, our similarity function is the Morgan fingerprint similarity (radius 2), implemented in RDKit. $^{12}$  IntDiv ranks the hmm among the most diverse models, while VS ranks it as the least diverse (see Section 4.2).

# 9.3 Evaluating image generative models for diversity

In Table 5, we replicate the table described in Section 4.4 and add an additional column, which evaluates diversity using the cosine similarity between pixel vectors as the similarity function.

vs should be understood as the diversity with respect to a specific similarity function, in this case, the Inception ImageNet similarity. We illustrate this point in Figure 6 by comparing the top eigenvalues of the kernel matrices corresponding to the Inception similarity and the pixel similarity, which we calculate by resizing the images to  $32 \times 32$

<table><tr><td>Model</td><td>IntDiv</td><td>vs</td></tr><tr><td>Original</td><td>0.855</td><td>403.9</td></tr><tr><td>aaa</td><td>0.859</td><td>501.1</td></tr><tr><td>char-rnn</td><td>0.856</td><td>482.4</td></tr><tr><td>Combinatorial</td><td>0.873</td><td>536.9</td></tr><tr><td>hmm</td><td>0.871</td><td>250.9</td></tr><tr><td>jtn</td><td>0.856</td><td>489.5</td></tr><tr><td>Latent gan</td><td>0.857</td><td>486.4</td></tr><tr><td>N-gram</td><td>0.874</td><td>479.8</td></tr><tr><td>vae</td><td>0.856</td><td>475.3</td></tr></table>

Table 4: IntDiv and vs for generative models of molecules. The hmm has one of the highest IntDiv scores, but scores much lower on vs . An analysis of 250 molecules from the hmm reveals vs is more accurate in this case. (See 3.)

pixels and taking the cosine similarity between pixel vectors. Inception similarity provides a form of semantic similarity, with components corresponding to particular cat breeds, while the pixel kernel provides a simple form of visual similarity, with components corresponding to broad differences in lightness, darkness, and color.

# 9.4 Evaluating decoding algorithms for text for diversity

In Figure 7, we plot the relationship between VS and n-gram diversity using the MS-COCO captioning data and the n-gram overlap kernel described in Section 4.5. The figure shows that VS is highly correlated with n-gram diversity, which is expected given that our similarity function is based on n-gram overlap. Nonetheless, there are some data points that the metrics rank differently. This is because n-gram diversity conflates two properties: the diversity of n-grams within a single sentences and the n-gram overlap between sentences. We highlight two examples in Figure 8. In general, the instances that n-gram diversity ranks lower compared to VS contain individual sentences that repeat phrases. On the other hand, n-gram diversity can be inflated in cases when one sentence in the sample is much longer than the others, even if the other sentences are not diverse.

# 9.5 Diagnosing datasets for diversity

Molecules We evaluate the diversity scores of molecules in the GoodScents database of perfume materials, $^{13}$  which has been used in prior machine learning research on odor modeling (Sanchez-Lengeling et al., 2019). We use the standardized version of the data provided by the Pyrfume library. $^{14}$  Each molecule in the dataset is labeled with one or more odor descriptors (for example, "clean, oily, waxy" or "floral, fruity, green"). We form groups of molecules corresponding to the seven most common odor descriptors, with each group consisting of 500 randomly sampled

<table><tr><td>Model</td><td>IS↑</td><td>FID↓</td><td>Prec↑</td><td>Rec↑</td><td>VS1↑</td><td>VSp↑</td></tr><tr><td colspan="7">CIFAR-10</td></tr><tr><td>Original</td><td></td><td></td><td></td><td></td><td>19.50</td><td>3.52</td></tr><tr><td>VDVAE</td><td>5.82</td><td>40.05</td><td>0.63</td><td>0.35</td><td>12.87</td><td>3.34</td></tr><tr><td>DenseFlow</td><td>6.01</td><td>34.54</td><td>0.62</td><td>0.38</td><td>13.55</td><td>2.94</td></tr><tr><td>IDDPM</td><td>9.24</td><td>4.39</td><td>0.66</td><td>0.60</td><td>16.86</td><td>3.27</td></tr><tr><td colspan="7">ImageNet 64×64</td></tr><tr><td>Original</td><td></td><td></td><td></td><td></td><td>43.93</td><td>4.43</td></tr><tr><td>VDVAE</td><td>9.68</td><td>57.57</td><td>0.47</td><td>0.37</td><td>18.04</td><td>4.24</td></tr><tr><td>DenseFlow</td><td>5.62</td><td>102.90</td><td>0.36</td><td>0.17</td><td>12.71</td><td>3.51</td></tr><tr><td>IDDPM</td><td>15.59</td><td>19.24</td><td>0.59</td><td>0.58</td><td>24.28</td><td>4.57</td></tr></table>

<table><tr><td>Model</td><td>IS↑</td><td>FID↓</td><td>Prec↑</td><td>Rec↑</td><td>VS1↑</td><td>VSPP↑</td></tr><tr><td colspan="7">LSUN Bedroom 256×256</td></tr><tr><td>Original</td><td></td><td></td><td></td><td></td><td>8.99</td><td>3.10</td></tr><tr><td>StyleGAN</td><td>2.55</td><td>2.35</td><td>0.59</td><td>0.48</td><td>8.76</td><td>3.09</td></tr><tr><td>ADM</td><td>2.38</td><td>1.90</td><td>0.66</td><td>0.51</td><td>7.97</td><td>3.27</td></tr><tr><td>RQ-VT</td><td>2.56</td><td>3.16</td><td>0.60</td><td>0.50</td><td>8.48</td><td>3.67</td></tr><tr><td colspan="7">LSUN Cat 256×256</td></tr><tr><td>Original</td><td></td><td></td><td></td><td></td><td>15.12</td><td>4.58</td></tr><tr><td>StyleGAN2</td><td>4.84</td><td>7.25</td><td>0.58</td><td>0.43</td><td>13.55</td><td>4.53</td></tr><tr><td>ADM</td><td>5.19</td><td>5.57</td><td>0.63</td><td>0.52</td><td>13.09</td><td>4.81</td></tr><tr><td>RQ-VT</td><td>5.76</td><td>10.69</td><td>0.53</td><td>0.48</td><td>14.91</td><td>5.83</td></tr></table>

Table 5: We evaluate samples from several recent models, measuring similarity using either Inception representations  $(\mathrm{VS}_I)$  or pixels  $(\mathrm{VS}_P)$ . The pixel similarity score is the cosine similarity between pixel vectors, calculated after resizing the images to  $32 \times 32$  pixels. The pixel similarity and Inception similarity scores do not always agree—for example, if the images in a sample represent a variety of ImageNet classes by share a similar color palette, we might expect the sample to have high Inception diversity but low pixel diversity. The pixel diversity scores are on a lower scale, indicating that this similarity metric is less capable of making fine-grained distinctions between the images in these samples.

Figure 6: The choice of similarity function provides a way of specifying the notion of diversity that is relevant for a given application. We project lsun Cat images along the top eigenvectors of the kernel matrix, using either Inception features or pixels to define similarity. Inception similarity provides a form of semantic similarity, with components corresponding to particular cat breeds, while the pixel kernel captures visual similarity. For each eigenvector  $\pmb{u}$ , we show the four images with the highest and lowest entries in  $\pmb{u}$ . For both kernels, every similarity score is positive, so all entries in the top eigenvector have the same sign; the images with the highest weights in this component have the highest expected similarity scores. The remaining eigenvectors partition the images along different dimensions of variation.


Figure 7: VS is correlated with N-gram diversity. Each point represents a group of five captions for a particular image.  
Figure 8: Two sets of captions that receive different ranks according Vendi Score and n-gram diversity. We manually highlight some features contributing to the different scores. On the left, a sentence contains repeated n-grams, which are penalized by n-gram diversity. On the right, one long outlier sentence contributes most of the n-grams for this group, greatly increasing the n-gram diversity.

High Vendi Score, low n-gram diversity:

- two men in bow ties standing next to steel rafter.  
- several men in suits talking together in a room.  
- an older man in a tuxedo standing next to a younger man in a tuxedo wearing glasses.  
- two men wearing tuxedos glance at each other.  
- older man in tuxedo sitting next to another younger man in tuxedo.

Low Vendi Score, high n-gram diversity:

- a man and woman cutting a slice of cake by trees.  
- a couple of people standing cutting a cake.  
- the dork with the earring stands next to the asian beauty who is way out of his league.  
- a newly married couple cutting a cake in a park.  
- a bride and groom are cutting a cake as they smile.

molecules. We evaluate VS using two similarity functions: the Morgan fingerprint similarity (radius 2), and the similarity between odor descriptors, defined as the cosine similarity between descriptor indicator vectors  $\phi(x)$ , where  $\phi_i(x)$  is equal to one if descriptor  $i$  is associated with molecule  $x$  and zero otherwise.

The diversity scores are plotted in Figure 9. The molecular diversity score and the odor-descriptor diversity scores are correlated, meaning that words like "woody" and "green" are used to describe molecules that vary in molecular structure and also elicit diverse odor descriptions, while words like "waxy" and "fatty" are used for molecules that are similar to each other and elicit similar odor descriptions. For example, the word "green" appears in tag sets such as "aldehydic, citrus, cortex, green, herbal, tart" and "floral, green, terpenic, tropical, vegetable, woody", whereas the word "waxy" tends to co-occur with the same tags ("fresh, waxy"; "fresh, green, melon rind, mushroom, tropical, waxy"; "fruity, green, musty, waxy"). Molecules from the

Figure 9: The Vendi Scores of samples containing 500 molecules with different scent labels, calculating diversity using two similarity functions: Morgan molecular fingerprint similarity, and the similarity between odor descriptors. Each molecule is associated with one or more human-written tags (e.g. "floral, fruity, green, sweet"), and the odor-descriptor similarity is the cosine similarity between binary tag indicator vectors.

categories with the highest and lowest scores are illustrated in Figure 10.

Figure 11: The Vendi Scores of samples containing 500 MultiNLI sentences with different genres (left) or Amazon reviews with different star ratings (right), defining similarity using either n-gram overlap or SimCSE (Gao et al., 2021).


Text In Figure 11, we evaluate the diversity scores of samples sentences with different genres, from the MultiNLI dataset (Williams et al., 2018), and Amazon product reviews with different star ratings (Keung et al., 2020), using either the n-gram overlap similarity or SimCSE (Gao et al., 2021). SimCSE is a Transformer-based sentence encoder that achieves state-of-the-art scores on semantic similarity benchmarks. The model we use initialized from the uncased BERT-base model (Devlin et al., 2019) and trained with a contrastive learning objective to assign high similarity scores to

Most diverse

Least diverse  
Figure 10: The scent categories in Goodscents dataset with the highest (top) and lowest (bottom) vendi score (vs), using the molecular fingerprint similarity. We show 100 examples from each category, in decreasing order of average similarity, with the image at the top left having the highest average similarity scores.

pairs of MultiNLI sentences that have a logical entailment relationship.

In MultiNLI, both models assign the highest score to Slate, which consists of sentences from articles published on slate.com. SimCSE assigns a higher score to the "Fiction" category, possibly because it is less sensitive to common n-grams (e.g. "he said"), that appear in many sentences in this genre and contribute to the low N-gram diversity score. In the Amazon review dataset, the 5-star reviews have the highest N-gram diversity but the lowest SimCSE diversity, perhaps because SimCSE assigns high similarity scores to sentences that have the same strong sentiment. SimCSE assigns the highest diversity score to 3-star reviews, which can vary in

sentiment.

Images Following the setting in 4.6, we evaluate two additional datasets, Fashion mnist (Xiao et al., 2017) and celeba (Liu et al., 2015). We use the same similarity scores as in 4.6. Images in CelebA are associated with 40-dimensional binary attribute vectors. We use these attributes as an additional similarity score, defining the attribute similarity as the cosine similarity between attribute vectors. These illustrations highlight the importance of the choice of similarity function in defining a diversity metric.

Figure 12: The categories in Fashion MNIST with the lowest (left) and highest (right) Vendi Scores, defining similarity as the cosine similarity between either Inception embeddings (top) or pixel vectors (bottom). We show 100 examples from each category, in decreasing order of average similarity, with the image at the top left having the highest average similarity scores according to the corresponding kernel.

Figure 13: The attributes in celeba with the lowest (left) and highest (right) vs, defining similarity as the cosine similarity between either Inception embeddings (top), pixel vectors (middle), or binary attribute vectors (bottom). We show 100 examples from each category, in decreasing order of average similarity, with the image at the top left having the highest average similarity scores according to the corresponding kernel. These examples illustrate the importance of the choice of similarity function for defining the notion of diversity that is relevant for a given application. However, almost all choices of similarity functions show that the celeba dataset is more diverse for men than for women.

# Footnotes:

Page 1: $^{1}$ Code for calculating the Vendi Score is available at https://github.com/verteaix/Vendi-Score. 
Page 9: $^{2}$ RDKit: Open-source Cheminformatics. https://www.rdkit.org. 
Page 13: <sup>3</sup>https://huggingface.co/ydshieh/vit-gpt2-coco-en-ckpts 
Page 22: <sup>4</sup>https://github.com/adjidieng/PresGANs <sup>5</sup>https://github.com/stevliu/self-conditioned-gan <sup>6</sup>https://github.com/openai/vdvae/ <sup>7</sup>https://github.com/matejgrcic/DenseFlow <sup>8</sup>https://github.com/openai/improved-diffusion <sup>9</sup>https://github.com/openai/guided-diffusion $^{10}$ https://github.com/kakaobrain/rq-vae-transformer 
Page 23: <sup>11</sup>https://huggingface.co/ydshieh/vit-gpt2-coco-en-ckpts 
Page 24: $^{12}$ RDKit: Open-source Cheminformatics. https://www.rdkit.org. 
Page 25: <sup>13</sup>http://www.thegoodscentscompany.com/ <sup>14</sup>https://pyrfume.org/ 
