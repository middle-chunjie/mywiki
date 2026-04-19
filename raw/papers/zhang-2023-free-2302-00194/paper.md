Free Lunch for Domain Adversarial Training: Environment Label Smoothing
=======================================================================

Yi-Fan Zhang1,2, Xue Wang3, Jian Liang1,2,  
Zhang Zhang1,2, Liang Wang1,2, Rong Jin3, Tieniu Tan1,2  
1National Laboratory of Pattern Recognition (NLPR), Institute of Automation  
2School of Artificial Intelligence, University of Chinese Academy of Sciences (UCAS)  
3 Machine Intelligence Technology, Alibaba GroupWork done during an internship at Alibaba Group.Work done at Alibaba Group, and now affiliated with Twitter.

###### Abstract

A fundamental challenge for machine learning models is how to generalize learned models for out-of-distribution (OOD) data. Among various approaches, exploiting invariant features by Domain Adversarial Training (DAT) received widespread attention. Despite its success, we observe training instability from DAT, mostly due to over-confident domain discriminator and environment label noise. To address this issue, we proposed Environment Label Smoothing (ELS), which
encourages the discriminator to output soft probability, which thus reduces the confidence of the discriminator and alleviates the impact of noisy environment labels. We demonstrate, both experimentally and theoretically, that ELS can improve training stability, local convergence, and robustness to noisy environment labels. By incorporating ELS with DAT methods, we are able to yield the state-of-art results on a wide range of domain generalization/adaptation tasks, particularly when the environment labels are highly noisy. The code is avaliable at https://github.com/yfzhang114/Environment-Label-Smoothing.

1 Introduction
--------------

Despite being empirically effective on visual recognition benchmarks*(Russakovsky et al., [2015](#bib.bib60 ""))*, modern neural networks are prone to learning shortcuts that stem from spurious correlations*(Geirhos et al., [2020](#bib.bib23 ""))*, resulting in poor generalization for out-of-distribution (OOD) data. A popular thread of methods, minimizing domain divergence by Domain Adversarial Training (DAT)*(Ganin et al., [2016](#bib.bib22 ""))*, has shown better domain transfer performance, suggesting that it is potential to be an effective candidate to extract domain-invariant features. Despite its power for domain adaptation and domain generalization, DAT is known to be difficult to train and converge *(Roth et al., [2017](#bib.bib59 ""); Jenni \& Favaro, [2019](#bib.bib33 ""); Arjovsky \& Bottou, [2017](#bib.bib4 ""); Sønderby et al., [2016](#bib.bib68 ""))*.

<img src='x1.png' alt='Refer to caption' title='' width='106' height='101' />

*Figure 1: A motivating example of ELS with 3 domains on the VLCS dataset.*

The main difficulty for stable training is
to maintain healthy competition between the encoder and the domain discriminator. Recent work seeks to attain this goal by designing novel optimization methods*(Acuna et al., [2022](#bib.bib2 ""); Rangwani et al., [2022](#bib.bib58 ""))*, however, most of them require additional optimization steps and slow the convergence. In this work, we aim to tackle the challenge from a totally different aspect from previous works, i.e., the environment label design.

Two important observations that lead to the training instability of DAT motivate this work:
(i) The environment label noise from environment partition*(Creager et al., [2021](#bib.bib17 ""))* and training*(Thanh-Tung et al., [2019](#bib.bib74 ""))*. As shown in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), different domains of the VLCS benchmark have no significant difference in image style and some images are indistinguishable for which domain they belong. Besides, when the encoder gets better, the generated features from different domains are more similar. However, regardless of their quality, features are still labeled differently. As shown in*(Thanh-Tung et al., [2019](#bib.bib74 ""); Brock et al., [2019](#bib.bib12 ""))*, discriminators will overfit these mislabelled examples and then has poor generalization capability. (ii) To our best knowledge, DAT methods all assign one-hot environment labels to each data sample for domain discrimination, where the output probabilities will be highly confident. For DAT, a very confident domain discriminator leads to highly oscillatory gradients *(Arjovsky \& Bottou, [2017](#bib.bib4 ""); Mescheder et al., [2018](#bib.bib49 ""))*, which is harmful to training stability. The first observation inspires us to force the training process to be robust with regard to environment-label noise, and the second observation encourages the discriminator to estimate soft probabilities rather than confident classification. To this end, we propose Environment Label Smoothing (ELS), which is a simple method to tackle the mentioned obstacles for DAT. Next, we summarize the main methodological, theoretical, and experimental contributions.

Methodology: To our best knowledge, this is the first work to smooth environment labels for DAT. The proposed ELS yields three main advantages: (i) it does not require any extra parameters and optimization steps and yields faster convergence speed, better training stability, and more robustness to label noise theoretically and empirically; (ii) despite its efficiency, ELS is also easily to implement. People can easily incorporate ELS with any DAT methods in very few lines of code; (iii) ELS equipped DAT methods attain superior generalization performance compared to their native counterparts;

Theories: The benefit of ELS is theoretically verified in the following aspects. (i) Training stability. We first connect DAT to Jensen–Shannon/Kullback–Leibler divergence minimization, where ELS is shown able to extend the support of training distributions and relieve both the oscillatory gradients and gradient vanishing phenomenons, which results in stable and well-behaved training.
(ii) Robustness to noisy labels. We theoretically verify that the negative effect caused by noisy labels can be reduced or even eliminated by ELS with a proper smooth parameter.
(iii) Faster non-asymptotic convergence speed. We analyze the non-asymptotic convergence properties of DANN. The results indicate that incorporating with ELS can further speed up the convergence process.
In addition, we also provide the empirical gap and analyze some commonly used DAT tricks.

Experiments: (i) Experiments are carried out on various benchmarks with different backbones, including image classification, image retrieval, neural language processing, genomics data, graph, and sequential data. ELS brings consistent improvement when incorporated with different DAT methods and achieves competitive or SOTA performance on various benchmarks, e.g., average accuracy on Rotating MNIST ($52.1\%\rightarrow 62.1\%$), worst group accuracy on CivilComments ($61.7\%\rightarrow 65.9\%$), test ID accuracy on RxRx1 ($22.9\%\rightarrow 26.7\%$), average accuracy on Spurious-Fourier dataset ($11.1\%\rightarrow 15.6\%$). (ii) Even if the environment labels are random or partially known, the performance of ELS + DANN will not degrade much and is superior to native DANN. (iii) Abundant analyzes on training dynamics are conducted to verify the benefit of ELS empirically. (iv) We conduct thorough ablations on hyper-parameter for ELS and some useful suggestions about choosing the best smooth parameter considering the dataset information are given.

2 Methodology
-------------

For domain generalization tasks, there are $M$ source domains ${\mathcal{D}_{i}}_{i\=1}^{M}$. Let the hypothesis $h$ be the composition of $h\=\hat{h}\circ g$, where $g\in\mathcal{G}$ pushes forward the data samples to a representation space $\mathcal{Z}$ and $\hat{h}\=(\hat{h}_{1}(\cdot),\dots,\hat{h}_{M}(\cdot))\in\hat{\mathcal{H}}:\mathcal{Z}\rightarrow[0,1]^{M};\sum_{i\=1}^{M}\hat{h}_{i}(\cdot)\=1$ is the domain discriminator with softmax activation function. The classifier is defined as $\hat{h}^{\prime}\in\hat{\mathcal{H}^{\prime}}:\mathcal{Z}\rightarrow[0,1]^{C};\sum_{i\=1}^{C}\hat{h}^{\prime}_{i}(\cdot)\=1$, where $C$ is the number of classes. The cost used for the discriminator can be defined as:

|  | $\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})\=\max_{\hat{h}\in{\mathcal{H}}}\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{1}}\log\hat{h}_{1}\circ g(\mathbf{x})+\dots+\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{M}}\log\hat{h}_{M}\circ g(\mathbf{x}),$ |  | (1) |
| --- | --- | --- | --- |

where $\hat{h}_{i}\circ g(\mathbf{x})$ is the prediction probability that $\mathbf{x}$ is belonged to $\mathcal{D}_{i}$. Denote $y$ the class label, then the overall objective of DAT is

|  | $\min_{\hat{h}^{\prime},g}\max_{\hat{h}}\frac{1}{M}\sum_{i\=1}^{M}\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{i}}[\ell(\hat{h}^{\prime}\circ g(\mathbf{x}),y)]+\lambda d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M}),$ |  | (2) |
| --- | --- | --- | --- |

where $\ell$ is the cross-entropy loss for classification tasks and MSE for regression tasks, and $\lambda$ is the tradeoff weight. We call the first term empirical risk minimization (ERM) part and the second term adversarial training (AT) part.
Applying ELS, the target in Equ. ([1](#S2.E1 "In 2 Methodology ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) can be reformulated as

|  | $\displaystyle\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g,\gamma}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})\=\max_{\hat{h}\in\hat{\mathcal{H}}}\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{1}}$ | $\displaystyle\left[\gamma\log\hat{h}_{1}\circ g(\mathbf{x})+\frac{(1-\gamma)}{M-1}\sum_{j\=1;j\neq 1}^{M}\log\left(\hat{h}_{j}\circ g(\mathbf{x})\right)\right]+\dots+$ |  | (3) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{M}}\left[\gamma\log\hat{h}_{M}\circ g(\mathbf{x})+\frac{(1-\gamma)}{M-1}\sum_{j\=1;j\neq M}^{M}\log\left(\hat{h}_{j}\circ g(\mathbf{x})\right)\right].$ | | |

3 Theoretical validation
------------------------

In this section, we first assume the discriminator is optimized with no constraint, providing a theoretical interpretation of applying ELS. Then how ELS makes the training process more stable is discussed based on the interpretation and some analysis of the gradients. We next theoretically show that with ELS, the effect of label noise can be eliminated. Finally, to mitigate the impact of the no constraint assumption, the empirical gap, parameterization gap, and non-asymptotic convergence property are analyzed respectively. All omitted proofs can be found in the Appendix.

### 3.1 Divergence Minimization Interpretation

In this subsection, the connection between ELS/one-sided ELS and divergence minimization is studied. The advantages brought by ELS and why GANs prefer one-sided ELS are theoretically claimed. We begin with the two-domain setting, which is used in domain adaptation and generative adversarial networks. Then the result in the multi-domain setting is further developed.

###### Proposition 1.

Given two domain distributions $\mathcal{D}_{S},\mathcal{D}_{T}$ over $X$, and a hypothesis class $\mathcal{H}$. We suppose $\hat{h}\in\hat{\mathcal{H}}$ the optimal discriminator with no constraint, denote the mixed distributions with hyper-parameter $\gamma\in[0.5,1]$ as $\left{\begin{array}[]{c}\mathcal{D}_{S^{\prime}}\=\gamma\mathcal{D}_{S}+(1-\gamma)\mathcal{D}_{T}\\
\mathcal{D}_{T^{\prime}}\=\gamma\mathcal{D}_{T}+(1-\gamma)\mathcal{D}_{S}\\
\end{array}\right.$. Then minimizing domain divergence by adversarial training with ELS is equal to minimizing $2D_{JS}(\mathcal{D}_{S^{\prime}}||\mathcal{D}_{T^{\prime}})-2\log 2$, where $D_{JS}$ is the Jensen-Shanon (JS) divergence.

Compared to Proposition 2 in*(Acuna et al., [2021](#bib.bib1 ""))* that adversarial training in DANN is equal to minimize $2D_{JS}(\mathcal{D}_{S}||\mathcal{D}_{T})-2\log 2$. The only difference here is the mixed distributions $\mathcal{D}_{S^{\prime}},\mathcal{D}_{T^{\prime}}$, which allows more flexible control on divergence minimization. For example, when $\gamma\=1$, $\mathcal{D}_{S^{\prime}}\=\mathcal{D}_{S},\mathcal{D}_{T^{\prime}}\=\mathcal{D}_{T}$ which is the same as the original adversarial training; when $\gamma\=0.5$, $\mathcal{D}_{S^{\prime}}\=\mathcal{D}_{T^{\prime}}\=0.5(\mathcal{D}_{S}+\mathcal{D}_{T})$ and $D_{JS}(\mathcal{D}_{S^{\prime}}||\mathcal{D}_{T^{\prime}})\=0$, which means that this term will not supply gradients during training and the training process will convergence like ERM. In other words, $\gamma$ controls the tradeoff between algorithm convergence and adversarial divergence minimization. One main argue that adjusting the tradeoff weight $\lambda$ can also balance AT and ERM, however, $\lambda$ can only adjust the gradient contribution of AT part, i.e., $2\lambda\nabla D_{JS}(\mathcal{D}_{S},\mathcal{D}_{T})$ and cannot affect the training dynamic of $D_{JS}(\mathcal{D}_{S},\mathcal{D}_{T})$. For example, when $\mathcal{D}_{S},\mathcal{D}_{T}$ have disjoint support, $\nabla D_{JS}(\mathcal{D}_{S},\mathcal{D}_{T})$ is always zero no matter what $\lambda$ is given. On the contrary, the proposed technique smooths the optimization distribution $\mathcal{D}_{S},\mathcal{D}_{T}$ of AT, making the whole training process more stable, but controlling $\lambda$ cannot do. In the experimental section, we show that in some benchmarks, the model cannot converge even if the tradeoff weight is small enough, however, when ELS is applied, DANN+ELS attains superior results and without the need for small tradeoff weights or small learning rate.

As shown in*(Goodfellow, [2016](#bib.bib26 ""))*, GANs always use a technique called one-sided label smoothing, which is a simple modification of the label smoothing technique and only replaces the target for real examples with a value slightly less than one, such as 0.9. Here we connect one-sided label smoothing to JS divergence and seek the difference between native and one-sided label smoothing techniques. See Appendix[A.2](#A1.SS2 "A.2 Connect One-sided Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") for proof and analysis. We further extend the above theoretical analysis to multi-domain settings, e.g., domain generalization, and multi-source GANs*(Trung Le et al., [2019](#bib.bib76 ""))* (See Proposition[3](#Thmprop3 "Proposition 3. ‣ A.3 Connect Multi-Domain Adversarial Training to KL Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") in Appendix[A.3](#A1.SS3 "A.3 Connect Multi-Domain Adversarial Training to KL Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") for detailed proof and analysis.). We find that with ELS, a flexible control on algorithm convergence and divergence minimization tradeoffs can be attained.

### 3.2 Training Stability

Noise injection for extending distribution supports. The main source of training instability of GANs is the real and the generated distributions have disjoint supports or lie on low dimensional manifolds*(Arjovsky \& Bottou, [2017](#bib.bib4 ""); Roth et al., [2017](#bib.bib59 ""))*. Adding noise from an arbitrary distribution to the data is shown to be able to extend the support of both distributions*(Jenni \& Favaro, [2019](#bib.bib33 ""); Arjovsky \& Bottou, [2017](#bib.bib4 ""); Sønderby et al., [2016](#bib.bib68 ""))* and will protect the discriminator against measure 0 adversarial examples*(Jenni \& Favaro, [2019](#bib.bib33 ""))*, which result in stable and well-behaved training. Environment label smoothing can be viewed as a kind of noise injection, e.g., in Proposition[1](#Thmprop1 "Proposition 1. ‣ 3.1 Divergence Minimization Interpretation ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), $\mathcal{D}_{S^{\prime}}\=\mathcal{D}_{T}+\gamma(\mathcal{D}_{S}-\mathcal{D}_{T})$ where the noise is $\gamma(\mathcal{D}_{S}-\mathcal{D}_{T})$ and the two distributions will be more likely to have joint supports.

ELS relieves the gradient vanishing phenomenon. As shown in Section[3.1](#S3.SS1 "3.1 Divergence Minimization Interpretation ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), the adversarial target is approximating KL or JS divergence, and when the discriminator is not optimal, a such approximation is inaccurate. We show that in vanilla DANN, as the discriminator gets better, the gradient passed from discriminator to the encoder vanishes (Proposition[25](#A1.E25 "In Proposition 4. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Proposition[32](#A1.E32 "In Proposition 5. ‣ A.5 Training Stability Analysis of Multi-Domain settings ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")). Namely, either the approximation is inaccurate, or the gradient vanishes, which will make adversarial training extremely hard*(Arjovsky \& Bottou, [2017](#bib.bib4 ""))*. Incorporating ELS is shown able to relieve the

<img src='x2.png' alt='Refer to caption' title='' width='126' height='94' />

*Figure 2: The sum of gradients provided to the encoder by the adversarial loss.*

gradient vanishing phenomenon when the discriminator is close to the optimal one and stabilizes the training process.

ELS serves as a data-driven regularization and stabilizes the oscillatory gradients. Gradients of the encoder with respect to adversarial loss remain highly oscillatory in native DANN, which is an important reason for the instability of adversarial training*(Mescheder et al., [2018](#bib.bib49 ""))*. Figure[2](#S3.F2 "Figure 2 ‣ 3.2 Training Stability ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") shows the gradient dynamics throughout the training process, where the PACS dataset is used as an example. With ELS, the gradient brought by the adversarial loss is smoother and more stable. The benefit is theoretically supported in Section[A.6](#A1.SS6 "A.6 ELS stabilize the oscillatory gradient ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), where applying ELS is shown similar to adding a regularization term on discriminator parameters, which stabilizes the supplied gradients compared to the vanilla adversarial loss.

### 3.3 ELS meets noisy labels

To analyze the benefits of ELS when noisy labels exist, we adopt the symmetric noise model*(Kim et al., [2019](#bib.bib37 ""))*. Specifically, given two environments with a high-dimensional feature $x$ and environment label $y\in{0,1}$, assume that noisy labels $\tilde{y}$ are generated by random noise transition with noise rate $e\=P(\tilde{y}\=1|y\=0)\=P(\tilde{y}\=0|y\=1)$. Denote $f:\=\hat{h}\circ g$, $\ell$ the cross-entropy loss and $\tilde{y}^{\gamma}$ the smoothed noisy label, then minimizing the smoothed loss with noisy labels can be converted to

|  |  | $\displaystyle\min_{f}\mathbb{E}_{(x,\tilde{y})\sim\tilde{\mathcal{D}}}[\ell(f(x),\tilde{y}^{\gamma})]\=\min_{f}\mathbb{E}_{(x,\tilde{y})\sim\tilde{\mathcal{D}}}\left[\gamma\ell(f(x),\tilde{y})+(1-\gamma)\ell(f(x),1-\tilde{y})\right]$ |  | (4) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\min_{f}\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),y^{\gamma^{*}})]+(\gamma^{*}-\gamma-e+2\gamma e)\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),1-y)-\ell(f(x),y)]$ | | |

where $\gamma^{*}$ is the optimal smooth parameter that makes the classifier return the best performance on unseen clean data*(Wei et al., [2022](#bib.bib83 ""))*. The first term in Equ. ([4](#S3.E4 "In 3.3 ELS meets noisy labels ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is the risk under the clean label. The influence of both noisy labels and ELS are reflected in the last term of the Equ. ([4](#S3.E4 "In 3.3 ELS meets noisy labels ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")). $\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),1-y)-\ell(f(x),y)]$ is the opposite of the optimization process as we expect. Without label smoothing, the weight will be $\gamma^{*}-1+e$ and a high noisy rate $e$ will let this harmful term contributes more to our optimization. On the contrary, by choosing a smooth parameter $\gamma\=\frac{\gamma^{*}-e}{1-2e}$, the second term will be removed. For example, if $e\=0$, the best smooth parameter is just $\gamma^{*}$.

### 3.4 Empirical Gap and Parameterization Gap

Propositions in Section[3.1](#S3.SS1 "3.1 Divergence Minimization Interpretation ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Section[3.2](#S3.SS2 "3.2 Training Stability ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") are based on two unrealistic assumptions. (i) Infinite data samples, and (ii) the discriminator is optimized without a constraint, namely, the discriminator is optimized over infinite-dimensional space. In practice, only empirical distributions with finite samples are observed and the discriminator is always constrained to smaller classes such as neural networks*(Goodfellow et al., [2014](#bib.bib27 ""))* or reproducing kernel Hilbert spaces (RKHS)*(Li et al., [2017a](#bib.bib40 ""))*. Besides, as shown in*(Arora et al., [2017](#bib.bib7 ""); Schäfer et al., [2019](#bib.bib67 ""))*, JS divergence has a large empirical gap, e.g., let $\mathcal{D}_{\mu},\mathcal{D}_{\nu}$ be uniform Gaussian distributions $\mathcal{N}(0,\frac{1}{d}I)$, and $\hat{\mathcal{D}}_{\mu},\hat{\mathcal{D}}_{\nu}$ be empirical versions of $\mathcal{D}_{\mu},\mathcal{D}_{\nu}$ with $n$ examples. Then we have $|d_{JS}(\mathcal{D}_{\mu},\mathcal{D}_{\nu})-d_{JS}(\hat{\mathcal{D}}_{\mu},\hat{\mathcal{D}}_{\nu})|\=\log 2$ with high probability. Namely, the empirical divergence cannot reflect the true distribution divergence.

A natural question arises: “Given finite samples to multi-domain AT over finite-dimensional parameterized space, whether the expectation over the empirical distribution converges to the expectation over the true distribution?”. In this subsection, we seek to answer this question by analyzing the empirical gap and parameterization gap, which is $|d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})-d_{\hat{h},g}(\hat{\mathcal{D}}_{1},\dots,\hat{\mathcal{D}}_{M})|$, where $\hat{D}_{i}$ is the empirical distribution of $\mathcal{D}_{i}$ and $\hat{h}$ is constrained. We first show that, let $\mathcal{H}$ be a hypothesis class of VC dimension $d$, then for any $\delta\in(0,1)$, with probability at least $1-\delta$, the gap is less than $4\sqrt{({d\log(2n^{*})+\log{2}/{\delta}})/{n^{*}}}$, where $n^{*}\=\min(n_{1},\dots,n_{M})$ and $n_{i}$ is the number of samples in $\mathcal{D}_{i}$ (Appendix[A.8](#A1.SS8 "A.8 Empirical Gap Analysis Adopted from Vapnik-Chervonenkis framework ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")). The above analysis is based on $\mathcal{H}$ divergence and the VC dimension; we further analyze the gap when the discriminator is constrained to the Lipschitz continuous and build a connection between the gap and the model parameters. Specifically, suppose that each $\hat{h}_{i}$ is $L$-Lipschitz with respect to the parameters and use $p$ to denote the number of parameters of $\hat{h}_{i}$. Then given a universal constant $c$ such that when $n^{*}\geq{cpM\log(Lp/\epsilon)}/{\epsilon}$, we have with probability at least $1-\exp(-p)$, the gap is less than $\epsilon$ (Appendix[A.9](#A1.SS9 "A.9 Empirical Gap Analysis Adopted from Neural Net Distance ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")). Although the analysis cannot support the benefits of ELS, as far as we know, it is the first attempt to study the empirical and parameterization gap of multi-domain AT.

### 3.5 Non-Asymptotic Convergence

As mentioned in Section[3.4](#S3.SS4 "3.4 Empirical Gap and Parameterization Gap ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), the analysis in Section[3.1](#S3.SS1 "3.1 Divergence Minimization Interpretation ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Section[3.2](#S3.SS2 "3.2 Training Stability ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") assumes that the optimal discriminator can be obtained, which implies that both the hypothesis set has infinite modeling capacity and the training process can converge to the optimal result. If the objective of AT is convex-concave, then many works can support the global convergence behaviors*(Nowozin et al., [2016](#bib.bib54 ""); Yadav et al., [2017](#bib.bib89 ""))*. However, the convex-concave assumption is too unrealistic to hold true*(Nie \& Patel, [2020](#bib.bib53 ""); Nagarajan \& Kolter, [2017](#bib.bib51 ""))*, namely, the updates of DAT are no longer guaranteed to converge. In this section, we focus on the local convergence behaviors of DAT of points near the equilibrium. Specifically, we focus on the non-asymptotic convergence, which is shown able to more precisely reveal the convergence of the dynamic system than the asymptotic analysis*(Nie \& Patel, [2020](#bib.bib53 ""))*.

We build a toy example to help us understand the convergence of DAT. Denote $\eta$ the learning rate, $\gamma$ the parameter for ELS, and $c$ a constant. We conclude our theoretical results here (which are detailed in Appendix[A.10](#A1.SS10 "A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")): (1) Simultaneous Gradient Descent (GD) DANN, which trains the discriminator and encoder simultaneously, has no guarantee of the non-asymptotic convergence. (2) If we train the discriminator $n_{d}$ times once we train the encoder $n_{e}$ times, the resulting alternating Gradient Descent (GD) DANN could converge with a sublinear convergence rate only when the $\eta\leq\frac{4}{\sqrt{n_{d}n_{e}}c}$. Such results support the importance of alternating GD training, which is commonly used during DANN implementation*(Gulrajani \& Lopez-Paz, [2021](#bib.bib29 ""))*. (3) Incorporate ELS into alternating GD DANN speed up the convergence rate by a factor $\frac{1}{2\gamma-1}$, that is, when $\eta\leq\frac{4}{\sqrt{n_{d}n_{e}}c}\frac{1}{2\gamma-1}$, the model could converge.

Remark. In the above analysis, we made some assumptions e.g., in Section[3.5](#S3.SS5 "3.5 Non-Asymptotic Convergence ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), we assume the algorithms are initialized in a neighborhood of a unique equilibrium point, and in Section[3.4](#S3.SS4 "3.4 Empirical Gap and Parameterization Gap ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") we assume that the NN is L-Lipschitz. These assumptions may not hold in practice, and they are computationally hard to verify. To this end, we empirically support our theoretical results, namely, verifying the benefits to convergence, training stability, and generalization results in the next section.

4 Experiments
-------------

To demonstrate the effectiveness of our ELS, in this section, we select a broad range of tasks (in Table[1](#S4.T1 "Table 1 ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")), which are image classification, image retrieval, neural language processing, genomics, graph, and sequential prediction tasks. Our target is to include benchmarks with (i) various numbers of domains (from $3$ to $120,084$); (ii) various numbers of classes (from $2$ to $18,530$); (iii) various dataset sizes (from $3,200$ to $448,000$); (iv) various dimensionalities and backbones (Transformer, ResNet, MobileNet, GIN, RNN). See Appendix[C](#A3 "Appendix C Additional Experimental Setups ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") for full details of all experimental settings, including dataset details, hyper-parameters, implementation details, and model structures. We conduct all the experiments on a machine with i7-8700K, 32G RAM, and four GTX2080ti. All experiments are repeated $3$ times with different seeds and the full experimental results can be found in the appendix.

*Table 1: A summary on evaluation benchmarks. Wg. acc. denotes worst group accuracy, 10 %/ acc. denotes 10th percentile accuracy. GIN*(Xu et al., [2018](#bib.bib87 ""))* denotes Graph Isomorphism Networks, and CRNN*(Gagnon-Audet et al., [2022](#bib.bib20 ""))* denotes convolutional recurrent neural networks.*

TaskDatasetDomainsClassesMetricBackbone# Data ExamplesRotated MNIST6 rotated angles10Avg. acc.MNIST ConvNet70,000PACS4 image styles7Avg. acc.ResNet509,991VLCS4 image styles5Avg. acc.ResNet5010,729Office-313 image styles31Avg. acc.ResNet50/ResNet184,110Office-Home4 image styles65Avg. acc.ResNet50/ViT15,500Images ClassificationRotating MNIST8 rotated angles10Avg. acc.EncoderSTN60,000Image RetrievalMS5 locations18,530mAP, Rank $m$MobileNet$\times 1.4$121,738CivilComments8 demographic groups2Avg/Wg acc.DistillBERT448,000Neural Language ProcessingAmazon7676 reviewers510 %/Avg/Wg acc.DistillBERT100,124RxRx151 experimental batch1139Wg/Avg/Test ID acc.ResNet-50125,510Genomics and GraphOGB-MolPCBA120,084 molecular scaffold128Avg. acc.GIN437,929Spurious-Fourier3 spurious correlations2Avg. acc.LSTM12,000Sequential PredictionHHAR5 smart devices6Avg. acc.Deep ConvNets13,674

### 4.1 Numerical Results on Different Settings and Benchmarks

Domain Generalization and Domain Adaptation on Image Classification Tasks. We first incorporate ELS into SDAT, which is a variant of the DAT method and achieves the state-of-the-art performance on the Office-Home dataset. Table[2](#S4.T2 "Table 2 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Table[4](#S4.T4 "Table 4 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") show that with the simple smoothing trick, the performance of SDAT is consistently improved, and on many of the domain pairs, the improvement is greater than $1\%$. Besides, the ELS can also bring consistent improvement both with ResNet-18, ResNet-50, and ViT backbones. The average domain generalization results on other benchmarks are shown in Table[3](#S4.T3 "Table 3 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"). We observe consistent improvements achieved by DANN+ELS compared to DANN and the average accuracy on VLCS achieved by DANN+ELS ($81.5\%$) clearly outperforms all other methods. See Appendix[D.1](#A4.SS1 "D.1 Additional Numerical Results ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") for Multi-Source Domain Generalization performance, DG performance on Rotated MNIST and on Image Retrieval benchmarks.

Domain Generalization with Partial Environment labels. One of the main advantages brought by ELS is the robustness to environment label noise. As shown in Figure[4](#S4.F4 "Figure 4 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), when all environment labels are known (GT), DANN+ELS is slightly better than DANN. When partial environment labels are known, for example, $30\%$ means the environment labels of $30\%$ training data are known and others are annotated differently than the ground truth annotations, DANN+ELS outperform DANN by a large margin (more than $5\%$ accuracy when only $20\%$ correct environment labels are given). Besides, we further assume the total number of environments is also unknown and the environment number is generated randomly. M\=2 in Figure[4](#S4.F4 "Figure 4 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") means we partition all the training data randomly into two domains, which are used for training then. With random environment partitions, DANN+ELS consistently beats DANN by a large margin, which verifies that the smoothness of the discrimination loss brings significant robustness to environment label noise for DAT.

*Table 2: The domain adaptation accuracies (%) on Office-31. $\uparrow$ denotes improvement of a method with ELS compared to that wo/ ELS.*

|  | A - W | D - W | W - D | A - D | D - A | W - A | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | ResNet18 | | | | | | |
| ERM(Vapnik, [1999](#bib.bib78 "")) | 72.2 | 97.7 | 100.0 | 72.3 | 61.0 | 59.9 | 77.2 |
| DANN(Ganin et al., [2016](#bib.bib22 "")) | 84.1 | 98.1 | 99.8 | 81.3 | 60.8 | 63.5 | 81.3 |
| DANN+ELS | 85.5 | 99.1 | 100.0 | 82.7 | 62.1 | 64.5 | 82.4 |
| $\uparrow$ | 1.4 | 1.0 | 0.2 | 1.4 | 1.3 | 1.1 | 1.1 |
| SDAT(Rangwani et al., [2022](#bib.bib58 "")) | 87.8 | 98.7 | 100.0 | 82.5 | 73.0 | 72.7 | 85.8 |
| SDAT+ELS | 88.9 | 99.3 | 100.0 | 83.9 | 74.1 | 73.9 | 86.7 |
| $\uparrow$ | 1.1 | 0.5 | 0.0 | 1.4 | 1.1 | 1.2 | 0.9 |
|  | ResNet50 | | | | | | |
| ERM(Vapnik, [1999](#bib.bib78 "")) | 75.8 | 95.5 | 99.0 | 79.3 | 63.6 | 63.8 | 79.5 |
| ADDA(Tzeng et al., [2017](#bib.bib77 "")) | 94.6 | 97.5 | 99.7 | 90.0 | 69.6 | 72.5 | 87.3 |
| CDAN(Long et al., [2018](#bib.bib47 "")) | 93.8 | 98.5 | 100.0 | 89.9 | 73.4 | 70.4 | 87.7 |
| MCC(Jin et al., [2020](#bib.bib35 "")) | 94.1 | 98.4 | 99.8 | 95.6 | 75.5 | 74.2 | 89.6 |
| DANN(Ganin et al., [2016](#bib.bib22 "")) | 91.3 | 97.2 | 100.0 | 84.1 | 72.9 | 73.6 | 86.5 |
| DANN$+$ELS | 92.2 | 98.5 | 100.0 | 85.9 | 74.3 | 75.3 | 87.7 |
| $\uparrow$ | 0.9 | 1.3 | 0.0 | 1.8 | 1.4 | 1.7 | 1.2 |
| SDAT(Rangwani et al., [2022](#bib.bib58 "")) | 92.7 | 98.9 | 100.0 | 93.0 | 78.5 | 75.7 | 89.8 |
| SDAT$+$ELS | 93.6 | 99.0 | 100.0 | 93.4 | 78.7 | 77.5 | 90.4 |
| $\uparrow$ | 0.9 | 0.1 | 0.0 | 0.4 | 0.2 | 1.8 | 0.6 |

*Table 3: The domain generalization accuracies (%) on VLCS, and PACS. $\uparrow$ denotes improvement of DANN+ELS compared to DANN.*

| Algorithm | PACS | | | | | VLCS | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | A | C | P | S | Avg | C | L | S | V | Avg |
| ERM(Vapnik, [1999](#bib.bib78 "")) | 87.8 ± 0.4 | 82.8 ± 0.5 | 97.6 ± 0.4 | 80.4 ± 0.6 | 87.2 | 97.7 ± 0.3 | 65.2 ± 0.4 | 73.2 ± 0.7 | 75.2 ± 0.4 | 77.8 |
| IRM(Arjovsky et al., [2019](#bib.bib6 "")) | 85.7 ± 1.0 | 79.3 ± 1.1 | 97.6 ± 0.4 | 75.9 ± 1.0 | 84.6 | 97.6 ± 0.5 | 64.7 ± 1.1 | 69.7 ± 0.5 | 76.6 ± 0.7 | 77.2 |
| DANN(Ganin et al., [2016](#bib.bib22 "")) | 85.4 $\pm$ 1.2 | 83.1 $\pm$ 0.8 | 96.3 $\pm$ 0.4 | 79.6 $\pm$ 0.8 | 86.1 | 98.6 $\pm$ 0.8 | 73.2 $\pm$ 1.1 | 72.8 $\pm$ 0.8 | 78.8 $\pm$ 1.2 | 80.8 |
| ARM(Zhang et al., [2021b](#bib.bib92 "")) | 85.0 $\pm$ 1.2 | 81.4 $\pm$ 0.2 | 95.9 $\pm$ 0.3 | 80.9 $\pm$ 0.5 | 85.8 | 97.6 $\pm$ 0.6 | 66.5 $\pm$ 0.3 | 72.7 $\pm$ 0.6 | 74.4 $\pm$ 0.7 | 77.8 |
| Fisher(Rame et al., [2021](#bib.bib57 "")) | —— | —— | —— | —— | 86.9 | —— | —— | —— | —— | 76.2 |
| DDG (Zhang et al., [2021a](#bib.bib91 "")) | 88.9 ± 0.6 | 85.0 ± 1.9 | 97.2 ± 1.2 | 84.3 ± 0.7 | 88.9 | 99.1 ± 0.6 | 66.5 ± 0.3 | 73.3 ± 0.6 | 80.9 ± 0.6 | 80.0 |
| DANN+ELS | 87.8 $\pm$ 0.8 | 83.8 $\pm$ 1.6 | 97.1 $\pm$ 0.4 | 81.4 $\pm$ 1.3 | 87.5 | 99.1 $\pm$ 0.3 | 73.2 $\pm$ 1.1 | 73.8 $\pm$ 0.9 | 79.9 $\pm$ 0.9 | 81.5 |
| $\uparrow$ | 2.4 | 0.7 | 0.8 | 1.8 | 1.4 | 0.5 | 0 | 1 | 1.1 | 0.7 |

*Table 4: Accuracy ($\%$) on Office-Home for unsupervised DA (with ResNet-50 and ViT backbone). SDAT$+$ELS outperforms other SOTA DA techniques and improves SDAT consistently.*

| Method | Backbone | A-C | A-P | A-R | C-A | C-P | C-R | P-A | P-C | P-R | R-A | R-C | R-P | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50(He et al., [2016](#bib.bib30 "")) |  | 34.9 | 50.0 | 58.0 | 37.4 | 41.9 | 46.2 | 38.5 | 31.2 | 60.4 | 53.9 | 41.2 | 59.9 | 46.1 |
| DANN(Ganin et al., [2016](#bib.bib22 "")) |  | 45.6 | 59.3 | 70.1 | 47.0 | 58.5 | 60.9 | 46.1 | 43.7 | 68.5 | 63.2 | 51.8 | 76.8 | 57.6 |
| CDAN(Long et al., [2018](#bib.bib47 "")) |  | 49.0 | 69.3 | 74.5 | 54.4 | 66.0 | 68.4 | 55.6 | 48.3 | 75.9 | 68.4 | 55.4 | 80.5 | 63.8 |
| MMD(Zhang et al., [2019](#bib.bib95 "")) |  | 54.9 | 73.7 | 77.8 | 60.0 | 71.4 | 71.8 | 61.2 | 53.6 | 78.1 | 72.5 | 60.2 | 82.3 | 68.1 |
| f-DAL(Acuna et al., [2021](#bib.bib1 "")) |  | 56.7 | 77.0 | 81.1 | 63.1 | 72.2 | 75.9 | 64.5 | 54.4 | 81.0 | 72.3 | 58.4 | 83.7 | 70.0 |
| SRDC(Tang et al., [2020](#bib.bib72 "")) |  | 52.3 | 76.3 | 81.0 | 69.5 | 76.2 | 78.0 | 68.7 | 53.8 | 81.7 | 76.3 | 57.1 | 85.0 | 71.3 |
| SDAT(Rangwani et al., [2022](#bib.bib58 "")) |  | 57.8 | 77.4 | 82.2 | 66.5 | 76.6 | 76.2 | 63.3 | 57.0 | 82.2 | 75.3 | 62.6 | 85.2 | 71.8 |
| SDAT$+$ELS |  | 58.2 | 79.7 | 82.5 | 67.5 | 77.2 | 77.2 | 64.6 | 57.9 | 82.2 | 75.4 | 63.1 | 85.5 | 72.6 |
| $\uparrow$ | ResNet-50 | 0.4 | 2.3 | 0.3 | 1.0 | 0.6 | 1.0 | 1.3 | 0.9 | 0.0 | 0.1 | 0.5 | 0.3 | 0.8 |
| TVT(Yang et al., [2021](#bib.bib90 "")) |  | 74.9 | 86.6 | 89.5 | 82.8 | 87.9 | 88.3 | 79.8 | 71.9 | 90.1 | 85.5 | 74.6 | 90.6 | 83.6 |
| CDAN(Long et al., [2018](#bib.bib47 "")) |  | 62.6 | 82.9 | 87.2 | 79.2 | 84.9 | 87.1 | 77.9 | 63.3 | 88.7 | 83.1 | 63.5 | 90.8 | 79.3 |
| SDAT(Rangwani et al., [2022](#bib.bib58 "")) |  | 70.8 | 87.0 | 90.5 | 85.2 | 87.3 | 89.7 | 84.1 | 70.7 | 90.6 | 88.3 | 75.5 | 92.1 | 84.3 |
| SDAT$+$ELS | ViT | 72.1 | 87.3 | 90.6 | 85.2 | 88.1 | 89.7 | 84.1 | 70.7 | 90.8 | 88.4 | 76.5 | 92.1 | 84.6 |
| $\uparrow$ |  | 1.3 | 0.3 | 0.1 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.2 | 0.1 | 1.0 | 0.0 | 0.3 |

Continuously Indexed Domain Adaptation. We compare DANN+ELS with state-of-the-art continuously indexed domain adaptation methods. Table[5](#S4.T5 "Table 5 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") compares the accuracy of various methods. DANN shows an inferior performance to CIDA. However, with ELS, DANN+ELS boosts the generalization performance by a large margin and beats the SOTA method CIDA*(Wang et al., [2020](#bib.bib80 ""))*. We also visualize the classification results on Circle Dataset (See Appendix[C.1.1](#A3.SS1.SSS1 "C.1.1 Images Classification Datasets ‣ C.1 Dataset Details and Experimental Settings ‣ Appendix C Additional Experimental Setups ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") for dataset details). Figure[3](#S4.F3 "Figure 3 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") shows that the representative DA method (ADDA) performs poorly when asked to align domains with continuous indices. However, the proposed DANN+ELS can get a near-optimal decision boundary.

*Table 5: Rotating MNIST accuracy (%) at the source domain and each target domain. $X^{\circ}$ denotes the domain whose images are Rotating by $[X^{\circ},X^{\circ}+45^{\circ}]$.*

| Rotating MNIST | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Algorithm | $0^{\circ}$(Source) | 45∘ | 90∘ | 135∘ | 180∘ | 225∘ | 270∘ | 315∘ | Average |
| ERM(Vapnik, [1999](#bib.bib78 "")) | 99.2 | 79.7 | 26.8 | 31.6 | 35.1 | 37.0 | 28.6 | 76.2 | 45.0 |
| ADDA(Tzeng et al., [2017](#bib.bib77 "")) | 97.6 | 70.7 | 22.2 | 32.6 | 38.2 | 31.5 | 20.9 | 65.8 | 40.3 |
| DANN(Ganin et al., [2016](#bib.bib22 "")) | 98.4 | 81.4 | 38.9 | 35.4 | 40.0 | 43.4 | 48.8 | 77.3 | 52.1 |
| CIDA(Wang et al., [2020](#bib.bib80 "")) | 99.5 | 80.0 | 33.2 | 49.3 | 50.2 | 51.7 | 54.6 | 81.0 | 57.1 |
| DANN+ELS | 98.4 | 81.4 | 55.0 | 39.9 | 43.7 | 45.9 | 53.7 | 78.7 | 62.1 |
| $\uparrow$ | 0.0 | 0.0 | 16.1 | 4.5 | 3.7 | 2.5 | 4.9 | 1.4 | 10.0 |

<img src='x3.png' alt='Refer to caption' title='' width='461' height='346' />

*(a) Domains.*

<img src='x4.png' alt='Refer to caption' title='' width='461' height='346' />

*(b) Ground Truth.*

<img src='x5.png' alt='Refer to caption' title='' width='461' height='346' />

*(c) ADDA.*

<img src='x6.png' alt='Refer to caption' title='' width='461' height='346' />

*(d) DANN+ELS*

*Figure 3: Results on the Circle dataset with 30 domains. (a) shows domain index by color, (b) shows label index by color, where red dots and blue crosses are positive and negative data sample. Source domains contain the first 6 domains and others are target domains.*

Generalization results on other structural datasets and Sequential Datasets. Table[6](#S4.T6 "Table 6 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") shows the generalization results on NLP datasets, and Table[7](#S4.T7 "Table 7 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"),[D.1](#A4.SS1 "D.1 Additional Numerical Results ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") show the results on genomics datasets. DANN+ELS bring huge performance improvement on most of the evaluation metrics, e.g., $4.17\%$ test worst-group accuracy on CivilComments, $3.79\%$ test ID accuracy on RxRx1, and $3.13\%$ test accuracy on OGB-MolPCBA. Generalization results on sequential prediction tasks are shown in Table[D.1](#A4.SS1 "D.1 Additional Numerical Results ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Table[18](#A4.T18 "Table 18 ‣ D.1 Additional Numerical Results ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), where DANN works poorly but DANN+ELS brings consistent improvement and beats all baselines on the Spurious-Fourier dataset.

*Table 6: Domain generalization performance on neural language datasets. The backbone is DistillBERT-base-uncased and all results are reported over 3 random seed runs.*

Amazon-WildsAlgorithmVal Avg AccTest Avg AccVal 10% AccTest 10% AccVal Worst-group accTest Worst-group accERM*(Vapnik, [1999](#bib.bib78 ""))*72.7 $\pm$ 0.171.9 $\pm$ 0.155.2 $\pm$ 0.753.8 $\pm$ 0.820.3 $\pm$ 0.14.2 $\pm$ 0.2Group DRO*(Sagawa et al., [2019](#bib.bib62 ""))*70.7 $\pm$ 0.670.0 $\pm$ 0.654.7 $\pm$ 0.053.3 $\pm$ 0.054.2 $\pm$ 0.36.3 $\pm$ 0.2CORAL*(Sun \& Saenko, [2016](#bib.bib70 ""))*72.0 $\pm$ 0.371.1 $\pm$ 0.354.7 $\pm$ 0.052.9 $\pm$ 0.830.0 $\pm$ 0.26.1 $\pm$ 0.1IRM*(Arjovsky et al., [2019](#bib.bib6 ""))*71.5 $\pm$ 0.370.5 $\pm$ 0.354.2 $\pm$ 0.852.4 $\pm$ 0.832.2 $\pm$ 0.85.3 $\pm$ 0.2Reweight69.1 $\pm$ 0.568.6 $\pm$ 0.652.1 $\pm$ 0.252.0 $\pm$ 0.034.9 $\pm$ 1.29.1 $\pm$ 0.4DANN*(Ganin et al., [2016](#bib.bib22 ""))*72.1 $\pm$ 0.271.3 $\pm$ 0.154.6 $\pm$ 0.052.9 $\pm$ 0.64.4 $\pm$ 1.38.0 $\pm$ 0.0DANN+ELS72.3 $\pm$ 0.171.5 $\pm$ 0.154.7 $\pm$ 0.053.8 ± 0.04.9 ± 0.69.4 ± 0.0$\uparrow$0.20.20.10.90.51.4

CivilComments-WildsAlgorithmVal Avg AccVal Worst-Group AccTest Avg AccTest Worst-Group AccGroup DRO*(Sagawa et al., [2019](#bib.bib62 ""))*90.4 $\pm$ 0.465.0 $\pm$ 3.890.2 $\pm$ 0.369.1 $\pm$ 1.8Reweighted90.0 $\pm$ 0.763.7 $\pm$ 2.789.8 $\pm$ 0.866.6 $\pm$ 1.6IRM*(Arjovsky et al., [2019](#bib.bib6 ""))*89.0 $\pm$ 0.765.9 $\pm$ 2.888.8 $\pm$ 0.766.3 $\pm$ 2.1ERM*(Vapnik, [1999](#bib.bib78 ""))*92.3 $\pm$ 0.250.5 $\pm$ 1.992.2 $\pm$ 0.156.0 $\pm$ 3.6DANN*(Ganin et al., [2016](#bib.bib22 ""))*87.0 ± 0.364.0 ± 2.087.0 ± 0.361.7 ± 2.2DANN+ELS88.5 ± 0.465.9 ± 1.188.4 ± 0.466.0 ± 2.2$\uparrow$1.41.91.44.3

*Table 7: Domain generalization performance on genomics dataset, RxRx1.*

RxRx1-WildsAlgorithmVal AccTest ID AccTest AccVal Worst-Group AccTest ID Worst-Group AccTest Worst-Group AccERM*(Vapnik, [1999](#bib.bib78 ""))*19.4 ± 0.235.9 ± 0.429.9 ± 0.4———Group DRO*(Sagawa et al., [2019](#bib.bib62 ""))*15.2 ± 0.128.1 ± 0.323.0 ± 0.3———IRM*(Arjovsky et al., [2019](#bib.bib6 ""))*5.6 ± 0.49.9 ± 1.48.2 ± 1.10.8 ± 0.21.9 ± 0.41.5 ± 0.2DANN*(Ganin et al., [2016](#bib.bib22 ""))*12.7 ± 0.222.9 ± 0.119.2 ± 0.11.0 ± 0.14.6 ± 0.43.6 ± 0.0DANN+ELS14.1 ± 0.126.7 ± 0.121.2 ± 0.21.1 ± 0.17.2 ± 0.34.2 ± 0.1$\uparrow$1.43.820.12.60.6

<img src='x7.png' alt='Refer to caption' title='' width='461' height='346' />

<img src='x8.png' alt='Refer to caption' title='' width='461' height='368' />

<img src='x9.png' alt='Refer to caption' title='' width='461' height='346' />

*Figure 4: (a) Generalization performance of DANN+ELS compared to DANN with partial correct environment label on the PACS dataset ($P$ as target domain). (b) The best $\gamma$ for each dataset. Civil is the CivilComments dataset and OGB is the OGB-MolPCBA dataset. (c) Average generalization accuracy on the PACS dataset with different smoothing policies.*

### 4.2 Interpretation and Analysis

To choose the best $\gamma$.Figure[4](#S4.F4 "Figure 4 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") visualizes the best $\gamma$ values in our experiments. For datasets like PACS and VLCS, where each domain will be set as a target domain respectively and has one best $\gamma$, we calculate the mean and standard deviation of all these $\gamma$ values. Our main observation is that, as the number of domains increases, the optimal $\gamma$ will also decrease, which is intuitive because more domains mean that the discriminator is more likely to overfit and thus needs a lower $\gamma$ to solve the problem. An interesting thing is that in Figure[4](#S4.F4 "Figure 4 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), PACS and VLCS both have $4$ domains, but VLCS needs a higher $\gamma$. Figure[6](#A4.F6 "Figure 6 ‣ D.2 Additional Analysis and Interpretation ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") shows that images from different domains in PACS are of great visual difference and can be easily discriminated. In contrast, domains in VLCS do not show significant visual differences, and it is hard to discriminate which domain one image belongs to. The discrimination difficulty caused by this inter-domain distinction is another important factor affecting the selection of $\gamma$.

Annealing $\gamma$. To achieve better generalization performance and avoid troublesome parametric searches, we propose to gradually decrease $\gamma$ as training progresses, specifically, $\gamma\=1.0-\frac{M-1}{M}\frac{t}{T}$, where $t,T$ are the current training step and the total training steps. Figure[4](#S4.F4 "Figure 4 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") shows that annealing $\gamma$ achieves a comparable or even better generalization performance than fine-grained searched $\gamma$.

<img src='x10.png' alt='Refer to caption' title='' width='461' height='352' />

*(a) Classification loss.*

<img src='x11.png' alt='Refer to caption' title='' width='461' height='357' />

*(b) Avg accuracy of source domains.*

<img src='x12.png' alt='Refer to caption' title='' width='461' height='353' />

*(c) Acc on the target domain.*

*Figure 5: Training statistics on PACS datasets. Alternating GD with $n_{d}\=5,n_{e}\=1$ is used. All other parameters setting are the same and only on the default hyperparameters
and without the fine-grained parametric search.*

Empirical Verification of our theoretical results. We use the PACS dataset as an example to empirically support our theoretical results, namely verifying the benefits to convergence, training stability, and generalization results. In Figure[5](#S4.F5 "Figure 5 ‣ 4.2 Interpretation and Analysis ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), ’A’ is set as the target domain and other domains as sources. Considering ELS, we can see that in all the experimental results, DANN+ELS with appropriate $\gamma$ attains high training stability, faster and stable convergence, and better performance compared to DANN. In comparison, the training dynamics of native DANN is highly oscillatory, especially in the middle and late stages of training.

5 Related Works
---------------

Label Smoothing and Analysis is a technique from the 1980s, and independently re-discovered by*(Szegedy et al., [2016](#bib.bib71 ""))*. Recently, label smoothing is shown to reduce the vulnerability of neural networks*(Warde-Farley \& Goodfellow, [2016](#bib.bib82 ""))* and reduce the risk of adversarial examples in GANs*(Salimans et al., [2016](#bib.bib63 ""))*. Several works seek to theoretically or empirically study the effect of label smoothing.*(Chen et al., [2020](#bib.bib13 ""))* focus on studying the minimizer of the training error and finding the optimal smoothing parameter.*(Xu et al., [2020](#bib.bib88 ""))* analyzes the convergence behaviors of stochastic gradient descent with label smoothing. However, as far as we know, no study focuses on the effect of label smoothing on the convergence speed and training stability of DAT.

Domain Adversarial Training *(Ganin et al., [2016](#bib.bib22 ""))* using a domain discriminator to distinguish the source and target domains and the gradients of the discriminator to the encoder are reversed by the Gradient Reversal layer (GRL), which achieves the goal of learning domain invariant features.*(Schoenauer-Sebag et al., [2019](#bib.bib66 ""); Zhao et al., [2018](#bib.bib96 ""))* extend generalization bounds in DANN*(Ganin et al., [2016](#bib.bib22 ""))* to multi-source domains and propose multisource domain adversarial networks.*(Acuna et al., [2022](#bib.bib2 ""))* interprets the DAT framework through the lens of game theory and proposes to replace gradient descent with high-order ODE solvers.*(Rangwani et al., [2022](#bib.bib58 ""))* finds that enforcing the smoothness of the classifier leads to better generalization on the target domain and presents Smooth Domain Adversarial Training (SDAT). The proposed method is orthogonal to existing DAT methods and yields excellent optimization properties theoretically and empirically.

For space limit, the related works about domain adaptation, domain generalization, and adversarial Training in GANs are in the appendix.

6 Conclusion
------------

In this work, we propose a simple approach, i.e., ELS, to optimize the training process of DAT methods from an environment label design perspective, which is orthogonal to most existing DAT methods. Incorporating ELS into DAT methods is empirically and theoretically shown to be capable of improving robustness to noisy environment labels, converge faster, attain more stable training and better generalization performance. As far as we know, our work takes a first step towards utilizing and understanding label smoothing for environmental labels. Although ELS is designed for DAT methods, reducing the effect of environment label noise and a soft environment partition may benefit all DG/DA methods, which is a promising future direction.

References
----------

* Acuna et al. (2021)David Acuna, Guojun Zhang, Marc T Law, and Sanja Fidler.f-domain adversarial learning: Theory and algorithms.In *International Conference on Machine Learning*, pp. 66–75.
PMLR, 2021.
* Acuna et al. (2022)David Acuna, Marc T Law, Guojun Zhang, and Sanja Fidler.Domain adversarial training: A game perspective.*ICLR*, 2022.
* Albuquerque et al. (2019)Isabela Albuquerque, João Monteiro, Mohammad Darvishi, Tiago H Falk, and
Ioannis Mitliagkas.Generalizing to unseen domains via distribution matching.*arXiv preprint arXiv:1911.00804*, 2019.
* Arjovsky \& Bottou (2017)Martin Arjovsky and Léon Bottou.Towards principled methods for training generative adversarial
networks.*arXiv preprint arXiv:1701.04862*, 2017.
* Arjovsky et al. (2017)Martin Arjovsky, Soumith Chintala, and Léon Bottou.Wasserstein generative adversarial networks.In *International conference on machine learning*, pp. 214–223. PMLR, 2017.
* Arjovsky et al. (2019)Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz.Invariant risk minimization.*arXiv preprint arXiv:1907.02893*, 2019.
* Arora et al. (2017)Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang.Generalization and equilibrium in generative adversarial nets (gans).In *International Conference on Machine Learning*, pp. 224–232. PMLR, 2017.
* Ben-David et al. (2006)Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira.Analysis of representations for domain adaptation.In *NIPS*, 2006.
* Ben-David et al. (2010)Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and
Jennifer Wortman Vaughan.A theory of learning from different domains.*Machine learning*, 2010.
* Bertsekas (1999)Dimitri P Bertsekas.Nonlinear programming.In *thena scientific Belmont*, 1999.
* Blanchard et al. (2021)Gilles Blanchard, Aniket Anand Deshmukh, Ürün Dogan, Gyemin Lee, and
Clayton Scott.Domain generalization by marginal transfer learning.*J. Mach. Learn. Res.*, 2021.
* Brock et al. (2019)Andrew Brock, Jeff Donahue, and Karen Simonyan.Large scale gan training for high fidelity natural image synthesis.*International Conference on Learning Representations (ICLR)*,
2019.
* Chen et al. (2020)Blair Chen, Liu Ziyin, Zihao Wang, and Paul Pu Liang.An investigation of how label smoothing affects generalization.*arXiv preprint arXiv:2010.12648*, 2020.
* Chen et al. (2021)Peixian Chen, Pingyang Dai, Jianzhuang Liu, Feng Zheng, Qi Tian, and Rongrong
Ji.Dual distribution alignment network for generalizable person
re-identification.*AAAI Conference on Artificial Intelligence*, 2021.
* Chen et al. (2018)Xu Chen, Jiang Wang, and Hao Ge.Training generative adversarial networks via primal-dual subgradient
methods: a lagrangian perspective on gan.*arXiv preprint arXiv:1802.01765*, 2018.
* Choi et al. (2021)Seokeon Choi, Taekyung Kim, Minki Jeong, Hyoungseob Park, and Changick Kim.Meta batch-instance normalization for generalizable person
re-identification.In *Computer Vision and Pattern Recognition (CVPR)*, 2021.
* Creager et al. (2021)Elliot Creager, Jörn-Henrik Jacobsen, and Richard Zemel.Environment inference for invariant learning.In *ICML*, 2021.
* Deng et al. (2009)Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.Imagenet: A large-scale hierarchical image database.In *Computer Vision and Pattern Recognition (CVPR)*, 2009.
* Farnia \& Ozdaglar (2020)Farzan Farnia and Asuman Ozdaglar.Do GANs always have Nash equilibria?In Hal Daumé III and Aarti Singh (eds.), *Proceedings of the
37th International Conference on Machine Learning*, volume 119 of*Proceedings of Machine Learning Research*, pp. 3029–3039. PMLR,
13–18 Jul 2020.URL <https://proceedings.mlr.press/v119/farnia20a.html>.
* Gagnon-Audet et al. (2022)Jean-Christophe Gagnon-Audet, Kartik Ahuja, Mohammad-Javad Darvishi-Bayazi,
Guillaume Dumas, and Irina Rish.Woods: Benchmarks for out-of-distribution generalization in time
series tasks.*arXiv preprint arXiv:2203.09978*, 2022.
* Gan et al. (2017)Zhe Gan, Liqun Chen, Weiyao Wang, Yuchen Pu, Yizhe Zhang, Hao Liu, Chunyuan Li,
and Lawrence Carin.Triangle generative adversarial networks.*Advances in neural information processing systems*, 30, 2017.
* Ganin et al. (2016)Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo
Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky.Domain-adversarial training of neural networks.*The journal of machine learning research*, 2016.
* Geirhos et al. (2020)Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel,
Wieland Brendel, Matthias Bethge, and Felix A. Wichmann.Shortcut learning in deep neural networks.*Nature Machine Intelligence*, 2020.
* Ghifary et al. (2015)Muhammad Ghifary, W Bastiaan Kleijn, Mengjie Zhang, and David Balduzzi.Domain generalization for object recognition with multi-task
autoencoders.In *ICCV*, 2015.
* Gidel et al. (2018)Gauthier Gidel, Hugo Berard, Gaëtan Vignoud, Pascal Vincent, and Simon
Lacoste-Julien.A variational inequality perspective on generative adversarial
networks.*arXiv preprint arXiv:1802.10551*, 2018.
* Goodfellow (2016)Ian Goodfellow.Nips 2016 tutorial: Generative adversarial networks.*arXiv preprint arXiv:1701.00160*, 2016.
* Goodfellow et al. (2014)Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozair, Aaron Courville, and Yoshua Bengio.Generative adversarial nets.*Advances in neural information processing systems*, 27, 2014.
* Gray et al. (2007)D. Gray, S. Brennan, and H. Tao.Evaluating Appearance Models for Recognition, Reacquisition, and
Tracking.*Proc. IEEE International Workshop on Performance Evaluation for
Tracking and Surveillance (PETS)*, 2007.
* Gulrajani \& Lopez-Paz (2021)Ishaan Gulrajani and David Lopez-Paz.In search of lost domain generalization.In *ICLR*, 2021.
* He et al. (2016)Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition.In *Proceedings of the IEEE conference on computer vision and
pattern recognition*, pp. 770–778, 2016.
* Hirzer et al. (2011)Martin Hirzer, Csaba Beleznai, Peter M Roth, and Horst Bischof.Person re-identification by descriptive and discriminative
classification.In *Scandinavian Conference on Image Analysis*, 2011.
* Jaderberg et al. (2015)Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al.Spatial transformer networks.*Advances in neural information processing systems*, 28, 2015.
* Jenni \& Favaro (2019)Simon Jenni and Paolo Favaro.On stabilizing generative adversarial training with noise.In *Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition*, pp. 12145–12153, 2019.
* Jia et al. (2019)Jieru Jia, Qiuqi Ruan, and Timothy M Hospedales.Frustratingly easy person re-identification: Generalizing person
re-id in practice.*arXiv preprint arXiv:1905.03422*, 2019.
* Jin et al. (2020)Ying Jin, Ximei Wang, Mingsheng Long, and Jianmin Wang.Less confusion more transferable: Minimum class confusion for
versatile domain adaptation.In *ECCV*, 2020.
* Khalil. (1996)Hassan K Khalil.*Non-linear Systems.*Prentice-Hall, New Jersey,, 1996.
* Kim et al. (2019)Youngdong Kim, Junho Yim, Juseung Yun, and Junmo Kim.Nlnl: Negative learning for noisy labels.In *Proceedings of the IEEE/CVF International Conference on
Computer Vision*, pp. 101–110, 2019.
* Koh et al. (2021)Pang Wei Koh, Shiori Sagawa, Sang Michael Xie, Marvin Zhang, Akshay
Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena
Gao, Tony Lee, et al.Wilds: A benchmark of in-the-wild distribution shifts.In *ICML*, 2021.
* Krueger et al. (2021)David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan
Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville.Out-of-distribution generalization via risk extrapolation (rex).In *ICML*, 2021.
* Li et al. (2017a)Chun-Liang Li, Wei-Cheng Chang, Yu Cheng, Yiming Yang, and Barnabás
Póczos.Mmd gan: Towards deeper understanding of moment matching network.*Advances in neural information processing systems*, 30,
2017a.
* Li et al. (2017b)Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales.Deeper, broader and artier domain generalization.In *ICCV*, 2017b.
* Li et al. (2018a)Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy Hospedales.Learning to generalize: Meta-learning for domain generalization.In *AAAI*, 2018a.
* Li \& Wang (2013)Wei Li and Xiaogang Wang.Locally aligned feature transforms across views.In *Computer Vision and Pattern Recognition (CVPR)*, June 2013.
* Li et al. (2014)Wei Li, Rui Zhao, Tong Xiao, and Xiaogang Wang.Deepreid: Deep filter pairing neural network for person
re-identification.In *Computer Vision and Pattern Recognition (CVPR)*, June 2014.
* Li et al. (2018b)Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, and
Dacheng Tao.Deep domain generalization via conditional invariant adversarial
networks.In *ECCV*, 2018b.
* Liu et al. (2012)Chunxiao Liu, Shaogang Gong, Chen Change Loy, and Xinggang Lin.Person re-identification: What features are important?In *European Conference on Computer Vision (ECCV)*. Springer,
2012.
* Long et al. (2018)Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan.Conditional adversarial domain adaptation.*Advances in neural information processing systems*, 31, 2018.
* Mescheder et al. (2017)Lars Mescheder, Sebastian Nowozin, and Andreas Geiger.The numerics of gans.*Advances in neural information processing systems*, 30, 2017.
* Mescheder et al. (2018)Lars Mescheder, Andreas Geiger, and Sebastian Nowozin.Which training methods for gans do actually converge?In *ICML*, 2018.
* Muandet et al. (2013)K. Muandet, D. Balduzzi, and B. Schölkopf.Domain generalization via invariant feature representation.In *ICML*, 2013.
* Nagarajan \& Kolter (2017)Vaishnavh Nagarajan and J Zico Kolter.Gradient descent gan optimization is locally stable.*Advances in neural information processing systems*, 30, 2017.
* Nguyen et al. (2017)Tu Nguyen, Trung Le, Hung Vu, and Dinh Phung.Dual discriminator generative adversarial nets.*Advances in neural information processing systems*, 30, 2017.
* Nie \& Patel (2020)Weili Nie and Ankit B Patel.Towards a better understanding and regularization of gan training
dynamics.In *Uncertainty in Artificial Intelligence*, pp. 281–291.
PMLR, 2020.
* Nowozin et al. (2016)Sebastian Nowozin, Botond Cseke, and Ryota Tomioka.f-gan: Training generative neural samplers using variational
divergence minimization.*Advances in neural information processing systems*, 29, 2016.
* Pezeshki et al. (2021)Mohammad Pezeshki, Oumar Kaba, Yoshua Bengio, Aaron C Courville, Doina Precup,
and Guillaume Lajoie.Gradient starvation: A learning proclivity in neural networks.*Advances in Neural Information Processing Systems*, 34, 2021.
* Pu et al. (2018)Yunchen Pu, Shuyang Dai, Zhe Gan, Weiyao Wang, Guoyin Wang, Yizhe Zhang,
Ricardo Henao, and Lawrence Carin Duke.Jointgan: Multi-domain joint distribution learning with generative
adversarial nets.In *International Conference on Machine Learning*, pp. 4151–4160. PMLR, 2018.
* Rame et al. (2021)Alexandre Rame, Corentin Dancette, and Matthieu Cord.Fishr: Invariant gradient variances for out-of-distribution
generalization.*arXiv preprint arXiv:2109.02934*, 2021.
* Rangwani et al. (2022)Harsh Rangwani, Sumukh K Aithal, Mayank Mishra, Arihant Jain, and R Venkatesh
Babu.A closer look at smoothness in domain adversarial training.*ICML*, 2022.
* Roth et al. (2017)Kevin Roth, Aurélien Lucchi, Sebastian Nowozin, and Thomas Hofmann.Stabilizing training of generative adversarial networks through
regularization.In *NIPS*, 2017.
* Russakovsky et al. (2015)Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma,
Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al.Imagenet large scale visual recognition challenge.*IJCV*, 2015.
* Saenko et al. (2010)Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell.Adapting visual category models to new domains.In *European conference on computer vision*, pp. 213–226.
Springer, 2010.
* Sagawa et al. (2019)Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang.Distributionally robust neural networks for group shifts: On the
importance of regularization for worst-case generalization.*arXiv preprint arXiv:1911.08731*, 2019.
* Salimans et al. (2016)Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and
Xi Chen.Improved techniques for training gans.*Advances in neural information processing systems*, 29, 2016.
* Sandler et al. (2018)Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh
Chen.Mobilenetv2: Inverted residuals and linear bottlenecks.In *Computer Vision and Pattern Recognition (CVPR)*, 2018.
* Schirrmeister et al. (2017)Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas Dominique Josef
Fiederer, Martin Glasstetter, Katharina Eggensperger, Michael Tangermann,
Frank Hutter, Wolfram Burgard, and Tonio Ball.Deep learning with convolutional neural networks for eeg decoding and
visualization.*Human brain mapping*, 38(11):5391–5420,
2017.
* Schoenauer-Sebag et al. (2019)Alice Schoenauer-Sebag, Louise Heinrich, Marc Schoenauer, Michele Sebag, Lani F
Wu, and Steve J Altschuler.Multi-domain adversarial learning.*arXiv preprint arXiv:1903.09239*, 2019.
* Schäfer et al. (2019)Florian Schäfer, Hongkai Zheng, and Anima Anandkumar.Implicit competitive regularization in gans, 2019.
* Sønderby et al. (2016)Casper Kaae Sønderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc
Huszár.Amortised map inference for image super-resolution.*arXiv preprint arXiv:1610.04490*, 2016.
* Song et al. (2019)Jifei Song, Yongxin Yang, Yi-Zhe Song, Tao Xiang, and Timothy M. Hospedales.Generalizable person re-identification by domain-invariant mapping
network.In *Computer Vision and Pattern Recognition (CVPR)*, June 2019.
* Sun \& Saenko (2016)Baochen Sun and Kate Saenko.Deep coral: Correlation alignment for deep domain adaptation.In *European conference on computer vision*, pp. 443–450.
Springer, 2016.
* Szegedy et al. (2016)Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew
Wojna.Rethinking the inception architecture for computer vision.In *Proceedings of the IEEE conference on computer vision and
pattern recognition*, pp. 2818–2826, 2016.
* Tang et al. (2020)Hui Tang, Ke Chen, and Kui Jia.Unsupervised domain adaptation via structurally regularized deep
clustering.In *Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition*, pp. 8725–8735, 2020.
* Tao et al. (2018)Chenyang Tao, Liqun Chen, Ricardo Henao, Jianfeng Feng, and Lawrence Carin
Duke.Chi-square generative adversarial network.In *International conference on machine learning*, pp. 4887–4896. PMLR, 2018.
* Thanh-Tung et al. (2019)Hoang Thanh-Tung, Truyen Tran, and Svetha Venkatesh.Improving generalization and stability of generative adversarial
networks.*arXiv preprint arXiv:1902.03984*, 2019.
* Torralba \& Efros (2011)Antonio Torralba and Alexei A Efros.Unbiased look at dataset bias.In *CVPR*, 2011.
* Trung Le et al. (2019)Quan Hoang Trung Le, Hung Vu, Tu Dinh Nguyen, Hung Bui, and Dinh Phung.Learning generative adversarial networks from multiple data sources.In *Proceedings of the 28th International Joint Conference on
Artificial Intelligence*, 2019.
* Tzeng et al. (2017)Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell.Adversarial discriminative domain adaptation.In *Proceedings of the IEEE conference on computer vision and
pattern recognition*, pp. 7167–7176, 2017.
* Vapnik (1999)Vladimir Vapnik.*The nature of statistical learning theory*.Springer science \& business media, 1999.
* Venkateswara et al. (2017)Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman
Panchanathan.Deep hashing network for unsupervised domain adaptation.In *Proceedings of the IEEE conference on computer vision and
pattern recognition*, pp. 5018–5027, 2017.
* Wang et al. (2020)Hao Wang, Hao He, and Dina Katabi.Continuously indexed domain adaptation.*ICML*, 2020.
* (81)Jindong Wang and Wenxin Hou.Deepda: Deep domain adaptation toolkit.<https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA>.
* Warde-Farley \& Goodfellow (2016)David Warde-Farley and Ian Goodfellow.11 adversarial perturbations of deep neural networks.*Perturbations, Optimization, and Statistics*, 311:5,
2016.
* Wei et al. (2022)Jiaheng Wei, Hangyu Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama, and Yang
Liu.To smooth or not? when label smoothing meets noisy labels.*ICML*, 2022.
* Wei-Shi et al. (2009)Zheng Wei-Shi, Gong Shaogang, and Xiang Tao.Associating groups of people.In *British Machine Vision Conference (BMVC)*, 2009.
* Wolf et al. (2019)Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue,
Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
et al.Huggingface’s transformers: State-of-the-art natural language
processing.*arXiv preprint arXiv:1910.03771*, 2019.
* Xiao et al. (2016)Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, and Xiaogang Wang.End-to-end deep learning for person search.*arXiv preprint arXiv:1604.01850*, 2(2), 2016.
* Xu et al. (2018)Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.How powerful are graph neural networks?*arXiv preprint arXiv:1810.00826*, 2018.
* Xu et al. (2020)Yi Xu, Yuanhong Xu, Qi Qian, Hao Li, and Rong Jin.Towards understanding label smoothing.*arXiv preprint arXiv:2006.11653*, 2020.
* Yadav et al. (2017)Abhay Yadav, Sohil Shah, Zheng Xu, David Jacobs, and Tom Goldstein.Stabilizing adversarial nets with prediction methods.*arXiv preprint arXiv:1705.07364*, 2017.
* Yang et al. (2021)Jinyu Yang, Jingjing Liu, Ning Xu, and Junzhou Huang.Tvt: Transferable vision transformer for unsupervised domain
adaptation.*arXiv preprint arXiv:2108.05988*, 2021.
* Zhang et al. (2021a)Hanlin Zhang, Yi-Fan Zhang, Weiyang Liu, Adrian Weller, Bernhard Schölkopf,
and Eric P Xing.Towards principled disentanglement for domain generalization.*arXiv preprint arXiv:2111.13839*, 2021a.
* Zhang et al. (2021b)Marvin Zhang, Henrik Marklund, Nikita Dhawan, Abhishek Gupta, Sergey Levine,
and Chelsea Finn.Adaptive risk minimization: Learning to adapt to domain shift.*NeurIPS*, 2021b.
* Zhang et al. (2021c)Yi-Fan Zhang, Zhang Zhang, Da Li, Zhen Jia, Liang Wang, and Tieniu Tan.Learning domain invariant representations for generalizable person
re-identification.*arXiv preprint arXiv:2103.15890*, 2021c.
* Zhang et al. (2022)YiFan Zhang, Feng Li, Zhang Zhang, Liang Wang, Dacheng Tao, and Tieniu Tan.Generalizable person re-identification without demographics, 2022.URL [https://openreview.net/forum?id\=VNdFPD5wqjh](https://openreview.net/forum?id=VNdFPD5wqjh "").
* Zhang et al. (2019)Yuchen Zhang, Tianle Liu, Mingsheng Long, and Michael Jordan.Bridging theory and algorithm for domain adaptation.In *International Conference on Machine Learning*, pp. 7404–7413. PMLR, 2019.
* Zhao et al. (2018)Han Zhao, Shanghang Zhang, Guanhang Wu, José MF Moura, Joao P Costeira, and
Geoffrey J Gordon.Adversarial multiple source domain adaptation.In *NeurIPS*, 2018.
* Zheng et al. (2015)Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian.Scalable person re-identification: A benchmark.In *International Conference on Computer Vision (ICCV)*,
December 2015.
* Zheng et al. (2017)Zhedong Zheng, Liang Zheng, and Yi Yang.Unlabeled samples generated by gan improve the person
re-identification baseline in vitro.In *International Conference on Computer Vision (ICCV)*, Oct
2017.

Appendix

###### Contents

Appendix A Proofs of Theoretical Statements
-------------------------------------------

*Table 8: Notations.*

| Symbol | Description |
| --- | --- |
| $\mathcal{D}_{S},\mathcal{D}_{T},\mathcal{D}_{i}$ | Distributions for source domain, target domain, and domain $i$. |
| $\hat{\mathcal{D}}_{S},\hat{\mathcal{D}}_{T},\hat{\mathcal{D}}_{i}$ | Empirical distributions for source domain, target domain, and domain $i$. |
| $p_{s},p_{t},p_{i}$ | Density functions for source domain, target domain, and domain $i$. |
| $\mathbf{x}_{s},\mathbf{x}_{t},\mathbf{x}_{i}$ | Data samples from source domain, target domain, and domain $i$. |
| $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T},\mathcal{D}^{z}_{i}$ | | Feature distributions of $\mathcal{D}_{S},\mathcal{D}_{T},\mathcal{D}_{i}$ respectively, | | --- | | which is also termed $g\circ\mathcal{D}_{S},g\circ\mathcal{D}_{T},g\circ\mathcal{D}_{i}$. | |
| $p^{z}_{s},p^{z}_{t},p^{z}_{i}$ | Density functions for $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T},\mathcal{D}^{z}_{i}$ respectively. |
| $\mathbf{z}_{s},\mathbf{z}_{t},\mathbf{z}_{i}$ | Data samples from $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T},\mathcal{D}^{z}_{i}$. |
| $\mathcal{H},\hat{\mathcal{H}},\mathcal{G}$ | Support sets for hypothesis, discriminator, and feature encoder. |
| $h,\hat{h},\hat{h}^{*},g$ | Hypothesis, discriminator, the optimal discriminator, and feature encoder. |
| $M,n_{i}$ | Number of training distributions, number of data samples in $\mathcal{D}_{i}$. |
| $\gamma$ | Hyper-parameter for the environment label smoothing. |
| $d_{\mathcal{H}},\hat{d}_{\mathcal{H}}$ | $\mathcal{H}$-divergence and Empirical $\mathcal{H}$-divergence. |

The commonly used notations and their corresponding descriptions are concluded in Table[8](#A1.T8 "Table 8 ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing").

### A.1 Connect Environment Label Smoothing to JS Divergence Minimization

To complete the proofs, we begin by introducing some necessary definitions and assumptions.

###### Definition 1.

($\mathcal{H}$-divergence*(Ben-David et al., [2006](#bib.bib8 ""))*). Given two domain distributions $\mathcal{D}_{S},\mathcal{D}_{T}$ over $X$, and a hypothesis class $\mathcal{H}$, the $\mathcal{H}$-divergence between $\mathcal{D}_{S},\mathcal{D}_{T}$ is

|  | $d_{\mathcal{H}}(\mathcal{D}_{S},\mathcal{D}_{T})\=2\sup_{h\in\mathcal{H}}\left\mid\mathbb{E}_{\mathbf{x}\sim\mathcal{D}_{S}}[h(\mathbf{x})\=1]-\mathbb{E}_{\mathbf{x}\sim\mathcal{D}_{T}}[h(\mathbf{x})\=1]\right\mid$ |  | (5) |
| --- | --- | --- | --- |

###### Definition 2.

(Empirical $\mathcal{H}$-divergence*(Ben-David et al., [2006](#bib.bib8 ""))*.)
For an symmetric hypothesis class $\mathcal{H}$, one can compute the empirical $\mathcal{H}$-divergence between two empirical distributions $\hat{\mathcal{D}}_{S}$ and $\hat{\mathcal{D}}_{T}$ by computing

|  | $\hat{d}_{\mathcal{H}}(\hat{\mathcal{D}}_{S},\hat{\mathcal{D}}_{T})\=2\left(1-\min_{h\in\mathcal{H}}\left[\frac{1}{m}\sum_{i\=1}^{m}I[h(\mathbf{x}_{i})\=0]+\frac{1}{n}\sum_{i\=1}^{n}I[h(\mathbf{x}_{i})\=1]\right]\right),$ |  | (6) |
| --- | --- | --- | --- |

where $m,n$ is the number of data samples of $\hat{\mathcal{D}}_{S}$ and $\hat{\mathcal{D}}_{T}$ respectively and $I[a]$ is the indicator function which is 1 if predicate $a$ is true, and 0 otherwise.

Vanilla DANN estimating the “min” part of Equ. ([6](#A1.E6 "In Definition 2. ‣ A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) by a domain discriminator, that models the probability that a given input is from the source domain or the target domain. Specially, let the hypothesis $h$ be the composition of $h\=\hat{h}\circ g$, where $\hat{h}\in\hat{\mathcal{H}}$ is a additional hypothesis and $g\in\mathcal{G}$ pushes forward the data samples to a representation space $\mathcal{Z}$. DANN*(Ben-David et al., [2006](#bib.bib8 ""))* seeks to approximate the $\mathcal{H}$-divergence of Equ. ([6](#A1.E6 "In Definition 2. ‣ A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) by

|  | $\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g}(\mathcal{D}_{S},\mathcal{D}_{T})\=\max_{\hat{h}\in\hat{\mathcal{H}}}\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\log\hat{h}\circ g(\mathbf{x}_{s})+\mathbb{E}_{\mathbf{x}_{t}\sim\mathcal{D}_{T}}\log\left(1-\hat{h}\circ g(\mathbf{x}_{t})\right),$ |  | (7) |
| --- | --- | --- | --- |

where the sigmoid activate function is ignored for simplicity, $\hat{h}\circ g(\mathbf{x})$ is the prediction probability that $\mathbf{x}$ is belonged to $\mathcal{D}_{S}$ and $1-\hat{h}\circ g(\mathbf{x})$ is the prediction probability that $\mathbf{x}$ is belonged to $\mathcal{D}_{T}$. Applying environment label smoothing, the target can be reformulated to

|  | $\displaystyle\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})\=\max_{\hat{h}\in\hat{\mathcal{H}}}\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}$ | $\displaystyle\left[\gamma\log\hat{h}\circ g(\mathbf{x}_{s})+(1-\gamma)\log\left(1-\hat{h}\circ g(\mathbf{x}_{s})\right)\right]+$ |  | (8) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\mathbb{E}_{\mathbf{x}_{t}\sim\mathcal{D}_{T}}\left[(1-\gamma)\log\hat{h}\circ g(\mathbf{x}_{t})+\gamma\log\left(1-\hat{h}\circ g(\mathbf{x}_{t})\right)\right]$ | | |

When $\gamma\in{0,1}$, Equ. ([8](#A1.E8 "In A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is equal to Equ. ([7](#A1.E7 "In A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) and no environment label smoothing is applied. Then we prove the proposition[1](#Thmprop1 "Proposition 1. ‣ 3.1 Divergence Minimization Interpretation ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")

Proposition [1](#Thmprop1 "Proposition 1. ‣ 3.1 Divergence Minimization Interpretation ‣ 3 Theoretical validation ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"). Suppose $\hat{h}$ the optimal domain classifier with no constraint and mixed distributions $\left{\begin{array}[]{c}\mathcal{D}_{S^{\prime}}\=\gamma\mathcal{D}_{S}+(1-\gamma)\mathcal{D}_{T}\\
\mathcal{D}_{T^{\prime}}\=\gamma\mathcal{D}_{T}+(1-\gamma)\mathcal{D}_{S}\\
\end{array}\right.$ with hyper-parameter $\gamma$, then $\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})\=2D_{JS}(\mathcal{D}_{S^{\prime}}||\mathcal{D}_{T^{\prime}})-2\log 2$, where $D_{JS}$ is the Jensen-Shanon (JS) divergence.

###### Proof.

Denote the injected source/target density as $p_{s}^{z}:\=g\circ p_{s},p_{t}^{z}:\=g\circ p_{t}$, where $p_{s},p_{t}$ is the density of $\mathcal{D}_{S},\mathcal{D}_{T}$ respectively.
We can rewrite Equ. ([8](#A1.E8 "In A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) as:

|  | $\displaystyle d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})\=\int_{\mathcal{Z}}p^{z}_{s}(\mathbf{z})\log\left[\gamma\log\hat{h}(\mathbf{z})+(1-\gamma)\log\left(1-\hat{h}(\mathbf{z})\right)\right]+$ |  | (9) |
| --- | --- | --- | --- |
| | $\displaystyle p_{t}^{z}(\mathbf{z})\left[(1-\gamma)\log\hat{h}(\mathbf{z})+\gamma\log\left(1-\hat{h}(\mathbf{z})\right)\right]$ | | |

We first take derivatives and find the optimal $\hat{h}^{*}$:

|  | $\displaystyle\frac{\partial d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})}{\partial\hat{h}(\mathbf{z})}$ | $\displaystyle\=p^{z}_{s}(\mathbf{z})\left[\gamma\frac{1}{\hat{h}(\mathbf{z})}+(1-\gamma)\frac{-1}{1-\hat{h}(\mathbf{z})}\right]+p_{t}^{z}(\mathbf{z})\left[(1-\gamma)\log\frac{1}{\hat{h}(\mathbf{z})}+\gamma\frac{-1}{1-\hat{h}(\mathbf{z})}\right]\=0$ |  | (10) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\Rightarrow p_{s}^{z}(\mathbf{z})\left[\gamma(1-\hat{h}(\mathbf{z}))-(1-\gamma)\hat{h}(\mathbf{z})\right]]+p_{t}^{z}(\mathbf{z})\left[(1-\gamma)(1-\hat{h}(\mathbf{z}))-\gamma\hat{h}(\mathbf{z})\right]\=0$ | | |
|  |  | $\displaystyle\Rightarrow p_{s}^{z}(\mathbf{z})\left[\gamma-\hat{h}(\mathbf{z})\right]+p_{t}^{z}(\mathbf{z})\left[1-\gamma-\hat{h}(\mathbf{z})\right]\=0$ |  |
|  |  | $\displaystyle\Rightarrow\hat{h}^{*}(\mathbf{z})\=\frac{p_{t}^{z}(\mathbf{z})+\gamma(p_{s}^{z}(\mathbf{z})-p_{t}^{z}(\mathbf{z}))}{p_{s}^{z}(\mathbf{z})+p_{t}^{z}(\mathbf{z})}$ |  |

For simplicity, we use $p_{s},p_{t}$ denote $p_{s}^{z}(\mathbf{z}),p_{t}^{z}(\mathbf{z})$ respectively and ignore the $\int_{\mathcal{Z}}$. Plugging Equ. ([10](#A1.E10 "In Proof. ‣ A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) into Equ. ([8](#A1.E8 "In A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) we can get

|  | $\displaystyle\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})$ | $\displaystyle\=\int_{\mathcal{Z}}p_{s}\left[\gamma\log\left[\frac{p_{t}+\gamma(p_{s}-p_{t})}{p_{s}+p_{t}}\right]+(1-\gamma)\log\left[\frac{p_{s}+\gamma(p_{t}-p_{s})}{p_{s}+p_{t}}\right]\right]$ |  | (11) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad+p_{t}\left[(1-\gamma)\log\left[\frac{p_{t}+\gamma(p_{s}-p_{t})}{p_{s}+p_{t}}\right]+\gamma\log\left[\frac{p_{s}+\gamma(p_{t}-p_{s})}{p_{s}+p_{t}}\right]\right]d_{\mathbf{z}}$ | | |
|  |  | $\displaystyle\=\int_{\mathcal{Z}}\underbrace{p_{s}\log\frac{p_{s}+\gamma(p_{t}-p_{s})}{p_{s}+p_{t}}+p_{t}\log\frac{p_{t}+\gamma(p_{s}-p_{t})}{p_{s}+p_{t}}}_{\rm\lower 2.1097pt\hbox{ \leavevmode\hbox to11.67pt{\vbox to9.18pt{\pgfpicture\makeatletter\hbox{\hskip 5.83311pt\lower-4.58867pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}{{}}{}{{{}}{}{}{}{}{}{}{}{}}{{}}{}\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@moveto{3.50002pt}{0.0pt}\pgfsys@curveto{3.50002pt}{1.93303pt}{1.93303pt}{3.50002pt}{0.0pt}{3.50002pt}\pgfsys@curveto{-1.93303pt}{3.50002pt}{-3.50002pt}{1.93303pt}{-3.50002pt}{0.0pt}\pgfsys@curveto{-3.50002pt}{-1.93303pt}{-1.93303pt}{-3.50002pt}{0.0pt}{-3.50002pt}\pgfsys@curveto{1.93303pt}{-3.50002pt}{3.50002pt}{-1.93303pt}{3.50002pt}{0.0pt}\pgfsys@closepath\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@stroke\pgfsys@invoke{ }\hbox{\hbox{{\pgfsys@beginscope\pgfsys@invoke{ }{{}{}{{ {}{}}}{ {}{}} {{}{{}}}{{}{}}{}{{}{}} { }{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{-3.5pt}{-2.25555pt}\pgfsys@invoke{ }\hbox{{\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\hbox{{\makebox[7.00002pt][c]{\rm 1}}} }}\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}+$ |  |
|  |  | $\displaystyle\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\underbrace{p_{s}\gamma\log\frac{p_{t}+\gamma(p_{s}-p_{t})}{p_{s}+\gamma(p_{t}-p_{s})}+p_{t}\gamma\frac{p_{s}+\gamma(p_{t}-p_{s})}{p_{t}+\gamma(p_{s}-p_{t})}}_{\rm\lower 2.1097pt\hbox{ \leavevmode\hbox to11.67pt{\vbox to9.18pt{\pgfpicture\makeatletter\hbox{\hskip 5.83311pt\lower-4.58867pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}{{}}{}{{{}}{}{}{}{}{}{}{}{}}{{}}{}\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@moveto{3.50002pt}{0.0pt}\pgfsys@curveto{3.50002pt}{1.93303pt}{1.93303pt}{3.50002pt}{0.0pt}{3.50002pt}\pgfsys@curveto{-1.93303pt}{3.50002pt}{-3.50002pt}{1.93303pt}{-3.50002pt}{0.0pt}\pgfsys@curveto{-3.50002pt}{-1.93303pt}{-1.93303pt}{-3.50002pt}{0.0pt}{-3.50002pt}\pgfsys@curveto{1.93303pt}{-3.50002pt}{3.50002pt}{-1.93303pt}{3.50002pt}{0.0pt}\pgfsys@closepath\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@stroke\pgfsys@invoke{ }\hbox{\hbox{{\pgfsys@beginscope\pgfsys@invoke{ }{{}{}{{ {}{}}}{ {}{}} {{}{{}}}{{}{}}{}{{}{}} { }{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{-3.5pt}{-2.25555pt}\pgfsys@invoke{ }\hbox{{\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\hbox{{\makebox[7.00002pt][c]{\rm 2}}} }}\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}d_{\mathbf{z}}$ |  |

Let $\left{\begin{array}[]{c}p_{s^{\prime}}\=p_{s}+(1-\gamma)p_{t}\\
p_{t^{\prime}}\=p_{t}+(1-\gamma)p_{s}\\
\end{array}\right.$
two distribution densities that are the convex combinations of $p_{s},p_{t}$, we have
$\left{\begin{array}[]{c}p_{s}\=\frac{\gamma p_{s^{\prime}}+(\gamma-1)p_{t^{\prime}}}{2\gamma-1}\\
p_{t}\=\frac{\gamma p_{t^{\prime}}+(\gamma-1)p_{s^{\prime}}}{2\gamma-1}\\
\end{array}\right.$, and $p_{s^{\prime}}+p_{t^{\prime}}\=p_{s}+p_{t}$.

|  |  | $\displaystyle\frac{\gamma}{2\gamma-1}\left(p_{s^{\prime}}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}+p_{t^{\prime}}\log\frac{p_{s^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}\right)+\frac{\gamma-1}{2\gamma-1}\left(p_{t^{\prime}}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}+p_{s^{\prime}}\log\frac{p_{s^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}\right)$ |  | (12) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\frac{\gamma}{2\gamma-1}\left(p_{s^{\prime}}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}+p_{s^{\prime}}\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}-p_{s^{\prime}}\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}+p_{t^{\prime}}\log\frac{p_{s^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}\right)+\frac{\gamma-1}{2\gamma-1}\left(p_{t^{\prime}}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}+p_{s^{\prime}}\log\frac{p_{s^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}\right)$ | | |
|  |  | $\displaystyle\=\left(p_{t^{\prime}}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}+p_{s^{\prime}}\log\frac{p_{s^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}\right)-\frac{\gamma}{2\gamma-1}\left(p_{s^{\prime}}\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}+p_{t^{\prime}}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}}\right)$ |  |
|  |  | $\displaystyle\=2\frac{1}{2}\left(p_{t^{\prime}}\log\frac{2p_{t^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}+p_{s^{\prime}}\log\frac{2p_{s^{\prime}}}{p_{s^{\prime}}+p_{t^{\prime}}}-2\log 2\right)-\frac{\gamma}{2\gamma-1}\left(p_{s^{\prime}}\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}+p_{t^{\prime}}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}}\right)$ |  |
|  |  | $\displaystyle\=2D_{JS}(\mathcal{D}_{S^{\prime}}||\mathcal{D}_{T^{\prime}})-2\log 2-\frac{\gamma}{2\gamma-1}\left(p_{s^{\prime}}-p_{t^{\prime}}\right)\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}$ |  |


|  |  | $\displaystyle\gamma\left(p_{s}\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}+p_{t}\log\frac{p_{t^{\prime}}}{p_{s^{\prime}}}\right)$ |  | (13) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\gamma\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}\left(\frac{\gamma p_{s^{\prime}}+(\gamma-1)p_{t^{\prime}}}{2\gamma-1}-\frac{\gamma p_{t^{\prime}}+(\gamma-1)p_{s^{\prime}}}{2\gamma-1}\right)$ | | |
|  |  | $\displaystyle\=\frac{\gamma}{2\gamma-1}(p_{s^{\prime}}-p_{t^{\prime}})\log\frac{p_{s^{\prime}}}{p_{t^{\prime}}}$ |  |


|  | $\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})\=2D_{JS}(\mathcal{D}_{S^{\prime}}||\mathcal{D}_{T^{\prime}})-2\log 2$ |  | (14) |
| --- | --- | --- | --- |

∎

### A.2 Connect One-sided Environment Label Smoothing to JS Divergence Minimization

###### Proposition 2.

Given two domain distributions $\mathcal{D}_{S},\mathcal{D}_{T}$ over $X$, where $\mathcal{D}_{S}$ is the read data distribution and $\mathcal{D}_{T}$ is the generated data distribution. The cost used for the discriminator is:

|  | $\max_{h\in\mathcal{H}}d_{h}(\mathcal{D}_{S},\mathcal{D}_{T})\=\max_{h\in{\mathcal{H}}}\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\log h(\mathbf{x}_{s})+\mathbb{E}_{\mathbf{x}_{t}\sim\mathcal{D}_{T}}\log\left(1-h(\mathbf{x}_{t})\right),$ |  | (15) |
| --- | --- | --- | --- |

where $h\in\mathcal{H}:\mathcal{X}\rightarrow[0,1]$. Suppose ${h}\in{\mathcal{H}}$ the optimal discriminator with no constraint and mixed distributions $\left{\begin{array}[]{l}\mathcal{D}_{S^{\prime}}\=\gamma\mathcal{D}_{S}\\
\mathcal{D}_{T^{\prime}}\=\mathcal{D}_{T}+(1-\gamma)\mathcal{D}_{S}\\
\end{array}\right.$ with hyper-parameter $\gamma$. Then to minimize domain divergence by adversarial training with one-sided environment label smoothing is equal to minimize $2D_{JS}(\mathcal{D}_{S^{\prime}}||\mathcal{D}_{T^{\prime}})-2\log 2$, where $D_{JS}$ is the Jensen-Shanon (JS) divergence.

###### Proof.

Applying one-sided environment label smoothing, the target can be reformulated to

|  | $\displaystyle\max_{h\in\mathcal{H}}d_{h,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})$ | $\displaystyle\=\max_{h\in\mathcal{H}}\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\gamma\log h(\mathbf{x}_{s})+(1-\gamma)\log\left(1-h(\mathbf{x}_{s})\right)\right]+\mathbb{E}_{\mathbf{x}_{t}\sim\mathcal{D}_{T}}\left[\log\left(1-h(\mathbf{x}_{t})\right)\right]$ |  | (16) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\max_{h\in\mathcal{H}}\int_{\mathcal{X}}p_{s}(\mathbf{x})\log\left[\gamma\log h(\mathbf{x})+(1-\gamma)\log(1-h(\mathbf{x}))\right]+p_{t}(\mathbf{x})\log(1-h(\mathbf{x}))$ | | |

where $\gamma$ is a value slightly less than one, $p_{s}(\mathbf{x}),p_{t}(\mathbf{x})$ is the density of $\mathcal{D}_{S},\mathcal{D}_{T}$ respectively. By taking derivatives and finding the optimal $h$ we can get $h^{*}\=\frac{\gamma p_{s}(\mathbf{x})}{p_{s}(\mathbf{x})+p_{t}(\mathbf{x})}$. Plugging the optimal $h^{*}$ into the original target we can get:

|  |  | $\displaystyle\=\int_{\mathcal{X}}p_{s}(\mathbf{x})\left[\gamma\log\frac{\gamma p_{s}(\mathbf{x})}{p_{s}(\mathbf{x})+p_{t}(\mathbf{x})}+(1-\gamma)\log\frac{p_{t}(\mathbf{x})+(1-\gamma)p_{s}(\mathbf{x})}{p_{s}(\mathbf{x})+p_{t}(\mathbf{x})}\right]+p_{t}(\mathbf{x})\log\frac{p_{t}(\mathbf{x})+(1-\gamma)p_{s}(\mathbf{x})}{p_{s}(\mathbf{x})+p_{t}(\mathbf{x})}d_{\mathbf{x}}$ |  | (17) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\int_{\mathcal{X}}p_{s}(\mathbf{x})\gamma\log\frac{\gamma p_{s}(\mathbf{x})}{p_{s}(\mathbf{x})+p_{t}(\mathbf{x})}+\left[p_{s}(1-\gamma)+p_{t}(\mathbf{x})\right]\log\frac{p_{t}(\mathbf{x})+(1-\gamma)p_{s}(\mathbf{x})}{p_{s}(\mathbf{x})+p_{t}(\mathbf{x})}d_{\mathbf{x}}$ | | |
|  |  | $\displaystyle\=\int_{\mathcal{X}}p_{s^{\prime}}(\mathbf{x})\log\frac{p_{s^{\prime}}(\mathbf{x})}{p_{s^{\prime}}(\mathbf{x})+p_{t^{\prime}}(\mathbf{x})}+p_{t^{\prime}}(\mathbf{x})\log\frac{p_{t^{\prime}}(\mathbf{x})}{p_{s^{\prime}}(\mathbf{x})+p_{t^{\prime}}(\mathbf{x})}d_{\mathbf{x}}$ |  |
|  |  | $\displaystyle\=2D_{JS}(\mathcal{D}_{S^{\prime}}||\mathcal{D}_{T^{\prime}})-2\log 2,$ |  |

where $\left{\begin{array}[]{l}\mathcal{D}_{S^{\prime}}\=\gamma\mathcal{D}_{S}\\
\mathcal{D}_{T^{\prime}}\=\mathcal{D}_{T}+(1-\gamma)\mathcal{D}_{S}\\
\end{array}\right.$ are two mixed distributions and $\left{\begin{array}[]{l}p_{s^{\prime}}\=\gamma p_{s}\\
p_{t^{\prime}}\=p_{t}+(1-\gamma)p_{s}\\
\end{array}\right.$ are their densities.
∎

Our result supplies an explanation to “why GANs only use one-sided label smoothing rather than native label smoothing”. That is, if the density of real data in a region is near zero $p_{s}(\mathbf{x})\rightarrow 0$, native environment label smoothing will be dominated by only the generated sample densities because $\left{\begin{array}[]{l}p_{s^{\prime}}\=p_{t}+\gamma(p_{s}-p_{t})\approx(1-\gamma)p_{t}\\
p_{t^{\prime}}\=p_{s}+\gamma(p_{t}-p_{s})\approx\gamma p_{t}\\
\end{array}\right.$.
Namely, the discriminator will not align the distribution between generated samples and real samples, but enforce the generator to produce samples that follow the fake mode $\mathcal{D}_{T}$. In contrast, one-sided label smoothing reserves the real distribution density as far as possible, that is, $p_{s^{\prime}}\=\gamma p_{s},p_{t^{\prime}}\approx\gamma p_{t}$, which avoids divergence minimization between fake mode to fake mode and relieves model collapse.

### A.3 Connect Multi-Domain Adversarial Training to KL Divergence Minimization

###### Proposition 3.

Given domain distributions ${\mathcal{D}_{i}}_{i\=1}^{M}$ over $X$, and a hypothesis class $\mathcal{H}$. Suppose $\hat{h}\in\hat{\mathcal{H}}$ the optimal discriminator with no constraint and mixed distributions $\mathcal{D}_{Mix}\=\sum_{i\=1}^{M}\mathcal{D}_{i}$, and ${\mathcal{D}_{i^{\prime}}\=\gamma\mathcal{D}_{i}+\frac{1-\gamma}{M-1}\sum^{M}_{j\=1;j\neq i}\mathcal{D}}_{i\=1}^{M}$ with hyper-parameter $\gamma\in[0.5,1]$. Then to minimize domain divergence by adversarial training w/wo environment label smoothing is equal to minimize $\sum_{i\=1}^{M}D_{KL}(\mathcal{D}_{i}||\mathcal{D}_{Mix})$, and $\sum_{i\=1}^{M}D_{KL}(\mathcal{D}_{i^{\prime}}||\mathcal{D}_{Mix})$ respectively, where $D_{KL}$ is the Kullback–Leibler (KL) divergence.

###### Proof.

We restate corresponding notations and definitions as follows. Given $M$ domains ${\mathcal{D}_{i}}_{i\=1}^{M}$. Let the hypothesis $h$ be the composition of $h\=\hat{h}\circ g$, where $g\in\mathcal{G}$ pushes forward the data samples to a representation space $\mathcal{Z}$ and the domain discriminator with softmax activation function is defined as $\hat{h}\=(\hat{h}_{1}(\cdot),\dots,\hat{h}_{M}(\cdot))\in\hat{\mathcal{H}}:\mathcal{Z}\rightarrow[0,1]^{M};\sum_{i\=1}^{M}\hat{h}_{i}(\cdot)\=1$. Denote $g\circ\mathcal{D}_{i}$ the feature distribution of $\mathcal{D}_{i}$ which is encoded by encoder $g$. The cost used for the discriminator can be defined as:

|  | $\displaystyle\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})\=$ | $\displaystyle\max_{\hat{h}\in{\mathcal{H}}}\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\mathbf{z})+\dots+\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{M}}\log\hat{h}_{M}(\mathbf{z}),\text{s.t. }\sum_{i\=1}^{M}\hat{h}_{i}(\mathbf{z})\=1$ |  | (18) |
| --- | --- | --- | --- | --- |

Denote $p_{i}^{z}(\mathbf{z})$ the density of feature distribution $g\circ\mathcal{D}_{i}$. For simplicity, we ignore $\int_{\mathcal{Z}}$. Applying lagrange multiplier and taking the first derivative with respect to each $\hat{h}_{i}$, we can get

|  | $\left{\begin{array}[]{c}\frac{\partial d_{\hat{h},g}}{\partial\hat{h}_{1}}\=p_{1}^{z}(\mathbf{z})\frac{1}{\hat{h}_{1}(z)}-\lambda\=0\\ \vdots\\ \frac{\partial d_{\hat{h},g}}{\partial\hat{h}_{M}}\=p_{M}^{z}(\mathbf{z})\frac{1}{\hat{h}_{M}(z)}-\lambda\=0\\ \end{array}\right.\Rightarrow\left{\begin{array}[]{c}\hat{h}_{1}(\mathbf{z})\=\frac{p_{1}^{z}(\mathbf{z})}{\lambda}\\ \vdots\\ \hat{h}_{M}(\mathbf{z})\=\frac{p_{M}^{z}(\mathbf{z})}{\lambda}\\ \end{array}\right.\Rightarrow^{\rm\lower 2.1097pt\hbox{ \leavevmode\hbox to11.67pt{\vbox to9.18pt{\pgfpicture\makeatletter\hbox{\hskip 5.83311pt\lower-4.58867pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}{{}}{}{{{}}{}{}{}{}{}{}{}{}}{{}}{}\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@moveto{3.50002pt}{0.0pt}\pgfsys@curveto{3.50002pt}{1.93303pt}{1.93303pt}{3.50002pt}{0.0pt}{3.50002pt}\pgfsys@curveto{-1.93303pt}{3.50002pt}{-3.50002pt}{1.93303pt}{-3.50002pt}{0.0pt}\pgfsys@curveto{-3.50002pt}{-1.93303pt}{-1.93303pt}{-3.50002pt}{0.0pt}{-3.50002pt}\pgfsys@curveto{1.93303pt}{-3.50002pt}{3.50002pt}{-1.93303pt}{3.50002pt}{0.0pt}\pgfsys@closepath\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@stroke\pgfsys@invoke{ }\hbox{\hbox{{\pgfsys@beginscope\pgfsys@invoke{ }{{}{}{{ {}{}}}{ {}{}} {{}{{}}}{{}{}}{}{{}{}} { }{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{-3.5pt}{-2.25555pt}\pgfsys@invoke{ }\hbox{{\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\hbox{{\makebox[7.00002pt][c]{\rm 1}}} }}\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}\left{\begin{array}[]{c}\hat{h}_{1}^{*}(\mathbf{z})\=\frac{p_{1}^{z}(\mathbf{z})}{p_{1}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}\\ \vdots\\ \hat{h}_{M}^{*}(\mathbf{z})\=\frac{p_{M}^{z}(\mathbf{z})}{p_{1}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}\\ \end{array}\right.$ |  | (19) |
| --- | --- | --- | --- |


|  | $\displaystyle\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})$ | $\displaystyle\=\int_{\mathcal{Z}}p_{1}^{z}(\mathbf{z})\log\frac{p_{1}^{z}(\mathbf{z})}{p^{z}_{Mix}(\mathbf{z})}+p_{2}^{z}(\mathbf{z})\log\frac{p_{2}^{z}(\mathbf{z})}{p^{z}_{Mix}(\mathbf{z})}+\cdot+p_{M}^{z}(\mathbf{z})\log\frac{p_{M}^{z}(\mathbf{z})}{p^{z}_{Mix}(\mathbf{z})}d_{\mathbf{z}}$ |  | (20) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\sum_{i\=1}^{M}D_{KL}(\mathcal{D}_{i}\ |\ | |

where $D_{KL}$ is the KL divergence. With environment label smoothing, the target is

|  | $\displaystyle\max_{\hat{h}\in\hat{\mathcal{H}}}d_{\hat{h},g,\gamma}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})\=\max_{\hat{h}\in\hat{\mathcal{H}}}\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}$ | $\displaystyle\left[\gamma\log\hat{h}_{1}(\mathbf{z})+\frac{(1-\gamma)}{M-1}\sum_{j\=1;j\neq 1}^{M}\log\left(\hat{h}_{j}(\mathbf{z})\right)\right]+\dots+$ |  | (21) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{M}}\left[\gamma\log\hat{h}_{M}(\mathbf{z})+\frac{(1-\gamma)}{M-1}\sum_{j\=1;j\neq M}^{M}\log\left(\hat{h}_{j}(\mathbf{z})\right)\right],\text{s.t. }\sum_{i\=1}^{M}\hat{h}_{i}(\mathbf{z})\=1$ | | |

Take the same operation as Equ. ([19](#A1.E19 "In Proof. ‣ A.3 Connect Multi-Domain Adversarial Training to KL Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) we can get

|  | $\left{\begin{array}[]{c}\frac{\partial d_{\hat{h},g,\gamma}}{\partial\hat{h}_{1}}\=\gamma p_{1}^{z}(\mathbf{z})\frac{1}{\hat{h}_{1}(z)}+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq 1}^{M}p_{j}^{z}(\mathbf{z})\frac{1}{\hat{h}_{1}(z)}-\lambda\=0\\ \vdots\\ \frac{\partial d_{\hat{h},g,\gamma}}{\partial\hat{h}_{M}}\=\gamma p_{M}^{z}(\mathbf{z})\frac{1}{\hat{h}_{M}(z)}+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq M}^{M}p_{j}^{z}(\mathbf{z})\frac{1}{\hat{h}_{M}(z)}-\lambda\=0\\ \end{array}\right.\Rightarrow\left{\begin{array}[]{c}\hat{h}_{1}^{*}(\mathbf{z})\=\frac{\gamma p_{1}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq 1}^{M}p_{j}^{z}(\mathbf{z})}{p_{1}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}\\ \vdots\\ \hat{h}_{M}^{*}(\mathbf{z})\=\frac{\gamma p_{M}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum^{M}_{j\=1;j\neq M}p_{j}^{z}(\mathbf{z})}{p_{1}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}\\ \end{array}\right.$ |  | (22) |
| --- | --- | --- | --- |

Denote ${\mathcal{D}_{i^{\prime}}\=\gamma\mathcal{D}_{i}+\frac{1-\gamma}{M-1}\sum^{M}_{j\=1;j\neq i}\mathcal{D}}_{i\=1}^{M}$ a set of mixed distributions and ${p_{i^{\prime}}(\mathbf{z})\=\gamma p_{i}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum^{M}_{j\=1;j\neq i}p_{j}^{z}(\mathbf{z})}_{i\=1}^{M}$ the corresponding densities. Plugging Equ. ([22](#A1.E22 "In Proof. ‣ A.3 Connect Multi-Domain Adversarial Training to KL Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) to the target we can get

|  |  | $\displaystyle\sum_{i\=1}^{M}\left[\int_{\mathcal{Z}}\gamma p_{i}^{z}(z)\log\frac{\gamma p_{i}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq i}^{M}p_{j}^{z}(\mathbf{z})}{p_{i}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}+\frac{(1-\gamma)}{M-1}\sum_{k\=1;k\neq i}^{M}p_{i}^{z}(\mathbf{z})\log\frac{\gamma p_{k}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq i}^{M}p_{j}^{z}(\mathbf{z})}{p_{i}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}d_{\mathbf{z}}\right]$ |  | (23) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\sum_{i\=1}^{M}\left[\int_{\mathcal{Z}}\gamma p_{i}^{z}(z)\log\frac{\gamma p_{i}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq i}^{M}p_{j}^{z}(\mathbf{z})}{p_{i}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}+\frac{(1-\gamma)}{M-1}\sum_{k\=1;k\neq i}^{M}p_{k}^{z}(\mathbf{z})\log\frac{\gamma p_{i}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq i}^{M}p_{j}^{z}(\mathbf{z})}{p_{i}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}d_{\mathbf{z}}\right]$ | | |
|  |  | $\displaystyle\=\sum_{i\=1}^{M}\left[\int_{\mathcal{Z}}\left(\gamma p_{i}^{z}(z)+\frac{(1-\gamma)}{M-1}\sum_{k\=1;k\neq i}^{M}p_{k}^{z}(\mathbf{z})\right)\log\frac{\gamma p_{i}^{z}(\mathbf{z})+\frac{1-\gamma}{M-1}\sum_{j\=1;j\neq i}^{M}p_{j}^{z}(\mathbf{z})}{p_{i}^{z}(\mathbf{z})+\dots+p_{M}^{z}(\mathbf{z})}d_{\mathbf{z}}\right]$ |  |
|  |  | $\displaystyle\=\sum_{i\=1}^{M}D_{KL}(\mathcal{D}_{i^{\prime}}||\mathcal{D}_{Mix})$ |  |

∎

### A.4 Training Stability Brought by Environment Label Smoothing

Let $\mathcal{D}_{S},\mathcal{D}_{T}$ two distributions and $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T}$ their induced distributions projected by encoder $g:\mathcal{X}\rightarrow\mathcal{Z}$ over feature space. We first show that if $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T}$ are disjoint or lie in low dimensional manifolds, there is always a perfect discriminator between them.

###### Theorem 1.

(Theorem 2.1. in*(Arjovsky \& Bottou, [2017](#bib.bib4 ""))*.) If two distribution $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T}$ have support contained on two disjoint compact subsets $\mathcal{M}$ and $\mathcal{P}$ respectively, then there is a smooth optimal discriminator $\hat{h}^{*}:\mathcal{Z}\rightarrow[0,1]$ that has accuracy $1$ and $\nabla_{\mathbf{z}}\hat{h}^{*}(\mathbf{z})\=0$ for all $\mathbf{z}\sim\mathcal{M}\bigcup\mathcal{P}$.

###### Theorem 2.

(Theorem 2.2. in*(Arjovsky \& Bottou, [2017](#bib.bib4 ""))*.) Assume two distribution $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T}$ have support contained in two closed manifolds $\mathcal{M}$ and $\mathcal{P}$ that don’t perfectly align and don’t have full dimension. Both $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T}$ are assumed to be continuous in their respective manifolds. Then, there is a smooth optimal discriminator $\hat{h}^{*}:\mathcal{Z}\rightarrow[0,1]$ that has accuracy $1$, and for almost all $\mathbf{z}\sim\mathcal{M}\bigcup\mathcal{P}$, $\hat{h}^{*}$ is smooth in a neighbourhood of $\mathbf{z}$ and $\nabla_{\mathbf{z}}\hat{h}^{*}(\mathbf{z})\=0$.

Namely, if the two distributions have supports that are disjoint or lie on low dimensional manifolds, the optimal discriminator will be accurate on all samples and its gradient will be zero almost everywhere. Then we can study the gradients we pass to the generator through a discriminator.

###### Proposition 4.

Denote $g(\theta;\cdot):\mathcal{X}\rightarrow\mathcal{Z}$ a differentiable function that induces distributions $\mathcal{D}^{z}_{S},\mathcal{D}^{z}_{T}$ with parameter $\theta$, and $\hat{h}$ a differentiable discriminator. If Theorem[1](#Thmtheo1 "Theorem 1. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") or[2](#Thmtheo2 "Theorem 2. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") holds, given a $\epsilon$-optimal discriminator $\hat{h}$, that is $\sup_{\mathbf{z}\in\mathcal{Z}}\parallel\nabla_{\mathbf{z}}\hat{h}(\mathbf{z})\parallel_{2}+|\hat{h}(\mathbf{z})-\hat{h}^{*}(\mathbf{z})|<\epsilon$111The constraint on $\parallel\nabla_{\mathbf{z}}\hat{h}(\mathbf{z})\parallel_{2}$ is because the optimal discriminator has zero gradients almost everywhere, and $|\hat{h}(\mathbf{z})-\hat{h}^{*}(\mathbf{z})|$ is a constraint on the prediction accuracy., assume the Jacobian matrix of $g(\theta;\mathbf{x})$ given $\mathbf{x}$ is bounded by $\sup_{\mathbf{x}\in\mathcal{X}}[\parallel J_{\theta}(g(\theta;\mathbf{x}))\parallel_{2}]\leq C$, then we have

|  | $\displaystyle\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g}(\mathcal{D}_{S},\mathcal{D}_{T})\parallel_{2}\=0$ |  | (24) |
| --- | --- | --- | --- |

|  | $\displaystyle\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})\parallel_{2}<2(1-\gamma)C$ |  | (25) |
| --- | --- | --- | --- |

###### Proof.

Theorem[1](#Thmtheo1 "Theorem 1. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") or[2](#Thmtheo2 "Theorem 2. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") show that in Equ. ([8](#A1.E8 "In A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")), $\hat{h}^{*}$ is locally one on the support of $\mathcal{D}^{z}_{S}$ and zero on the support of $\mathcal{D}^{z}_{T}$. Then, using Jensen’s inequality, triangle inequality, and the chain rule on these supports, the gradients we pass to the generator through a discriminator given $\mathbf{x}_{s}\sim\mathcal{D}_{S}$ is

|  |  | $\displaystyle\parallel\nabla_{\theta}\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\gamma\log\hat{h}\circ g(\theta;\mathbf{x}_{s})+(1-\gamma)\log\left(1-\hat{h}\circ g(\theta;\mathbf{x}_{s})\right)\right]\parallel_{2}$ |  | (26) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\leq\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\parallel\nabla_{\theta}\gamma\log\hat{h}\circ g(\theta;\mathbf{x}_{s})\parallel_{2}\right]+\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\parallel\nabla_{\theta}(1-\gamma)\log\left(1-\hat{h}\circ g(\theta;\mathbf{x}_{s})\right)\parallel_{2}\right]$ | | |
|  |  | $\displaystyle\leq\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\gamma\frac{\parallel\nabla_{\theta}\hat{h}\circ g(\theta;\mathbf{x}_{s})\parallel_{2}}{|\hat{h}\circ g(\theta;\mathbf{x}_{s})|}\right]+\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[(1-\gamma)\frac{\parallel\nabla_{\theta}\hat{h}\circ g(\theta;\mathbf{x}_{s})\parallel_{2}}{\left|1-\hat{h}\circ g(\theta;\mathbf{x}_{s})\right|}\right]$ |  |
|  |  | $\displaystyle\leq\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\gamma\frac{\parallel\nabla_{\mathbf{z}}\hat{h}(\mathbf{z})\parallel_{2}\parallel J_{\theta}(g(\theta;\mathbf{x}_{s}))\parallel_{2}}{|\hat{h}\circ g(\theta;\mathbf{x}_{s})|}\right]+\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[(1-\gamma)\frac{\parallel\nabla_{\mathbf{z}}\hat{h}(\mathbf{z})\parallel_{2}\parallel J_{\theta}(g(\theta;\mathbf{x}_{s}))\parallel_{2}}{\left|1-\hat{h}\circ g(\theta;\mathbf{x}_{s})\right|}\right]$ |  |
|  |  | $\displaystyle<\gamma\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\frac{\epsilon\parallel J_{\theta}(g(\theta;\mathbf{x}_{s}))\parallel_{2}}{|\hat{h}^{*}\circ g(\theta;\mathbf{x}_{s})-\epsilon|}\right]+(1-\gamma)\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\frac{\epsilon\parallel J_{\theta}(g(\theta;\mathbf{x}_{s}))\parallel_{2}}{\left|1-\hat{h}^{*}\circ g(\theta;\mathbf{x}_{s})+\epsilon\right|}\right]$ |  |
|  |  | $\displaystyle\leq\gamma\frac{\epsilon C}{1-\epsilon}+(1-\gamma)C,$ |  |

where the fifth line is because we have $\hat{h}(z)\approx\hat{h}^{*}(z)-\epsilon$ when $\epsilon$ is small enough and $\parallel\nabla_{\mathbf{z}}\hat{h}(\mathbf{z})\parallel_{2}<\epsilon$. Similarly we can get the gradients given $\mathbf{x}_{t}\sim\mathcal{D}_{T}$ is

|  |  | $\displaystyle\parallel\nabla_{\theta}\mathbb{E}_{\mathbf{x}_{t}\sim\mathcal{D}_{T}}\left[(1-\gamma)\log\hat{h}\circ g(\mathbf{x}_{t})+\gamma\log\left(1-\hat{h}\circ g(\mathbf{x}_{t})\right)\right]\parallel_{2}$ |  | (27) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle<(1-\gamma)\mathbb{E}_{\mathbf{x}_{t}\sim\mathcal{D}_{T}}\left[\frac{\epsilon\parallel J_{\theta}(g(\theta;\mathbf{x}_{t}))\parallel_{2}}{\ |\hat{h}^{*}\circ g(\theta;\mathbf{x}_{t})+\epsilon\ | |
|  |  | $\displaystyle\leq(1-\gamma)C+\gamma\frac{\epsilon C}{1-\epsilon}$ |  |

Here $\hat{h}(z)\approx\hat{h}^{*}(z)+\epsilon$ because $\hat{h}^{*}$ is locally zero on the support of $\mathcal{D}^{z}_{T}$. Then we have

|  |  | $\displaystyle\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})\parallel_{2}$ |  | (28) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\leq\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}\mathbb{E}_{\mathbf{x}_{s}\sim\mathcal{D}_{S}}\left[\gamma\log\hat{h}\circ g(\theta;\mathbf{x}_{s})+(1-\gamma)\log\left(1-\hat{h}\circ g(\theta;\mathbf{x}_{s})\right)\right]\parallel_{2}$ | | |
|  |  | $\displaystyle\quad\quad\quad\quad\quad+\parallel\nabla_{\theta}\mathbb{E}_{\mathbf{x}_{t}\sim\mathcal{D}_{T}}\left[(1-\gamma)\log\hat{h}\circ g(\mathbf{x}_{t})+\gamma\log\left(1-\hat{h}\circ g(\mathbf{x}_{t})\right)\right]\parallel_{2}$ |  |
|  |  | $\displaystyle<\lim_{\epsilon\rightarrow 0}\underbrace{\gamma\frac{\epsilon C}{1-\epsilon}+\gamma\frac{\epsilon C}{1-\epsilon}}_{\rm\lower 2.1097pt\hbox{ \leavevmode\hbox to11.67pt{\vbox to9.18pt{\pgfpicture\makeatletter\hbox{\hskip 5.83311pt\lower-4.58867pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}{{}}{}{{{}}{}{}{}{}{}{}{}{}}{{}}{}\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@moveto{3.50002pt}{0.0pt}\pgfsys@curveto{3.50002pt}{1.93303pt}{1.93303pt}{3.50002pt}{0.0pt}{3.50002pt}\pgfsys@curveto{-1.93303pt}{3.50002pt}{-3.50002pt}{1.93303pt}{-3.50002pt}{0.0pt}\pgfsys@curveto{-3.50002pt}{-1.93303pt}{-1.93303pt}{-3.50002pt}{0.0pt}{-3.50002pt}\pgfsys@curveto{1.93303pt}{-3.50002pt}{3.50002pt}{-1.93303pt}{3.50002pt}{0.0pt}\pgfsys@closepath\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@stroke\pgfsys@invoke{ }\hbox{\hbox{{\pgfsys@beginscope\pgfsys@invoke{ }{{}{}{{ {}{}}}{ {}{}} {{}{{}}}{{}{}}{}{{}{}} { }{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{-3.5pt}{-2.25555pt}\pgfsys@invoke{ }\hbox{{\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\hbox{{\makebox[7.00002pt][c]{\rm 1}}} }}\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}+\underbrace{(1-\gamma)C+(1-\gamma)C}_{\rm\lower 2.1097pt\hbox{ \leavevmode\hbox to11.67pt{\vbox to9.18pt{\pgfpicture\makeatletter\hbox{\hskip 5.83311pt\lower-4.58867pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}{{}}{}{{{}}{}{}{}{}{}{}{}{}}{{}}{}\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@moveto{3.50002pt}{0.0pt}\pgfsys@curveto{3.50002pt}{1.93303pt}{1.93303pt}{3.50002pt}{0.0pt}{3.50002pt}\pgfsys@curveto{-1.93303pt}{3.50002pt}{-3.50002pt}{1.93303pt}{-3.50002pt}{0.0pt}\pgfsys@curveto{-3.50002pt}{-1.93303pt}{-1.93303pt}{-3.50002pt}{0.0pt}{-3.50002pt}\pgfsys@curveto{1.93303pt}{-3.50002pt}{3.50002pt}{-1.93303pt}{3.50002pt}{0.0pt}\pgfsys@closepath\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@stroke\pgfsys@invoke{ }\hbox{\hbox{{\pgfsys@beginscope\pgfsys@invoke{ }{{}{}{{ {}{}}}{ {}{}} {{}{{}}}{{}{}}{}{{}{}} { }{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{-3.5pt}{-2.25555pt}\pgfsys@invoke{ }\hbox{{\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\hbox{{\makebox[7.00002pt][c]{\rm 2}}} }}\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}$ |  |
|  |  | $\displaystyle\=2(1-\gamma)C,$ |  |


|  | $\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g}(\mathcal{D}_{S},\mathcal{D}_{T})\parallel_{2}\=0,$ |  | (29) |
| --- | --- | --- | --- |

which shows that as our discriminator gets better, the gradient of the encoder vanishes. With environment label smoothing, we have

|  | $\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g,\gamma}(\mathcal{D}_{S},\mathcal{D}_{T})\parallel_{2}\=2(1-\gamma)C,$ |  | (30) |
| --- | --- | --- | --- |

which alleviates the problem of gradients vanishing.
∎

### A.5 Training Stability Analysis of Multi-Domain settings

Let ${\mathcal{D}_{i}}_{i\=1}^{M}$ a set of data distributions and ${\mathcal{D}^{z}_{i}}_{i\=1}^{M}$ their induced distributions projected by encoder $g:\mathcal{X}\rightarrow\mathcal{Z}$ over feature space. Recall that the domain discriminator with softmax activation function is defined as $\hat{h}\=(\hat{h}_{1},\dots,\hat{h}_{M})\in\hat{\mathcal{H}}:\mathcal{Z}\rightarrow[0,1]^{M}$, where $\hat{h}_{i}(\mathbf{z})$ denotes the probability that $\mathbf{z}$ belongs to $\mathcal{D}^{z}_{i}$. To verify the existence of each optimal discriminator $\hat{h}_{i}^{*}$, we can easily replace $\mathcal{D}_{s}^{z},\mathcal{D}_{t}^{z}$ in Theorem[1](#Thmtheo1 "Theorem 1. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Theorem[2](#Thmtheo2 "Theorem 2. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") by $\mathcal{D}^{z}_{i},\sum_{j\=1;j\neq i}^{M}\mathcal{D}^{z}_{j}$ respectively. Namely, if distribution $\mathcal{D}^{z}_{i}$ and $\sum_{j\=1;j\neq i}^{M}\mathcal{D}^{z}_{j}$ have supports that are disjoint or lie on low dimensional manifolds, $\hat{h}_{i}^{*}$ can perfectly discriminate samples within and beyond $\mathcal{D}^{z}_{i}$ and its gradient will be zero almost everywhere.

###### Proposition 5.

Denote $g(\theta;\cdot):\mathcal{X}\rightarrow\mathcal{Z}$ a differentiable function that induces distributions ${\mathcal{D}^{z}_{i}}_{i\=1}^{M}$ with parameter $\theta$, and ${\hat{h}_{i}}_{i\=1}^{M}$ corresponding differentiable discriminators. If optimal discriminators for induced distributions exist, given any $\epsilon$-optimal discriminator $\hat{h}_{i}$, we have $\sup_{\mathbf{z}\in\mathcal{Z}}\parallel\nabla_{\mathbf{z}}\hat{h}_{i}(\mathbf{z})\parallel_{2}+|\hat{h}_{i}(\mathbf{z})-\hat{h}_{i}^{*}(\mathbf{z})|<\epsilon$, assume the Jacobian matrix of $g(\theta;\mathbf{x})$ given $\mathbf{x}$ is bounded by $\sup_{\mathbf{x}\in\mathcal{X}}[\parallel J_{\theta}(g(\theta;\mathbf{x}))\parallel_{2}]\leq C$, then we have

|  | $\displaystyle\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})\parallel_{2}\=0$ |  | (31) |
| --- | --- | --- | --- |

|  | $\displaystyle\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g,\gamma}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})\parallel_{2}<{M}(1-\gamma)C$ |  | (32) |
| --- | --- | --- | --- |

###### Proof.

Following the proof in Proposition[25](#A1.E25 "In Proposition 4. ‣ A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), we have

|  |  | $\displaystyle\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{i}}\left[\gamma\log\hat{h}_{i}\circ g(\mathbf{x})+\frac{(1-\gamma)}{M-1}\sum_{j\=1;j\neq i}^{M}\log\left(\hat{h}_{j}\circ g(\mathbf{x})\right)\right]\parallel_{2}$ |  | (33) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\leq\lim_{\epsilon\rightarrow 0}\mathbb{E}_{\mathbf{x}\sim\mathcal{D}_{i}}\left[\gamma\frac{\parallel\nabla_{\theta}\hat{h}_{i}\circ g(\theta;\mathbf{x})\parallel_{2}}{\ |\hat{h}_{i}\circ g(\theta;\mathbf{x})\ | |
|  |  | $\displaystyle<\lim_{\epsilon\rightarrow 0}\gamma\mathbb{E}_{\mathbf{x}\sim\mathcal{D}_{i}}\left[\frac{\epsilon\parallel J_{\theta}(g(\theta;\mathbf{x}))\parallel_{2}}{|\hat{h}^{*}_{i}\circ g(\theta;\mathbf{x})-\epsilon|}\right]+\frac{(1-\gamma)}{M-1}\sum_{j\=1;j\neq i}^{M}\gamma\mathbb{E}_{\mathbf{x}\sim\mathcal{D}_{j}}\left[\frac{\epsilon\parallel J_{\theta}(g(\theta;\mathbf{x}))\parallel_{2}}{|\hat{h}_{j}^{*}\circ g(\theta;\mathbf{x})+\epsilon|}\right]$ |  |
|  |  | $\displaystyle\leq\lim_{\epsilon\rightarrow 0}\left[\gamma\frac{\epsilon C}{1-\epsilon}+(1-\gamma)C\right]$ |  |
|  |  | $\displaystyle\=(1-\gamma)C$ |  |

where the second line is because for $\mathbf{z}\sim\mathcal{D}_{i}^{z}$, $\hat{h}_{i}^{*}(\mathbf{z})$ is locally one and other optimal discriminators $\hat{h}_{j}^{*}(\mathbf{z})|j\neq i,j\in[M]$ are all locally zero, thus we have $\hat{h}_{i}(\mathbf{z})\approx\hat{h}_{i}^{*}(\mathbf{z})-\epsilon$, and $\hat{h}_{j}(\mathbf{z})\approx\hat{h}_{j}^{*}(\mathbf{z})+\epsilon$. $\lim_{\epsilon\rightarrow 0}\frac{\epsilon C}{1-\epsilon}\=0$ is the gradient that passed to the generator by native multi-domain DANN (Equ. ([1](#S2.E1 "In 2 Methodology ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"))). Environment label smoothing leads to another term, that is $(1-\gamma)C$ and avoid gradients vanishing. Consider all distributions, we have

|  |  | $\displaystyle\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}d_{\hat{h},g,\gamma}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})\parallel_{2}$ |  | (34) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\leq\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{1}}\left[\gamma\log\hat{h}_{1}\circ g(\mathbf{x})+\frac{(1-\gamma)}{M-1}\sum_{j\=2}^{M}\log\left(\hat{h}_{j}\circ g(\mathbf{x})\right)\right]\parallel_{2}$ | | |
|  |  | $\displaystyle\quad\quad\quad\quad\quad+\dots+\lim_{\epsilon\rightarrow 0}\parallel\nabla_{\theta}\mathbb{E}_{\mathbf{x}\in\mathcal{D}_{M}}\left[\gamma\log\hat{h}_{M}\circ g(\mathbf{x})+\frac{(1-\gamma)}{M-1}\sum_{j\=1}^{M-1}\log\left(\hat{h}_{j}\circ g(\mathbf{x})\right)\right]\parallel_{2}$ |  |
|  |  | $\displaystyle\=M(1-\gamma)C,$ |  |

∎

### A.6 ELS stabilize the oscillatory gradient

For the clarity of our proof, the notations here is a little different compared to other sections. Let $ec(i)$ be the cross-entropy loss for class $i$, we denote $g$ is the encoder and ${w_{i}}_{i\=1}^{M}$ is the classification parameter for all domains, then the adversarial loss function for a given sample $x$ with domain index $i$ here is

|  | $\displaystyle F(x,i)\=$ | $\displaystyle(1-\gamma)ec(i)+\frac{\gamma}{M}\sum_{j\neq i}ec(j)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\=$ | $\displaystyle ec(i)+\frac{\gamma}{M-1}\sum_{j}\left(ec(j)-ec(i)\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\=$ | $\displaystyle ec(i)+\frac{\gamma}{M-1}\sum_{j}\left(-\log\left(\frac{\exp(w_{j}^{\top}g(x))}{\sum_{k}\exp(w_{k}^{\top}g(x))}\right)+\log\left(\frac{\exp(w_{i}^{\top}g(x))}{\sum_{k}\exp(w_{k}^{\top}g(x))}\right)\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\=$ | $\displaystyle ec(i)+\frac{\gamma}{M-1}\sum_{j}\left(-w_{j}^{\top}g(x)+\log\left(\sum_{k}\exp(w_{k}^{\top}g(x))\right)+w_{i}^{\top}g(x))-\log\left(\sum_{k}\exp(w_{k}^{\top}g(x))\right)\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\=$ | $\displaystyle ec(i)+\frac{\gamma}{M-1}\sum_{j}\left((w_{i}-w_{j})^{\top}g(x)\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\=$ | $\displaystyle-w_{i}^{\top}g(x)+\log\left(\sum_{k}\exp(w_{k}^{\top}g(x))\right)+\frac{\gamma}{M-1}\sum_{j}\left((w_{i}-w_{j})^{\top}g(x)\right)$ |  |
| --- | --- | --- | --- |

We compute the gradient:

|  | $\displaystyle\frac{\partial F(x,i)}{\partial w_{i}}\=-g(x)+\frac{\exp(w_{i}^{\top}g(x)}{\sum_{k}\exp(w_{k}^{\top}g(x))}g(x)+\frac{\gamma}{M-1}g(x)\=\left(-1+p(i)+\frac{\gamma}{M-1}\right)g(x),$ |  | (36) |
| --- | --- | --- | --- |

where $p(i)$ denotes $\frac{\exp(w_{i}^{\top}g(x)}{\sum_{k}\exp(w_{k}^{\top}g(x))}$. When $\gamma$ is small (e.g., $\gamma\lesssim M(1-p(i))$), the gradient will be further pullback towards 0.
Similarly, for $w_{j}$ and $g(x)$, we have

|  | $\displaystyle\frac{\partial F(x,i)}{\partial w_{j}}\=\frac{\exp(w_{j}^{\top}g(x)}{\sum_{k}\exp(w_{k}^{\top}g(x))}g(x)-\frac{\gamma}{M-1}g(x)\=\left(p(j)-\frac{\gamma}{M-1}\right)g(x)$ |  |
| --- | --- | --- |
|  | $\displaystyle\frac{\partial F(x,i)}{\partial g(x)}\=-w_{i}+\sum_{j}\frac{\exp(w_{j}^{\top}g(x)}{\sum_{k}\exp(w_{k}^{\top}g(x))}w_{j}+\frac{\gamma}{M-1}\sum_{j}(w_{i}-w_{j})\=-(1-\frac{\gamma}{M-1})w_{i}+\sum_{j}\left(p(j)-\frac{\gamma}{M-1}\right)w_{j},$ |  | (37) |
| --- | --- | --- | --- |

then with proper choice of $\gamma$ (e.g., $\gamma\lesssim\min_{j}Mp(j)$), the gradient w.r.t $w_{j}$ and $g(x)$ will also shrink towards zero.

### A.7 Environment label smoothing meets noisy labels

In this subsection, we focus on binary classification settings and adopt the symmetric noise model*(Kim et al., [2019](#bib.bib37 ""))*. Some of our proofs follow*(Wei et al., [2022](#bib.bib83 ""))* but different results and analyses are given. The symmetric noise model is widely accepted in the literature on learning with noisy labels and generates the noisy labels by randomly flipping the clean label to the other possible classes. Specifically, given two environment with high-dimensional feature $x$ environment label $y\in{0,1}$, denote noisy labels $\tilde{y}$ is generated by a noise transition matrix $T$, where $T_{ij}$ denotes denotes the probability of flipping the clean label $y\=i$ to the noisy label $\tilde{y}\=j$, i.e., $T_{ij}\=P(\tilde{y}\=j|y\=i)$. Let $e\=P(\tilde{y}\=1|y\=0)\=P(\tilde{y}\=0|y\=1)$ denote the noisy rate, the binary symmetric transition matrix becomes:

|  | $T\=\left(\begin{array}[]{cc}1-e\&e\\ e\&1-e\\ \end{array}\right),$ |  | (38) |
| --- | --- | --- | --- |

Suppose $(x,y)$ are drawn from a joint distribution $\mathcal{D}$, but during training, only samples with noisy labels are accessible from $(x,\tilde{y})\sim\tilde{\mathcal{D}}$. Denote $f:\=\hat{h}\circ g$ and $\ell$ the cross-entropy loss, minimizing the smoothed loss with noisy labels can then be converted to

|  | $\displaystyle\min_{f}\mathbb{E}_{(x,\tilde{y})\sim\tilde{\mathcal{D}}}[\ell(f(x),\tilde{y}^{\gamma})]\=\min_{f}\mathbb{E}_{(x,\tilde{y})\sim\tilde{\mathcal{D}}}\left[\gamma\ell(f(x),\tilde{y})+(1-\gamma)\ell(f(x),1-\tilde{y})\right]$ |  | (39) |
| --- | --- | --- | --- |

Let $c_{1}\=\gamma,c_{2}\=1-\gamma$, according to the law of total probability, we have Equ. ([39](#A1.E39 "In A.7 Environment label smoothing meets noisy labels ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is equal to

|  | $\displaystyle\min_{f}$ | $\displaystyle\mathbb{E}_{x.y\=0}[P(\tilde{y}\=0|y\=0)(c_{1}\ell(f(x),0)+c_{2}\ell(f(x),1)$ |  | (40) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\quad\quad+P(\tilde{y}\=1\ |y\=0)(c_{1}\ell(f(x),1)+c_{2}\ell(f(x),0)]$ | |
|  |  | $\displaystyle+\mathbb{E}_{x.y\=1}[P(\tilde{y}\=0|y\=1)(c_{1}\ell(f(x),0)+c_{2}\ell(f(x),1)$ |  |
|  |  | $\displaystyle\quad\quad+P(\tilde{y}\=1|y\=1)(c_{1}\ell(f(x),1)+c_{2}\ell(f(x),0)]$ |  |

recall that $e\=P(\tilde{y}\=1|y\=0)\=P(\tilde{y}\=0|y\=1)$, the above equation is equal to

|  |  | $\displaystyle\min_{f}\mathbb{E}_{x.y\=0}\left[(1-e)(c_{1}\ell(f(x),0)+c_{2}\ell(f(x),1)+e(c_{1}\ell(f(x),1)+c_{2}\ell(f(x),0)\right]$ |  | (41) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\quad\quad+\mathbb{E}_{x.y\=1}\left[e(c_{1}\ell(f(x),0)+c_{2}\ell(f(x),1)+(1-e)(c_{1}\ell(f(x),1)+c_{2}\ell(f(x),0)\right]$ | | |
|  |  | $\displaystyle\=\min_{f}\mathbb{E}_{x.y\=0}\left[[(1-e)c_{1}+ec_{2}]\ell(f(x),0)+[(1-e)c_{2}+ec_{1}]\ell(f(x),1)\right]$ |  |
|  |  | $\displaystyle\quad\quad+\mathbb{E}_{x.y\=1}\left[[ec_{2}+(1-e)c_{1}]\ell(f(x),1)+[ec_{1}+(1-ec_{2})]\ell(f(x),0)\right]$ |  |
|  |  | $\displaystyle\=\min_{f}\mathbb{E}_{x.y\=0}\left[[(1-e)c_{1}+ec_{2}]\ell(f(x),0)+[(1-e)c_{2}+ec_{1}]\ell(f(x),1)\right]$ |  |
|  |  | $\displaystyle\quad\quad+\mathbb{E}_{x.y\=1}\left[[(1-e)c_{1}+ec_{2}]\ell(f(x),1)+[(1-e)c_{2}+ec_{1}]\ell(f(x),0)\right]$ |  |
|  |  | $\displaystyle\quad\quad+\mathbb{E}_{x.y\=1}\left[[(e-e)(c_{2}-c_{1})]\ell(f(x),1)-[(e-e)(c_{2}-c_{1})]\ell(f(x),0)\right]$ |  |
|  |  | $\displaystyle\=\min_{f}\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[[(1-e)c_{1}+ec_{2}]\ell(f(x),y)+[(1-e)c_{2}+ec_{1}]\ell(f(x),1-y)\right]$ |  |
|  |  | $\displaystyle\=\min_{f}\mathbb{E}_{(x,y)\sim\mathcal{D}}[(c_{1}+c_{2})\ell(f(x),y)]$ |  |
|  |  | $\displaystyle\quad\quad+[(1-e)c_{2}+ec_{1}]\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\ell(f(x),1-y)-\ell(f(x),y)\right]$ |  |
|  |  | $\displaystyle\=\min_{f}\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),y)]+(1-\gamma-e+2\gamma e)\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),1-y)-\ell(f(x),y)]$ |  |

Assume $\gamma^{*}$ is the optimal smooth parameter that makes the corresponding classifier return the best performance on unseen clean data distribution*(Wei et al., [2022](#bib.bib83 ""))*. Then the above equation can be converted to

|  |  | $\displaystyle\=\min_{f}\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),y^{\gamma^{*}})]$ |  | (42) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\quad\quad+(\gamma^{*}-\gamma-e+2\gamma e))\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),1-y)-\ell(f(x),y)],$ | | |

namely minimizing the smoothed loss with noisy labels is equal to optimizing two terms,

|  |  | $\displaystyle\min_{f}\underbrace{\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),y^{\gamma^{*}})]}_{\rm\lower 2.1097pt\hbox{ \leavevmode\hbox to11.67pt{\vbox to9.18pt{\pgfpicture\makeatletter\hbox{\hskip 5.83311pt\lower-4.58867pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}{{}}{}{{{}}{}{}{}{}{}{}{}{}}{{}}{}\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@moveto{3.50002pt}{0.0pt}\pgfsys@curveto{3.50002pt}{1.93303pt}{1.93303pt}{3.50002pt}{0.0pt}{3.50002pt}\pgfsys@curveto{-1.93303pt}{3.50002pt}{-3.50002pt}{1.93303pt}{-3.50002pt}{0.0pt}\pgfsys@curveto{-3.50002pt}{-1.93303pt}{-1.93303pt}{-3.50002pt}{0.0pt}{-3.50002pt}\pgfsys@curveto{1.93303pt}{-3.50002pt}{3.50002pt}{-1.93303pt}{3.50002pt}{0.0pt}\pgfsys@closepath\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@stroke\pgfsys@invoke{ }\hbox{\hbox{{\pgfsys@beginscope\pgfsys@invoke{ }{{}{}{{ {}{}}}{ {}{}} {{}{{}}}{{}{}}{}{{}{}} { }{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{-3.5pt}{-2.25555pt}\pgfsys@invoke{ }\hbox{{\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\hbox{{\makebox[7.00002pt][c]{\rm 1}}} }}\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}\text{ Risk under clean label}}+\underbrace{(\gamma^{*}-\gamma-e+2\gamma e))\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),1-y)-\ell(f(x),y)]}_{\rm\lower 2.1097pt\hbox{ \leavevmode\hbox to11.67pt{\vbox to9.18pt{\pgfpicture\makeatletter\hbox{\hskip 5.83311pt\lower-4.58867pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}{{}}{}{{{}}{}{}{}{}{}{}{}{}}{{}}{}\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@moveto{3.50002pt}{0.0pt}\pgfsys@curveto{3.50002pt}{1.93303pt}{1.93303pt}{3.50002pt}{0.0pt}{3.50002pt}\pgfsys@curveto{-1.93303pt}{3.50002pt}{-3.50002pt}{1.93303pt}{-3.50002pt}{0.0pt}\pgfsys@curveto{-3.50002pt}{-1.93303pt}{-1.93303pt}{-3.50002pt}{0.0pt}{-3.50002pt}\pgfsys@curveto{1.93303pt}{-3.50002pt}{3.50002pt}{-1.93303pt}{3.50002pt}{0.0pt}\pgfsys@closepath\pgfsys@moveto{0.0pt}{0.0pt}\pgfsys@stroke\pgfsys@invoke{ }\hbox{\hbox{{\pgfsys@beginscope\pgfsys@invoke{ }{{}{}{{ {}{}}}{ {}{}} {{}{{}}}{{}{}}{}{{}{}} { }{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{-3.5pt}{-2.25555pt}\pgfsys@invoke{ }\hbox{{\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\hbox{{\makebox[7.00002pt][c]{\rm 2}}} }}\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}\text{Reverse optimization}}$ |  | (43) |
| --- | --- | --- | --- | --- |


### A.8 Empirical Gap Analysis Adopted from Vapnik-Chervonenkis framework

###### Theorem 3.

(Lemma 1 in*(Ben-David et al., [2010](#bib.bib9 ""))*) Given Definition[5](#A1.E5 "In Definition 1. ‣ A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Definition[2](#Thmdefinition2 "Definition 2. ‣ A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), let $\mathcal{H}$ be a hypothesis class of VC dimension $d$. If empirical distributions $\hat{\mathcal{D}}_{S}$ and $\hat{\mathcal{D}}_{T}$ all have at least $n$ samples, then for any $\delta\in(0,1)$, with probability at least $1-\delta$,

|  | $d_{\mathcal{H}}(\mathcal{D}_{S},\mathcal{D}_{T})\leq\hat{d}_{\mathcal{H}}(\hat{\mathcal{D}}_{S},\hat{\mathcal{D}}_{T})+4\sqrt{\frac{d\log(2n)+\log\frac{2}{\delta}}{n}}$ |  | (44) |
| --- | --- | --- | --- |

Denote convex hull $\Lambda$ the set of mixture distributions, $\Lambda\={\bar{\mathcal{D}}_{Mix}:\bar{\mathcal{D}}_{Mix}\=\sum_{i\=1}^{M}\pi_{i}\mathcal{D}_{i},\pi_{i}\in\Delta}$, where $\Delta$ is standard $M-1$-simplex. The convex hull assumption is commonly used in domain generalization setting*(Zhang et al., [2021a](#bib.bib91 ""); Albuquerque et al., [2019](#bib.bib3 ""))*, while none of them focus on the empirical gap. Note that $d_{\mathcal{H}}(\bar{\mathcal{D}}_{Mix},\mathcal{D}_{T})$ in domain generalization setting is intractable for the unseen target domain $\mathcal{D}_{T}$ is unavailable during training. We thus need to convert $d_{\mathcal{H}}(\bar{\mathcal{D}}_{Mix},\mathcal{D}_{T})$ to a tractable objective. Let $\bar{\mathcal{D}}_{Mix}^{*}\=\sum_{i\=1}^{M}\pi_{i}^{*}\mathcal{D}_{i},(\pi^{*}_{0},\dots,\pi^{*}_{M})\in\Delta$, where $\pi^{*}_{0},\dots,\pi^{*}_{M}\=\arg\min_{\pi_{0},\dots,\pi_{M}}d_{\mathcal{H}}(\bar{\mathcal{D}}_{Mix},\mathcal{D}_{T})$, and $\bar{\mathcal{D}}_{Mix}^{*}$ is the element within $\Lambda$ which is closest to the unseen target domain. Then we have

|  | $\displaystyle d_{\mathcal{H}}(\bar{\mathcal{D}}_{Mix},\mathcal{D}_{T})$ | $\displaystyle\=2\sup_{h\in\mathcal{H}}\left\mid\mathbb{E}_{\mathbf{x}\sim\bar{\mathcal{D}}_{Mix}}[h(\mathbf{x})\=1]-\mathbb{E}_{\mathbf{x}\sim\mathcal{D}_{T}}[h(\mathbf{x})\=1]\right\mid$ |  | (45) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=2\sup_{h\in\mathcal{H}}\mid\mathbb{E}_{\mathbf{x}\sim\bar{\mathcal{D}}_{Mix}}[h(\mathbf{x})\=1]-\mathbb{E}_{\mathbf{x}\sim\bar{\mathcal{D}}_{Mix}^{*}}[h(\mathbf{x})\=1]$ | | |
|  |  | $\displaystyle\quad+\mathbb{E}_{\mathbf{x}\sim\bar{\mathcal{D}}_{Mix}^{*}}[h(\mathbf{x})\=1]-\mathbb{E}_{\mathbf{x}\sim\mathcal{D}_{T}}[h(\mathbf{x})\=1]\mid$ |  |
|  |  | $\displaystyle\leq d_{\mathcal{H}}(\bar{\mathcal{D}}_{Mix}^{*},\mathcal{D}_{T})+d_{\mathcal{H}}(\bar{\mathcal{D}}_{Mix},\bar{\mathcal{D}}_{Mix}^{*})$ |  |

The explanation follows*(Zhang et al., [2021a](#bib.bib91 ""))* that the first term corresponds to “To what extent can the convex combination of the source domain approximate the target domain”. The minimization of the first term requires diverse data or strong data augmentation, such that the unseen distribution lies within the convex combination of source domains. We dismiss this term in the following because it includes $\mathcal{D}_{T}$ and cannot be optimized.
Follows Lemma 1 in*(Albuquerque et al., [2019](#bib.bib3 ""))*, the second term can be bounded by,

|  | $d_{\mathcal{H}}(\bar{\mathcal{D}}_{Mix},\bar{\mathcal{D}}_{Mix}^{*})\leq\sum_{i\=1}^{M}\sum_{j\=1}^{M}\pi_{i}\pi_{j}^{*}d_{\mathcal{H}}(\mathcal{D}_{i},\mathcal{D}_{j})\leq\max_{i,j\in[M]}d_{\mathcal{H}}(\mathcal{D}_{i},\mathcal{D}_{j}),$ |  | (46) |
| --- | --- | --- | --- |

namely the second term can be bounded by the combination of pairwise $\mathcal{H}$-divergence between source domains. The cost (Equ. ([1](#S2.E1 "In 2 Methodology ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"))) used for the multi-domain adversarial training can be seen as an approximation of such a target. Until now, we can bound the empirical gap with the help of Theorem[44](#A1.E44 "In Theorem 3. ‣ A.8 Empirical Gap Analysis Adopted from Vapnik-Chervonenkis framework ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")

|  | $\displaystyle\sum_{i\=1}^{M}\sum_{j\=1}^{M}\pi_{i}\pi_{j}^{*}d_{\mathcal{H}}(\mathcal{D}_{i},\mathcal{D}_{j})\leq\sum_{i\=1}^{M}\sum_{j\=1}^{M}\pi_{i}\pi_{j}^{*}\left[\hat{d}_{\mathcal{H}}(\hat{\mathcal{D}}_{i},\hat{\mathcal{D}}_{j})+4\sqrt{\frac{d\log(2\min(n_{i},n_{j}))+\log\frac{2}{\delta}}{\min(n_{i},n_{j})}}\right]$ |  | (47) |
| --- | --- | --- | --- |
| | $\displaystyle\left\mid\sum_{i\=1}^{M}\sum_{j\=1}^{M}\pi_{i}\pi_{j}^{*}d_{\mathcal{H}}(\mathcal{D}_{i},\mathcal{D}_{j})-\sum_{i\=1}^{M}\sum_{j\=1}^{M}\pi_{i}\pi_{j}^{*}\hat{d}_{\mathcal{H}}(\hat{\mathcal{D}}_{i},\hat{\mathcal{D}}_{j}\right\mid\leq 4\sqrt{\frac{d\log(2n^{*})+\log\frac{2}{\delta}}{n^{*}}}$ | | |

where $n_{i}$ is the number of samples in $\mathcal{D}_{i}$ and $n^{*}\=\min(n_{1},\dots,n_{M})$.

### A.9 Empirical Gap Analysis Adopted from Neural Net Distance

###### Proposition 6.

(Adapted from Theorem A.2 in*(Arora et al., [2017](#bib.bib7 ""))*) Let ${\mathcal{D}_{i}}_{i\=1}^{M}$ a set of distributions and ${\hat{\mathcal{D}}_{i}}_{i\=1}^{M}$ be empirical versions with at least $n^{*}$ samples each. We assume that the set of discriminators with softmax activation function $\hat{h}(\theta;\cdot)\=(\hat{h}_{1}(\theta_{1},\cdot),\dots,\hat{h}_{M}(\theta_{M},\cdot))\in\hat{\mathcal{H}}:\mathcal{Z}\rightarrow[0,1]^{M};\sum_{i\=1}^{M}\hat{h}_{i}(\theta_{i};\cdot)\=1$222There might be some confusion here because in Section[A.4](#A1.SS4 "A.4 Training Stability Brought by Environment Label Smoothing ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") we use $\theta$ as the parameters of encoder $h$. The usage is just for simplicity but does not mean that $h,g$ have the same parameters. are $L$-Lipschitz with respect to the parameters $\theta$ and use $p$ denote the number of parameter $\theta_{i}$. There is a universal constant $c$ such that when $n^{*}\geq\frac{cpM\log(Lp/\epsilon)}{\epsilon}$, we have with probability at least $1-\exp(-p)$ over the randomness of ${\hat{\mathcal{D}}_{i}}_{i\=1}^{M}$,

|  | $\mid d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})-d_{\hat{h},g}(\hat{\mathcal{D}}_{1},\dots,\hat{\mathcal{D}}_{M})\mid\leq\epsilon$ |  | (48) |
| --- | --- | --- | --- |

###### Proof.

For simplicity, we ignore the parameter $\theta_{i}$ when using $h_{i}(\cdot)$. According to the following triangle inequality, below we focus on the term $\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\mathbf{z})\mid$ and other terms have the same results.

|  |  | $\displaystyle\mid d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})-d_{\hat{h},g}(\hat{\mathcal{D}}_{1},\dots,\hat{\mathcal{D}}_{M})\mid$ |  | (49) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\left\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\mathbf{z})+\dots+\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{M}}\log\hat{h}_{M}(\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\mathbf{z})-\dots-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{M}}\log\hat{h}_{M}(\mathbf{z})\right\mid$ | | |
|  |  | $\displaystyle\leq\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\mathbf{z})\mid+\dots+\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{M}}\log\hat{h}_{M}(\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{M}}\log\hat{h}_{M}(\mathbf{z})\mid$ |  |

Let $\Phi$ be a finite set such that every $\theta_{1}\in\Theta$ is within distance $\frac{\epsilon}{4LM}$ of a $\theta_{1}\in\Phi$, which is also termed a $\frac{\epsilon}{4LM}$-net. Standard construction given a $\Phi$ satisfying $\log|\Phi|\leq O(p\log(Lp/\epsilon))$, namely there aren’t too many distinct discriminators in $\Phi$. By Chernoff bound, we have

|  | $\text{Pr}\left[\left\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\mathbf{z})\right\mid\geq\frac{\epsilon}{2M}\right]\leq 2\exp({-\frac{n^{*}\epsilon}{2M}})$ |  | (50) |
| --- | --- | --- | --- |

Therefore, when $n^{*}\geq\frac{cpM\log(Lp/\epsilon)}{\epsilon}$ for large enough constant $c$, we can union bound over all $\theta_{1}\in\Phi$. With probability at least $1-\exp(-p)$, for all $\theta_{1}\in\Phi$, we have $\left\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\mathbf{z})\right\mid\leq\frac{\epsilon}{2M}$. Then for every $\theta_{1}\in\Theta$, we can find a $\theta_{1}^{\prime}\in\Phi$ such that $||\theta_{1}-\theta_{1}^{\prime}||\leq\epsilon/4LM$. Therefore

|  |  | $\displaystyle\left\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\theta_{1};\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\theta_{1};\mathbf{z})\right\mid$ |  | (51) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\leq\left\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\theta_{1}^{\prime};\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\theta_{1}^{\prime};\mathbf{z})\right\mid$ | | |
|  |  | $\displaystyle\quad\quad\quad+\left\mid\mathbb{E}_{\mathbf{z}\sim g\circ\mathcal{D}_{1}}\log\hat{h}_{1}(\theta_{1}^{\prime};\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ{\mathcal{D}}_{1}}\log\hat{h}_{1}(\theta_{1};\mathbf{z})\right\mid$ |  |
|  |  | $\displaystyle\quad\quad\quad+\left\mid\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\theta_{1}^{\prime};\mathbf{z})-\mathbb{E}_{\mathbf{z}\sim g\circ\hat{\mathcal{D}}_{1}}\log\hat{h}_{1}(\theta_{1};\mathbf{z})\right\mid$ |  |
|  |  | $\displaystyle\leq\frac{\epsilon}{2M}+\frac{\epsilon}{4M}+\frac{\epsilon}{4M}\=\frac{\epsilon}{M}$ |  |

Namely we have

|  | $\mid d_{\hat{h},g}(\mathcal{D}_{1},\dots,\mathcal{D}_{M})-d_{\hat{h},g}(\hat{\mathcal{D}}_{1},\dots,\hat{\mathcal{D}}_{M})\mid\leq M\times\frac{\epsilon}{M}\=\epsilon$ |  | (52) |
| --- | --- | --- | --- |

The result verifies that for the multi-domain adversarial training, the expectation over the empirical distribution converges to the expectation over the true distribution for all discriminators given enough data samples.
∎

### A.10 Convergence theory

In this subsection, we first provide some preliminaries before domain adversarial training convergence analysis. We then show simultaneous gradient descent DANN is not stable near the equilibrium but alternating gradient descent DANN could converge with a sublinear convergence rate, which support the importance of training encoder and discriminator separately. Finally, when incorporated with environment label smoothing, alternating gradient descent DANN is shown able to attain a faster convergence speed.

#### A.10.1 Preliminaries

The asymptotic convergence analysis is defined as applying the “ordinary differential equation (ODE) method” to analyze the convergence properties of dynamic systems. Given a discrete-time system characterized by the gradient descent:

|  | $F_{\eta}(\theta^{t}):\=\theta^{t+1}\=\theta^{t}+\eta h(\theta^{t}),$ |  | (53) |
| --- | --- | --- | --- |

where $h(\cdot):\mathbb{R}\rightarrow\mathbb{R}$ is the gradient and $\eta$ is the learning rate. The important technique for analyzing asymptotic convergence analysis is Hurwitz condition *(Khalil., [1996](#bib.bib36 ""))*: if the Jacobian of the dynamic system $A\triangleq h^{\prime}(\theta)_{|\theta\=\theta^{*}}$ at a stationary point $\theta^{*}$ is Hurwitz, namely the real part of every eigenvalue of $A$ is positive then the continuous gradient dynamics are asymptotically stable.

Given the same discrete-time system and Jacobian $A$, to ensure the non-asymptotic convergence, we need to provide an appropriate range of $\eta$ by solving $|1+\lambda_{i}(A)|<1,\forall\lambda_{i}\in Sp(A)$, where $Sp(A)$ is the spectrum of $A$. Namely, we can get constraint of the learning rate, which thus is able to evaluate the minimum number of iterations for an $\epsilon$-error solution and could more precisely reveal the convergence performance of the dynamic system than the asymptotic analysis*(Nie \& Patel, [2020](#bib.bib53 ""))*.

###### Theorem 4.

(Proposition 4.4.1 in*(Bertsekas, [1999](#bib.bib10 ""))*.) Let $F:\Omega\rightarrow\Omega$ be a continuously differential function on an open subset $\Omega$ in $\mathbb{R}$ and let $\theta\in\Omega$ be so that

1. $F_{\eta}(\theta^{*})\=\theta^{*}$, and

2. the absolute values of the eigenvalues of the Jacobian $|\lambda_{i}|<1,\forall\lambda_{i}\in Sp(F_{\eta}^{\prime}(\theta^{*}))$.

Then there is an open neighborhood $U$ of $\theta^{*}$ so that for all $\theta^{0}\in U$, the iterates $\theta^{k+1}\=F_{\eta}(\theta^{k})$ is locally converge to $\theta^{*}$. The rate of convergence is at least linear. More precisely, the error $\parallel\theta^{k}-\theta^{*}\parallel$ is in $\mathcal{O}(|\lambda_{max}|^{k})$ for $k\rightarrow\infty$ where $\lambda_{max}$ is the eigenvalue of $F_{\eta}^{\prime}(\theta^{*})$ with the largest absolute value. When $|\lambda_{i}|>1$, $F$ will not converge and when $|\lambda_{i}|\=1$, $F$ is either converge with a sublinear convergence rate or cannot converge.

Finding fixed points of $F_{\eta}(\theta)\=\theta+\eta h(\theta)$ is equivalent to finding solutions to the nonlinear equation $h(\theta)\=0$ and the Jacobian is given by:

|  | $F_{\eta}^{\prime}(\theta)\=I+\eta h^{\prime}(\theta),$ |  | (54) |
| --- | --- | --- | --- |

where both $F_{\eta}^{\prime}(\theta),h^{\prime}(\theta)$ are not symmetric and can therefore have complex eigenvalues. The following Theorem shows when a fixed point of $F$ satisfies the conditions of Theorem[4](#Thmtheo4 "Theorem 4. ‣ A.10.1 Preliminaries ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing").

###### Theorem 5.

(Lemma 4 in*(Mescheder et al., [2017](#bib.bib48 ""))*.) Assume $A\triangleq h^{\prime}(\theta)_{|\theta\=\theta^{*}}$ only has eigenvalues with negative real-part and let $\eta>0$, then the eigenvalues of the matrix $I+\eta A$ lie in the unit ball if and only if

|  | $\eta<\frac{2a}{a^{2}+b^{2}}\=\frac{1}{|a|}\frac{2}{1+(\frac{b}{a})^{2}};\forall\lambda\=-a+bi\in Sp(A)$ |  | (55) |
| --- | --- | --- | --- |

Namely, both the maximum value of $a$ and $b/a$ determine the maximum possible learning rate. Although*(Acuna et al., [2021](#bib.bib1 ""))* shows domain adversarial training is indeed a three-player game among classifier, feature encoder, and domain discriminator, it also indicates that the complex eigenvalues with a large imaginary component are originated from encoder-discriminator adversarial training. Hence here we only focus on the two-player zero-sum game between the feature encoder, and domain discriminator. One interesting thing is that, from non-asymptotic convergence analysis, we can get a result (Theorem [55](#A1.E55 "In Theorem 5. ‣ A.10.1 Preliminaries ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) that is very similar to that from the Hurwitz condition (Corollary 1 in*(Acuna et al., [2021](#bib.bib1 ""))*: $\eta<\frac{-2a}{b^{2}-a^{2}};\forall\lambda\=a+bi\in Sp(A)$ and $|a|<|b|$).

#### A.10.2 A Simple Adversarial Training Example

According to Ali Rahimi’s test of times award speech at NIPS 17, simple experiments, simple theorems are the building blocks that help us understand more complicated systems. Along this line, we propose this toy example to understand the convergence of domain adversarial training. Denote $\mathcal{D}_{S}\=x_{s},\mathcal{D}_{t}\=x_{t}$ two Dirac distribution where both $x_{1}$ and $x_{2}$ are float number. In this setting, both the encoder and discriminator have exactly one parameter, which is $\theta_{e},\theta_{d}$ respectively333One may argue that neural networks are non-linear, but Theorem 4.5 from*(Khalil., [1996](#bib.bib36 ""))* shows that one can “linearize” any non-linear system near equilibrium and analyze the stability of the linearized system to comment on the local stability of the original system.. The DANN training objective in Equ. ([7](#A1.E7 "In A.1 Connect Environment Label Smoothing to JS Divergence Minimization ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is given by

|  | $d_{\theta}\=f(\theta_{d}\theta_{e}x_{s})+f(-\theta_{d}\theta_{e}x_{t}),$ |  | (56) |
| --- | --- | --- | --- |

where $f(t)\=\log\left(1/(1+\exp(-t))\right)$ and the unique equilibrium point of the training objective in Equ. ([56](#A1.E56 "In A.10.2 A Simple Adversarial Training Example ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is given by $\theta^{*}_{e}\=\theta^{*}_{d}\=0$. We then recall the update operators of simultaneous and alternating Gradient Descent, for the former, we have

|  | $F_{\eta}(\theta)\=\left(\begin{array}[]{c}\theta_{e}-\eta\nabla_{\theta_{e}}d_{\theta}\\ \theta_{d}+\eta\nabla_{\theta_{d}}d_{\theta}\\ \end{array}\right)$ |  | (57) |
| --- | --- | --- | --- |

For the latter, we have $F_{\eta}\=F_{\eta,2}(\theta)\circ F_{\eta,1}(\theta)$, and $F_{\eta,1},F_{\eta,2}$ are defined as

|  | $F_{\eta,1}(\theta)\=\left(\begin{array}[]{c}\theta_{e}-\eta\nabla_{\theta_{e}}d_{\theta}\\ \theta_{d}\\ \end{array}\right),F_{\eta,2}(\theta)\=\left(\begin{array}[]{c}\theta_{e}\\ \theta_{d}+\eta\nabla_{\theta_{d}}d_{\theta}\\ \end{array}\right),$ |  | (58) |
| --- | --- | --- | --- |

If we update the discriminator $n_{d}$ times after we update the encoder $n_{e}$ times, then the update operator will be $F_{\eta}\=F^{n_{e}}_{\eta,1}(\theta)\circ F^{n_{d}}_{\eta,1}(\theta)$. To understand convergence of simultaneous and alternating gradient descent, we have to understand when the Jacobian of the corresponding update operator has only eigenvalues with absolute value smaller than 1.

#### A.10.3 Simultaneous gradient descent DANN

###### Proposition 7.

The unique equilibrium point of the training objective in Equ. ([56](#A1.E56 "In A.10.2 A Simple Adversarial Training Example ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is given by $\theta^{*}_{e}\=\theta^{*}_{d}\=0$. Moreover, the Jacobian of $F_{\eta}(\theta)\=\left(\begin{array}[]{c}\theta_{e}-\eta\nabla_{\theta_{e}}d_{\theta}\\
\theta_{d}+\eta\nabla_{\theta_{d}}d_{\theta}\\
\end{array}\right)$ at the equilibrium point has the two eigenvalues

|  | $\lambda_{1/2}\=1\pm\frac{\eta}{2}|x_{s}-x_{t}|i,$ |  | (59) |
| --- | --- | --- | --- |

namely $F_{\eta}(\theta)$ will never satisfies the second conditions of Theorem[4](#Thmtheo4 "Theorem 4. ‣ A.10.1 Preliminaries ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") whatever $\eta$ is, which shows that this continuous system is generally not linearly convergent to the equilibrium point.

###### Proof.

The Jacobian of $F_{\eta}(\theta)\=\left(\begin{array}[]{c}\theta_{e}-\eta\nabla_{\theta_{e}}d_{\theta}\\
\theta_{d}+\eta\nabla_{\theta_{d}}d_{\theta}\\
\end{array}\right)$ is

|  |  | $\displaystyle\nabla_{\theta}F_{\eta}(\theta)\=\nabla_{\theta}\left(\begin{array}[]{c}\theta_{e}-\eta\left(\theta_{d}x_{s}f^{\prime}(\theta_{d}\theta_{e}x_{s})-\theta_{d}x_{t}f^{\prime}(\theta_{d}\theta_{e}x_{t})\right)\\ \theta_{d}+\eta\left(\theta_{e}x_{s}f^{\prime}(\theta_{d}\theta_{e}x_{s})-\theta_{e}x_{t}f^{\prime}(\theta_{d}\theta_{e}x_{t})\right)\\ \end{array}\right)$ |  | (60) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\=\left(\begin{array}[]{cc}1\&-\eta\left(x_{s}f^{\prime}(\theta_{d}\theta_{e}x_{s})-x_{t}f^{\prime}(\theta_{d}\theta_{e}x_{t})\right)\\ \eta\left(x_{s}f^{\prime}(\theta_{d}\theta_{e}x_{s})-x_{t}f^{\prime}(\theta_{d}\theta_{e}x_{t})\right)\&1\\ \end{array}\right)$ | | |
|  |  | $\displaystyle\=\left(\begin{array}[]{cc}1\&-\frac{\eta}{2}\left(x_{s}-x_{t}\right)\\ \frac{\eta}{2}\left(x_{s}-x_{t}\right)\&1\\ \end{array}\right),$ |  |

The derivation result of $\nabla_{\theta_{e}}\theta_{e}-\eta\left(\theta_{d}x_{s}f^{\prime}(\theta_{d}\theta_{e}x_{s})-\theta_{d}x_{t}f^{\prime}(\theta_{d}\theta_{e}x_{t})\right)$ should have been

|  | $1-\eta\left(\theta^{2}_{d}x^{2}_{s}f^{\prime\prime}(\theta_{d}\theta_{e}x_{s})-\theta^{2}_{d}x^{2}_{t}f^{\prime\prime}(\theta_{d}\theta_{e}x_{t})\right)$ |  | (61) |
| --- | --- | --- | --- |

Since the equilibrium point $(\theta_{e}^{*},\theta_{d}^{*})\=(0,0)$, for points near the equilibrium, we ignore high-order infinitesimal terms e.g., $\theta_{e}^{2},\theta_{d}^{2},\theta_{e}\theta_{d}$. We can thus obtain the derivation of the second line. The eigenvalues of the second-order matrix $A\=\left(\begin{array}[]{cc}a\&b\\
c\&d\\
\end{array}\right)$
are $\lambda\=\frac{a+d\pm\sqrt{(a+d)^{2}-4(ad-bc)}}{2}$, and then the eigenvalues of $\nabla_{\theta}F_{\eta}(\theta)$ is $1\pm\frac{\eta}{2}|x_{s}-x_{t}|i$. Obviously $|\lambda|>1$ and the proposition is completed.
∎

#### A.10.4 Alternating gradient descent DANN

###### Proposition 8.

The unique equilibrium point of the training objective in Equ. ([56](#A1.E56 "In A.10.2 A Simple Adversarial Training Example ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is given by $\theta^{*}_{e}\=\theta^{*}_{d}\=0$. If we update the discriminator $n_{d}$ times after we update the encoder $n_{e}$ times. Moreover, the Jacobian of $F_{\eta}\=F_{\eta,2}(\theta)\circ F_{\eta,1}(\theta)$ (Equ. ([58](#A1.E58 "In A.10.2 A Simple Adversarial Training Example ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"))) has eigenvalues

|  | $\lambda_{1/2}\=1-\frac{\alpha^{2}}{2}\pm\sqrt{\left(1-\frac{\alpha^{2}}{2}\right)^{2}-1},$ |  | (62) |
| --- | --- | --- | --- |

where $\alpha\=\frac{1}{2}\sqrt{n_{d}n_{e}}\eta|x_{s}-x_{t}|$. $|\lambda_{1/2}|\=1$ for $\eta\leq\frac{4}{\sqrt{n_{e}n_{d}}|x_{s}-x_{t}|}$ and $|\lambda_{1/2}|>1$ otherwise. Such result indicates that although alternating gradient descent does not converge linearly to the Nash-equilibrium, it could converge with a sublinear convergence rate.

###### Proof.

The Jacobians of alternating gradient descent DANN operators (Equ. ([58](#A1.E58 "In A.10.2 A Simple Adversarial Training Example ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"))) near the equilibrium are given by:

|  | $\nabla_{\theta}F_{\eta,1}(\theta)\=\left(\begin{array}[]{cc}1\&-\eta\left(x_{s}f^{\prime}(\theta_{d}\theta_{e}x_{s})-x_{t}f^{\prime}(\theta_{d}\theta_{e}x_{t})\right)\\ 0\&1\\ \end{array}\right)\=\left(\begin{array}[]{cc}1\&-\frac{\eta}{2}\left(x_{s}-x_{t}\right)\\ 0\&1\\ \end{array}\right),$ |  | (63) |
| --- | --- | --- | --- |

similarly we can get
$\nabla_{\theta}F_{\eta,2}(\theta)\=\left(\begin{array}[]{cc}1\&0\\
\frac{\eta}{2}\left(x_{s}-x_{t}\right)\&1\\
\end{array}\right).$
As a result, the Jacobian of the combined update operator $\nabla_{\theta}F_{\eta}(\theta)$ is

|  | $\nabla_{\theta}F_{\eta}(\theta)\=\nabla_{\theta}F^{n_{e}}_{\eta,2}(\theta)\nabla_{\theta}F^{n_{d}}_{\eta,1}(\theta)\=\left(\begin{array}[]{cc}1\&-\frac{\eta n_{e}}{2}\left(x_{s}-x_{t}\right)\\ \frac{\eta n_{d}}{2}\left(x_{s}-x_{t}\right)\&-\frac{\eta n_{d}n_{e}}{4}\left(x_{s}-x_{t}\right)^{2}+1\\ \end{array}\right).$ |  | (64) |
| --- | --- | --- | --- |

An easy calculation shows that the eigenvalues of this matrix are

|  | $\lambda_{1/2}\=1-\frac{n_{e}n_{d}}{8}\eta^{2}(x_{s}-x_{t})^{2}\pm\sqrt{\left(1-\frac{n_{e}n_{d}}{8}\eta^{2}(x_{s}-x_{t})^{2}\right)^{2}-1}$ |  | (65) |
| --- | --- | --- | --- |

Let $\alpha\=\frac{1}{2}\sqrt{n_{d}n_{e}}\eta|x_{s}-x_{t}|$, we can get $\lambda_{1/2}\=1-\frac{\alpha^{2}}{2}\pm\sqrt{\left(1-\frac{\alpha^{2}}{2}\right)^{2}-1}$. If $\left(1-\frac{\alpha^{2}}{2}\right)^{2}>1$, namely $\alpha>2$, then $|\lambda_{1/2}|\=\sqrt{2\left(1-\frac{\alpha^{2}}{2}\right)^{2}-1}$. To satisfy $|\lambda|<1$, we have $\left(1-\frac{\alpha^{2}}{2}\right)^{2}<1$, which conflicts with the assumption. That is $\alpha\leq 2$, and in this case $|\lambda_{1/2}|\=1$.
∎

#### A.10.5 Alternating gradient descent DANN+ELS

Incorporate environment label smoothing to Equ. ([56](#A1.E56 "In A.10.2 A Simple Adversarial Training Example ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")), the target is revised into:

|  | $d_{\theta,\gamma}\=\gamma f(\theta_{d}\theta_{e}x_{s})+(1-\gamma)f(-\theta_{d}\theta_{e}x_{s})+\gamma f(-\theta_{d}\theta_{e}x_{t})+(1-\gamma)f(\theta_{d}\theta_{e}x_{t}),$ |  | (66) |
| --- | --- | --- | --- |

###### Proposition 9.

The unique equilibrium point of the training objective in Equ. ([66](#A1.E66 "In A.10.5 Alternating gradient descent DANN+ELS ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing")) is given by $\theta^{*}_{e}\=\theta^{*}_{d}\=0$. If we update the discriminator $n_{d}$ times after we update the encoder $n_{e}$ times. Moreover, the Jacobian of $F_{\eta}\=F_{\eta,2}(\theta)\circ F_{\eta,1}(\theta)$ (Equ. ([58](#A1.E58 "In A.10.2 A Simple Adversarial Training Example ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"))) has eigenvalues

|  | $\lambda_{1/2}\=1-\frac{\alpha^{2}}{2}\pm\sqrt{\left(1-\frac{\alpha^{2}}{2}\right)^{2}-1},$ |  | (67) |
| --- | --- | --- | --- |

where $\alpha\=\frac{2\gamma-1}{2}\sqrt{n_{d}n_{e}}\eta|x_{s}-x_{t}|$. $|\lambda_{1/2}|\=1$ for $\eta\leq\frac{4}{\sqrt{n_{d}n_{e}}|x_{s}-x_{t}|}\frac{1}{2\gamma-1}$ and $|\lambda_{1/2}|>1$ otherwise. Such result indicates that alternating gradient descent DANN+ELS could converge faster than alternating gradient descent DANN.

###### Proof.

The operator for alternating gradient descent DANN+ELS is $F_{\eta}\=F_{\eta,2}(\theta)\circ F_{\eta,1}(\theta)$, and $F_{\eta,1},F_{\eta,2}$ near the equilibrium are given by:

|  |  | $\displaystyle F_{\eta,1}(\theta)\=\left(\begin{array}[]{c}\theta_{e}-\eta\nabla_{\theta_{e}}d_{\theta,\gamma}\\ \theta_{d}\\ \end{array}\right)\=\left(\begin{array}[]{c}\theta_{e}-\eta\left(\gamma\theta_{d}x_{s}f^{\prime}(0)-(1-\gamma)\theta_{d}x_{s}f^{\prime}(0)-\gamma\theta_{d}x_{t}f^{\prime}(0)+(1-\gamma)\theta_{d}x_{t}f^{\prime}(0)\right)\\ \theta_{d}\\ \end{array}\right)$ |  | (68) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle F_{\eta,2}(\theta)\=\left(\begin{array}[]{c}\theta_{e}\\ \theta_{d}+\eta\nabla_{\theta_{d}}d_{\theta,\gamma}\\ \end{array}\right)\=\left(\begin{array}[]{c}\theta_{e}\\ \theta_{d}+\eta\left(\gamma\theta_{e}x_{s}f^{\prime}(0)-(1-\gamma)\theta_{e}x_{s}f^{\prime}(0)-\gamma\theta_{e}x_{t}f^{\prime}(0)+(1-\gamma)\theta_{e}x_{t}f^{\prime}(0)\right)\\ \end{array}\right),$ | | |

The Jacobians of alternating gradient descent DANN+ELS operators near the equilibrium are given by:

|  | $\nabla_{\theta}F_{\eta,1}(\theta)\=\left(\begin{array}[]{cc}1\&-\frac{\eta(2\gamma-1)}{2}\left(x_{s}-x_{t}\right)\\ 0\&1\\ \end{array}\right),\nabla_{\theta}F_{\eta,2}(\theta)\=\left(\begin{array}[]{cc}1\&0\\ \frac{\eta(2\gamma-1)}{2}\left(x_{s}-x_{t}\right)\&1\\ \end{array}\right),$ |  | (69) |
| --- | --- | --- | --- |

As a result, the Jacobian of the combined update operator $\nabla_{\theta}F_{\eta}(\theta)$ is

|  | $\nabla_{\theta}F_{\eta}(\theta)\=\nabla_{\theta}F^{n_{e}}_{\eta,2}(\theta)\nabla_{\theta}F^{n_{d}}_{\eta,1}(\theta)\=\left(\begin{array}[]{cc}1\&-\frac{\eta n_{e}(2\gamma-1)}{2}\left(x_{s}-x_{t}\right)\\ \frac{\eta n_{d}(2\gamma-1)}{2}\left(x_{s}-x_{t}\right)\&-\frac{\eta n_{d}n_{e}(2\gamma-1)^{2}}{4}\left(x_{s}-x_{t}\right)^{2}+1\\ \end{array}\right).$ |  | (70) |
| --- | --- | --- | --- |

An easy calculation shows that the eigenvalues of this matrix are

|  | $\lambda_{1/2}\=1-\frac{n_{e}n_{d}}{8}\eta^{2}(2\gamma-1)^{2}(x_{s}-x_{t})^{2}\pm\sqrt{\left(1-\frac{n_{e}n_{d}}{8}\eta^{2}(2\gamma-1)^{2}(x_{s}-x_{t})^{2}\right)^{2}-1}$ |  | (71) |
| --- | --- | --- | --- |

Similarly to the proof of Proposition[8](#Thmprop8 "Proposition 8. ‣ A.10.4 Alternating gradient descent DANN ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), let $\alpha\=\frac{2\gamma-1}{2}\sqrt{n_{d}n_{e}}\eta|x_{s}-x_{t}|$, we can get $\lambda_{1/2}\=1-\frac{\alpha^{2}}{2}\pm\sqrt{\left(1-\frac{\alpha^{2}}{2}\right)^{2}-1}$. Only when $\alpha\leq 2$, $\lambda_{1/2}$ are on the unit circle, namely $\eta\leq\frac{4}{\sqrt{n_{d}n_{e}}|x_{s}-x_{t}|}\frac{1}{2\gamma-1}$. Compared to the result in Proposition[8](#Thmprop8 "Proposition 8. ‣ A.10.4 Alternating gradient descent DANN ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), which is $\eta\leq\frac{4}{\sqrt{n_{d}n_{e}}|x_{s}-x_{t}|}$, the additional $\frac{1}{2\gamma-1}>1$ enables us to choose more large learning rate and could converge to an small error solution by fewer iterations.
∎

*Table 9: Hyper-parameters for different benchmarks. $Lr_{g},Decay_{g}$: learning rate and weight decay for the encoder and classifier; $Lr_{d},Decay_{d}$: learning rate and weight decay for the domain discriminator; $bsz$: batch size during training; $d_{steps}$: the discriminator is trained $d_{steps}$ times once the encoder and classifier are trained; $W_{reg}$: tradeoff weight for the gradient penalty; $\lambda$: tradeoff weight for the adversarial loss. The default $\beta_{2}$ for Adam and AdamW optimizer is 0.99 and the momentum for SGD optimizer is 0.9. / means domain discriminators trained on the dataset use GRL but not alternating gradient descent.*

TaskDatsets$Lr_{g}$$Lr_{d}$$\beta_{1}$$Decay_{g}$$Decay_{d}$bsz$d_{steps}$$W_{reg}$$\lambda$ImagesClassificationRotated MNIST1E-031E-030.50E+000.064110.5PACS5E-055E-050.50E+000.032510.5VLCS5E-055E-050.50E+000.032510.5Office-31(ResNet50)1E-021E-02SGD1E-31E-332/01.00Office-Home (ResNet50)1E-021E-02SGD1E-31E-332/01.00Office-31 (ViT)2E-032E-03SGD1E-31E-324/01.00Office-Home (ViT)2E-032E-03SGD1E-31E-324/01.00Rotating MNIST2E-042E-040.95E-045E-04100102.00ImageRetrievalMS1E-021E-02SGD5E-045E-0480101.00Neural LanguageProcessingCivilComments1E-051E-05SGD1E-02016101.00Amazon1E-042E-04AdamW1E-0208100.11Genomicsand GraphRxRx11E-042E-040.91E-05072100.11OGB-MolPCBA8E-041E-020.91E-05032100.11SequentialPredictionSpurious-Fourier4E-044E-0401E-0307831.251.56HHAR3E-031E-030.50E+0001343.512

Appendix B Extended Related Works
---------------------------------

Domain adaptation and domain generalization *(Muandet et al., [2013](#bib.bib50 ""); Sagawa et al., [2019](#bib.bib62 ""); Li et al., [2018a](#bib.bib42 ""); Blanchard et al., [2021](#bib.bib11 ""); Li et al., [2018b](#bib.bib45 ""); Zhang et al., [2021a](#bib.bib91 ""))* aims to learn a model that can extrapolate well in unseen environments. Representative methods like AT method*(Ganin et al., [2016](#bib.bib22 ""))* proposed the idea of learning domain-invariant representations as an adversarial game. This approach led to a plethora of methods including state-of-the-art approaches*(Zhang et al., [2019](#bib.bib95 ""); Acuna et al., [2021](#bib.bib1 ""); [2022](#bib.bib2 ""))*. In this paper, we propose a simple but effective trick, ELS, which benefits the generalization performance of methods by using soft environment labels.

Adversarial Training in GANs is well studied and many theoretical results of GANs motivate the analysis in this paper. e.g., divergence minimization interpretation*(Goodfellow et al., [2014](#bib.bib27 ""); Nguyen et al., [2017](#bib.bib52 ""))*, generalization of the discriminator*(Arora et al., [2017](#bib.bib7 ""); Thanh-Tung et al., [2019](#bib.bib74 ""))*, training stability*(Thanh-Tung et al., [2019](#bib.bib74 ""); Schäfer et al., [2019](#bib.bib67 ""); Arjovsky \& Bottou, [2017](#bib.bib4 ""); Arjovsky et al., [2017](#bib.bib5 ""))*, nash equilibrium*(Farnia \& Ozdaglar, [2020](#bib.bib19 ""); Nagarajan \& Kolter, [2017](#bib.bib51 ""))*, and gradient descent in GAN optimization*(Nagarajan \& Kolter, [2017](#bib.bib51 ""); Gidel et al., [2018](#bib.bib25 ""); Chen et al., [2018](#bib.bib15 ""))*. Multi-domain image generation is also related to this work, generalization to the JSD metric has been explored to address this challenge*(Gan et al., [2017](#bib.bib21 ""); Pu et al., [2018](#bib.bib56 ""); Trung Le et al., [2019](#bib.bib76 ""))*. However, most of them have to build $\frac{M(M-1)}{2}$ pairwise critics, which is expensive when $M$ is large. $\chi^{2}$ GAN*(Tao et al., [2018](#bib.bib73 ""))* firstly attempts to tackle the challenge and only needs $M-1$ critics.

Appendix C Additional Experimental Setups
-----------------------------------------

### C.1 Dataset Details and Experimental Settings

In this subsection, we introduce all the used datasets and the hyper-parameters for reproducing the experimental results in this work. We have uploaded the codes for all experiments in the supplementary materials to make sure that all the results are reproducible. All the main hyper-parameters for reproducing the experimental results in this work are shown in Table[9](#A1.T9 "Table 9 ‣ A.10.5 Alternating gradient descent DANN+ELS ‣ A.10 Convergence theory ‣ Appendix A Proofs of Theoretical Statements ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing").

#### C.1.1 Images Classification Datasets

Experimental settings. For DG and multi-source DG tasks, all the baselines are implemented using the codebase of Domainbed *(Gulrajani \& Lopez-Paz, [2021](#bib.bib29 ""))* and we use as encoders ConvNet for RotatedMNIST (detailed in Appdendix D.1 in *(Gulrajani \& Lopez-Paz, [2021](#bib.bib29 ""))*) and ResNet-50 for the remaining datasets. The model selection that we use is test-domain validation, one of the three selection methods in *(Gulrajani \& Lopez-Paz, [2021](#bib.bib29 ""))*. That is, we choose the model maximizing the accuracy on a validation set that follows the same distribution of the test domain. For DA tasks, all baselines implementation and hyper-parameters follows*([Wang \& Hou,](#bib.bib81 "") )*. For Continuously Indexed Domain Adaptation tasks, all baselines are implemented using PyTorch with the same architecture as*(Wang et al., [2020](#bib.bib80 ""))*. Note that although our theoretical analysis on non-asymptotic convergence is based on alternating Gradient Descent, current DA methods mainly build on Gradient Reverse Layer. For a fair comparison, in our experiments considering domain adaptation benchmarks, we also use GRL as default and let the analysis in future work.

Rotated MNIST *(Ghifary et al., [2015](#bib.bib24 ""))* consists of 70,000 digits in MNIST with different rotated angles where domain is determined by the degrees $d\in{0,15,30,45,60,75}$.

PACS *(Li et al., [2017b](#bib.bib41 ""))* includes 9, 991 images with 7 classes $y\in{$ dog, elephant, giraffe, guitar, horse, house, person $}$ from 4 domains $d\in$ ${$art, cartoons, photos, sketches$}$.

VLCS *(Torralba \& Efros, [2011](#bib.bib75 ""))* is composed of 10,729 images, 5 classes $y\in{$ bird, car, chair, dog, person $}$ from domains $d\in{$Caltech101, LabelMe, SUN09, VOC2007$}$.

Office-31 *(Saenko et al., [2010](#bib.bib61 ""))* contains contains $4,110$ images, 31 object categories in three domains: $d\in{$ Amazon, DSLR, and Webcam$}$.

Office-Home *(Venkateswara et al., [2017](#bib.bib79 ""))*: consists of 15,500 images from 65 classes and 4 domains: $d\in{$ Art (Ar), Clipart (Cl), Product (Pr) and Real World (Rw) $}$.

Rotating MNIST *(Wang et al., [2020](#bib.bib80 ""))* is adapted from regular MNIST digits with mild rotation to significantly Rotating MNIST digits. In our experiments, $[0^{\circ},45^{\circ})$ is set as the source domain and others are unlabeled target domains. The chosen baselines include Adversarial Discriminative Domain Adaptation (ADDA*(Tzeng et al., [2017](#bib.bib77 ""))*), and CIDA*(Wang et al., [2020](#bib.bib80 ""))*. ADDA merges data with different domain indices into one source and one target domain. DANN divides the continuous domain spectrum into several separate domains and performs adaptation between multiple source and target domains. For Rotating MNIST, the seven target domains contain images rotating by $d\in${$[0^{\circ},45^{\circ}),[45^{\circ},90^{\circ}),[90^{\circ},135^{\circ}),\dots,[315^{\circ},360^{\circ})$} degrees, respectively.

Circle Dataset *(Wang et al., [2020](#bib.bib80 ""))* includes 30 domains indexed from 1 to 30 and Figure[3(a)](#S4.F3.sf1 "In Figure 3 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") shows the 30 domains in different colors (from right to left is $1,\dots,30$ respectively). Each domain contains data on a circle and the task is binary classification. Figure[3(b)](#S4.F3.sf2 "In Figure 3 ‣ 4.1 Numerical Results on Different Settings and Benchmarks ‣ 4 Experiments ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") shows positive samples as red dots and negative samples as blue crosses. In our experiments, We use domains 1 to 6 as source domains and the rest as target domains.

#### C.1.2 Image Retrieval Datasets

Experimental settings. Following previous generalizable person ReID methods, we use MobileNetV2*(Sandler et al., [2018](#bib.bib64 ""))* with a multiplier of $1.4$ as the backbone network, which is pretrained on ImageNet*(Deng et al., [2009](#bib.bib18 ""))*. Images are resized to $256\times 128$ and the training batch size $N$ is set to $80$. The SGD optimizer is used to train all the components with a learning rate of $0.01$, a momentum of $0.9$ and a weight decay of $5\times 10^{-4}$. The learning rate is warmed up in the first $10$ epochs and decayed to its $0.1\times$ and $0.01\times$ at $40$ and $70$ epochs.

We evaluate the proposed method by Person re-identification (ReID) tasks, which aims to find the correspondences between person images from the same identity across multiple camera views. The training datasets include CUHK02*(Li \& Wang, [2013](#bib.bib43 ""))*, CUHK03*(Li et al., [2014](#bib.bib44 ""))*, Market1501*(Zheng et al., [2015](#bib.bib97 ""))*, DukeMTMC-ReID*(Zheng et al., [2017](#bib.bib98 ""))*, and CUHK-SYSU PersonSearch*(Xiao et al., [2016](#bib.bib86 ""))*. The unseen test domains are VIPeR*(Gray et al., [2007](#bib.bib28 ""))*, PRID*(Hirzer et al., [2011](#bib.bib31 ""))*, QMUL GRID*(Liu et al., [2012](#bib.bib46 ""))*, and i-LIDS*(Wei-Shi et al., [2009](#bib.bib84 ""))*. Details of the training datasets are summarized in Table[10](#A3.T10 "Table 10 ‣ C.1.2 Image Retrieval Datasets ‣ C.1 Dataset Details and Experimental Settings ‣ Appendix C Additional Experimental Setups ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and the test datasets are summarized in Table[11](#A3.T11 "Table 11 ‣ C.1.2 Image Retrieval Datasets ‣ C.1 Dataset Details and Experimental Settings ‣ Appendix C Additional Experimental Setups ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"). All the assets (i.e., datasets and the codes for baselines) we use include a MIT license containing a copyright notice and this permission notice shall be included in all copies or substantial portions of the software.

*Table 10: Training Datasets Statistics.*

| Dataset | IDs | Images |
| --- | --- | --- |
| CUHK02 | 1,816 | 7,264 |
| CUHK03 | 1,467 | 14,097 |
| DukeMTMC-Re-Id | 1,812 | 36,411 |
| Market-1501 | 1,501 | 29,419 |
| CUHK-SYSU | 11,934 | 34,547 |

*Table 11: Testing Datasets statistics.*

| Dataset | Probe | | Gallery | |
| --- | --- | --- | --- | --- |
| | Pr. IDs | Pr. Imgs | Ga. IDs | Ga. imgs |
| PRID | 100 | 100 | 649 | 649 |
| GRID | 125 | 125 | 1025 | 1,025 |
| VIPeR | 316 | 316 | 316 | 316 |
| i-LIDS | 60 | 60 | 60 | 60 |

GRID *(Liu et al., [2012](#bib.bib46 ""))* contains $250$ probe images and $250$ true match images of the probes in the gallery. Besides, there are a total of $775$ additional images that do not belong to any of the probes. We randomly take out $125$ probe images. The remaining $125$ probe images and $1025(775+250)$ images in the gallery are used for testing.

i-LIDS *(Wei-Shi et al., [2009](#bib.bib84 ""))* has two versions, images and sequences. The former is used in our experiments. It involves $300$ different pedestrian pairs observed across two disjoint camera views $1$ and $2$ in public open space. We randomly select $60$ pedestrian pairs, two images per pair are randomly selected as probe image and gallery image respectively.

PRID2011 *(Hirzer et al., [2011](#bib.bib31 ""))* has single-shot and multi-shot versions. We use the former in our experiments. The single-shot version has two camera views $A$ and $B$, which capture $385$ and $749$ pedestrians respectively. Only $200$ pedestrians appear in both views. During the evaluation, $100$ randomly identities presented in both views are selected, the remaining $100$ identities in view $A$ constitute probe set and the remaining $649$ identities in view $B$ constitute gallery set.

VIPeR *(Gray et al., [2007](#bib.bib28 ""))* contains $632$ pedestrian image pairs. Each pair contains two images of the same individual seen from different camera views $1$ and $2$. Each image pair was taken from an arbitrary viewpoint under varying illumination conditions. To compare to other methods, we randomly select half of these identities from camera view $1$ as probe images and their matched images in view $2$ as gallery images.

We follow the single-shot setting. The average rank-k (R-k) accuracy and mean Average Precision ($m$AP) over $10$ random splits are reported based on the evaluation protocol

#### C.1.3 Neural Language Datasets

CivilComments-Wilds *(Koh et al., [2021](#bib.bib38 ""))* contains $448,000$ comments on online articles taken from the Civil Comments platform. The input is a text comment and the task is to predicate whether the comment was rated as toxic, e.g., , the comment Maybe you should learn to write a coherent sentence so we can understand WTF your point is is rated as toxic and I applaud your father. He was a good man! We need more like him. is not. Domain in CivilComments-Wilds dataset is an 8-dimensional binary vector where each component corresponds to whether the comment mentions one of the 8 demographic identities {male, female, LGBTQ, Christian, Muslim, other religions, Black, White}.

Amazon-Wilds *(Koh et al., [2021](#bib.bib38 ""))* contains $539,520$ reviews from disjoint sets of users. The input is the review text and the task is to predict the corresponding 1-to-5 star rating from reviews of Amazon products. Domain $d$ identifies the user who wrote the review and the training set has $3,920$ domains. The 10-th percentile of per-user accuracies metric is used for evaluation, which is standard to measure model performance on devices and users at various percentiles in an effort to encourage good performance across many devices.

#### C.1.4 Genomics and Graph datasets

RxRx1-wilds *(Koh et al., [2021](#bib.bib38 ""))* comprises images of cells that have been genetically perturbed by siRNA, which comprises $125,510$ images of cells obtained by fluorescent microscopy. The output $y$ indicates which of the $1,139$ genetic treatments (including no treatment) the cells received, and $d$ specifies $51$ batches in which the imaging experiment was run.

OGB-MolPCBA *(Koh et al., [2021](#bib.bib38 ""))* is a multi-label classification dataset, which comprises $437,929$ molecules with $120,084$ different structural scaffolds. The input is a molecular graph, the label is a 128-dimensional binary vector where each component corresponds to a biochemical assay result, and the domain $d$ specifies the scaffold (i.e., a cluster of molecules with similar structure). The training and test sets contain molecules with disjoint scaffolds; The training set has molecules from over $40,000$ scaffolds. We evaluate models by averaging the Average Precision (AP) across each of the 128 assays.

#### C.1.5 Sequential data

Spurious-Fourier *(Gagnon-Audet et al., [2022](#bib.bib20 ""))* is a binary classification dataset ($y\in{$low-frequency peak (L) and high-frequency peak (H).$}$), which is composed of one-dimensional signal. Domains $d\in{10\%,80\%,90\%}$ contain signal-label pairs, where the label is a noisy function of the low- and high-frequencies such that low-frequency peaks bear a varying correlation of $d$ with the label and high-frequency peaks bear an invariant correlation of $75\%$ with the label.

HHAR *(Gagnon-Audet et al., [2022](#bib.bib20 ""))* is a 6 activities classification dataset ($y\in{$Stand, Sit, Walk, Bike, Stairs up, and Stairs Down $}$), which is composed of recordings of 3-axis accelerometer and 3-axis gyroscope data. Specifically, the input $x$ is recordings of 500 time-steps of a 6-dimensional signal sampled at 100Hz. Domain $d$ consist of five smart device models: $d\in{$Nexus 4, Galaxy S3, Galaxy S3 Mini, LG Watch, and Samsung Galaxy Gears$}$.

### C.2 Backbone Structures

Most of the backbones are ResNet-50/ResNet-18 and we follow the same setting as the reference works. Here we briefly introduce some special backbones used in our experiments,i.e., ConvNet for Rotated MNIST, EncoderSTN for Rotating MNIST, DistillBERT for Neural Language datasets, and GIN for OGB-MoIPCBA.

MNIST ConvNet. is detailed in Table.[12](#A3.T12 "Table 12 ‣ C.2 Backbone Structures ‣ Appendix C Additional Experimental Setups ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing").

DistillBERT. We use the implementation from*(Wolf et al., [2019](#bib.bib85 ""))* and finetune a BERT-base-uncased models for neural language datasets.

*Table 12: Details of our MNIST ConvNet architecture. All convolutions use 3×3 kernels and “same” padding*

| # | Layer |
| --- | --- |
| 1 | Conv2D (in\=d, out\=64) |
| 2 | ReLU |
| 3 | GroupNorm (groups\=8) |
| 4 | Conv2D (in\=64, out\=128, stride\=2) |
| 5 | ReLU |
| 6 | GroupNorm (8 groups) |
| 7 | Conv2D (in\=128, out\=128) |
| 8 | ReLU |
| 9 | GroupNorm (8 groups) |
| 10 | Conv2D (in\=128, out\=128) |
| 11 | ReLU |
| 12 | GroupNorm (8 groups) |
| 13 | Global average-pooling |

EncoderSTN use a four-layer convolutional neural network for the encoder and a three-layer MLP to make the prediction. The domain discriminator is a four-layer MLP. The encoder is incorporated with a Spacial Transfer Network (STN)*(Jaderberg et al., [2015](#bib.bib32 ""))*, which takes the image and the domain index as input and outputs a set of rotation parameters which are then applied to rotate the given image.

Graph Isomorphism Networks (GIN)*(Xu et al., [2018](#bib.bib87 ""))* combined with virtual nodes is used for OGB-MoIPCBA dataset, as this is currently the model with the highest performance in the Open Graph Benchmark.

Deep ConvNets*(Schirrmeister et al., [2017](#bib.bib65 ""))* for HHAR combines temporal and spatial convolution,which fits this data well and we use the implementation in the BrainDecode Schirrmeister*(Schirrmeister et al., [2017](#bib.bib65 ""))* Toolbox.

Appendix D Additional Experimental Results
------------------------------------------

### D.1 Additional Numerical Results

*Table 13: The domain generalization/adaptation accuracy on Rotated MNIST.*

Rotated MNISTAlgorithm01530456075AvgERM*(Vapnik, [1999](#bib.bib78 ""))*95.3 $\pm$ 0.298.7 $\pm$ 0.198.9 $\pm$ 0.198.7 $\pm$ 0.298.9 $\pm$ 0.096.2 $\pm$ 0.297.8IRM*(Arjovsky et al., [2019](#bib.bib6 ""))*94.9 $\pm$ 0.698.7 $\pm$ 0.298.6 $\pm$ 0.198.6 $\pm$ 0.298.7 $\pm$ 0.195.2 $\pm$ 0.397.5DANN*(Ganin et al., [2016](#bib.bib22 ""))*95.9 $\pm$ 0.198.6 $\pm$ 0.198.7 $\pm$ 0.299.0 $\pm$ 0.198.7 $\pm$ 0.096.5 $\pm$ 0.397.9ARM*(Zhang et al., [2021b](#bib.bib92 ""))*95.9 $\pm$ 0.499.0 $\pm$ 0.198.8 $\pm$ 0.198.9 $\pm$ 0.199.1 $\pm$ 0.196.7 $\pm$ 0.298.1DANN+ELS96.3 $\pm$ 0.198.7 $\pm$ 0.198.9 $\pm$ 0.399.1 $\pm$ 0.198.7 $\pm$ 0.096.9 $\pm$ 0.598.1$\uparrow$0.40.10.20.10.00.40.2

Multi-Source Domain Generalization. IRM*(Arjovsky et al., [2019](#bib.bib6 ""))* introduces specific conditions for an upper bound on the number of training environments required such that an invariant optimal model can be obtained, which stresses the importance of the number of training environments. In this paper, we reduce the training environments on the Rotated MNIST from five to three. As shown in Table[17](#A4.T17 "Table 17 ‣ D.1 Additional Numerical Results ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), as the number of training environment decreases, the performance of IRM fall sharply (e.g., the averaged accuracy from $97.5\%$ to $91.8\%$), and the performance on the most challenging domains $d\={0,5}$ decline the most ($94.9\%\rightarrow 80.9\%$ and $95.2\%\rightarrow 91.1\%$). In contrast, both ERM and DANN+ELS retain high generalization performances and DANN+ELS outperforms ERM in most domains.

Image Retrieval. We compare the proposed DANN+ELS with methods on a typical DG-ReID setting. As shown in Table[16](#A4.T16 "Table 16 ‣ D.1 Additional Numerical Results ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), we implement DANN with various hyper-parameters while DANN always fails to converge on ReID benchmarks. As illustrated in Appendix Figure[8](#A4.F8 "Figure 8 ‣ D.2 Additional Analysis and Interpretation ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), we compare the training statistics with the baseline, where DANN is highly unstable and attains inferior results. However, equipped with ELS and following the same hyper-parameter as DANN, DANN+ELS attains well-training stability and achieves either comparable or better performance when compared with recent state-of-the-art DG-ReID methods. See Appendix[D.2](#A4.SS2 "D.2 Additional Analysis and Interpretation ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") for t-sne visualization and comparison.

*Table 14: Domain generalization performance on the OGB-MolPCBA dataset.*

| OGB-MolPCBA | | |
| --- | --- | --- |
| Algorithm | Val Avg Acc | Test Avg Acc |
| ERM(Vapnik, [1999](#bib.bib78 "")) | 27.8 ± 0.1 | 27.2 ± 0.3 |
| Group DROSagawa et al. ([2019](#bib.bib62 "")) | 23.1 ± 0.6 | 22.4 ± 0.6 |
| CORALSun \& Saenko ([2016](#bib.bib70 "")) | 18.4 ± 0.2 | 17.9 ± 0.5 |
| IRM(Arjovsky et al., [2019](#bib.bib6 "")) | 15.8 ± 0.2 | 15.6 ± 0.3 |
| DANN(Ganin et al., [2016](#bib.bib22 "")) | 15.0 ± 0.6 | 14.1 ± 0.5 |
| DANN+ELS | 18.0 ± 0.3 | 17.2 ± 0.3 |
| $\uparrow$ | 3.0 | 3.1 |

*Table 15: Domain generalization performance on the Spurious-Fourier dataset.*

| Spurious-Fourier dataset | | |
| --- | --- | --- |
| Algorithm | Train validation | Test validation |
| ERM(Vapnik, [1999](#bib.bib78 "")) | 9.7 ± 0.3 | 9.3 ± 0.1 |
| IRM(Arjovsky et al., [2019](#bib.bib6 "")) | 9.3 ± 0.1 | 57.6 ± 0.8 |
| SDPezeshki et al. ([2021](#bib.bib55 "")) | 10.2 ± 0.1 | 9.2 ± 0.0 |
| VRExKrueger et al. ([2021](#bib.bib39 "")) | 9.7 ± 0.2 | 65.3 ± 4.8 |
| DANN(Ganin et al., [2016](#bib.bib22 "")) | 9.7 ± 0.1 | 11.1 ± 1.5 |
| DANN+ELS | 10.7 ± 0.6 | 15.6 ± 2.8 |
| $\uparrow$ | 1.0 | 4.5 |

*Table 16: Comparison with recent state-of-the-art DG-ReID methods. —— denotes DANN cannot converge and attains infinite loss.*

MethodsAverageVIPeRPRIDGRIDi-LIDSR-1$m$APR-1R-5R-10$m$APR-1R-5R-10$m$APR-1R-5R-10$m$APR-1R-5R-10$m$APDIMN*(Song et al., [2019](#bib.bib69 ""))*47.557.951.270.276.060.139.267.076.752.029.353.365.841.170.289.794.578.4DualNorm*(Jia et al., [2019](#bib.bib34 ""))*57.661.853.962.575.358.060.473.684.864.941.447.464.745.774.882.091.578.5DDAN*(Chen et al., [2021](#bib.bib14 ""))*59.063.152.360.671.856.454.562.774.958.950.662.173.855.778.585.392.581.5DIR-ReID*(Zhang et al., [2021c](#bib.bib93 ""))*63.871.258.576.983.367.069.785.891.077.148.267.176.357.679.094.897.283.4MetaBIN*(Choi et al., [2021](#bib.bib16 ""))*64.271.959.376.881.967.670.686.591.578.247.366.074.056.479.593.097.585.5Group DRO*(Sagawa et al., [2019](#bib.bib62 ""))*57.165.948.568.477.257.866.186.590.674.838.758.866.648.674.890.896.881.9Unit-DRO*(Zhang et al., [2022](#bib.bib94 ""))*65.472.860.078.282.868.473.585.391.779.447.569.377.457.280.794.097.086.2DANN*(Ganin et al., [2016](#bib.bib22 ""))*——————————DANN+ELS64.272.159.376.482.767.469.687.791.777.748.167.577.857.279.894.797.286.1

*Table 17: Generalization performance on multiple unseen target domains. $\uparrow$ denotes improvement of DANN+ELS compared to DANN, and $\gamma$ is the hyper-parameter for environment label smoothing.*

Rotated MNISTTarget domains ${0^{\circ},30^{\circ},60^{\circ}}$Target domains ${15^{\circ},45^{\circ},75^{\circ}}$Method0∘30∘60∘15∘45∘75∘AvgERM*(Vapnik, [1999](#bib.bib78 ""))*96.0 $\pm$ 0.398.8 $\pm$ 0.498.7 $\pm$ 0.198.8 $\pm$ 0.399.1 $\pm$ 0.196.7 $\pm$ 0.398.0IRM*(Arjovsky et al., [2019](#bib.bib6 ""))*80.9 $\pm$ 3.294.7 $\pm$ 0.994.3 $\pm$ 1.394.3 $\pm$ 0.895.5 $\pm$ 0.591.1 $\pm$ 3.191.8DANN*(Ganin et al., [2016](#bib.bib22 ""))*96.6 $\pm$ 0.298.8 $\pm$ 0.398.7 $\pm$ 0.198.6 $\pm$ 0.498.8 $\pm$ 0.296.9 $\pm$ 0.198.1DANN+ELS96.7 $\pm$ 0.498.9 $\pm$ 0.298.8 $\pm$ 0.198.8 $\pm$ 0.199.0 $\pm$ 0.297.0 $\pm$ 0.498.2$\uparrow$0.10.10.10.20.20.10.1

*Table 18: Generalization performance on sequential benchmarks. $\uparrow$ denotes improvement of DANN+ELS compared to DANN.*

HHARTrain-domain validationAlgorithmNexus 4Galazy S3Galaxy S3 MiniLG watchSam. GearAverageID ERM98.91±0.2498.44±0.1598.68±0.1590.08±0.2880.63±1.3393.35ERM97.64±0.1597.64±0.0992.51±0.4671.69±0.1461.94±1.0484.28IRM96.02±0.1795.75±0.2289.46±0.5066.49±0.9457.66±0.3781.08SD98.14±0.0198.32±0.1992.71±0.0975.12±0.1863.85±0.2885.63VREx95.81±0.5095.92±0.2390.72±0.1069.04±0.2356.42±1.5781.58DANN$94.45\pm 0.44$$95.05\pm 0.10$$88.70\pm 0.56$$68.33\pm 0.49$$58.45\pm 1.24$80.99DANN+ELS$95.95\pm 0.39$$95.65\pm 0.42$$90.50\pm 0.39$$69.55\pm 0.36$$58.45\pm 0.24$82.02$\uparrow$1.50.61.81.220.01.03Oracle train-domain validationAlgorithmNexus 4Galazy S3Galaxy S3 MiniLG watchSam. GearAverageID ERM98.91±0.2498.44±0.1598.68±0.1590.08±0.2880.63±1.3393.35ERM97.98±0.0297.92±0.0593.09±0.1571.96±0.0464.08±0.6685.01IRM96.02±0.1795.75±0.2289.91±0.2568.00±0.3457.77±0.4281.49SD98.48±0.0198.67±0.1194.36±0.2475.12±0.1864.86±0.2886.3VREx96.65±0.1896.30±0.0590.98±0.1669.39±0.2759.12±0.8082.49DANN$95.95\pm 0.21$$96.20\pm 0.07$$89.91\pm 0.73$$72.70\pm 0.63$$58.45\pm 1.77$82.64DANN+ELS$96.79\pm 0.13$$96.94\pm 0.13$$91.57\pm 0.22$$72.70\pm 0.63$$59.80\pm 0.84$83.56$\uparrow$0.840.741.660.01.350.92

### D.2 Additional Analysis and Interpretation

T-sne visualization. We compare the proposed DANN+ELS with MetaBIN and ERM through $t$-SNE visualization. We observe a distinct division of different domains in Figure[7(a)](#A4.F7.sf1 "In Figure 7 ‣ D.2 Additional Analysis and Interpretation ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing") and Figure[7(d)](#A4.F7.sf4 "In Figure 7 ‣ D.2 Additional Analysis and Interpretation ‣ Appendix D Additional Experimental Results ‣ Free Lunch for Domain Adversarial Training: Environment Label Smoothing"), which indicates that a domain-specific feature space is learned by the ERM. MetaBIN perform better than ERM and the proposed DANN+ELS can learn more domain-invariant representations while keeping discriminative capability for ReID tasks.

<img src='x13.png' alt='Refer to caption' title='' width='322' height='269' />

*Figure 6: Data examples from the PACS and the VLCS datasets.*

<img src='imgs/tsne/erm_all.png' alt='Refer to caption' title='' width='200' height='200' />

*(a) ERM (All).*

<img src='x14.png' alt='Refer to caption' title='' width='153' height='153' />

*(b) MetaBIN (All).*

<img src='imgs/tsne/dann_all.png' alt='Refer to caption' title='' width='200' height='200' />

*(c) DANN+ELS (All).*

<img src='imgs/tsne/erm_test.png' alt='Refer to caption' title='' width='200' height='200' />

*(d) ERM (Test)*

<img src='x15.png' alt='Refer to caption' title='' width='153' height='153' />

*(e) MetaBIN (Test).*

<img src='imgs/tsne/dann_test.png' alt='Refer to caption' title='' width='200' height='200' />

*(f) DANN+ELS (Test).*

*Figure 7: Visualization of the embeddings on training and test datasets. Query and gallery samples of these unseen datasets are shown using different types of mark. Best viewed in color.*

<img src='x16.png' alt='Refer to caption' title='' width='153' height='115' />

*(a) Domain discrimination loss.*

<img src='x17.png' alt='Refer to caption' title='' width='153' height='115' />

*(b) Identity classification loss.*

<img src='x18.png' alt='Refer to caption' title='' width='153' height='115' />

*(c) Average $m$AP on the test set.*

*Figure 8: Training statistics on ReID datasets.*

### D.3 Ablation Studies
