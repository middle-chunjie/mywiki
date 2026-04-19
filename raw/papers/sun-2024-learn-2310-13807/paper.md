Learning to (Learn at Test Time)
================================

Yu Sun†1,
 Xinhao Li∗2,
 Karan Dalal3,
 Chloe Hsu3,
 Sanmi Koyejo1,  
Carlos Guestrin1,
 Xiaolong Wang2,
 Tatsunori Hashimoto‡1,
 Xinlei Chen‡4  
1 Stanford University,2 UC San Diego,3 UC Berkeley,4 Meta AIEqual technical contribution.†Project lead.‡Equal advising.

###### Abstract

We reformulate the problem of supervised learning as learning to learn with two nested loops (i.e. learning problems).
The inner loop learns on each individual instance with self-supervision before final prediction.
The outer loop learns the self-supervised task used by the inner loop, such that its final prediction improves.
Our inner loop turns out to be equivalent to linear attention when the inner-loop learner is only a linear model, and to self-attention when it is a kernel estimator.
For practical comparison with linear or self-attention layers, we replace each of them in a transformer with an inner loop, so our outer loop is equivalent to training the architecture.
When each inner-loop learner is a neural network, our approach vastly outperforms transformers with linear attention on ImageNet from $224\times 224$ raw pixels in both accuracy and FLOPs, while (regular) transformers cannot run.111Code release: [https://github.com/test-time-training/mttt](https://github.com/test-time-training/mttt "").

1 Introduction
--------------

Test-time training (TTT) is an algorithmic framework for machine learning.
The core idea is that each test instance defines its own learning problem, with its own target of generalization*(Sun et al., [2020](#bib.bib51 ""))*.
Since the test instance comes without its label, TTT is performed with a self-supervised task such as reconstruction.
Performance should improve on this particular instance for the self-supervised task, because that is the objective optimized by TTT.
But will such a process lead to better performance for the main task we actually care about?

If improvement for a self-supervised task transfers to a given main task, we say the two tasks are *aligned* *(Sun et al., [2020](#bib.bib51 ""))*.
In prior work, task alignment has been an art, combining ingenuity with trial and error *(Gandelsman et al., [2022](#bib.bib23 ""); Wang et al., [2023](#bib.bib61 ""))*.
Crucially, the amount of ingenuity in task design does not scale with more data and compute.
Our main approach is to learn an aligned self-supervised task from data, instead of handcrafting it from human priors.
Specifically, we learn a self-supervised task such that TTT on it actually improves performance on the main task.

Since TTT already defines a learning problem, learning its self-supervised task is a form of *learning to learn*, i.e. meta-learning or bi-level optimization*(Schmidhuber, [1987](#bib.bib48 ""))*.
The literature refers to the two nested learning problems as the inner and outer loop.
At training time, the *inner loop* learns with self-supervision on each training instance individually, as if it were a test instance. The *outer loop* learns to align the self-supervised task with the main task on the entire training set.
At test time, we only invoke the inner loop, *i.e.* TTT.
We name our algorithm MTTT, with M for meta.

To better understand MTTT, we look at its simplest nontrivial instantiation,
where all components are linear models, and the inner loop takes only one gradient step.
Given fixed outer-loop parameters, the inner loop turns out to be equivalent to forward inference with linear attention, i.e. self-attention without softmax*(Katharopoulos et al., [2020](#bib.bib33 ""))*.
For a linear transformer, i.e. transformer with only linear attention layers, we can replace each with an inner loop.
Nesting multiple such inner loops into one outer loop, the most naive case of MTTT is equivalent to training a linear transformer.

It also turns out that our inner loop with a particular kernel estimator is theoretically equivalent to self-attention (with softmax), so MTTT with multiple such inner loops is equivalent to training a transformer.
This suggests that our framework is compatible with existing, successful architectures.
To extend beyond existing equivalences, we investigate TTT with neural networks.
This performs much better than TTT with linear models (i.e. linear transformers), in settings where transformers run out of memory and time.
Given the freedom inside our inner loop, we can augment it with heuristics like output normalization and stochastic gradient descent that improve results even more.

Our inner loop *mirrors* regular (non-meta) learning in design, because it breaks each instance into pieces, i.e. tokens, that are explicitly treated as data.
This perspective is further validated by our empirical evidence, which is not explained through any existing perspective for architecture design.
Given the historic success of deep learning over kernels and linear models, we conjecture that such success can potentially be replicated in our inner loop, with more compute and data under MTTT.

2 Inner Loop: Test-Time Training with Reconstruction
-----------------------------------------------------

The architecture for TTT has a shared feature extractor with two output heads.
The self-supervised task has a head $g$, and the main task has a head $h$.
At test time, the model can only learn from the self-supervised task, so the heads share a feature extractor $f$. This way, TTT can update the shared features, thus helping the main task if it uses the same kind of features as the self-supervised task.
Altogether, this architecture looks like the letter ‘Y’, where $f$ is the stem, $g$ and $h$ are the branches.

In principle, TTT is compatible with any choice of self-supervised task.
Here we focus on one general-purpose and domain-agnostic family of self-supervised tasks – reconstruction, since it has been highly effective in prior work*(Vincent et al., [2008](#bib.bib58 ""); Pathak et al., [2016](#bib.bib44 ""); Brown et al., [2020](#bib.bib12 ""); Bao et al., [2021](#bib.bib5 ""); He et al., [2021](#bib.bib26 ""))*.
For reconstruction, the feature extractor $f$ is also known as the encoder, and the self-supervised head $g$ as the decoder; $g\circ f$ together is called an autoencoder.

Following a standard process called tokenization, each instance is always broken into a sequence of $n$ tokens, so we denote both the instance and sequence by $X\=(x_{1},\dots,x_{n})$, with token $x_{i}\in\mathbb{R}^{d}$.222To be precise, $x_{i}\in\mathbb{R}^{d}$ is actually the token’s embedding, not the token itself.
For $X$ a paragraph of text, each token is usually a (sub-)word; for $X$ an image, each token is usually a patch or pixel.
While the type of tokens can potentially be non-numeric, standard techniques are available to embed them into vectors. Our basic unit of reconstruction is each individual token $x_{i}$.
The reconstruction target is $x_{i}$ itself, but the input is transformed by a given function $\phi$, such as adding noise*(Vincent et al., [2008](#bib.bib58 ""))* and random masking*(He et al., [2021](#bib.bib26 ""))*.
For each $X$, we optimize the parameters of $f$, denoted by $W$.
Overall, the self-supervised loss is

|  | $\ell(W;X)\=\frac{1}{2n}\sum_{i\=1}^{n}\big{\|}g\circ f\left(\phi(x_{i});W\right)-x_{i}\big{\|}^{2}.$ |  | (1) |
| --- | --- | --- | --- |

Note that the decoder $g$ is also considered given within the scope of TTT, which only updates $W$.333While the decoder $g$ also contains learnable parameters, we do not optimize them during TTT in this paper.
Our choice, although nonstandard for autoencoders, makes learning to learn conceptually easier in Section[3](#S3 "3 Outer Loop: Learning the Self-Supervised Task for TTT ‣ Learning to (Learn at Test Time)").
Moreover, *Sun et al. ([2020](#bib.bib51 ""))* and *Gandelsman et al. ([2022](#bib.bib23 ""))* have shown that whether or not $g$ is optimized during TTT makes little empirical difference.
In fact, for $T\=1$ (using notations defined for Equation[2](#S2.E2 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)")), whether or not a gradient step is taken on $g$ does not matter at all, because $g$ affects the final prediction only through $W_{1}$. Optimization is performed with $T$ gradient steps.
For each $t\=1,\dots,T$,

|  | $W_{t}\=W_{t-1}-\eta\nabla\ell(W_{t-1};X),$ |  | (2) |
| --- | --- | --- | --- |

where the initial value $W_{0}$ and the learning rate $\eta$ are given, like $\phi$ and $g$.

For the main task, we also transform its input $x_{i}$ by a given function $\psi$, in the spirit of symmetry to $\phi$ for the self-supervised task.
In prior work, $\psi$ has mostly been the identity transform, but Section[3](#S3 "3 Outer Loop: Learning the Self-Supervised Task for TTT ‣ Learning to (Learn at Test Time)") will make $\psi$ nontrivial, adding expressiveness to the outer loop.
Next, we produce the main task outputs by applying $h\circ f$ individually on each $\psi(x_{i})$.
For convenience, we overload $h,f$ and $\phi$ so they can produce an output sequence from an input sequence:

|  | $X_{\text{out}}\=h\circ f\left(\psi(X);W_{T}\right)\=\bigg{(}h\circ f\left(\psi(x_{1});W_{T}\right),\dots,h\circ f\left(\psi(x_{n});W_{T}\right)\bigg{)}.$ |  | (3) |
| --- | --- | --- | --- |

Equation[3](#S2.E3 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)") could be the last step for main tasks that require $n$ predictions (e.g. language modeling), but for other tasks that require a single prediction (e.g. object recognition),
it is standard to apply an aggregation function across the output sequence, predicting $\hat{y}\=\texttt{aggregate}(X_{\text{out}})$ in the end.

### 2.1 Context Window as a Dataset

In standard terminology, $X\=(x_{1},\dots,x_{n})$ is called the context window, and $n$ the window length.
But for TTT, $X$ is a dataset of size $n$, where each token $x_{i}$ is actually a non-independent and non-identically distributed piece of data.
This intuition is consistent with our algorithm: Equation[1](#S2.E1 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)") simply sums the losses individually across tokens, just like across pieces of data;
Equation[3](#S2.E3 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)") also processes each $x_{i}$ individually as a “test token”, like how a fixed model processes each test instance.

Tokenization enables us to reuse $f$ on $n$ different parts (tokens) of $X$, by treating them as pieces of data, and $X$ as a dataset.
It brings the units of operation for TTT “one level below” their traditional sense in machine learning, where $X$ is a piece of data, and a collection of $X$s is a dataset.
TTT can be applied without tokenization, but then $X$ would be singleton, unless augmentations are used to create an artificial batch like in *Sun et al. ([2020](#bib.bib51 ""))*.

3 Outer Loop: Learning the Self-Supervised Task for TTT
--------------------------------------------------------

As noted above, TTT does not modify
the initialization $W_{0}$ for encoder $f$,
the transformations $\phi$ and $\psi$, or the decoder $g$ and main task head $h$.
Altogether, these important components must be determined outside of the scope of TTT.
Prior work has tried various heuristics, discussed in Subsection[6.2](#S6.SS2 "6.2 Learning at Test Time ‣ 6 Related Work ‣ Learning to (Learn at Test Time)").
Here we take the more principled approach of directly optimizing the final prediction loss on the main task after $T$ steps of TTT.

We first explicitly express the learnable parameters that were hidden in Section[2](#S2 "2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)") because they were considered given within the scope of the inner loop.
These are the parameters of $g,h$, $\phi$ and $\psi$, denoted by $\theta_{g}$, $\theta_{h}$, $\theta_{\phi}$ and $\theta_{\psi}$.
We group them together with $W_{0}$ into $\bm{\theta}\=(\theta_{g},\theta_{h},\theta_{\phi},\theta_{\psi},W_{0})$, since they will all be learned in the outer loop.
Technically, $\bm{\theta}$ should also contain the learnable parameters of aggregate, which we omit for convenience.

Now we derive the outer-loop objective $\mathcal{L}_{T}$.
Denote the main task loss by $\mathcal{L}$, e.g. the cross-entropy loss.
In the trivial case, for $T\=0$, i.e. without TTT, the final prediction loss is exactly $\mathcal{L}$.
To be precise, for each instance $X$ with unknown label $y$,

|  | $\mathcal{L}_{0}\big{(}\bm{\theta};X,y\big{)}\=\mathcal{L}\big{(}h\circ f(\psi(X);W_{0}),y\big{)}.$ |  | (4) |
| --- | --- | --- | --- |

For $T\=1$, the parameters of $f$ become $W_{1}\=W_{0}-\eta\nabla\ell(W_{0};X)$, as defined in Equation[1](#S2.E1 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)").
Therefore, the final prediction loss for the main task is

|  | $\mathcal{L}_{1}\big{(}\bm{\theta};X,y\big{)}\=\mathcal{L}\big{(}h\circ f\left(\psi(X);W_{1}\right),y\big{)}\=\mathcal{L}\big{(}h\circ f\left(\psi(X);W_{0}-\eta\nabla\ell(W_{0};X)\right),y\big{)}.$ |  | (5) |
| --- | --- | --- | --- |

For any $T\geq 1$, $\theta_{g}$ and $\theta_{\phi}$ implicitly determine the inner-loop loss function $\ell$ defined in Equation[1](#S2.E1 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)"), therefore affect $\mathcal{L}_{T}$ through $\nabla\ell$.
In other words, $\theta_{g}$ and $\theta_{\phi}$ parameterize the self-supervised task.444Note that even though $\theta_{g}$ and $\theta_{\phi}$ are included as arguments of $\mathcal{L}_{T}$ for all values of $T$, they do not actually matter for $\mathcal{L}_{0}$.
When the inner loop is trivial, i.e runs for 0 iteration, learning to learn collapses to regular (non-meta) learning, and the self-supervised task does not matter. Going further, for $T\geq 2$,

|  | $\mathcal{L}_{T}\big{(}\bm{\theta};X,y\big{)}\=\mathcal{L}\big{(}h\circ f\left(\psi(X);W_{T}\right),y\big{)}$ |  | (6) |
| --- | --- | --- | --- |

would be cumbersome to write out in terms of $W_{0}$, but can be expressed recursively, with $W_{t}$ defined in Equation[2](#S2.E2 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)") for each $t\=1,\dots,T$.

At training time, the outer loop calculates $\mathcal{L}_{T}$ individually for each labeled training instance $X$, then optimizes the average $\mathcal{L}_{T}$ on the entire training set with (a variant of) stochastic gradient descent.
Calculating $\nabla\mathcal{L}(\bm{\theta};X,y)$ requires taking gradients through $\nabla\ell(W_{t};X)$ for $t\=0,\dots,T-1$.
, since the latter is implicitly a function of $W_{0}$, $\theta_{g}$ and $\theta_{\phi}$.
This turns out to be easily programmable in JAX, and surprisingly efficient in practice, as we will show in Section[5](#S5 "5 Experiments ‣ Learning to (Learn at Test Time)").

4 Choice of Learner for Inner Loop
----------------------------------

While our inner loop is a sequence of forward and backward operations, it can also be represented as a single forward operation on its unrolled computation graph, so the outer loop becomes regular (non-meta) learning using this graph as a fixed model.
It turns out that for simple choices of the inner-loop learner, this equivalent graph can be interpreted through the lens of architecture design.

### 4.1 TTT with Linear Models: Equivalence to Linear Attention

The simplest choice for the feature extractor $f$ is a linear model:

|  | $f(x;W)\=Wx.$ |  | (7) |
| --- | --- | --- | --- |

And the outer-loop components $g$, $h$, $\phi$ and $\psi$ are linear as well. Specifically,

|  | $g(x;\theta_{g})\=\theta_{g}^{T}x,~{}~{}h(x;\theta_{h})\=\theta_{h}x,~{}~{}\phi(x;\theta_{\phi})\=\theta_{\phi}x,~{}~{}\psi(x;\theta_{\psi})\=\theta_{\psi}x.$ |  | (8) |
| --- | --- | --- | --- |

To make the math even simpler, we always initialize the feature extractor with $W_{0}\=0$.
Under this construction, the self-supervised loss in Equation[1](#S2.E1 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)") becomes

|  | $\ell\big{(}W;X\big{)}\=\frac{1}{2n}\sum_{i\=1}^{n}\|g\circ f\left(\phi(x_{i});W\right)-x_{i}\|^{2}\=\frac{1}{2n}\sum_{i\=1}^{n}\|\theta_{g}^{T}W\theta_{\phi}x_{i}-x_{i}\|^{2}.$ |  | (9) |
| --- | --- | --- | --- |

For $W_{0}\=0$, one gradient step with learning rate $\eta\=1$ produces

|  | $W_{1}\=W_{0}-\nabla\ell\left(W_{0};X\right)\=\frac{1}{n}\sum_{i\=1}^{n}(\theta_{g}x_{i})(\theta_{\phi}x_{i})^{T}.$ |  | (10) |
| --- | --- | --- | --- |

Using $W_{1}$ as the updated weights for the feature extractor, the updated features for each token $x_{j}$, $j\=1,\dots,n$, becomes

|  | $f\left(\psi(x_{j});W_{1}\right)\=\frac{1}{n}\sum_{i\=1}^{n}(\theta_{g}x_{i})(\theta_{\phi}x_{i})^{T}\theta_{\psi}x_{j}.$ |  | (11) |
| --- | --- | --- | --- |

This happens to be linear attention (explained in Appendix[A](#A1 "Appendix A Linear Attention and Linear Transformers ‣ Learning to (Learn at Test Time)")), where $\theta_{\phi}$, $\theta_{\psi}$, $\theta_{g}$ are the key, query, value weights.
$h$ is the projection operation used for multi-head attention, discussed in Appendix[B](#A2 "Appendix B Multi-Head Attention ‣ Learning to (Learn at Test Time)").

### 4.2 TTT with Kernels: Equivalence to Self-Attention

So far, we have considered $f$ with explicit parameters.
But machine learning is more than just parametric models and gradient-based optimization.
Here we consider $f$ as a non-parametric learner.

Recall that non-parametric learning produces an algorithmic function controlled by the training data $x_{1},\dots,x_{n}$, without explicit parameters of a fixed shape.
So our notation for the encoder changes from $f(x;W)$ to $f(x;x_{1},\dots,x_{n})$.
For example, the nearest neighbor $f(x;x_{1},\dots,x_{n})$ simply looks for the most similar piece of training data.
Some other non-parametric learners are: support vector machines (SVMs), radial basis function networks, and kernel ridge regression.

But unlike most cases of non-parametric learning, our data for TTT come without labels, since $x_{1},\dots,x_{n}$ are just tokens of an unlabeled test instance $X$.
Analogous to parametric learners, non-parametric ones can also learn with self-supervision to produce better features for a main task downstream.
So for each $i\=1,\dots,n$, we create each label $z_{i}\=\theta_{V}x_{i}$ from the unlabeled input $x_{i}$ itself, where $\theta_{V}$ is an outer-loop parameter like $\theta_{g}$ in the parametric case.

The popular self-attention (with softmax) is equivalent to TTT with $f$ as the time-honored Nadaraya-Watson estimator*(Bierens, [1988](#bib.bib9 ""); Cai, [2001](#bib.bib13 ""))*, which outputs a locally weighted average of labels $z_{i}$, $i\=1,\dots,n$, using a kernel $\kappa$ as the weighting function:

|  | $f(x;x_{1},\dots,x_{n})\=\frac{1}{\sum_{i\=1}^{n}\kappa(x,x_{i})}\sum_{i\=1}^{n}\kappa(x,x_{i})~{}z_{i}.$ |  | (12) |
| --- | --- | --- | --- |

See Appendix[C](#A3 "Appendix C Our Kernel Estimator ‣ Learning to (Learn at Test Time)") for a detailed derivation of this estimator.
We choose the kernel $\kappa$ to be

|  | $\kappa(x,x^{\prime};\theta_{K},\theta_{Q})\propto e^{(\theta_{K}x)^{T}\theta_{Q}x^{\prime}}$ |  | (13) |
| --- | --- | --- | --- |

where $\theta_{K}$ and $\theta_{Q}$ are known as bandwidth hyper-parameters for kernels. But for MTTT, they are outer-loop parameters like $\theta_{V}$.
As detailed in Appendix[C](#A3 "Appendix C Our Kernel Estimator ‣ Learning to (Learn at Test Time)"),
asymmetric kernels like our $\kappa$ above have enjoyed a long tradition*(Breiman et al., [1977](#bib.bib11 ""); Chen, [2017](#bib.bib15 ""))*.
Altogether, Equation[12](#S4.E12 "In 4.2 TTT with Kernels: Equivalence to Self-Attention ‣ 4 Choice of Learner for Inner Loop ‣ Learning to (Learn at Test Time)") and [13](#S4.E13 "In 4.2 TTT with Kernels: Equivalence to Self-Attention ‣ 4 Choice of Learner for Inner Loop ‣ Learning to (Learn at Test Time)") combined is the same as self-attention, where $\theta_{K},\theta_{Q}$, $\theta_{V}$ are the key, query, value weights.

Unlike the parametric case, TTT with kernels does not solve an optimization problem, therefore does not produce a different implementation from self-attention.
While our equivalence here only provides an alternative interpretation, the fact that both linear models and kernels are empirically effective as inner-loop learners suggests that other learners might also be effective.

### 4.3 TTT with Neural Networks

From the past three decades of progress in machine learning, we observe that the performance of

|  | $\textit{deep~{}learning}~{}>~{}\textit{kernels}~{}>~{}\textit{linear models}$ |  |
| --- | --- | --- |

given enough data and compute.
In Subsection[2.1](#S2.SS1 "2.1 Context Window as a Dataset ‣ 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)"), we discussed the perspective that our inner loop mirrors regular (non-meta) learning, at least in terms of algorithmic design.
To collect empirical evidence for this perspective, we investigate if the ordering above is preserved within our inner loop.

It is well known that transformers with self-attention (TTT with kernels) often outperform those with linear attention (TTT with linear models), i.e. linear transformers*(Katharopoulos et al., [2020](#bib.bib33 ""))*.
This validates the rightmost link of the ordering within our inner loop.
But TTT with neural networks has no existing equivalence, so we devote the rest of the paper to taking a small step in this huge search space.
We delay implementation details such as architecture and optimization to Section[5](#S5 "5 Experiments ‣ Learning to (Learn at Test Time)"), and end this subsection with one remaining conceptual implication.

TTT with neural networks and linear models, or any parametric learner, has complexity linear in $n$ for each test instance $X\=(x_{1},\dots,x_{n})$, since complexity for each token is constant in $n$, and only proportional to the number of parameters.
TTT with any non-parametric learner, however, cannot have linear complexity by definition, since its complexity for each token cannot be constant in $n$, i.e. amount of training data.
For Nadaraya-Watson, complexity for each token happens to be linear.
This serves as an alternative explanation for the quadratic complexity of self-attention.

5 Experiments
-------------

The goal of our experiments is not to be the top on leaderboards, but to evaluate our key perspective, that the inner loop mirrors regular (non-meta) learning, in terms of three qualities.
1) *Descriptive*: Does our equivalence to linear attention hold in practice?
2) *Prescriptive*: Does our perspective show a path for new methods with better performance?
3) *Predictive*: Does our perspective accurately explain the empirical behaviors of new methods?

##### TTT layers.

The cleanest and most practical way to answer these questions is to replace every attention layer in an architecture with a TTT inner loop, because ultimately, attention layers are only used as parts of an architecture.
Since the inner loop here functions as a *drop-in replacement* for attention, we call it a *TTT layer*, which can also be thought of as an equivalent computation graph (discussed in Section[4](#S4 "4 Choice of Learner for Inner Loop ‣ Learning to (Learn at Test Time)")).
After dropping in the TTT layers, the entire architecture can be trained with MTTT, using the same recipe as that with attention layers, without TTT.

##### Variants of MTTT.

We call our method *MTTT-Linear* when encoder $f$ is linear in each TTT layer, and *MTTT-MLP* when $f$ is a multi-layer perception (MLP).
We always keep $g,h,\phi,\psi$ linear following Subsection[4.1](#S4.SS1 "4.1 TTT with Linear Models: Equivalence to Linear Attention ‣ 4 Choice of Learner for Inner Loop ‣ Learning to (Learn at Test Time)").
For MTTT-Linear, we always keep $W_{0}\=0$ fixed to ensure equivalence to linear attention, since MTTT-Linear is only used to investigate descriptiveness.
For MTTT-MLP, we experiment with the two design choices below, to investigate the prescriptive power of our perspective.
For simplicity, we always set the inner-loop learning rate $\eta\=1$.

##### Inner-loop architecture.

For MTTT-MLP, the MLP architecture simply follows standard design in transformers.
Concretely, our MLP has 2 linear layers with GELU activation in between; the input and output dimension are the same, and the hidden dimension is $4\times$ as large.
The only architectural change, called *Decoder LN*, is that we add a layer norm (LN) after the output of $g$, to normalize the reconstruction outputs, in the spirit of *He et al. ([2021](#bib.bib26 ""))*.
We explain this design choice in Figure[2](#A2.F2 "Figure 2 ‣ Appendix B Multi-Head Attention ‣ Learning to (Learn at Test Time)"), deferred to the appendix due to space constraints.

##### Inner-loop optimization.

When the inner loop takes $T>1$ steps, each gradient step, by default, uses the average loss over all the tokens, defined in Equation[1](#S2.E1 "In 2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)").
But $T$ steps make the inner loop $T\times$ slower.
Given the popularity of stochastic gradient descent (SGD) in deep learning, we use it for our inner loop.
Specifically, we randomly split the $n$ tokens into $T$ mini-batches, each of size $T/n$, and take one inner-loop step per mini-batch.
Therefore, $T$ steps of SGD combined consumes the same amount of compute as a full-batch gradient step over all the $n$ tokens together.

| Drop-in layer | Acc. (%) | Params. (M) | FLOPs |
| --- | --- | --- | --- |
| Linformer(Wang et al., [2020b](#bib.bib62 "")) | 71.9 | 22.2 | 0.9$\times$ |
| Longformer(Beltagy et al., [2020](#bib.bib6 "")) | 76.3 | 27.4 | 1.1$\times$ |
| SOFT(Lu et al., [2021](#bib.bib37 "")) | 74.6 | 23.5 | 0.9$\times$ |
| Hyena(Poli et al., [2023](#bib.bib45 "")) | 74.8 | 23.5 | 1.0$\times$ |
| Self-attn.(Beyer et al., [2022](#bib.bib8 "")) | 76.5 | 22.1 | 1.1$\times$ |
| Linear attn. (Katharopoulos et al.) | 73.2 | 22.1 | 1.0$\times$ |
| Linear attn. identity map | 73.0 | 22.1 | 1.0$\times$ |
| MTTT-Linear | 72.8 | 22.1 | 1.1$\times$ |
| MTTT-MLP | 74.6 | 24.6 | 1.5$\times$ |

*Table 1:  Results on ImageNet.
FLOPs are presented as relative to linear attention.
Our inner-loop dataset is tiny, with $n\=196$.
MTTT-Linear matches linear attention with identity map, as expected.
MTTT-MLP outperforms both by a nontrivial margin, but is $1.5\times$ slower than linear attention.
Also as expected, self-attention, i.e. the original ViT performs the best.
See Subsection[5.2](#S5.SS2 "5.2 ImageNet from 224×224 Raw Pixels ‣ 5 Experiments ‣ Learning to (Learn at Test Time)") for details.*

### 5.1 ImageNet

We first experiment with the standard setting of ImageNet object recognition*(Deng et al., [2009](#bib.bib19 ""))*.
Our benchmark architecture is Vision Transformer (ViT)*(Dosovitskiy et al., [2020](#bib.bib20 ""))*.
We adopt the well-known recipe of *Beyer et al. ([2022](#bib.bib8 ""))* by the ViT authors, and their recommended setup for fast research turnaround – training ViT-Small for 90 epochs.
With an accuracy of 76.5%, it is often regarded as a fast and competitive baseline.
Its recipe splits each image into $14\times 14$ patches, then embeds each patch with a learned projection.
So each $X$ becomes $n\=196$ tokens.

Thinking of the context window as training data for TTT, a dataset of size 196 is not nearly enough for deep learning, if adequate for a linear model.
Since over-parameterized neural networks are known to be able to regularize themselves*(Zhang et al., [2021](#bib.bib64 ""))*,
MTTT-MLP should not do poorly, but might not justify the extra compute.
In addition, small $n$ means our linear complexity is less of an advantage, in comparison to self-attention (with softmax).

Our results in Table[1](#S5.T1 "Table 1 ‣ Inner-loop optimization. ‣ 5 Experiments ‣ Learning to (Learn at Test Time)") confirm those expectations.
MTTT-MLP outperforms MTTT-Linear by a small margin, but uses more FLOPs.
If MTTT-MLP was using a smaller architecture that matches the FLOPs of MTTT-Linear, it would have performed worse.
Self-attention, for which the training recipe was originally designed, performs the best.

In terms of descriptiveness, MTTT-Linear almost exactly matches linear attention (identity map) – the 0.2% difference is likely due to random noise and loss of numeric precision.
However, MTTT-Linear uses $0.1\times$ more FLOPs as linear attention.
This extra factor exists because the JAX compiler is unaware that the compiled inner loop will receive $W_{0}\=0$ so all those terms involved can be eliminated. We manually calculated the total number of FLOPs for those terms involving $W_{0}$, and found that it matches the difference in FLOPs between MTTT-Linear and linear attention.

Taking more gradient steps in the inner loop significantly improves accuracy of MTTT-MLP up to $T\=4$, as shown in the left panel of Figure[1](#S5.F1 "Figure 1 ‣ 5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)").
However, $T$ steps on the full batch costs $T\times$ number of FLOPs.
So this improvement is predictive but not practically useful.
We have experimented with SGD and found that it does not help here.
Since $n\=196$ is already a small batch size, splitting 196 tokens into even smaller mini-batches for SGD is usually considered bad practice for deep learning.

The right panel of Figure[1](#S5.F1 "Figure 1 ‣ 5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)") shows the average $\ell(W_{t};X)$ across the test set, for TTT layer 6 (out of 12 in total).
The plot for all layers is deferred to Figure[3](#A3.F3 "Figure 3 ‣ Appendix C Our Kernel Estimator ‣ Learning to (Learn at Test Time)") in the appendix due to space constraints, but the overall behavior is essentially the same across layers.
The five lines are for $t\=0,\dots,T$, where $T\=4$, i.e. the optimal choice of $T$ according to the left panel.
For every epoch of outer-loop learning, average inner-loop loss decreases monotonically with more steps.
The behavior of this novel inner loop matches that of regular learning with successful optimization.

While MTTT has not been practically useful in this setting, its behavior matches our expectations, indicating that our perspective is predictive on top of descriptive.
Note that every hyper-parameter is set according to *Beyer et al. ([2022](#bib.bib8 ""))*, and we have not changed any to get the expected behavior.
Our inner-loop learning rate $\eta$ has always been 1, derived from equivalence to linear attention.

In Table[2](#S5.T2 "Table 2 ‣ 5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)"), we ablate MTTT-MLP with the four combinations of whether or not to use Decoder LN and train $W_{0}$ in the outer loop.
We choose these two factors since Decoder LN is our own design, and training $W_{0}$ goes a step further from equivalence to linear attention, which requires fixing $W_{0}\=0$.
Empirically, both components prove to be important for good performance.
Therefore, we always keep them for future experiments, without spending more resources to ablate them.

For additional context around our results, we run a few baselines that also have linear complexity. Linear attention as proposed by *Katharopoulos et al. ([2020](#bib.bib33 ""))* uses manually engineered features of the input tokens, instead of the input tokens themselves. We label the former with citation, and the latter with *identity map*. Other baselines have roughly the same accuracy as MTTT-MLP. Longformer stands out with the same accuracy as self-attention, but we find that the default window size for its sliding attention is $512>196$, so it happens to be the same as self-attention for $n\=196$.

<img src='x1.png' alt='[Uncaptioned image]' title='' width='423' height='317' />

<img src='x2.png' alt='[Uncaptioned image]' title='' width='423' height='258' />

*Figure 1: More inner-loop steps improve accuracy up to $T\=4$ (left).
Behavior of inner-loop loss mirrors regular (non-meta) learning (right).*

| Dec. LN | Train $W_{0}$ | Acc. (%) |
| --- | --- | --- |
| ✗ | ✗ | 72.9 |
| ✗ | ✓ | 73.0 |
| ✓ | ✗ | 73.8 |
| ✓ | ✓ | 74.6 |

*Table 2: Ablations on ImageNet. 
See Subsection[5.1](#S5.SS1 "5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)") for details.*

### 5.2 ImageNet from $224\times 224$ Raw Pixels

To better evaluate our perspective that the inner loop mirrors regular (non-meta) learning, we need a setting where the sequence length $n$, i.e. amount of training data for the inner loop, is actually comparable to the amount in typical applications of deep learning.
Inspired by*Chen et al. ([2020](#bib.bib14 ""))*, we experiment with ImageNet object recognition using raw pixels instead of patches as input tokens.
This gives us $n\=224\times 224\=50,176$.

For *Chen et al. ([2020](#bib.bib14 ""))*, the point of using pixels is to eliminate image-specific prior knowledge.555While transformers have already eliminated the locality prior in convolutions, most papers on ImageNet still use patches instead of pixels as input tokens.
This is equivalent to a first layer of convolutions where the filter size and stride size both equal to the patch size, and is in fact often implemented as such.
Using raw pixels as input tokens eliminates locality prior completely. At a high level, the progress in deep learning over the past decade can be seen as gradually eliminating human priors, in favor of general methods that take advantage of data and compute.
Following their setting, we use learned positional embeddings, instead of engineered positional encoding.
Therefore, our entire system is permutation invariant.

While *Chen et al. ([2020](#bib.bib14 ""))* do not use any data augmentation, they use a much larger collection of images.
We have been able to remove the augmentations except one – random resize crop*(Szegedy et al., [2015](#bib.bib53 ""))*, without which all methods fail to get more than 40% accuracy.
Since random resize crop does not add any synthetic artifact to natural images, we justify it as using more data without actually using another dataset.
We always use random resize crop for the rest of the subsection.

Experiments in this subsection are conducted with ViT-Tiny unless noted otherwise, because training with 50k tokens per instance is very compute-intensive.
Every other aspect of our recipe follows *Beyer et al. ([2022](#bib.bib8 ""))*, like in Subsection[5.1](#S5.SS1 "5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)").
Our results are in Table[3](#S5.T3 "Table 3 ‣ 5.2 ImageNet from 224×224 Raw Pixels ‣ 5 Experiments ‣ Learning to (Learn at Test Time)").
Self-attention, which performed the best with patches, cannot fit in memory. Even if memory was not an issue, it would still need at least $200\times$ more FLOPs than linear attention according to our estimations.

We highlight two results.
First, taking $T\=4$ steps of SGD improves accuracy by 3.3% on top of MTTT-MLP with $T\=1$, without costing extra FLOPs.
To the best of our knowledge, this improvement cannot be explained through any existing perspective without an explicit inner loop.
Like in Figure[1](#S5.F1 "Figure 1 ‣ 5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)"), our inner-loop loss with SGD steps also behaves like regular learning, as shown in Figure[4](#A3.F4 "Figure 4 ‣ Appendix C Our Kernel Estimator ‣ Learning to (Learn at Test Time)") of the appendix.
Second, MTTT-MLP with SGD improves almost 10% on top of even a ViT-Small with linear attention, which uses more than $3\times$ parameters and $2\times$ FLOPs.
For SGD, $T\=4$ was simply chosen according to the optimal on patches.

These pieces of empirical evidence indicate that our perspective is prescriptive, by showing a path to new methods with better performance.
It is also predictive, since expectations derived from regular learning accurately explain novel behaviors of the inner loop, without any hyper-parameter tuning.
In terms of descriptiveness, MTTT-Linear matches linear attention (identity map) within 0.1%.

| Model | Drop-in layer | Acc. (%) | Params. (M) | FLOPs |
| --- | --- | --- | --- | --- |
| ViT-Tiny | Self-attn.(Beyer et al., [2022](#bib.bib8 "")) | - | 5.6 | $200\times$ |
| | Linear attn. (Katharopoulos et al.) | 53.7 | 5.6 | 1.0$\times$ |
| Linear attn. identity map | 49.9 | 5.6 | 1.0$\times$ |
| MTTT-Linear | 50.0 | 5.6 | 1.1$\times$ |
| MTTT-MLP | 61.9 | 6.8 | 1.8$\times$ |
| MTTT-MLP SGD $T\=4$ | 65.2 | 6.8 | 1.8$\times$ |
| ViT-Small | Linear attn. (Katharopoulos et al.) | 54.4 | 21.8 | 3.9$\times$ |
| | Linear attn. identity map | 55.7 | 21.8 | 3.9$\times$ |

*Table 3:  Results on ImageNet from pixels. FLOPs are presented as relative to linear attention.
MTTT-MLP with SGD outperforms without by 3.3%, and does not cost extra FLOPs.
It improves almost 10% on top of a ViT-Small with linear attention, which uses more than $3\times$ parameters and $2\times$ FLOPs.
See Subsection[5.2](#S5.SS2 "5.2 ImageNet from 224×224 Raw Pixels ‣ 5 Experiments ‣ Learning to (Learn at Test Time)") for details.*

6 Related Work
--------------

### 6.1 In-Context Learning as Explicit Learning

To the best of our knowledge, three pieces of prior work*(Akyürek et al., [2022](#bib.bib2 ""); Dai et al., [2022](#bib.bib18 ""); Von Oswald et al., [2023](#bib.bib59 ""))* have independently proposed the idea that linear transformers can simulate some variant of linear regression on in-context data, as an explanation for in-context learning.
Take *Von Oswald et al. ([2023](#bib.bib59 ""))* as an example. Given a labeled dataset, their work first trains a linear regression model with $T$ gradient steps, then constructs the weights of a $T$-layer linear transformer to produce the same output as the trained linear model.

Our work differs in two main aspects: self-supervision and direction of claims.
First, prior work focuses on showing that (linear) transformers can simulate learning on specific, supervised objectives, e.g. ridge regression, so their constructions rely on labeled pairs of in-context training data.
If there is a meta-learning component, it is restricted to specific hyper-parameters, e.g. the learning rate.
On the other hand, our inner loop implements a general objective that itself is mostly learned, so it does not need labeled data.
This makes our inner loop less interpretable but more practical.

At a higher level, transformers are complex models, and linear models are simple.
Prior work uses the complex to construct the simple.
Our construction takes the converse direction.
In prior work, empirical performance of meta-learning with linear regression has been significantly worse than linear transformers, even on labeled in-context data.
Again, with the goal of explaining transformers, their claims often indicate that linear transformers are superior to meta-learning.
Our experiments also point towards the converse.

Recently, *Mahankali et al. ([2023](#bib.bib39 "")); Zhang et al. ([2023](#bib.bib66 "")); Ahn et al. ([2023](#bib.bib1 ""))* and *Tarzanagh et al. ([2023](#bib.bib54 ""))* have further extended the arguments in prior work, therefore inheriting their two aspects above. *Tarzanagh et al. ([2023](#bib.bib54 ""))*, in particular, argues that transformers implement non-parametric learners (SVMs) on labeled data, supporting our intuition in the converse direction.
In summary, our paper complements prior work, with the different goal of inspiring potentially more powerful systems.

### 6.2 Learning at Test Time

The idea of learning at test time has a long history in machine learning.
One of the earliest instantiations of this idea is *Bottou \& Vapnik ([1992](#bib.bib10 ""))*:
For each test input, train on its neighbors before making a prediction.
This idea continues to be effective for SVMs*(Zhang et al., [2006](#bib.bib65 ""))* and large language models*(Hardt \& Sun, [2023](#bib.bib25 ""))*.
In computer vision, the general idea of learning at test time has also been applied to specific applications*(Jain \& Learned-Miller, [2011](#bib.bib31 ""); Shocher et al., [2018](#bib.bib50 ""); Mullapudi et al., [2018](#bib.bib42 ""); Luo et al., [2020](#bib.bib38 ""); Nitzan et al., [2022](#bib.bib43 ""))*.

*Transductive learning* *(Gammerman et al., [1998](#bib.bib22 ""))* is the first to articulate our philosophy in Section[1](#S1 "1 Introduction ‣ Learning to (Learn at Test Time)").
As stated by *Vapnik ([2013](#bib.bib57 ""))*:
”Try to get the answer that you really need, but not a more general one.”
Implementation-wise, it uses test data to add constraints to the margin of SVMs*(Joachims, [2002](#bib.bib32 ""); Collobert et al., [2006](#bib.bib17 ""))*.
This is an example of non-parametric learning at test time, similar to our kernel estimator in Subsection[4.2](#S4.SS2 "4.2 TTT with Kernels: Equivalence to Self-Attention ‣ 4 Choice of Learner for Inner Loop ‣ Learning to (Learn at Test Time)").
However, transductive learning usually needs multiple test instances to be practically effective, unlike our method, which only needs a single instance at a time.

Next we have an in-depth discussion of two particular relevant lines of work: TTT and fast weights.

#### 6.2.1 Test-Time Training with Self-Supervision

Our inner loop performs TTT with self-supervision, discussed in Section[2](#S2 "2 Inner Loop: Test-Time Training with Reconstruction ‣ Learning to (Learn at Test Time)").
This general framework was first proposed by *Sun et al. ([2020](#bib.bib51 ""))*, with results for supervised learning under distribution shifts.
Unlike previous lines of work, TTT can be used in principle with any self-supervised task, on any type of data, for any application, making it particularly suitable for deep learning.
Follow-up work has applied TTT to
batches of data*(Wang et al., [2020a](#bib.bib60 ""); Liu et al., [2021](#bib.bib36 ""))*, and other main tasks like
robot manipulation*(Hansen et al., [2020](#bib.bib24 ""))* and locomotion*(Sun et al., [2021](#bib.bib52 ""))*,
among others.

Particularly relevant to our inner loop, *Gandelsman et al. ([2022](#bib.bib23 ""))* performs TTT with reconstruction as the self-supervised task, and *Wang et al. ([2023](#bib.bib61 ""))* applies this method online to video streams.
The biggest difference is that our reconstruction task is parameterized for meta-learning.
In addition, our inner loop obtains multiple units of learning, $x_{1},\dots,x_{n}$, out of a single test instance through tokenization.
In prior work, each unit of learning is created through either data augmentations or a randomized $\phi$, such as masking random patches*(He et al., [2021](#bib.bib26 ""))*.

#### 6.2.2 Fast Weights

The general idea of *fast weights* is to update the parameters of a “fast” model on the most relevant data, as opposed to a “slow” model on all data*(Hinton \& Plaut, [1987](#bib.bib28 ""); Tieleman \& Hinton, [2009](#bib.bib56 ""))*, which most people today simply refer to as training or learning.
The most relevant data can be the test instance itself, where the update is performed without human supervision at test time.
Our work shares the same general idea, but formulates an explicit learning problem for each inner-loop update, with the goal of generalizing to that test instance.

To make fast weights “fast”, i.e. efficient, their update rules avoid forming an optimization problem with explicit objectives on the training data, i.e. a learning problem.
For example, given each input $x$, one popular update rule for fast weights is to add $xx^{T}$ (or some variant thereof)*(Ba et al., [2016](#bib.bib4 ""))* like in Hebbian learning and Hopfield networks*(Hopfield, [1982](#bib.bib29 ""))*.
In contrast, our update rule for TTT is an explicit training process as its name suggests.

*Fast weight programmers* (FWPs)*(Schmidhuber, [1992](#bib.bib49 ""))* produce the updates to fast weights with a “slow” model.
MTTT’s outer loop can be seen as training the “slow” model, if its inner loop is viewed as updating fast weights.
In particular, FWPs with the Hebbian update rule above are equivalent to linear transformers*(Schlag et al., [2021](#bib.bib47 ""))*, therefore also to MTTT with linear models. *Clark et al. ([2022](#bib.bib16 ""))* add a final layer of fast weights to a transformer and train its initialization with a FWP to improve performance on language modeling.

Given the broadest definition of FWPs, MTTT with parametric models can be seen as a special case*(Kirsch \& Schmidhuber, [2021](#bib.bib34 ""))*.
But the difference in update rules between TTT and fast weights, as discussed, carries over to MTTT and FWPs. *Irie et al. ([2021](#bib.bib30 ""))* have tried “fast” networks with weights directly produced as output of a “slow” network, without forming a learning problem.
In contrast, our inner loop mirrors regular (non-meta) learning. This helps us with empirical intuitions like in Figure[1](#S5.F1 "Figure 1 ‣ 5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)"), and heuristics like output normalization and stochastic gradient descent.

### 6.3 Learning to Learn

For decades, researchers have been arguing that learning to learn should be an important component of intelligence*(Schmidhuber, [1987](#bib.bib48 ""); Bengio et al., [1990](#bib.bib7 ""); Thrun \& Pratt, [1998](#bib.bib55 ""); Lake et al., [2017](#bib.bib35 ""))*.

Most prior work on learning to learn, such as *Andrychowicz et al. ([2016](#bib.bib3 ""))*, *Finn et al. ([2017](#bib.bib21 ""))* and *Metz et al. ([2018](#bib.bib41 ""))*, try to generalize across datasets or tasks instead of instances, since meta-learning lifts its units “one level above”.
Their inner loop learns from an entire dataset at a time, instead of a single instance, so the outer loop needs a collection of datasets or tasks.
Since it is hard to collect millions of datasets or tasks, their outer loop is hard to scale.

But for TTT, each instance itself is a task and defines its own generalization problem, so MTTT is only a solution to the canonical problem of supervised learning, reformulated as learning to learn.
It does not propose a new problem setting like generalization across datasets or tasks.
Its inner loop only needs a single instance, so the outer loop only needs a collection of instances – a dataset in the traditional (non-meta) sense, e.g. the ImageNet training set.
This makes it easier to scale.

7 Limitations and Future work
-----------------------------

The search space for practically effective instantiations of MTTT is huge, and our paper has only taken a baby step.
We believe this search needs to be a community effort, and our biggest motivation for writing this paper is to inspire future steps.
Fortunately, if our perspective holds, then successful heuristics for regular learning can transfer to our inner loop, and search can be much more efficient.
Next we outline some especially promising directions for future work, given our current limitations.

##### Multi-level learning to learn.

We have already discussed the possibility of a more ambitious architecture for $f$ (e.g. larger MLP or CNN), but when $f$ is a transformer, it can be interpreted as yet another inner loop nested inside the existing one.
In this fashion, we can potentially build many levels of nested learning problems, instead of the existing two-level paradigm for learning to learn.
This possibility has been mentioned in *Irie et al. ([2021](#bib.bib30 ""))*, but might become practically useful under MTTT, given the functional programming capabilities of JAX.

##### Better infrastructure.

Since learning to learn has been relatively under-explored as a practical solution to supervised learning, its support in JAX is still primitive.
For example, MTTT-Linear only costs $0.1\times$ more FLOPs than linear attention (already unnecessary in principle), but turns out to be $2\times$ slower in wall-clock time.
SGD is slower by an additional factor of $2\times$ regardless of $T$, even though it costs the same number of FLOPs.
We believe these systems-level inefficiencies will eventually disappear once the community builds a better infrastructure.

##### Autoregressive language modeling.

For this application, $T\=n$ because each $W_{t}$ only updates on the gradient from $x_{t}$ in an online fashion, like in *Wang et al. ([2023](#bib.bib61 ""))*.
In this paper, we have not been able to try autoregressive tasks because of a rather technical reason:
JAX saves every intermediate $W_{t}$, taking $O(TD)$ memory when $W$ is of size $D$.
But in principle, only $O(D)$ memory is needed.
Implementing this solution turns out to be a large engineering effort beyond the scope of our paper.

##### Outer-loop parameterization and optimization.

There are many other ways to parameterize a family of reconstruction tasks, or even a more general family.
For clean comparison with attention, our way has been simple but probably far from optimal.
For the same reason, we have also refrained from searching the outer-loop optimization recipe, even though the best recipe with neural networks as inner loop is almost certainly different from that with kernels.

##### Gradient-free optimization.

Optimizing $\mathcal{L}_{T}$ with zeroth-order techniques*(Salimans et al., [2017](#bib.bib46 ""); Hinton, [2022](#bib.bib27 ""); Malladi et al., [2023](#bib.bib40 ""))* might sound radical, but can bring many practical benefits for MTTT.
It frees us from taking gradients of gradients, and the engineering challenges that follow, such as backpropagation through time.
We simply avoid the aforementioned problem with recovering intermediate weights $W_{t}$, and all the systems-level inefficiencies for learning to learn in JAX.
Altogether, we believe that MTTT could become a killer application for zeroth-order techniques.

Acknowledgements
----------------

We are grateful to Guandao Yang and Beidi Chen for helpful discussions, also to Yossi Gandelsman and Yutong Bai for their help at an early stage of this project.
Yu Sun is grateful to his PhD advisors, Alexei A. Efros and Moritz Hardt, for their many insights that eventually became part of this paper.
Yu Sun is supported in part by Oracle Cloud credits and related resources, generously provided by the Oracle for Research program.

References
----------

* Ahn et al. (2023)Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra.Transformers learn to implement preconditioned gradient descent for
in-context learning.*arXiv preprint arXiv:2306.00297*, 2023.
* Akyürek et al. (2022)Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou.What learning algorithm is in-context learning? investigations with
linear models.*arXiv preprint arXiv:2211.15661*, 2022.
* Andrychowicz et al. (2016)Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau,
Tom Schaul, Brendan Shillingford, and Nando De Freitas.Learning to learn by gradient descent by gradient descent.*Advances in neural information processing systems*, 29, 2016.
* Ba et al. (2016)Jimmy Ba, Geoffrey E Hinton, Volodymyr Mnih, Joel Z Leibo, and Catalin Ionescu.Using fast weights to attend to the recent past.*Advances in neural information processing systems*, 29, 2016.
* Bao et al. (2021)Hangbo Bao, Li Dong, and Furu Wei.Beit: BERT pre-training of image transformers.*CoRR*, abs/2106.08254, 2021.URL [https://arxiv.org/abs/2106.08254](https://arxiv.org/abs/2106.08254 "").
* Beltagy et al. (2020)Iz Beltagy, Matthew E Peters, and Arman Cohan.Longformer: The long-document transformer.*arXiv preprint arXiv:2004.05150*, 2020.
* Bengio et al. (1990)Yoshua Bengio, Samy Bengio, and Jocelyn Cloutier.*Learning a synaptic learning rule*.Citeseer, 1990.
* Beyer et al. (2022)Lucas Beyer, Xiaohua Zhai, and Alexander Kolesnikov.Better plain vit baselines for imagenet-1k.*arXiv preprint arXiv:2205.01580*, 2022.
* Bierens (1988)Hermanus Josephus Bierens.The nadaraya-watson kernel regression function estimator.*(Serie Research Memoranda; No. 1988-58). Faculty of Economics
and Business Administration, Vrije Universiteit Amsterdam.*, 1988.
* Bottou \& Vapnik (1992)Léon Bottou and Vladimir Vapnik.Local learning algorithms.*Neural computation*, 4(6):888–900, 1992.
* Breiman et al. (1977)Leo Breiman, William Meisel, and Edward Purcell.Variable kernel estimates of multivariate densities.*Technometrics*, 19(2):135–144, 1977.
* Brown et al. (2020)Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
et al.Language models are few-shot learners.*Advances in neural information processing systems*,
33:1877–1901, 2020.
* Cai (2001)Zongwu Cai.Weighted nadaraya–watson regression estimation.*Statistics \& probability letters*, 51(3):307–318, 2001.
* Chen et al. (2020)Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and
Ilya Sutskever.Generative pretraining from pixels.In *International conference on machine learning*, pp. 1691–1703. PMLR, 2020.
* Chen (2017)Yen-Chi Chen.A tutorial on kernel density estimation and recent advances.*Biostatistics \& Epidemiology*, 1(1):161–187, 2017.
* Clark et al. (2022)Kevin Clark, Kelvin Guu, Ming-Wei Chang, Panupong Pasupat, Geoffrey Hinton, and
Mohammad Norouzi.Meta-learning fast weight language models.*arXiv preprint arXiv:2212.02475*, 2022.
* Collobert et al. (2006)Ronan Collobert, Fabian Sinz, Jason Weston, Léon Bottou, and Thorsten
Joachims.Large scale transductive svms.*Journal of Machine Learning Research*, 7(8), 2006.
* Dai et al. (2022)Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei.Why can gpt learn in-context? language models secretly perform
gradient descent as meta optimizers.*arXiv preprint arXiv:2212.10559*, 2022.
* Deng et al. (2009)Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.Imagenet: A large-scale hierarchical image database.In *2009 IEEE conference on computer vision and pattern
recognition*, pp. 248–255. Ieee, 2009.
* Dosovitskiy et al. (2020)Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, et al.An image is worth 16x16 words: Transformers for image recognition at
scale.*arXiv preprint arXiv:2010.11929*, 2020.
* Finn et al. (2017)Chelsea Finn, Pieter Abbeel, and Sergey Levine.Model-agnostic meta-learning for fast adaptation of deep networks.In *International conference on machine learning*, pp. 1126–1135. PMLR, 2017.
* Gammerman et al. (1998)A. Gammerman, V. Vovk, and V. Vapnik.Learning by transduction.In *In Uncertainty in Artificial Intelligence*, pp. 148–155.
Morgan Kaufmann, 1998.
* Gandelsman et al. (2022)Yossi Gandelsman, Yu Sun, Xinlei Chen, and Alexei A. Efros.Test-time training with masked autoencoders.*Advances in Neural Information Processing Systems*, 2022.
* Hansen et al. (2020)Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenyà, Pieter Abbeel,
Alexei A Efros, Lerrel Pinto, and Xiaolong Wang.Self-supervised policy adaptation during deployment.*arXiv preprint arXiv:2007.04309*, 2020.
* Hardt \& Sun (2023)Moritz Hardt and Yu Sun.Test-time training on nearest neighbors for large language models.*arXiv preprint arXiv:2305.18466*, 2023.
* He et al. (2021)Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and
Ross B. Girshick.Masked autoencoders are scalable vision learners.*CoRR*, abs/2111.06377, 2021.
* Hinton (2022)Geoffrey Hinton.The forward-forward algorithm: Some preliminary investigations.*arXiv preprint arXiv:2212.13345*, 2022.
* Hinton \& Plaut (1987)Geoffrey E Hinton and David C Plaut.Using fast weights to deblur old memories.In *Proceedings of the ninth annual conference of the Cognitive
Science Society*, pp. 177–186, 1987.
* Hopfield (1982)John J Hopfield.Neural networks and physical systems with emergent collective
computational abilities.*Proceedings of the national academy of sciences*, 79(8):2554–2558, 1982.
* Irie et al. (2021)Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber.Going beyond linear transformers with recurrent fast weight
programmers.*Advances in Neural Information Processing Systems*,
34:7703–7717, 2021.
* Jain \& Learned-Miller (2011)Vidit Jain and Erik Learned-Miller.Online domain adaptation of a pre-trained cascade of classifiers.In *CVPR 2011*, pp. 577–584. IEEE, 2011.
* Joachims (2002)Thorsten Joachims.*Learning to classify text using support vector machines*,
volume 668.Springer Science \& Business Media, 2002.
* Katharopoulos et al. (2020)Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François
Fleuret.Transformers are rnns: Fast autoregressive transformers with linear
attention.In *International conference on machine learning*, pp. 5156–5165. PMLR, 2020.
* Kirsch \& Schmidhuber (2021)Louis Kirsch and Jürgen Schmidhuber.Meta learning backpropagation and improving it.*Advances in Neural Information Processing Systems*,
34:14122–14134, 2021.
* Lake et al. (2017)Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman.Building machines that learn and think like people.*Behavioral and brain sciences*, 40:e253, 2017.
* Liu et al. (2021)Yuejiang Liu, Parth Kothari, Bastien van Delft, Baptiste Bellot-Gurlet, Taylor
Mordan, and Alexandre Alahi.Ttt++: When does self-supervised test-time training fail or thrive?*Advances in Neural Information Processing Systems*, 34, 2021.
* Lu et al. (2021)Jiachen Lu, Jinghan Yao, Junge Zhang, Xiatian Zhu, Hang Xu, Weiguo Gao,
Chunjing Xu, Tao Xiang, and Li Zhang.Soft: Softmax-free transformer with linear complexity.*Advances in Neural Information Processing Systems*,
34:21297–21309, 2021.
* Luo et al. (2020)Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf.Consistent video depth estimation.*ACM Transactions on Graphics (ToG)*, 39(4):71–1, 2020.
* Mahankali et al. (2023)Arvind Mahankali, Tatsunori B Hashimoto, and Tengyu Ma.One step of gradient descent is provably the optimal in-context
learner with one layer of linear self-attention.*arXiv preprint arXiv:2307.03576*, 2023.
* Malladi et al. (2023)Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D Lee, Danqi
Chen, and Sanjeev Arora.Fine-tuning language models with just forward passes.*arXiv preprint arXiv:2305.17333*, 2023.
* Metz et al. (2018)Luke Metz, Niru Maheswaranathan, Brian Cheung, and Jascha Sohl-Dickstein.Meta-learning update rules for unsupervised representation learning.*arXiv preprint arXiv:1804.00222*, 2018.
* Mullapudi et al. (2018)Ravi Teja Mullapudi, Steven Chen, Keyi Zhang, Deva Ramanan, and Kayvon
Fatahalian.Online model distillation for efficient video inference.*arXiv preprint arXiv:1812.02699*, 2018.
* Nitzan et al. (2022)Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi
Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or.Mystyle: A personalized generative prior.*arXiv preprint arXiv:2203.17272*, 2022.
* Pathak et al. (2016)Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A
Efros.Context encoders: Feature learning by inpainting.In *Proceedings of the IEEE conference on computer vision and
pattern recognition*, pp. 2536–2544, 2016.
* Poli et al. (2023)Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y Fu, Tri Dao, Stephen
Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré.Hyena hierarchy: Towards larger convolutional language models.*arXiv preprint arXiv:2302.10866*, 2023.
* Salimans et al. (2017)Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever.Evolution strategies as a scalable alternative to reinforcement
learning.*arXiv preprint arXiv:1703.03864*, 2017.
* Schlag et al. (2021)Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber.Linear transformers are secretly fast weight programmers.In *International Conference on Machine Learning*, pp. 9355–9366. PMLR, 2021.
* Schmidhuber (1987)Jürgen Schmidhuber.*Evolutionary principles in self-referential learning, or on
learning how to learn: the meta-meta-… hook*.PhD thesis, Technische Universität München, 1987.
* Schmidhuber (1992)Jürgen Schmidhuber.Learning to control fast-weight memories: An alternative to dynamic
recurrent networks.*Neural Computation*, 4(1):131–139, 1992.
* Shocher et al. (2018)Assaf Shocher, Nadav Cohen, and Michal Irani.“zero-shot” super-resolution using deep internal learning.In *Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition*, pp. 3118–3126, 2018.
* Sun et al. (2020)Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt.Test-time training with self-supervision for generalization under
distribution shifts.In *International Conference on Machine Learning*, pp. 9229–9248. PMLR, 2020.
* Sun et al. (2021)Yu Sun, Wyatt L Ubellacker, Wen-Loong Ma, Xiang Zhang, Changhao Wang, Noel V
Csomay-Shanklin, Masayoshi Tomizuka, Koushil Sreenath, and Aaron D Ames.Online learning of unknown dynamics for model-based controllers in
legged locomotion.*IEEE Robotics and Automation Letters*, 6(4):8442–8449, 2021.
* Szegedy et al. (2015)Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir
Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich.Going deeper with convolutions.In *Proceedings of the IEEE conference on computer vision and
pattern recognition*, pp. 1–9, 2015.
* Tarzanagh et al. (2023)Davoud Ataee Tarzanagh, Yingcong Li, Christos Thrampoulidis, and Samet Oymak.Transformers as support vector machines.*arXiv preprint arXiv:2308.16898*, 2023.
* Thrun \& Pratt (1998)Sebastian Thrun and Lorien Pratt.Learning to learn: Introduction and overview.In *Learning to learn*, pp. 3–17. Springer, 1998.
* Tieleman \& Hinton (2009)Tijmen Tieleman and Geoffrey Hinton.Using fast weights to improve persistent contrastive divergence.In *Proceedings of the 26th annual international conference on
machine learning*, pp. 1033–1040, 2009.
* Vapnik (2013)Vladimir Vapnik.*The nature of statistical learning theory*.Springer science \& business media, 2013.
* Vincent et al. (2008)Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol.Extracting and composing robust features with denoising autoencoders.In *ICML*, pp. 1096–1103, 2008.
* Von Oswald et al. (2023)Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento,
Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov.Transformers learn in-context by gradient descent.In *International Conference on Machine Learning*, pp. 35151–35174. PMLR, 2023.
* Wang et al. (2020a)Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell.Tent: Fully test-time adaptation by entropy minimization.*arXiv preprint arXiv:2006.10726*, 2020a.
* Wang et al. (2023)Renhao Wang, Yu Sun, Yossi Gandelsman, Xinlei Chen, Alexei A Efros, and
Xiaolong Wang.Test-time training on video streams.*arXiv preprint arXiv:2307.05014*, 2023.
* Wang et al. (2020b)Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma.Linformer: Self-attention with linear complexity.*arXiv preprint arXiv:2006.04768*, 2020b.
* Williams \& Rasmussen (2006)Christopher KI Williams and Carl Edward Rasmussen.*Gaussian processes for machine learning*, volume 2.MIT press Cambridge, MA, 2006.
* Zhang et al. (2021)Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals.Understanding deep learning (still) requires rethinking
generalization.*Communications of the ACM*, 64(3):107–115,
2021.
* Zhang et al. (2006)Hao Zhang, Alexander C Berg, Michael Maire, and Jitendra Malik.Svm-knn: Discriminative nearest neighbor classification for visual
category recognition.In *2006 IEEE Computer Society Conference on Computer Vision and
Pattern Recognition (CVPR’06)*, volume 2, pp. 2126–2136. IEEE, 2006.
* Zhang et al. (2023)Ruiqi Zhang, Spencer Frei, and Peter L Bartlett.Trained transformers learn linear models in-context.*arXiv preprint arXiv:2306.09927*, 2023.

Appendix A Linear Attention and Linear Transformers
---------------------------------------------------

This section is intended as a very brief reference on linear attention.
For a more in-depth discussion, please see Section 3 of *Katharopoulos et al. ([2020](#bib.bib33 ""))*.
In this section, we use standard notations for transformers, where $X$ is the $n\times d$ matrix with $x_{i}$ as its $i$th column.
Recall that for self-attention, we first form the keys, queries and values by multiplying $X$ with the respective weight matrices:

|  | $K\=\theta_{K}X,~{}~{}Q\=\theta_{Q}X,~{}~{}V\=\theta_{V}X.$ |  | (14) |
| --- | --- | --- | --- |

Then we obtain the $i$th output embedding $Z_{i}$ as

|  | $Z_{i}\=\text{softmax}_{j\=1}^{n}\left(Q_{i}^{T}K_{j}\right)V_{j},$ |  | (15) |
| --- | --- | --- | --- |

where softmax makes $Q_{i}^{T}K_{j}$ sum to 1 over $j$.
Linear attention simply replaces softmax with mean:

|  | $V_{i}^{\prime}\=\frac{1}{n}\sum_{j\=1}^{n}\left(Q_{i}^{T}K_{j}\right)V_{j}\=\frac{1}{n}\sum_{j\=1}^{n}Q_{i}\left(K_{j}^{T}V_{j}\right)\=\frac{1}{n}\sum_{j\=1}^{n}\left(K_{j}^{T}V_{j}\right)Q_{i}.$ |  | (16) |
| --- | --- | --- | --- |

Since $\sum_{j\=1}^{n}\left(K_{j}^{T}V_{j}\right)$ is the same for each $i$, it can be pre-computed with linear complexity.

Linear attention in *Katharopoulos et al. ([2020](#bib.bib33 ""))* is slightly different. Before forming the keys, queries and values in Equation[14](#A1.E14 "In Appendix A Linear Attention and Linear Transformers ‣ Learning to (Learn at Test Time)"), it first passes $X$ through an engineered feature transformation ($\text{elu}+1$).
Then for Equation[16](#A1.E16 "In Appendix A Linear Attention and Linear Transformers ‣ Learning to (Learn at Test Time)"), instead of a simple mean, there is a data-dependent normalizer.
In Section[5](#S5 "5 Experiments ‣ Learning to (Learn at Test Time)"),
we label their modified linear attention with citation, and the one described in Equation[14](#A1.E14 "In Appendix A Linear Attention and Linear Transformers ‣ Learning to (Learn at Test Time)") and [16](#A1.E16 "In Appendix A Linear Attention and Linear Transformers ‣ Learning to (Learn at Test Time)") as linear attention with identity map.

In standard terms, a transformer uses self-attention unless noted otherwise.
A linear transformer is simply a transformer with every self-attention layer replaced by a linear attention layer. Our paper follows this convention.

Appendix B Multi-Head Attention
--------------------------------

For an attention layer with $H$ heads, we need $H$ inner loops in parallel. In the case of linear attention, there would be $H$ linear models, each with a weight matrix of size $(d/H)\times(d/H)$ instead of $d\times d$.
Under the MTTT perspective, this design naturally forms a bottleneck for compressing information, often critical for autoencoders.
Specifically, each $\phi$ now maps from dimension $d$ to ${d/H}$, and $g$ from ${d/H}$ back to $d$.
Each $h$ here is a projection operation, the de facto standard for multi-head attention.

![Refer to caption]()

*Figure 2: Illustration of Decoder Layer Norm (LN), presented in Section[5](#S5 "5 Experiments ‣ Learning to (Learn at Test Time)").
This diagram shows the first half of a transformer block, omitting the second half which does not contain any attention layer.
The input embedding is $z$, the output is $\hat{z}$.
The identity mapping is at the top, and the residual is learned at the bottom.
The dotted line from $W_{0}$ to $W_{1}$ represents an inner-loop gradient step.
Here we use $T\=1$, i.e. only one step in the inner loop, so the final prediction is made with $W_{1}$. Standard design: only use the blue LN.
The output of decoder $g$, in this case $z_{0}$, is expected to reconstruct $x$, causing a “type mismatch”. Our design: also use the red LN.
Now $\hat{x}$ is expected to reconstruct $x$, and both are outputs of LN.*

Appendix C Our Kernel Estimator
-------------------------------

Here is the derivation for the Nadaraya-Watson estimator.
Throughout this section of the appendix, we use $\mathbf{x}$ to denote the input token $x$ as a random variable, which is different from the test instance (i.e. sequence) $X$ in the main text of the paper.
Our goal is to produce the corresponding feature, another random variable $\mathbf{z}$.
This is formulated as estimating the conditional expectation of $\mathbf{z}$:

|  | $\mathbb{E}[\mathbf{z}|\mathbf{x}\=x]\=\int p(z|x)~{}z~{}dz\=\int\frac{p(x,z)}{p(x)}~{}z~{}dz.$ |  |
| --- | --- | --- |

Since the true probability distributions $p(x)$ and $p(x,z)$ are unknown, we replace them with their kernel density estimations.
Specifically, the kernel density estimation for $p(x)$ is:

|  | $\hat{p}(x)\=\frac{1}{n}\sum_{i\=1}^{n}\kappa(x,x_{i}),$ |  |
| --- | --- | --- |

where each $x_{i}$ is a piece of training data in general.
(Recall that for our paper, $x_{i}$ is specifically training data for the inner loop, i.e. a token, which matches our notation in the main text.)

For estimating $p(x,y)$, we use the product kernel:

|  | $\hat{p}(x,z)\=\frac{1}{n}\sum_{i\=1}^{n}\kappa(x,x_{i})~{}\kappa^{\prime}(z,z_{i}).$ |  |
| --- | --- | --- |

At first sight, it seem absurd to factor the joint probability into two seemingly independent kernels.
But in this case, $\kappa^{\prime}$ can actually be any $\kappa_{i}^{\prime}$ dependent on $x_{i}$, since it will be integrated out. So the two kernels do not actually need to be independent.

Plugging in those estimations, we obtain the Nadaraya-Watson estimator:

|  | $\displaystyle\hat{\mathbb{E}}[\mathbf{z}|\mathbf{x}\=x]$ | $\displaystyle\=\int\frac{\hat{p}(x,z)}{\hat{p}(x)}~{}z~{}dz$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{1}{\hat{p}(x)}\int\hat{p}(x,z)~{}z~{}dz$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{1}{\sum_{i\=1}^{n}\kappa(x,x_{i})}\int\sum_{i\=1}^{n}\kappa(x,x_{i})~{}\kappa^{\prime}(z,z_{i})~{}z~{}dz$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{1}{\sum_{i\=1}^{n}\kappa(x,x_{i})}\sum_{i\=1}^{n}\kappa(x,x_{i})~{}\int\kappa^{\prime}(z,z_{i})~{}z~{}dz$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{1}{\sum_{i\=1}^{n}\kappa(x,x_{i})}\sum_{i\=1}^{n}\kappa(x,x_{i})~{}z_{i}.$ |  |
| --- | --- | --- | --- |

Recall that in the main text, our kernel is chosen to be

|  | $\kappa(x,x^{\prime};\theta_{K},\theta_{Q})\propto e^{(\theta_{K}x)^{T}\theta_{Q}x^{\prime}}$ |  | (17) |
| --- | --- | --- | --- |

where $\theta_{K}$ and $\theta_{Q}$ have been known as bandwidth hyper-parameters*(Williams \& Rasmussen, [2006](#bib.bib63 ""))*.

In modern days, people think of kernels as positive semi-definite, which might not be guaranteed for $\kappa$ unless $\theta_{K}\=\theta_{Q}$.
However, people working on kernels decades ago, around the time when the Nadaraya-Watson estimator was popular, have been surprisingly lenient with the choice of kernels.
When an estimator uses $\theta_{K}\neq\theta_{Q}$, it is known as a balloon estimator*(Chen, [2017](#bib.bib15 ""))*.
Papers like *Breiman et al. ([1977](#bib.bib11 ""))* have even used $\theta_{Q}$ as a function of $x^{\prime}$, known as sample-adaptive smoothing.

<img src='x4.png' alt='Refer to caption' title='' width='422' height='282' />

*Figure 3: Inner-loop loss across the 12 TTT layers. Behavior across layers is roughly the same as in Figure[1](#S5.F1 "Figure 1 ‣ 5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)"). Method: MTTT-MLP performing full-batch gradient descent in the inner loop, $T\=4$. Setting: ImageNet from patches. See Subsection[5.1](#S5.SS1 "5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)").*

<img src='x5.png' alt='Refer to caption' title='' width='422' height='282' />

*Figure 4: Inner-loop loss across the 12 TTT layers. Behavior across layers is roughly the same as in Figure[1](#S5.F1 "Figure 1 ‣ 5.1 ImageNet ‣ 5 Experiments ‣ Learning to (Learn at Test Time)"). Method: MTTT-MLP performing stochastic gradient descent in the inner loop, $T\=4$. Setting: ImageNet from pixels. See Subsection[5.2](#S5.SS2 "5.2 ImageNet from 224×224 Raw Pixels ‣ 5 Experiments ‣ Learning to (Learn at Test Time)").*
