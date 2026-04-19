Scaling Neural Program Synthesis with Distribution-based Search
================================================================

Nathanaël FijalkowCNRS, LaBRI and Université de Bordeaux, FranceThe Alan Turing Institute of data science, United KingdomGuillaume LagardeCNRS, LaBRI and Université de Bordeaux, FranceThéo MatriconCNRS, LaBRI and Université de Bordeaux, FranceKevin EllisCornell University, United StatesPierre OhlmannUniversity of Paris, FranceAkarsh PottaIndian Institute of Technology Bombay, India

###### Abstract

We consider the problem of automatically constructing computer programs from input-output examples.
We investigate how to augment probabilistic and neural program synthesis methods with new search algorithms,
proposing a framework called distribution-based search.
Within this framework, we introduce two new search algorithms:Heap Search, an enumerative method, and SQRT Sampling, a probabilistic method.
We prove certain optimality guarantees for both methods,
show how they integrate with probabilistic and neural techniques,
and demonstrate how they can operate at scale across parallel compute environments.
Collectively these findings offer theoretical and applied studies of search algorithms for program synthesis that integrate with recent developments in machine-learned program synthesizers.

1 Introduction
--------------

Writing software is tedious, error-prone, and accessible only to a small share of the population
– yet coding grows increasingly important as the digital world plays larger and larger roles in peoples’ lives.
Program synthesis seeks to make coding more reliable and accessible by developing methods for automatically constructing programs*Gulwani et al. ([2017](#bib.bib13 ""))*.
For example, the FlashFill system*Gulwani et al. ([2017](#bib.bib13 ""))* in Microsoft Excel makes coding more accessible by allowing nontechnical users to synthesize spreadsheet programs by giving input-output examples,
while the TF-coder system*Shi et al. ([2020a](#bib.bib21 ""))* seeks to make coding neural networks more reliable by synthesizing TensorFlow code from input-outputs.
Where these systems have been most successful is when they pair a specialized *domain-specific language* (DSL) to a domain-specific search algorithm for synthesizing programs in that DSL.
A recent trend – both in industry*Kalyan et al. ([2018](#bib.bib14 ""))* and academia*Devlin et al. ([2017](#bib.bib9 ""))* – is to employ machine learning methods to learn to quickly search for a program in the DSL*Balog et al. ([2017](#bib.bib3 "")), Devlin et al. ([2017](#bib.bib9 "")), Lee et al. ([2018](#bib.bib17 "")), Zhang et al. ([2018](#bib.bib25 "")), Polosukhin and Skidanov ([2018](#bib.bib20 "")), Kalyan et al. ([2018](#bib.bib14 "")), Zohar and Wolf ([2018](#bib.bib26 "")), Chen et al. ([2018](#bib.bib5 ""))*.
Many such recent works have explored engineering better neural networks for guiding program search, effectively by training the network to act as a language model over source code that conditions on input-outputs*Polosukhin and Skidanov ([2018](#bib.bib20 ""))*.
Here, we ‘pop up’ a level and instead ask: given a neural net that probabilistically generates source code, how can we most efficiently deploy that model in order to find a program consistent with some input-outputs?
This concern arises because program synthesis requires solving a hard combinatorial search problem (exploring a possibly infinite space of programs),
so taming this combinatorial explosion makes the difference between a practically useful system, and a system which cannot scale to anything but the most trivial of programs.

At a high-level the approaches we develop in this work follow a 2-stage pipeline: in the first stage a learned model predicts probabilistic weights,
and in the second stage a symbolic search algorithm uses those weights to explore the space of source code.
Our contributions target the second stage of this pipeline,
and we focus on theoretical analysis of sampling-based search algorithms, new search algorithms based on neurally-informed enumeration, and empirical evaluations showing that recent neural program synthesizers can compose well with our methods.

This 2-stage pipelined approach has several benefits. The first is that the cost of querying the neural network is usually very small compared to the cost of combinatorial search, yet in practice the neural model learns to provide rough-and-ready probabilistic predictions to guide the search. A second benefit is that even if the probabilistic predictions are inaccurate, our methods maintain soundness and completeness (but may take longer to run). Another appeal is that it can be naturally combined with other classical approaches for program synthesis.

##### Our contributions:

* •

    A theoretical framework called distribution-based search for evaluating and comparing search algorithms
    in the context of machine-learned predictions.

* •

    Two new search algorithms: Heap Search, an enumerative method, and SQRT Sampling, a probabilistic method. We prove a number of theoretical results about them, in particular that they are both loss optimal.

* •

    A method for running any search algorithm across parallel compute environments.

We perform an empirical evaluation of existing and new search algorithms, showing how the new methods integrate with probabilistic and neural techniques.

*Figure 1: Pipeline for neural predictions for syntax guided program synthesis.*

2 Distribution-based search
----------------------------

We work within the syntax guided program synthesis (SyGuS) framework introduced by*Alur et al. ([2013](#bib.bib1 ""))*, see also*Alur et al. ([2018](#bib.bib2 ""))*.
In this setting, the DSL is given by a set of primitives together with their (possibly polymorphic) types and semantics.

We describe the machine learning pipeline for program synthesis, illustrated in
Figure[1](#S1.F1 "Figure 1 ‣ Our contributions: ‣ 1 Introduction ‣ Scaling Neural Program Synthesis with Distribution-based Search") on a toy DSL describing integer list manipulating programs.

The compilation phase constructs a context-free grammar (CFG) from the DSL together with a set of syntactic constraints.
The CFG may incorporate important information about the program being generated,
such as the $n$ last primitives (encompassing $n$-gram models) or semantic information (e.g. non-zero integer, sorted list).

A prediction model (typically a neural network) takes as inputs a set of I/O and outputs a probabilistic labelling of the CFG, inducing a probabilistic context-free grammar (PCFG).
The network is trained so that most likely programs (with respect to the PCFG) are the most likely to be solutions, meaning map the inputs to corresponding outputs.

We refer to Appendix[A](#A1 "Appendix A Machine-learned program synthesis ‣ Scaling Neural Program Synthesis with Distribution-based Search") for an in-depth technical discussion on program representations
and on the compilation phase.
In this work we focus on the search phase and start with defining a theoretical framework for analysing search algorithms.

The PCFG obtained through the predictions of the neural network defines a probabilistic distribution $\mathcal{D}$ over programs.
We make the theoretical assumption that the program we are looking for is actually sampled from $\mathcal{D}$,
and construct algorithms searching through programs which find programs sampled from $\mathcal{D}$ as quickly as possible.
Formally, the goal is to minimise the expected number of programs the algorithm outputs before finding the right program.

We write $A(n)$ for the $n$th program chosen by the algorithm $A$; since $A$ may be a randomised algorithm $A(n)$ is a random variable.
The performance $\mathcal{L}(A,\mathcal{D})$ of the algorithm $A$, which we call its loss, is the expected number of tries it makes before finding $x$:

|  | $\mathcal{L}(A,\mathcal{D})\=\mathbb{E}_{x\sim\mathcal{D}}\left[\inf\left{n\in\mathbb{N}:A(n)\=x\right}\right].$ |  |
| --- | --- | --- |

An algorithm $A^{*}$ is ‘loss optimal’ if $\mathcal{L}(A^{*},\mathcal{D})\=\inf_{A}\mathcal{L}(A,\mathcal{D})$.
Let us state a simple fact: an algorithm is loss optimal
if it generates each program once and in non-increasing order of probabilities.
Depending on $\mathcal{D}$ constructing an efficient loss optimal algorithm may be challenging,
pointing to a trade off between quantity and quality:
is it worth outputting a lot of possibly unlikely programs quickly,
or rather invest more resources into outputting fewer but more likely programs?

##### An example.

To illustrate the definitions let us consider the distribution $\mathcal{D}$ over the natural numbers such that $\mathcal{D}(n)\=\frac{1}{2^{n+1}}$; it is generated by the following PCFG:

|  | $S\to^{.5}f(S)\quad;\quad S\to^{.5}x,$ |  |
| --- | --- | --- |

when identifying $n$ with the program $f^{n}(x)$.
Let us analyse a few algorithms.

* •

    The algorithm $A_{1}$ enumerates in a deterministic fashion the natural numbers starting from <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S2.I1.i1.p1.2.m2.1"><semantics id="S2.I1.i1.p1.2.m2.1a"><mn id="S2.I1.i1.p1.2.m2.1.1" xref="S2.I1.i1.p1.2.m2.1.1.cmml">0</mn><annotation-xml encoding="MathML-Content" id="S2.I1.i1.p1.2.m2.1b"><cn id="S2.I1.i1.p1.2.m2.1.1.cmml" type="integer" xref="S2.I1.i1.p1.2.m2.1.1">0</cn></annotation-xml></semantics></math> -->00:
    $A_{1}(n)\=n$.
    Then $\mathcal{L}(A_{1},\mathcal{D})\=\sum_{n\geq 0}\frac{n+1}{2^{n+1}}\=2$.
    This enumeration algorithm $A_{1}$ is loss optimal.

* •

    The algorithm $A_{2}$ samples the natural numbers using the distribution $\mathcal{D}$.
    For $n\geq 0$, the value of $\mathbb{E}\left[\inf\left{n^{\prime}:A_{2}(n^{\prime})\=n\right}\right]$ is $2^{n+1}$:
    this is the expectation of the geometric distribution with parameter $\frac{1}{2^{n+1}}$.
    Then $\mathcal{L}(A_{2},\mathcal{D})\=\sum_{n\geq 0}\frac{2^{n+1}}{2^{n+1}}\=\infty$.
    Hence the naive sampling algorithm using $\mathcal{D}$ has infinite loss.

* •

    The algorithm $A_{3}$ samples the natural numbers using a distribution that we call $\sqrt{\mathcal{D}}$
    defined111 For the normalisation factor, note that $\sum_{n\geq 0}\frac{1}{2^{\frac{n+1}{2}}}\=1+\sqrt{2}$. by $\sqrt{\mathcal{D}}(n)\=\frac{1}{1+\sqrt{2}}\frac{1}{2^{\frac{n+1}{2}}}$.

    For $n\geq 0$, the value of $\mathbb{E}\left[\inf\left{n^{\prime}:A_{3}(n^{\prime})\=n\right}\right]$ is $(1+\sqrt{2})2^{\frac{n+1}{2}}$:
        this is the expectation of the geometric distribution with parameter $\frac{1}{1+\sqrt{2}}\frac{1}{2^{\frac{n+1}{2}}}$.
        Then

    |  | $\begin{array}[]{lll}\mathcal{L}(A_{3},\mathcal{D})\&\=\&\sum_{n\geq 0}\frac{(1+\sqrt{2})2^{\frac{n+1}{2}}}{2^{n+1}}\\ \&\=\&(1+\sqrt{2})\sum_{n\geq 0}\frac{1}{2^{\frac{n+1}{2}}}\\ \&\=\&(1+\sqrt{2})^{2}\approx 5.83.\end{array}$ |  |
        | --- | --- | --- |

    As we will prove in a more general statement (Theorem[2](#Thmtheorem2 "Theorem 2. ‣ 4 Sampling methods and the SQRT Sampling algorithm ‣ Scaling Neural Program Synthesis with Distribution-based Search")),
        the algorithm $A_{3}$ is loss optimal among sampling algorithms.
        Suprisingly it is not much worse than the loss optimal algorithm, yet offers many advantages:
        it is much easier to implement, and requires no memory at all.
        Last but not least in the case of PCFG it can be implemented using a new probabilistic labelling of the PCFG inducing $\mathcal{D}$.

The next two sections are devoted to the two natural classes of algorithms for distribution-based search: *enumeration* and *sampling*. We then study how they can run at scale accross parallel compute environments.

3 Enumerative methods and  the Heap Search algorithm
----------------------------------------------------

A number of enumerative methods have been investigated in previous works*Menon et al. ([2013](#bib.bib18 "")), Balog et al. ([2017](#bib.bib3 "")), Feng et al. ([2018](#bib.bib12 "")), Zohar and Wolf ([2018](#bib.bib26 ""))*.
They proceed in a top-down fashion, and can be understood as ways of exploring the tree of leftmost derivations of the grammar
as illustrated in Figure[2](#S3.F2 "Figure 2 ‣ 3 Enumerative methods and the Heap Search algorithm ‣ Scaling Neural Program Synthesis with Distribution-based Search").

<img src='tree_of_derivations.png' alt='Refer to caption' title='' width='592' height='489' />

*Figure 2: Illustration of the tree of leftmost derivations.*

We present a new efficient and loss optimal algorithm called Heap Search and following a bottom-up approach.
It uses a data structure based on heaps to efficiently enumerate all programs in non-increasing order of the
probabilities in the PCFG.

Let us write $\mathcal{D}$ for the distribution induced by a PCFG.
For a program $x$, we say that $x^{\prime}$ is the ‘successor of $x$’ if it is the most likely program after $x$,
meaning $\mathcal{D}(x)>\mathcal{D}(x^{\prime})$ and there are no programs $x^{\prime\prime}$ such that $\mathcal{D}(x)>\mathcal{D}(x^{\prime\prime})>\mathcal{D}(x^{\prime})$.
For a non-terminal $T$ in the grammar, the ‘successor of $x$ from $T$’ is the most likely program
after $x$ among those generated from $T$. We define ‘predecessor of
$x$’ and ‘predecessor of $x$ from $T$’ in a similar way.

We create a procedure $\textbf{Query}(T,x)$ which for any program $x$ generated from a non-terminal $T$ outputs the successor of $x$ from $T$.
Note that once this is defined, the algorithm performs successive calls to $\textbf{Query}(S,x)$ with $S$ the initial non-terminal and $x$ the latest generated program, yielding all programs in non-increasing order of the probabilities.

To explain how $\textbf{Query}(T,x)$ works, we first define the data structure.
For each non-terminal $T$ we have a hash table $\textsc{Succ}_{T}$ which stores the successors of all previously seen programs generated from $T$, and a heap $\textsc{Heap}_{T}$ which contains candidate programs, ordered by non-increasing probability.
The key invariant is the following:
the successor of $T$ from $x$ has either already been computed, hence is in $\textsc{Succ}_{T}$,
or is the maximum program in $\textsc{Heap}_{T}$.
This means that implementing $\textbf{Query}(T,x)$ is very simple: it checks whether the successor has already been computed and returns it in that case, and if not it pops the heap.
The difficulty is in maintaining the invariant; for this we need to add a number of candidate programs to the heaps.
They are obtained by substituting one argument from the returned successor, and propagate this recursively to the corresponding non-terminals.

###### Theorem 1.

The Heap Search algorithm is loss optimal: it enumerates every program exactly once and in non-increasing order of probabilities.

We refer to Appendix[C](#A3 "Appendix C Heap Search ‣ Scaling Neural Program Synthesis with Distribution-based Search") for a complete description of the algorithm with a pseudocode,
a proof of Theorem[1](#Thmtheorem1 "Theorem 1. ‣ 3 Enumerative methods and the Heap Search algorithm ‣ Scaling Neural Program Synthesis with Distribution-based Search"), and a computational complexity analysis.

Heap Search is related to $A^{*}$ from*Feng et al. ([2018](#bib.bib12 ""))*: they are both loss optimal algorithms,
meaning that they both enumerate programs in non-increasing order of probabilities.
As we will see in the experiments Heap Search is better in two aspects:
it is faster, and it is bottom-up, implying that program evaluations can be computed along with the programs and avoiding evaluating twice the same (sub)program.

4 Sampling methods and  the SQRT Sampling algorithm
---------------------------------------------------

A *sampling algorithm* takes random samples from a distribution $\mathcal{D}^{\prime}$; what is both a strength and a weakness is that a sampling algorithm is memoryless: a weakness because the algorithm does not remember the previous draws, which means that it may draw them again, but also a strength because it uses very little space and can be very easily implemented.

In the case of sampling algorithms, we identify algorithms with distributions.
The following theorem shows a dichotomy:
either there exists a loss optimal sampling algorithm among sampling algorithms,
and then it is characterised as the ‘square root’ of the distribution,
or all sampling algorithms have infinite loss.

###### Theorem 2.

Let $\mathcal{D}$ a distribution over a set $X$.
If $\sum_{x\in X}\sqrt{\mathcal{D}(x)}<\infty$, the distribution $\sqrt{\mathcal{D}}$ defined by

|  | $\sqrt{\mathcal{D}}(x)\=\frac{\sqrt{\mathcal{D}(x)}}{\sum_{y\in X}\sqrt{\mathcal{D}(y)}}$ |  |
| --- | --- | --- |

is loss optimal among all sampling algorithms.
If $\sum_{x\in X}\sqrt{\mathcal{D}(x)}\=\infty$, for all sampling algorithms $\mathcal{D}^{\prime}$ we have $\mathcal{L}(\mathcal{D}^{\prime},\mathcal{D})\=\infty$.

###### Proof.

Let $\mathcal{D}^{\prime}$ be a distribution.
For an element $x$, the expectation of the number of tries for $\mathcal{D}^{\prime}$ to draw $x$ is $\frac{1}{\mathcal{D}^{\prime}(x)}$:
this is the expectation of success for the geometric distribution with parameter $\mathcal{D}^{\prime}(x)$.
It follows that

|  | $\mathcal{L}(\mathcal{D}^{\prime},\mathcal{D})\=\mathbb{E}_{x\sim D}\left[\frac{1}{\mathcal{D}^{\prime}(x)}\right]\=\sum_{x\in X}\frac{\mathcal{D}(x)}{\mathcal{D}^{\prime}(x)}.$ |  |
| --- | --- | --- |

Let us assume that $\sum_{x\in X}\sqrt{\mathcal{D}(x)}<\infty$.
Thanks to Cauchy-Schwarz inequality we have:

|  | $\begin{array}[]{lll}\left(\sum_{x\in X}\sqrt{\mathcal{D}(x)}\right)^{2}\&\=\&\left(\sum_{x\in X}\sqrt{\frac{\mathcal{D}(x)}{\mathcal{D}^{\prime}(x)}}\sqrt{\mathcal{D}^{\prime}(x)}\right)^{2}\\ \&\leq\&\left(\sum_{x\in X}\frac{\mathcal{D}(x)}{\mathcal{D}^{\prime}(x)}\right)\cdot\underbrace{\left(\sum_{x\in X}\mathcal{D}^{\prime}(x)\right)}_{\=1}\\ \&\=\&\sum_{x\in X}\frac{\mathcal{D}(x)}{\mathcal{D}^{\prime}(x)}.\end{array}$ |  |
| --- | --- | --- |

We note that $\mathcal{L}(\sqrt{\mathcal{D}},\mathcal{D})\=\left(\sum_{x\in X}\sqrt{\mathcal{D}(x)}\right)^{2}$,
so the previous inequality reads $\mathcal{L}(\mathcal{D}^{\prime},\mathcal{D})\geq\mathcal{L}(\sqrt{\mathcal{D}},\mathcal{D})$.
Thus $\sqrt{\mathcal{D}}$ is loss optimal among sampling algorithms,
and if it is not defined, then for any $\mathcal{D}^{\prime}$ we have $\mathcal{L}(\mathcal{D}^{\prime},\mathcal{D})\=\infty$.
∎

Theorem[2](#Thmtheorem2 "Theorem 2. ‣ 4 Sampling methods and the SQRT Sampling algorithm ‣ Scaling Neural Program Synthesis with Distribution-based Search") characterises the loss optimal sampling algorithm, but does not explain how to implement it.
The following result answers that question.

###### Theorem 3.

If $\mathcal{D}$ is defined by a PCFG and $\sqrt{\mathcal{D}}$ is well defined,
then we can effectively construct a PCFG defining $\sqrt{\mathcal{D}}$.

The PCFG for $\sqrt{\mathcal{D}}$ is obtained from the PCFG for $\mathcal{D}$ by taking the square root of each transition probability,
and then globally renormalising.
Details of this procedure can be found in Appendix[D](#A4 "Appendix D SQRT Sampling ‣ Scaling Neural Program Synthesis with Distribution-based Search").

5 Parallel implementations
--------------------------

Harnessing parallel compute environments is necessary for scalable, future-proof search algorithms,
because combinatorial search bottlenecks on compute,
and both the present and likely future of massive compute is a parallel one.
Accordingly, we have taken care to design and evaluate extensions of our algorithms which can metabolize these compute resources through multiprocessing.

We introduce a new algorithm called the *grammar splitter*, which partitions a PCFG into a balanced family of $k$ sub-PCFGs.
Each of the $k$ threads is assigned a sub-PCFG and simply runs a search algorithm on it.
Two key advantages of our approach are that any search algorithm can be used in this very simple parallel architecture,
and that the theoretical gain of using $k$ threads is linear in $k$.
The output of the grammar splitter is illustrated in Figure[3](#S5.F3 "Figure 3 ‣ 5 Parallel implementations ‣ Scaling Neural Program Synthesis with Distribution-based Search"): the white PCFG is split into $4$ sub-PCFGs.

<img src='grammar_split.png' alt='Refer to caption' title='' width='592' height='362' />

*Figure 3: The grammar splitter: a balanced partition with quality $\alpha\=\frac{.3}{.25}\=1.2$.*

The two crucial properties of the grammar splitter are:

* •

    the space of programs is partitioned into $k$ subspaces.
    This implies that the threads do not carry out redundant work and that all programs are generated,

* •

    the $k$ program subspaces are balanced, meaning that their mass probabilities are (approximately) equal.
    This implies that all threads contribute equally to the search effort.

A split is a collection of partial programs, for instance `map (* 2) HOLE` and `fold + HOLE HOLE`,
it induces a PCFG.
A set of $k$ incomparable splits yields a partition of the PCFG.
Let us write $\alpha$ for the quality of a partition, defined as the ratio between the maximum and the minimum probability mass of a split.
We are looking for a balanced partition, i.e. one for which the quality $\alpha$ is close to $1$.

Our algorithm finds a balanced partition through a hill climbing process:
at each point the algorithm either looks for an improving swap or a refinement.
In the first case, the action of an improving swap is to transfer a partial program from one split to another,
and its goal is to lower the quality coefficient.
In the second case, we consider the partial program with maximal probability in a split
and refine it: for example `map (* 2) HOLE` could be replaced by `map (* 2) var0` and `map (* 2) (filter HOLE HOLE)`.

<img src='cumulative_probability_vs_time_100.png' alt='Refer to caption' title='' width='2315' height='1749' />

*(a)*

<img src='cumulative_probability_vs_number_programs_100.png' alt='Refer to caption' title='' width='2329' height='1748' />

*(b)*

*Figure 4: Comparing all search algorithms on random PCFGs*

6 Experiments
-------------

We study a range of search algorithms – both our new ones and prior work – across list processing and string manipulation domains, with the goal of answering the following questions:

* •

    Heap Search and $A^{*}$ are both loss optimal enumerative algorithms; beyond these theoretical guarantees, how do the two algorithms compare in practice?

* •

    How effective are our search algorithms for solving complex program synthesis benchmarks using neural guidance?

* •

    How do our algorithms scale with parallel compute?

We use a generic program synthesizer written from scratch in Python
(Appendix[A](#A1 "Appendix A Machine-learned program synthesis ‣ Scaling Neural Program Synthesis with Distribution-based Search")),
studying random PCFGs (more controlled) and machine-learned PCFGs (more naturalistic).

We report results on DSLs from DeepCoder*Balog et al. ([2017](#bib.bib3 ""))* and DreamCoder*Ellis et al. ([2021](#bib.bib11 ""))*.
Both target the classic program synthesis challenge of integer list processing programs, but with different properties.
DeepCoder’s DSL is larger and more specialized, with around $40$ high-level primitives, and does not use polymorphic types,
while DreamCoder’s is smaller and more generic, with basic functional programming primitives such as map, fold, unfold, car, cons, and cdr, etc., for a total of around $20$ primitives.
Both DSLs are compiled into a CFG with minimal syntactic constraints generating programs of depth $6$.

The search algorithms under consideration are:

* •

    Threshold from*Menon et al. ([2013](#bib.bib18 ""))*: iterative-deepening-search, where the threshold that is iteratively deepened is a bound on program description length (i.e. negative log probability),

* •

    Sort and add from*Balog et al. ([2017](#bib.bib3 ""))*: an inner loop of depth-first-search, with an outer loop that sorts productions by probability and runs depth-first-search with the top $k$ productions for increasing values of $k$,

* •

    $A^{*}$ from*Feng et al. ([2018](#bib.bib12 ""))*: best-first-search on the graph of (log probabilities of) tree derivations,

* •

    Beam search from*Zhang et al. ([2018](#bib.bib25 ""))*: breadth-first-search with bounded width that is iteratively increased.

As well as our new algorithms: Heap Search and SQRT Sampling.
We refer to Appendix[B](#A2 "Appendix B Review of enumerative methods ‣ Scaling Neural Program Synthesis with Distribution-based Search") for a description of the algorithms and their implementations.

Our implementation of SQRT Sampling uses the Alias method*Walker ([1977](#bib.bib24 ""))*,
which is an efficient data structure for sampling from a categorical distribution.
We associate to each non-terminal an Alias table, reducing the task of sampling a derivation rule with $n$ choices
to sampling uniformly in $[1,n]$ and in $[0,1]$.

All algorithms have been reimplemented and optimised in the codebase
to provide a fair and uniform comparison.
We also report on parallel implementations using our grammar splitter.

### 6.1 Random PCFGs

In this first set of experiments we run all search algorithms on random PCFGs until a timeout,
and compare the number of programs they output and the cumulative probability of all programs output.

To obtain random PCFGs from the CFGs we sample a probabilistic labeling with an exponential decrease (this is justified by the fact that machine-learned PCFGs feature exponential decrease in transition probabilities).
In this experiment the initialization cost of each algorithm is ignored.
The results presented here are averaged over $50$ samples of random PCFGs, the solid lines represent the average
and a lighter color indicates the standard deviation.
Details on the sampling procedure can be found in Appendix[F](#A6 "Appendix F Details on the experiments ‣ Scaling Neural Program Synthesis with Distribution-based Search").

Figure[4](#S5.F4 "Figure 4 ‣ 5 Parallel implementations ‣ Scaling Neural Program Synthesis with Distribution-based Search") shows the results for all algorithms in a non-parallel implementation.
On the lhs we see that Heap Search (almost always) has the highest cumulative probability against both time and number of distinct programs.
Note that since $A^{*}$ and Heap Search enumerate the same programs in the same order they produce the same curve in the rhs of Figure[4](#S5.F4 "Figure 4 ‣ 5 Parallel implementations ‣ Scaling Neural Program Synthesis with Distribution-based Search") so we did not include $A^{*}$.

To compare $A^{*}$ and Heap Search we refer to Figure[5](#S6.F5 "Figure 5 ‣ 6.1 Random PCFGs ‣ 6 Experiments ‣ Scaling Neural Program Synthesis with Distribution-based Search"),
showing that Heap Search generates $2.35$ times more programs than $A^{*}$, consistently over time.
The larger variations for $A^{*}$ are due to the manipulation of a single heap of growing size, requiring frequent memory reallocations.
The difference in performance can be explained by the fact that $A^{*}$ uses a single heap for storing past computations,
while Heap Search distributes this information in a family of connected heaps and hash tables.

<img src='comparison_heapsearch_astar_100.png' alt='Refer to caption' title='' width='592' height='441' />

*Figure 5: Comparing Heap Search and $A^{*}$*

We then turn to parallel implementation and perform the same experiments using a variable number of CPUs for Heap Search and SQRT Sampling using the grammar splitter.
We do not report on a baseline parallel implementation of SQRT Sampling which would simply sample using the same PCFG on multiple CPUs. Indeed this naive approach performs poorly in comparison, since thanks to the grammar splitter two CPUs cannot generate the same program.

The results are shown in Figure[6](#S6.F6 "Figure 6 ‣ 6.1 Random PCFGs ‣ 6 Experiments ‣ Scaling Neural Program Synthesis with Distribution-based Search"), where we count programs with repetitions.
We see that for SQRT Sampling the scale-up is linear, and it is mostly linear for Heap Search with an acceleration from the 0.2s mark.
This acceleration can be explained in two ways:
first, each sub-PCFG is shallower since it is split thus it is faster to enumerate program from it,
second, once all the successors have been computed Heap Search is a simple lookup table.
At the end of the experiment, SQRT Sampling has generated 2.8 times more programs with 6 CPUs than with 2 CPUs,
whereas Heap Search has generated 7.6 times more programs with 6 CPUs than with 2 CPUs.

This experiment suggests that the grammar splitter enables us to scale our search on multiple CPUs with a linear speed up in the number of CPUs.

<img src='programs_vs_time_100_parallel.png' alt='Refer to caption' title='' width='592' height='441' />

*Figure 6: Parallel implementations of Heap Search and SQRT Sampling using the grammar splitter*

### 6.2 Machine-learned PCFGs

In this second set of experiments we consider the benchmark suites of hand-picked problems and sets of I/O.
We extracted 218 problems from DreamCoder’s dataset*(Ellis et al., [2021](#bib.bib11 ""))*.
(The experiments can be easily replicated on DeepCoder’s dataset*(Balog et al., [2017](#bib.bib3 ""))* but we do not report on the results here.)

We train a neural network to make predictions from a set of I/O.
Our neural network is composed of a one layer GRU*Cho et al. ([2014](#bib.bib7 ""))* and a 3-layer MLP with sigmoid activation functions,
and trained on synthetic data generated from the DSL.
The details of the architecture and the training can be found in Appendix[F](#A6 "Appendix F Details on the experiments ‣ Scaling Neural Program Synthesis with Distribution-based Search").
Our network architecture induces some restrictions, for instance the types of the programs must be `int list -> int list`;
we removed tasks that did not fit our restrictions and obtained a filtered dataset of 137 tasks.
For each task we run every search algorithm on the PCFG induced by the neural predictions with a timeout of $100$s and a maximum of $1M$ programs. Unlike in the previous experiments the initialization costs of algorithms are not ignored.

<img src='machine_learned_pcfg_tasks_vs_time.png' alt='Refer to caption' title='' width='592' height='448' />

*Figure 7: Comparing all search algorithms on the DreamCoder reduced dataset with machine-learned PCFGs*

Figure[7](#S6.F7 "Figure 7 ‣ 6.2 Machine-learned PCFGs ‣ 6 Experiments ‣ Scaling Neural Program Synthesis with Distribution-based Search") shows the number of tasks solved within a time budget. Heap Search solves the largest number of tasks for any time budget, and in particular 97 tasks out of 137 before timeout.
The comparison between Threshold and $A^{*}$ is insightful:
$A^{*}$ solves a bit more tasks than Threshold (85 vs 83) but in twice the time. SQRT Sampling performs just a bit worse than $A^{*}$ despite being a sampling algorithm.

Table[1](#S6.T1 "Table 1 ‣ 6.2 Machine-learned PCFGs ‣ 6 Experiments ‣ Scaling Neural Program Synthesis with Distribution-based Search") shows for each algorithm how many programs were generated per second.
Recall that Heap Search and $A^{*}$ generate the same programs in the same order.
Overall in these experiments Heap Search is $6$ times faster than $A^{*}$.

Since Heap Search follows a bottom-up approach we save on program evaluations in two ways:
partial programs are evaluated along the search, and the results are cached.
On the other hand $A^{*}$ is a top-down enumeration method so every new program has to be evaluated from scratch.

It is interesting to compare the rates of SQRT Sampling and $A^{*}$:
although SQRT Sampling generates over two times more programs, their overall performances are similar.
This can be explained in two ways: SQRT Sampling may sample the same programs many times,
while $A^{*}$ enumerates each program once and starts with the most promising ones according to the predictions.

*Table 1: Number of programs generated*

| Algorithm | Number of programs generated |
| --- | --- |
| Heap Search | 38735 prog/s |
| Threshold | 25381 prog/s |
| DFS | 20281 prog/s |
| SQRT Sampling | 14020 prog/s |
| $A^{*}$ | 6071 prog/s |

Finally, we want to see how Heap Search improves with the quality of the predictions. According to the properties of Heap Search, we expect that the better the predictions the faster tasks are solved since Heap Search is loss optimal.
In order to show this, we ran Heap Search on the reduced DreamCoder dataset with a timeout of 5 minutes and kept the tasks where a solution was found since some tasks cannot be solved with our DSL.
Then we trained a neural network on this new reduced set with the solutions found.
At different epochs of the training we checked how fast and how many tasks Heap Search could solve with the predictions of the network with a timeout of 30 seconds and a limit of 1M programs.
We plot on Figure[8](#S6.F8 "Figure 8 ‣ 6.2 Machine-learned PCFGs ‣ 6 Experiments ‣ Scaling Neural Program Synthesis with Distribution-based Search") the number of tasks solved with respect to total time used by Heap Search after different number of training epochs with the uniform PCFG as a baseline.
We clearly observe the expected outcome, as the number of training epochs grows and the neural networks learn to better predict the solutions, Heap Search dramatically decreases the time required to solve the tasks. While this may be true for every algorithm, loss optimal algorithms like Heap Search benefit the most from it.

<img src='hs_time_evolution.png' alt='Refer to caption' title='' width='592' height='444' />

*Figure 8: Evolution of time and tasks solved by Heap Search at different epochs of the training of a machine-learning PCFG predictor on a reduced DreamCoder dataset*

7 Discussion
------------

### 7.1 Related work

The idea of guiding program search via probabilities is an old one*Solomonoff ([1989](#bib.bib23 ""))* but which recently has fast become a standard practice in the AI and program synthesis communities.
To the best of our knowledge, practical program synthesizers that learn to use probabilistic predictions originated in*(Menon et al., [2013](#bib.bib18 ""))*,
which was first extended with deep learning in*(Balog et al., [2017](#bib.bib3 ""))*,
and such methods are now winning program synthesis competitions*Lee et al. ([2018](#bib.bib17 ""))*.
To first approximation, such recent progress has drawn on advances in neural network architecture: e.g., early learned FlashFill-like systems*Parisotto et al. ([2017](#bib.bib19 ""))* benefited from sophisticated attention mechanisms*Devlin et al. ([2017](#bib.bib9 ""))*,
and procedural planning programs*Bunel et al. ([2018](#bib.bib4 ""))* benefited from feeding execution traces into the neural net*Chen et al. ([2018](#bib.bib5 ""))*.
While prior works have explored novel test-time strategies, from Sequential Monte Carlo*Ellis et al. ([2019](#bib.bib10 ""))* to ensembling methods*Chen et al. ([2018](#bib.bib5 ""))*,
here we sought a systematic empirical/theoretical study of two different families of inference strategies,
which we intend to mesh well with the larger body of work on neural program synthesis.
While the algorithms introduced here do not straightforwardly extend to neural autoregressive models (e.g. RobustFill*Devlin et al. ([2017](#bib.bib9 ""))*),
methods such as SQRT Sampling in principle apply to this setting too.
We hope that our work here spurs the development of the right tricks to get the theoretical benefits of SQRT Sampling for these more flexible model classes, just as DeepCoder paved the way for RobustFill.
Many extensions:*Lee et al. ([2018](#bib.bib17 "")), Zhang et al. ([2018](#bib.bib25 "")), Polosukhin and Skidanov ([2018](#bib.bib20 "")), Kalyan et al. ([2018](#bib.bib14 "")), Zohar and Wolf ([2018](#bib.bib26 ""))*.

*Shi et al. ([2020b](#bib.bib22 ""))* introduced Unique Randomizer in order to sample without replacement:
it is a general technique effectively turning a sampling method into an enumerative one
by updating the probabilistic weights along the search.
It is further improved through batching via Stochastic Beam Search*Kool et al. ([2019](#bib.bib16 ""))*.
It is possible to combine the SQRT Sampling algorithm with the Unique Randomizer and Stochastic Beam Search.
Our experiments did not yield interesting results in that direction,
possibly because of memory allocation issues.
We leave for future work to optimise this approach.

### 7.2 Contributions and Outlook

Learning to synthesize programs is a canonical neural-symbolic learning problem:
training high capacity statistical models to guide the construction of richly structured combinatorial objects, such as programs.
Yet while the neural side of this problem has received much deserved attention, the symbolic component is sometimes taken for granted–after all, symbolic program synthesis has received decades of attention.
But the entrance of powerful neural networks for synthesizing programs forces us to reconsider how we deploy symbolic methods for program synthesis.
We have considered both systematic and stochastic methods, from both theoretical angles (obtaining guarantees) and also engineering perspectives (such as how to parallelize our new techniques).
We hope this work helps contribute to thinking through the symbolic search back-end in this more modern context.

References
----------

* Alur et al. (2013)R. Alur, R. Bodík, G. Juniwal, M. M. K. Martin, M. Raghothaman, S. A.
Seshia, R. Singh, A. Solar-Lezama, E. Torlak, and A. Udupa.Syntax-guided synthesis.In *Formal Methods in Computer-Aided Design, FMCAD*, 2013.URL <http://ieeexplore.ieee.org/document/6679385/>.
* Alur et al. (2018)R. Alur, R. Singh, D. Fisman, and A. Solar-Lezama.Search-based program synthesis.*Communications of the ACM*, 61(12), 2018.URL [https://doi.org/10.1145/3208071](https://doi.org/10.1145/3208071 "").
* Balog et al. (2017)M. Balog, A. L. Gaunt, M. Brockschmidt, S. Nowozin, and D. Tarlow.Deepcoder: Learning to write programs.In *International Conference on Learning Representations,
ICLR*, 2017.URL [https://openreview.net/forum?id\=ByldLrqlx](https://openreview.net/forum?id=ByldLrqlx "").
* Bunel et al. (2018)R. Bunel, M. J. Hausknecht, J. Devlin, R. Singh, and P. Kohli.Leveraging grammar and reinforcement learning for neural program
synthesis.In *International Conference on Learning Representations,
ICLR*, 2018.URL [https://openreview.net/forum?id\=H1Xw62kRZ](https://openreview.net/forum?id=H1Xw62kRZ "").
* Chen et al. (2018)X. Chen, C. Liu, and D. Song.Execution-guided neural program synthesis.In *International Conference on Learning Representations,
ICLR*, 2018.
* Chi (1999)Z. Chi.Statistical properties of probabilistic context-free grammars.*Computational Linguistics*, 25(1), 1999.URL [https://www.aclweb.org/anthology/J99-1004](https://www.aclweb.org/anthology/J99-1004 "").
* Cho et al. (2014)K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares,
H. Schwenk, and Y. Bengio.Learning phrase representations using RNN encoder–decoder for
statistical machine translation.In *Conference on Empirical Methods in Natural Language
Processing, EMNLP*, 2014.URL [https://aclanthology.org/D14-1179](https://aclanthology.org/D14-1179 "").
* Clymo et al. (2020)J. Clymo, A. Gascón, B. Paige, N. Fijalkow, and H. Manukian.Data generation for neural programming by example.In *International Conference on Artificial Intelligence and
Statistics, AI\&STATS*, 2020.URL <http://proceedings.mlr.press/v108/clymo20a.html>.
* Devlin et al. (2017)J. Devlin, J. Uesato, S. Bhupatiraju, R. Singh, A. Mohamed, and P. Kohli.Robustfill: Neural program learning under noisy I/O.In *International Conference on Machine Learning, ICML*,
volume 70 of *Proceedings of Machine Learning Research*, 2017.URL <http://proceedings.mlr.press/v70/devlin17a.html>.
* Ellis et al. (2019)K. Ellis, M. Nye, Y. Pu, F. Sosa, J. Tenenbaum, and A. Solar-Lezama.Write, execute, assess: Program synthesis with a REPL.In *Neural Information Processing Systems, NeurIPS*, 2019.URL[https://proceedings.neurips.cc/paper/2019/hash/50d2d2262762648589b1943078712aa6-Abstract.html](https://proceedings.neurips.cc/paper/2019/hash/50d2d2262762648589b1943078712aa6-Abstract.html "").
* Ellis et al. (2021)K. Ellis, C. Wong, M. I. Nye, M. Sablé-Meyer, L. Morales, L. B. Hewitt,
L. Cary, A. Solar-Lezama, and J. B. Tenenbaum.Dreamcoder: bootstrapping inductive program synthesis with wake-sleep
library learning.In *International Conference on Programming Language Design and
Implementation, PLDI*, 2021.URL [https://doi.org/10.1145/3453483.3454080](https://doi.org/10.1145/3453483.3454080 "").
* Feng et al. (2018)Y. Feng, R. Martins, O. Bastani, and I. Dillig.Program synthesis using conflict-driven learning.In *ACM SIGPLAN Conference on Programming Language Design
and Implementation, PLDI*, 2018.URL [https://doi.org/10.1145/3192366.3192382](https://doi.org/10.1145/3192366.3192382 "").
* Gulwani et al. (2017)S. Gulwani, O. Polozov, and R. Singh.Program synthesis.*Foundations and Trends in Programming Languages*, 4(1-2), 2017.URL [https://doi.org/10.1561/2500000010](https://doi.org/10.1561/2500000010 "").
* Kalyan et al. (2018)A. Kalyan, A. Mohta, O. Polozov, D. Batra, P. Jain, and S. Gulwani.Neural-guided deductive search for real-time program synthesis from
examples.In *International Conference on Learning Representations,
ICLR*, 2018.URL [https://openreview.net/forum?id\=rywDjg-RW](https://openreview.net/forum?id=rywDjg-RW "").
* Kingma and Ba (2015)D. P. Kingma and J. Ba.Adam: A method for stochastic optimization.In *International Conference on Learning Representations,
ICLR*, 2015.URL [http://arxiv.org/abs/1412.6980](http://arxiv.org/abs/1412.6980 "").
* Kool et al. (2019)W. Kool, H. Van Hoof, and M. Welling.Stochastic beams and where to find them: The gumbel-top-k trick for
sampling sequences without replacement.In *International Conference on Machine Learning, ICML*, 2019.URL <https://proceedings.mlr.press/v97/kool19a.html>.
* Lee et al. (2018)W. Lee, K. Heo, R. Alur, and M. Naik.Accelerating search-based program synthesis using learned
probabilistic models.In *ACM SIGPLAN Conference on Programming Language Design
and Implementation, PLDI*, 2018.URL [https://doi.org/10.1145/3211992](https://doi.org/10.1145/3211992 "").
* Menon et al. (2013)A. K. Menon, O. Tamuz, S. Gulwani, B. W. Lampson, and A. Kalai.A machine learning framework for programming by example.In *International Conference on Machine Learning, ICML*, 2013.URL [http://proceedings.mlr.press/v28/menon13.html](http://proceedings.mlr.press/v28/menon13.html "").
* Parisotto et al. (2017)E. Parisotto, A. Mohamed, R. Singh, L. Li, D. Zhou, and P. Kohli.Neuro-symbolic program synthesis.In *International Conference on Learning Representations,
ICLR*, 2017.URL [https://openreview.net/forum?id\=rJ0JwFcex](https://openreview.net/forum?id=rJ0JwFcex "").
* Polosukhin and Skidanov (2018)I. Polosukhin and A. Skidanov.Neural program search: Solving programming tasks from description and
examples.In *International Conference on Learning Representations,
ICLR*, 2018.URL [https://openreview.net/forum?id\=BJTtWDyPM](https://openreview.net/forum?id=BJTtWDyPM "").
* Shi et al. (2020a)K. Shi, D. Bieber, and R. Singh.TF-coder: Program synthesis for tensor manipulations.In *Workshop on Computer-Assisted Programming, CAP*,
2020a.URL [https://openreview.net/forum?id\=nJ5Ij53umw2](https://openreview.net/forum?id=nJ5Ij53umw2 "").
* Shi et al. (2020b)K. Shi, D. Bieber, and C. Sutton.Incremental sampling without replacement for sequence models.In *International Conference on Machine Learning, ICML*,
2020b.URL <https://proceedings.mlr.press/v119/shi20a.html>.
* Solomonoff (1989)R. Solomonoff.A system for incremental learning based on algorithmic probability.In *Israeli Conference on Artificial Intelligence, Computer
Vision and Pattern Recognition*, 1989.URL [https://theworld.com/~rjs/publications/IncLrn89.pdf](https://theworld.com/~rjs/publications/IncLrn89.pdf "").
* Walker (1977)A. J. Walker.An efficient method for generating discrete random variables with
general distributions.*ACM Transactions on Mathematical Software*, 3(3),
1977.URL [https://doi.org/10.1145/355744.355749](https://doi.org/10.1145/355744.355749 "").
* Zhang et al. (2018)L. Zhang, G. Rosenblatt, E. Fetaya, R. Liao, W. E. Byrd, R. Urtasun, and R. S.
Zemel.Leveraging constraint logic programming for neural guided program
synthesis.In *International Conference on Learning Representations,
ICLR*, 2018.URL [https://openreview.net/forum?id\=HJIHtIJvz](https://openreview.net/forum?id=HJIHtIJvz "").
* Zohar and Wolf (2018)A. Zohar and L. Wolf.Automatic program synthesis of long programs with a learned garbage
collector.In *Neural Information Processing Systems, NeurIPS*, 2018.URL[https://proceedings.neurips.cc/paper/2018/hash/390e982518a50e280d8e2b535462ec1f-Abstract.html](https://proceedings.neurips.cc/paper/2018/hash/390e982518a50e280d8e2b535462ec1f-Abstract.html "").

Appendix A Machine-learned program synthesis
---------------------------------------------

### A.1 Program representation as trees

Programs are internally represented as trees as illustrated on the lhs of Figure[9](#A1.F9 "Figure 9 ‣ A.1 Program representation as trees ‣ Appendix A Machine-learned program synthesis ‣ Scaling Neural Program Synthesis with Distribution-based Search").
The program described by this tree is `(take (head (rev l1)) (rev l2))`.
We enforce a simple type system for programs consisting of basic types such as `int, bool, string, float` and two constructors: `list` and `arrow`.
Each primitive in the DSL comes with a possibly polymorphic type, for instance `take` has the type `take: int -> t0 list -> t0 list`, where `t0` can be instantiated to any type.
We only construct correctly typed programs using classical techniques from programming languages such as
type inference and de Bruijn’s indices for variables.

<img src='tree_string.png' alt='Refer to caption' title='' width='479' height='182' />

*Figure 9: Tree and straight line representation of the same program.*

### A.2 Compilation phase: from domain specific language to grammar

A domain specific language (DSL) is a set of primitives together with their type and semantics.
The compilation phase turns a DSL and a set of syntactic constraints into a CFG which generates the set of all correctly typed programs by following typing rules.
Syntactic constraints can be very diverse, they include maximum program depth, type request for the program, and other invariants to be satisfied.
The type request induces the initial non-terminal of the CFG:
for instance if the type request is `int -> int list -> int list` then the initial non-terminal carries the information that the program should output an `int list`,
and that we can use one variable of type `int` and another one of type `int list`.

Polymorphic types can be accommodated in two different ways: at compilation time or at generation time.
DreamCoder’s implementation*Ellis et al. ([2021](#bib.bib11 ""))* uses polymorphic types in the CFGs, and performs type inference when generating programs.
In our tool we instantiate types during the compilation into CFGs, implying that they do not contain polymorphic types.
So when adding the primitive `take: int -> t0 list -> t0 list` we create a number of primitives for each type of bounded size, for instance

`take[int]: int -> int list -> int list`.

This makes the CFGs larger, but considerably reduces the computation time for generating programs
by avoiding type inference algorithms.

The compilation phase consists in expanding typing rules as long as syntactic constraints are satisfied.
The CFGs are trimmed to keep their sizes manageable, removing non-productive and non-reachable non-terminals.

Appendix B Review of enumerative methods
----------------------------------------

We describe the different search algorithms constructed in*Menon et al. ([2013](#bib.bib18 "")), Balog et al. ([2017](#bib.bib3 "")), Feng et al. ([2018](#bib.bib12 "")), Zhang et al. ([2018](#bib.bib25 ""))*,
which can be understood as ways of exploring the tree of leftmost derivations of the grammar.

##### BFS and DFS for CFGs.

The first algorithms we mention do not make use of the weights in the PCFG, they are enumeration algorithms for CFGs.
We consider the tree of all leftmost partial derivations starting from the initial non-terminal; each node of this tree is a partial program.
There are two classical algorithms for searching this tree. Both maintain a tree of partial derivations.
To resolve ambiguity, for each non-terminal we fix an (arbitrary) order on the derivation rules from this non-terminal.
The breadth-first search (BFS) expands the *highest* node containing a non-terminal by applying the next derivation rule to the leftmost non-terminal in this node (next with respect to the order on derivation rules from this non-terminal).
The depth-first search (DFS) instead expands the *lowest* node containing a non-terminal.
The DFS may not terminate, we need to introduce a stopping criterion;
the most natural is the size of the program: we fix a target size $n$ and before applying a derivation rule we check whether the partial program has size at most $n$. With this stopping criterion the DFS enumerates all programs of size at most $n$.

The remaining algorithms take into account the probabilities attached to the PCFG.

##### The sort and add searchBalog et al. ([2017](#bib.bib3 "")).

The algorithm is based on two ideas: the first idea is called ‘biassing’; we specify the order on the derivation rules for each non-terminal,
naturally by non-increasing order of probabilities,
and the second idea is an incremental search: we fix a number $k$, restrict for each non-terminal to the $k$ most likely derivation rules,
runs the search on this restricted grammar, and iterates with a larger $k$ in case the program was not found.
The choice of the sequence of values for $k$ is both crucial for the performances and very hard to make.
In our experiments we use arithmetic progressions.

##### The maximal probability from a non-terminal.

The following subroutine was introduced in*Menon et al. ([2013](#bib.bib18 ""))* and will be used by all subsequent algorithms.
For each non-terminal $T$, we write $m_{T}$ for the most likely program generated from $T$, and $p_{T}$ for its probability.
Computing $m_{T}$ and $p_{T}$ is done using dynamic programming:
$p_{T}$ is the maximum of $p\cdot\prod_{i\in[1,k]}p_{T_{i}}$ over all derivation rules $T\to^{p}f(T_{1},\dots,T_{k})$.

##### The threshold algorithmMenon et al. ([2013](#bib.bib18 "")).

The algorithm fixes a threshold, enumerates all programs with probability beating the threshold, and iterates with a lower threshold.
Enumerating all programs beating a fixed threshold is achieved using a DFS with the following stopping criterion:
before expanding a partial program we check whether it may be completed into a program with probability beating the threshold
by replacing each non-terminal $T$ by the probability of the most likely program $p_{T}$
and checking whether that program has high enough probability.
In our experiments we use geometric progressions for the sequence of thresholds.

##### The beam searchZhang et al. ([2018](#bib.bib25 "")).

Beam search is a BFS algorithm with bounded memory: the size of the queue of partial program,
called the beam, is restricted to a constant, the beam width.

Appendix C Heap Search
----------------------

The pseudocode is given in Algorithm[1](#alg1 "Algorithm 1 ‣ Appendix C Heap Search ‣ Scaling Neural Program Synthesis with Distribution-based Search").

*Algorithm 1  Heap search*

1:procedure Initialization( )

2: for allnon-terminal symbols $T$do

3:Create an empty max heap $\textsc{Heap}_{T}$

4:Create an empty hash table $\textsc{Succ}_{T}$

5:Create an empty set $\textsc{Seen}_{T}$

6: for allderivation rules $T\to f(T_{1},\dots,T_{k})$do

7:Add $f(m_{1},\dots,m_{k})$ to $\textsc{Heap}_{T}$ with priority $\mathcal{D}(f(m_{1},\dots,m_{k}))$

8:Add $f(m_{1},\dots,m_{k})$ to $\textsc{Seen}_{T}$

9: end for

10: end for

11:end procedure

12:

13:procedure Query($T$,$x$)

14: if$x$ is a key of $\textsc{Succ}_{T}$then

15:Return $\textsc{Succ}_{T}[x]$

16: else

17:$x^{\prime}\leftarrow pop(\textsc{Heap}_{T})$ $\triangleright$ $x^{\prime}$ is the successor of $x$

18:$\textsc{Succ}_{T}[x]\leftarrow x^{\prime}$ $\triangleright$ updating the data structure

19:(Assumes $x^{\prime}\=f(x_{1},\dots,x_{k})$ is generated by $T\to f(T_{1},\dots,T_{k})$)

20: for all$i\in[1,k]$do $\triangleright$ add all potential successors

21:$y_{i}\=\textbf{Query}(T_{i},x_{i})$

22:$x^{\prime}_{i}\=f(x_{1},\dots,x_{i-1},y_{i},x_{i+1},\dots,x_{k})$

23: if$x^{\prime}_{i}$ is not in $\textsc{Seen}_{T}$then

24:Add $x^{\prime}_{i}$ to $\textsc{Heap}_{T}$ with priority $\mathcal{D}(x^{\prime}_{i})$

25:Add $x^{\prime}_{i}$ to $\textsc{Seen}_{T}$

26: end if

27: end for

28:Return $x^{\prime}$

29: end if

30:end procedure

### C.1 Correctness proof

The following lemma gives the correctness of the heap search algorithm.

###### Lemma 1.

For any non-terminal $T$ and program $x$ generated from $T$, if we
have already run Query(T,y) for any program y preceding x (among
those generated from T) then Query(T,x) outputs the successor of
$x$ from $T$

###### Proof.

Suppose the lemma is not true and consider the first time where it
fails, on say, input $(T,x)$. At step 14, $(T,x)$ was not queried
before and therefore the algorithm branches out to line 16. Saying
that the algorithm fails is equivalent to say that line 17 fails,
meaning that $x^{\prime}\=pop(\textsc{Heap}_{T})$ is not the successor of $x$ from
$T$. Let us denote by $y$ the correct successor. There are two cases
which can lead the algorithm the failure:

* •

    First case: $x^{\prime}$ is a program with higher
    probability than $y$. In this case, it means that $x^{\prime}$ was already
    the output of a previous query (because $(T,x)$ is the first query
    for which the algorithm fails, so programs with higher
    probability than $y$ have already been output before). Thus the
    program $x^{\prime}$ was pop at least two times from $\textsc{Heap}_{T}$, one time
    when $x^{\prime}$ was the correct output of a query, and another time when
    the program failed: this is not possible since we push programs
    only once into the heaps thanks to the condition at line 22.

* •

    Second case:  $x^{\prime}$ has a smaller probability than the
    probability of $y$. In this case, it means that $y$ was not in
    $\textsc{Heap}_{T}$ when the algorithm performed a pop at line 17 (otherwise
    $y$ would have pop since $y$ has higher probability than
    $x^{\prime}$). Suppose $y\=f(x_{1},\dots,x_{k})$, generated by the
    derivation rule $T\to f(T_{1},\dots,T_{k})$. If all $x_{i}$ are maximal
    (meaning that they don’t have a predecessor from $T_{i}$) then
    $f(x_{1},\dots,x_{k})$ is pushed in $\textsc{Heap}_{T}$ during the
    initialization procedure (line 6,7) so the algorithm cannot fail
    because of this case. Therefore there is at least one $x_{i}$, say
    w.l.o.g $x_{1}$, which has a predecessor $x^{\prime}_{1}$ from $T_{1}$. Consider
    the program $f(x^{\prime}_{1},x_{2},\dots,x_{k})$; this program has higher
    probability than $y$ and therefore has been seen before. Thus
    $f(\textbf{Query}(T_{1},x^{\prime}_{1}),x_{2},\dots,x_{k})$ was added to $\textsc{Heap}_{T}$
    because of line 23. To conclude, observe that
    $f(\textbf{Query}(T_{1},x^{\prime}_{1}),x_{2},\dots,x_{k})\=f(x_{1},x_{2},\dots,x_{k}))$
    so $y$ has previously been added to $\textsc{Heap}_{T}$.

∎

### C.2 Complexity analysis

###### Lemma 2.

Fix any non-terminal $T$. Suppose that we have already generated the
first i programs generated from $T$ (meaning that if $x$ is the
$j$-th program for $j<i$, then $x$ is a key of the hash table
$\textsc{Succ}_{T}$ and $\textsc{Succ}_{T}[x]$ is the $(j+1)$-th program generated from
$T$). Then querying the successor of the $i$-th program has a running
time of $O(\log i)$.

###### Proof.

First observe that a query can call recursively several others
queries. However, for any non-terminal symbol $T$ there is at most
one query of the form $\textbf{Query}(T,x)$ which leads the algorithm to
branch out to line 16 and thus to possibly rise other
queries. Indeed, this case happens only when the successor of $x$
has not been computed yet (otherwise the query stops at line 15);
this can happen for at most one program for any fixed symbol $T$:
the last program from $T$ already seen in any execution of the
algorithm. Forgetting about recursive queries, the running time of a
query going through line 16 is given by the pop and push operations
(line 17 and 23). The number of pops and pushs is at most $m+1$
where $m$ is the maximal arity of a function in the
grammar. Moreover, each such operation costs a running time of
$O(\log|\textsc{Heap}_{T}|)$, so the total time for the query is
$O(\log|\textsc{Heap}_{T}|)$.

Overall, the total running time is bounded by

|  | $\sum_{T}O(\log|\textsc{Heap}_{T}|).$ |  |
| --- | --- | --- |

To conclude, observe that when we query the successor of the i-th
program generated from $T$, the size of $\textsc{Heap}_{T}$ is bounded by
$m\cdot i$ since we push at most $m$ programs during any query not
already computed before.
∎

### C.3 Program evaluation

A key advantage of Heap Search is that it is bottom-up:
partial programs are composed from the leaves to the root.
This implies that partial programs can be evaluated and their evaluations cached.
Although memory hungry, this optimisation leads to major gains when taking evaluation into account.

Appendix D SQRT Sampling
------------------------

### D.1 An example

The simplest instantiation of this theorem is $X\=\left{\textrm{Head},\textrm{Tail}\right}$ and the Bernoulli distribution $\mathcal{D}$ with parameter $p$:
it draws Head with probability $p$ and Tail with probability $1-p$.

A sampling algorithm $\mathcal{D}^{\prime}$ is a Bernoulli distribution with parameter $p^{\prime}$,
and its loss is

|  | $\mathcal{L}(\mathcal{D}^{\prime},\mathcal{D})\=\mathbb{E}_{x\sim\mathcal{D}}\left[\frac{1}{\mathcal{D}^{\prime}(x)}\right]\=\frac{p}{p^{\prime}}+\frac{1-p}{1-p^{\prime}}.$ |  |
| --- | --- | --- |

Minimising for $p^{\prime}$ yields $p^{\prime}\=\frac{\sqrt{p}}{\sqrt{p}+\sqrt{1-p}}$,
inducing the loss optimal sampling algorithm $\sqrt{\mathcal{D}}$.

### D.2 Implementation details

The construction follows two steps: first construct a weighted CFG that recognises $\sqrt{\mathcal{D}}$,
and then normalise it into a PCFG using*Chi ([1999](#bib.bib6 ""))*.
The normalisation requires computing the partition function $Z$ defined by

|  | $Z(S)\=\sum_{P\text{ generated from }S}\mathcal{D}(P).$ |  |
| --- | --- | --- |

In general the partition function can be computed by solving a system of polynomial equations.
This is easier in our case since we restrict ourselves to acyclic PCFGs.

Appendix E Parallel implementation
----------------------------------

### E.1 Description and pseudocode

The pseudocode of the grammar splitter is given in Algorithm[2](#alg2 "Algorithm 2 ‣ E.1 Description and pseudocode ‣ Appendix E Parallel implementation ‣ Scaling Neural Program Synthesis with Distribution-based Search") using two procedures: split and find improving swap.
The split procedure describes at a high level how our grammar splitter works.
The find improving swap procedure is here to provide a clear method of finding an improving swap or refinement.

In our experiments, we initialized the splitting as follows:
we split the node with highest probability until the total number of nodes is greater than the number of splits required,
then assign one node to each split, and the remaining nodes to the last split.
We also limited the search of an improving swap or refinement to the most probable split and the least probable split unlike in the find improving swap procedure where all splits are considered.
Finally, in all of our experiments $\alpha_{desired}\=1.05$.

*Algorithm 2  Grammar splitter*

1:procedure split($G$, $nsplits$, $\alpha_{desired}$)

2:Create an initial splitting Splits

3:$\alpha\leftarrow\frac{\max_{sG\in\textsc{Splits}}{\text{probability mass}(sG)}}{\min_{sG\in\textsc{Splits}}{\text{probability mass}(sG)}}$

4: while$\alpha>\alpha_{desired}$do

5: ifan improving swap existsthen

6:Update Splits with the improving swap

7:$\alpha\leftarrow\frac{\max_{sG\in\textsc{Splits}}{\text{probability mass}(sG)}}{\min_{sG\in\textsc{Splits}}{\text{probability mass}(sG)}}$

8: else

9:Find the partial program with largest probability $P$

10:Replace $P$ in its split with its children

11: end if

12: end while

13:Return Splits

14:end procedure

15:procedure Find improving swap($G$, Splits)

16:$\alpha\leftarrow\frac{\max_{sG\in\textsc{Splits}}{\text{probability mass}(sG)}}{\min_{sG\in\textsc{Splits}}{\text{probability mass}(sG)}}$

17:$\alpha^{*}\leftarrow\alpha$ $\triangleright$ best improving swap $\alpha$

18:$s\leftarrow$ None $\triangleright$ best improving swap

19:$L\leftarrow\text{argmax}_{sG\in\textsc{Splits}}\ \text{probability mass}(sG)$

20:Sort Splits by increasing probability mass

21: for all$sG\in\textsc{Splits}\setminus{L}$do

22: for all$P^{\prime}\in L$do

23: for all$P\in G$do

24:$\beta\leftarrow$ Compute new $\alpha$ with Swap($L$, $sG$, $P$, $P^{\prime}$)

25: if$\beta<\alpha^{*}$then

26:$\alpha^{*}\leftarrow\beta$

27:$s\leftarrow$ Swap($L$, $sG$, $P$, $P^{\prime}$)

28: end if

29: end for

30:$\beta\leftarrow$ Compute new $\alpha$ with Gift($L$, $sG$, $P^{\prime}$)

31: if$\beta<\alpha^{*}$then

32:$\alpha^{*}\leftarrow\beta$

33:$s\leftarrow$ Gift($L$, $sG$, $P^{\prime}$)

34: end if

35: end for

36: end for

37:$l\leftarrow\text{argmin}_{sG\in\textsc{Splits}}\ \text{probability mass}(sG)$

38:Sort Splits by decreasing probability mass

39: for all$sG\in\textsc{Splits}\setminus{L,l}$do

40: for all$P\in sG$do

41: for all$P^{\prime}\in l$do

42:$\beta\leftarrow$ Compute new $\alpha$ with Swap($l$, $sG$, $P$, $P^{\prime}$)

43: if$\beta<\alpha^{*}$then

44:$\alpha^{*}\leftarrow\beta$

45:$s\leftarrow$ Swap($l$, $sG$, $P$, $P^{\prime}$)

46: end if

47: end for

48:$\beta\leftarrow$ Compute new $\alpha$ with Gift($sG$, $l$, $P$)

49: if$\beta<\alpha^{*}$then

50:$\alpha^{*}\leftarrow\beta$

51:$s\leftarrow$ Gift($sG$, $l$, $P$)

52: end if

53: end for

54: end for

55:Return $s$

56:end procedure

Appendix F Details on the experiments
-------------------------------------

### F.1 Random PCFG search

To turn the CFGs into PCFGs we sample a probabilistic labelling, meaning
a weight for each derivation rule such that the sum over all derivation rules from each non-terminal is one.
The distribution depends on a parameter $\alpha\in(0,1]$:
the $i$th weight is sampled uniformly at random in $[0,\alpha^{i}]$,
and eventually renormalised.
The smaller $\alpha$, the more biassed the distribution,
implying that the search for programs will be faster since we have better hints on the target program.
For $\alpha\=1$ the weights are sampled uniformly, resulting in a very unbiased PCFG, making the search more difficult.
In the random PCFGs experiment we use $\alpha\=0.7$.

### F.2 Machine-learned PCFG

We only work with tasks of type `int list -> int list`.
We remove examples where lists have length greater than $L_{max}\=10$ or where one of the elements of the input or output list are not in $L_{in}\=\left[-30;30\right]$.

We say that a task is solved if a program is found which satisfies all examples.
The timeout of $100$s only takes into account the search time and the evaluation times and not the time to query the neural network for predictions.

#### The neural network

The neural network takes as input (the encoding of) a list of examples and outputs a probabilistic labelling for the CFG.

##### Input encoding

Each list in the examples is encoded in a naive way by mapping each element of $L_{in}$ to $[0,60]$.
We additionally use two special symbols `PAD` and `NOPAD`, leading to
a fixed size encoding of a list into a vector of size $2L_{max}$.
Hence an example is encoded as a tensor of shape $(n_{inputs},2L_{max})$ where $n_{inputs}$ is the number of inputs in the example.

##### Embedding

The encoded examples are fed into an embedding consisting of a single layer GRU*Cho et al. ([2014](#bib.bib7 ""))* outputting a tensor of shape $(s_{GRU}\times 2L_{max})$.
In our experiments $s_{GRU}\=10$.

##### Main layers

The embedded examples are fed into an MLP with 3 layers.
The first two have output size $s_{MLP}\=64$ and the last layer outputs a tensor of dimension $k$
the number of derivation rules in the CFG $C$.
Assigning these (normalised) weights to the rules of the CFG yields a PCFG.

#### Training

We optimised end-to-end the neural network with Adam*Kingma and Ba ([2015](#bib.bib15 ""))* with default parameters and learning rate of $lr\=0.001$.
We trained for one epoch with a batch size of $128$ on a generated dataset of $10,000$ problems.
To generate the dataset, programs are sampled from the uniform PCFG and inputs by choosing a length at most $L_{max}$
and elements uniformly at random in $L_{in}$.
If the output of such a generated input does not fall in the lexicon $L_{in}$ then the program is discarded.

We use the binary cross entropy loss between the output of the neural network and the encoding of a solution program:
a rule in the CFG has probability 1 if it is used to derive the solution program, and 0 otherwise.
