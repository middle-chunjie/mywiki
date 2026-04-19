Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training
==========================================================================================================

Ran Xu1,* Tianci Liu2,* Zihan Dong3 Tony Yu4 Ilgee Hong4  
Carl Yang1 Linjun Zhang3 Tuo Zhao4 Haoyu Wang5  
1Emory University2Purdue University3Rutgers University4Georgia Institute of Technology5University at Albany

###### Abstract

Standard reward models typically predict scalar scores that fail to capture the multifaceted nature of response quality in non-verifiable domains, such as creative writing or open-ended instruction following. To address this limitation, we propose Rubric-ARM, a framework that jointly optimizes a rubric generator and a judge using reinforcement learning from preference feedback. Unlike existing methods that rely on static rubrics or disjoint training pipelines, our approach treats rubric generation as a latent action learned to maximize judgment accuracy. We introduce an alternating optimization strategy to mitigate the non-stationarity of simultaneous updates, providing theoretical analysis that demonstrates how this schedule reduces gradient variance during training. Extensive experiments show that Rubric-ARM achieves strong performance among baselines on multiple benchmarks and significantly improves downstream policy alignment in both offline and online reinforcement learning settings.

Keywords: Rubrics-as-Rewards, Reward Modeling, LLM Alignment, Synthetic Data

\= Date: February 2, 2026

\= Model Weights \& Checkpoints: <https://huggingface.co/collections/OpenRubrics/rubricarm>

\= Contact:[ran.xu@emory.edu](mailto:ran.xu@emory.edu ""); [liu3351@purdue.edu](mailto:liu3351@purdue.edu ""); [hwang28@albany.edu](mailto:hwang28@albany.edu "")

${}^{*}$${}^{*}$footnotetext: These authors contributed equally to this work, order was determined randomly (by rolling a die).

1 Introduction
--------------

Reward modeling serves as the compass for aligning large language models (LLMs) with human intents, typically by generating a scalar score or preference label to predict human preferences *(stiennon2020learning, wang2024secrets)*. However, in complex non-verifiable domain, such as creative writing or open-ended instruction following, these scalar or pairwise judgments often fail to capture the multifaceted nature of response quality *(ying2025beyond)*.
To address this limitation, recent advancements have shifted toward rubric-based reward modeling, where models explicitly generate structured criteria to ground their judgments *(gunjal2025rubrics, liu2025openrubrics, pathak2025rubric)*.
By decomposing evaluation into interpretable dimensions, rubric-based models offer transparency and improve generalization across prompt-specific evaluation axes.

Central to rubric-based evaluation is the availability of *high-quality rubrics*.
To ensure rubric quality, earlier work has primarily relied on human-authored rubrics, which are expensive to produce and difficult to scale to large datasets *(arora2025healthbench)*.
More recent approaches seek to automate rubric construction using LLMs *(viswanathan2025checklists, gunjal2025rubrics)*; however, these methods are largely prompting-based and rely on fixed, frozen models for both rubric generation and response quality judgment.
Consequently, they do not update the model’s intrinsic capabilities to the target domain or the underlying preference distribution, limiting their ability to generate in-domain, preference-aligned rubrics.
Moreover, even when learning-based components are introduced *(liu2025openrubrics, rezaei2025online)*, the rubric generator and the judge are treated as separate modules and trained independently rather than jointly optimized.
This decoupled training pipeline prevents deeper integration between rubric construction and judgment, leading to suboptimal evaluation signals.
Designing effective rubric-based reward models are still challenging.

In this work, we propose Rubric-ARM, an end-to-end framework that jointly optimizes the *rubric generator* and the *judge* via alternating reinforcement learning (RL), enabling the two components to co-evolve and mutually reinforce one another during training.
We formulate rubrics as *latent actions* that guide the reward model in recovering the underlying preference signal, and posit that improved rubric generation directly leads to more accurate preference predictions.
To ensure stable joint optimization, Rubric-ARM employs an alternating training strategy that decouples the learning dynamics while preserving a shared objective. Training alternates between (i) optimizing the reward model with a fixed rubric generator to align with target preference labels, and (ii) optimizing the rubric generator with a fixed reward model to produce discriminative rubrics that maximize prediction accuracy.

A key challenge of the alternating RL is the instability caused by simultaneous updates to both components. Our analysis reveals that early-stage exploration by the rubric generator can dominate the learning dynamics.
To mitigate this, we first stabilize the reward model under fixed rubrics before optimizing the rubric generator.
This alternating schedule reduces variance and ensures robust optimization.

Our contributions can be summarized as follows:

* •

    We develop Rubric-ARM, a rubric-based reward model to produce high-quality rubrics and precise judgments. To the best of our knowledge, this is the first approach that jointly optimizes rubric and judging via RL.

* •

    We introduce an *alternating RL* training algorithm that couples the rubric generator and judge through a shared correctness objective, enabling mutual improvement while stabilizing optimization.

* •

    We evaluate Rubric-ARM across diverse alignment settings (9 reward modeling and 6 policy benchmarks). Rubric-ARM outperforms strong reasoning-based judges and prior rubric-based reward models, achieving a $+4.7\%$ average gain on reward-modeling benchmarks, and consistently improves downstream policy post-training when used as the reward signal.

2 Related Works
---------------

LLM-based Reward and Judge Models. While *zheng2023judging* established the foundational utility of LLM-based judges.
Subsequent research expanded the scope of reasoning to include chain-of-thoughts *(zhang2025generative)*, self-critiques *(ankner2024critiqueoutloud, yu2025self, mahan2024generative)* or plan evaluations strategically *(evalplanner)*. *liu2025inference* explore inference-time reasoning for generative reward models. Recent studies *(chen2025judgelrm, chen2025rmr1, whitehouse2025j1, guo2025reward, hong2025thinkrm, xu2025incentivizing)* leverage online RL to directly incentivize detailed reasoning, aiming to mitigate bias and enhance the accuracy of pointwise and pairwise scoring.

Rubrics-based Reward Models. Recently, rubric-based approaches have emerged as a promising direction for LLM evaluation *(arora2025healthbench, hashemi2024llm, pathak2025rubric, akyurek2025prbench)*, alignment *(viswanathan2025checklists, zhang2025chasing)*, and reasoning *(gunjal2025rubrics, zhou2025breaking, huang2025reinforcement)*.
However, a unique challenge lies in generating *high-quality rubrics at scale*.
To address this, *li2026rubrichub, liu2025openrubrics, xie2025auto* extract rubrics from pairwise comparison signals, while *rezaei2025online, zhang2025chasing, shao2025dr* dynamically generate rubrics by leveraging policy model outputs in an online setting.

3 Preliminaries
---------------

We study rubric-based reward modeling in *non-verifiable* domains, where response quality cannot be directly validated against ground truth.
The rubric-based reward model contains two parts, namely *rubric generator* and *judge*. The key components of Rubric-ARM are described as follows.

#### Rubrics.

We define a rubric as a structured set of evaluation criteria conditioned on a prompt.
Formally, let $x$ denote a prompt, a rubric ${r}(x)\={c_{i}}_{i\=1}^{k}$ consists of $k$ criteria, where each $c_{i}$ specifies a distinct aspect of response quality (e.g., factual correctness, tone, or presentation).

For training rubric-based reward models in non-verifiable domains, a pairwise preference dataset is given as
$\mathcal{D}\={(x_{i},y^{(1)}_{i},y^{(2)}_{i},o_{i})}_{i\=1}^{N}$,
where $x$ is a prompt, $y^{(1)}$ and $y^{(2)}$ are two candidate responses, and $o\in{0,1}$ indicates which response is preferred
(e.g., $o\=1$ means $y^{(1)}\succ y^{(2)}$).
Formally, the rubric generator $\pi_{r}$ generates a rubric $r$ from the prompt as

|  | $r\sim\pi_{r}(\cdot\mid x;\theta_{r}),$ |  | (1) |
| --- | --- | --- | --- |

while a judge $\pi_{j}$ predicts a preference $o$ with the reasoning chain $c$ conditioned on the prompt, responses, and rubric as

|  | $(c,o)\sim\pi_{j}(\cdot\mid x,y^{(1)},y^{(2)},r;\theta_{j}).$ |  | (2) |
| --- | --- | --- | --- |

#### Learning Objective.

We define the preference-correctness reward

|  | $R(o,o)\;\=\;\mathbb{I}[o\=o],$ |  | (3) |
| --- | --- | --- | --- |

where $\mathbb{I}[o\=o]$ represents if the binary prediction extracted from $o$ aligns with ground truth $o$.

Denote $\theta_{r},\theta_{j}$ as the parameter for $\pi_{r}$ and $\pi_{j}$ respectively, our goal is to learn $(\theta_{r},\theta_{j})$ that maximize expected preference correctness under generated rubrics:

|  | $\max_{\theta_{r},\theta_{j}}\;\operatorname*{\mathbb{E}}_{(x,y^{(1)},y^{(2)},o)\sim\mathcal{D}}\operatorname*{\mathbb{E}}_{r\sim\pi_{r}(\cdot\mid x;\theta_{r})}\operatorname*{\mathbb{E}}_{(c,o)\sim\pi_{j}(\cdot\mid x,y^{(1)},y^{(2)},r;\theta_{j})}\big[R(o,o)\big].$ |  | (4) |
| --- | --- | --- | --- |

Since both $r$ (text) and $c,o$ (discrete decision with reasoning) are sampled actions, we optimize [eq.˜4] with RL.

<img src='x1.png' alt='Refer to caption' title='' width='581' height='287' />

*Figure 1: The overall framework for Rubric-ARM.*

4 Rubric-ARM: Alternating RL for Rubric Generation and Judging
---------------------------------------------------------------

In non-verifiable domains, supervision is limited to pairwise preference feedback and rubrics are not directly observed. Simultaneously updating the rubric generator $\pi_{r}$ and the judge $\pi_{j}$ leads to non-stationary learning targets and unstable optimization.
As shown in Figure [1], Rubric-ARM addresses this challenge using an alternating RL scheme that decouples the updates of two components.

### 4.1 Stage I: SFT Warmup

We equip both $\pi_{j}$ and $\pi_{r}$ with basic rubric generation and judging capabilities via leveraging open-source datasets.
Following the prior work *(liu2025openrubrics)*, we fine-tune on synthetic rubrics and judge trajectories derived from open-source datasets including *UltraFeedback* *(cui2024ultrafeedback)*, *SkyWork* *(liu2024skywork)*, *Magpie* *(xu2025magpie)*, and *Synthetic Instruction Following* *(lambert2025tulu)*.
Both $\pi_{r}(r\mid x;\theta_{r})$ and $\pi_{j}(c,o\mid x,y^{(1)},y^{(2)},r;\theta_{j})$ are trained with the standard next-token prediction objective.

### 4.2 Stage II: Alternating Reinforcement Learning

Stage I (SFT) warm-starts the rubric generator $\pi_{r}$ and judge $\pi_{j}$ by imitating synthetic rubric generation and judging trajectories, but optimizes the two components independently and does not directly target preference correctness.
We therefore optimize both components using *alternating reinforcement learning*.
Specifically, training switches between (i) *improving the judge with a fixed rubric generator* and (ii) *improving the rubric generator with a fixed judge*, providing each component with a clearer learning signal while preserving the same end objective $R(o,o)$.

#### (i) RL for Judge $\pi_{j}$ with the current $\pi_{r}$.

With the rubric generator parameters $\theta_{r}$ held fixed, we update $\theta_{j}$ to improve preference correctness under rubrics sampled from $\pi_{r}$:

|  | $\max_{\theta_{j}}\;J_{j}(\theta_{j};\theta_{r})\={}\operatorname*{\mathbb{E}}_{(x,y^{(1)},y^{(2)},o)\sim\mathcal{D}}\operatorname*{\mathbb{E}}_{r\sim\pi_{r}(\cdot\mid x;\theta_{r})}\operatorname*{\mathbb{E}}_{(c,o)\sim\pi_{j}(\cdot\mid x,y^{(1)},y^{(2)},r;\theta_{j})}\big[\mathbb{I}[o\=o]\big].$ |  | (5) |
| --- | --- | --- | --- |

This phase trains the judge to produce rubric-conditioned evaluations that recover the dataset preference.

Since $\pi_{r}(\cdot\mid x;\theta_{r})$ is fixed during judge updates, we cache rubrics to reduce sampling cost and stabilize optimization. For each training instance $(x_{i},y^{(1)}_{i},y^{(2)}_{i},o_{i})$, we sample a rubric $r_{i}\sim\pi_{r}(\cdot\mid x_{i};\theta_{r})$ once and reuse it for multiple judge optimization steps, yielding the Monte Carlo estimate:

|  | $J_{j}(\theta_{j};\theta_{r})\approx{}\operatorname*{\mathbb{E}}_{(x_{i},y^{(1)}_{i},y^{(2)}_{i},o_{i})\sim\mathcal{D},\,r_{i}}\operatorname*{\mathbb{E}}_{(c,o)\sim\pi_{j}(\cdot\mid x_{i},y^{(1)}_{i},y^{(2)}_{i},r_{i};\theta_{j})}\big[\mathbb{I}[o\=o_{i}]\big].$ |  | (6) |
| --- | --- | --- | --- |

In practice, we use a shaped reward that augments the final correctness signal $R_{\text{acc}}\=\mathbb{I}[o\=o_{i}]$ with *format-based* reward $R_{\text{fmt}}$ that enforces valid judging trajectories (i.e., addressing each rubric criterion with per-criterion explanations, followed by an overall justification and a final decision).
The final reward for the judge $\pi_{j}$ is $R_{j}\=R_{\text{acc}}+R_{\text{fmt}}$.

#### (ii) RL for Rubric Generator $\pi_{r}$ with the current $\pi_{j}$.

With the judge parameters $\theta_{j}$ fixed, we update $\theta_{r}$ to prefer rubrics that lead the current judge to make correct decisions. Concretely, we maximize the preference correctness under rubrics drawn from $\pi_{r}$ as:

|  | $\max_{\theta_{r}}\;J_{r}(\theta_{r};\theta_{j})\={}\operatorname*{\mathbb{E}}_{(x,y^{(1)},y^{(2)},o)\sim\mathcal{D}}\operatorname*{\mathbb{E}}_{r\sim\pi_{r}(\cdot\mid x;\theta_{r})}\operatorname*{\mathbb{E}}_{(c,o)\sim\pi_{j}(\cdot\mid x,y^{(1)},y^{(2)},r;\theta_{j})}\big[\mathbb{I}[o\=o]\big].$ |  | (7) |
| --- | --- | --- | --- |

Intuitively, $\pi_{r}$ learns to generate criteria that are discriminative for the given prompt and usable by the judge to recover the dataset preference.
In practice, we approximate the expectation with a single rollout by greedy decoding ($t\=0$), i.e., we generate one judging trajectory $(c,o)$ per rubric and use the Monte Carlo estimate

|  | ${R}_{r}\=\mathbb{I}[o\=o].$ |  | (8) |
| --- | --- | --- | --- |

Optimization (alternating RL).Rubric-ARM alternates between optimizing Eq. [5] RL for Judge 𝜋_𝑗 with the current 𝜋_𝑟. ‣ 4.2 Stage II: Alternating Reinforcement Learning ‣ 4 Rubric-ARM: Alternating RL for Rubric Generation and Judging ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training") and [7] RL for Rubric Generator 𝜋_𝑟 with the current 𝜋_𝑗. ‣ 4.2 Stage II: Alternating Reinforcement Learning ‣ 4 Rubric-ARM: Alternating RL for Rubric Generation and Judging ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training"). At iteration $t$, we run:

|  | $\displaystyle r_{i}^{t}$ | $\displaystyle\sim\pi_{r}(\cdot\mid x_{i};\theta_{r}^{t})\quad\forall(x_{i},y_{i}^{(1)},y_{i}^{(2)},o_{i})\in\mathcal{D},$ |  | (9) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\theta_{j}^{t+1}$ | $\displaystyle\leftarrow\mathrm{GRPO}!\left(\theta_{j}^{t}\,;\,{r_{i}^{t}},\mathcal{D}\right),$ |  | (10) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\theta_{r}^{t+1}$ | $\displaystyle\leftarrow\mathrm{GRPO}!\left(\theta_{r}^{t}\,;\,\theta_{j}^{t+1},\mathcal{D}\right).$ |  | (11) |
| --- | --- | --- | --- | --- |

Here we cache one rubric per instance during judge updates (since $\pi_{r}$ is fixed in that phase). In each phase, GRPO (*shao2024deepseekmath*, details in Appendix [A] ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training")) updates only the active policy while keeping the other fixed. Notably, we alternate training by updating the judge before the rubric generator in each cycle. In Sec. [5], we provide theoretical analysis proving the benefits of this ordering.

#### Connection to EM Algorithm.

Our alternating optimization can be viewed as a generalized EM procedure *(dempster1977maximum)* with rubrics $r$ as latent variables. For each preference instance $(x,y^{(1)},y^{(2)},o)$, the judge defines a conditional model $p_{\theta_{j}}(o\mid x,y^{(1)},y^{(2)},r)$, while the rubric generator $\pi_{r}(r\mid x;\theta_{r})$ acts as an amortized variational distribution over the latent rubric *(agrawal2021amortized)*.
With $\pi_{r}$ fixed, updating $\pi_{j}$ maximizes the expected correctness (or log-likelihood) under sampled rubrics, analogous to the M-step. With $\pi_{j}$ fixed, updating $\pi_{r}$ increases probability mass on rubrics that make the current judge more likely to recover $o$, analogous to an amortized E-step. Because rubrics are high-dimensional discrete text sequences, we use stochastic policy-gradient updates rather than exact posterior inference, yielding a stochastic EM-style coordinate ascent scheme.

### 4.3 Policy Model Post-training with Rubric-ARM

We use the trained rubric generator $\pi_{r}(\cdot\mid q;\theta_{r})$ and judge $\pi_{j}(\cdot\mid q,\cdot,\cdot,r;\theta_{j})$ to provide preference supervision for post-training a policy model $\pi_{\phi}(a\mid q)$, where $q$ denotes the prompt and $a$ denotes a candidate response. For any pair of responses $(a,b)$, Rubric-ARM samples a rubric $r\sim\pi_{r}(\cdot\mid q;\theta_{r})$ and predicts a preference label

|  | $\widehat{o}\;\=\;\mathrm{Judge}_{\theta_{j}}(q,a,b,r)\in{0,1},$ |  | (12) |
| --- | --- | --- | --- |

where $\widehat{o}\=0$ indicates $a\succ b$ and $\widehat{o}\=1$ indicates $b\succ a$.

Preference Optimization with Rubric-ARM. Given a prompt $q$, we sample two rollouts from the current policy,

|  | $a_{1},a_{2}\sim\pi_{\phi}(\cdot\mid q),$ |  | (13) |
| --- | --- | --- | --- |

and use Rubric-ARM to label which one is preferred via Eq. ([12]) and retain examples where the predictions are consistent for both orders.
We then update $\pi_{\phi}$ with the standard DPO objective *(rafailov2023direct)* relative to a fixed reference policy $\pi_{\mathrm{ref}}$.
For iterative DPO *(xiong2024iterative, pang2024iterative)*,
we repeat (i) sampling rollouts, (ii) labeling them with Rubric-ARM, and (iii) applying DPO updates for multiple rounds.

Online RL with Rubric-ARM. Following recent works on using pairwise judges to provide reward signals *(xu2025unified)*, we also consider online RL where Rubric-ARM provides rewards for optimizing $\pi_{\phi}$. For each prompt $q$, we adopt the ReMax-style baseline construction *(li2024remax)* by first generating a deterministic reference response via greedy decoding,

|  | $a^{(0)}\;\=\;\mathrm{Greedy}(\pi_{\phi}(\cdot\mid q))\quad(t\=0),$ |  | (14) |
| --- | --- | --- | --- |

and then sample $K$ additional rollouts,

|  | ${a^{(k)}}_{k\=1}^{K}\sim\pi_{\phi}(\cdot\mid q).$ |  | (15) |
| --- | --- | --- | --- |

To mitigate positional bias, we query the judge in both orders under the same rubric $r$.
Let $\widehat{o}^{(k)}\in{0,1}$ denote the judge outcome for $(q,a^{(k)},a^{(0)},r)$ and
$\widehat{o}^{(k)}\in{0,1}$ for the swapped order $(q,a^{(0)},a^{(k)},r)$.

We define the final reward for response $a^{(k)}$ as

|  | $R_{\phi}(q,a^{(k)})\=\frac{1}{2}\left(\mathbb{I}(\widehat{o}^{(k)}\=0)+\mathbb{I}(\widehat{o}^{(k)}\=1)\right).$ |  | (16) |
| --- | --- | --- | --- |

5 Theoretical Analysis
----------------------

We analyze the gradient variance to justify our training schedule. We compare two phases: Strategy A (Judge Warmup), where we optimize the judge with pre-generated, reused rubrics; and Strategy B (Rubric Generator Training), where we optimize the rubric generator against a fixed judge.

Setup. Let $u_{r}(r):\=\frac{\partial}{\partial\theta_{r}}\log\pi_{r}(r\,|\,x)$ and $u_{j}(o\,|\,r):\=\frac{\partial}{\partial\theta_{j}}\log\pi_{j}(o\,|\,c,r)$ be the score functions.
Let $p(r):\=\mathbb{P}(o\=o^{*}\,|\,c,r)$ be the judge’s correctness probability given a rubric.
We define the gradient variance as $\mathrm{Var}(\widehat{g}):\=\operatorname*{\mathbb{E}}\norm{\widehat{g}}^{2}-\norm{\operatorname*{\mathbb{E}}[\widehat{g}]}^{2}$.

### 5.1 Variance Decomposition

We first examine Strategy A. By freezing the rubric $\bar{r}$ (reuse) during judge updates, we eliminate inter-rubric variance.

###### Proposition 5.1(Judge Variance under Strategy A).

Conditioned on a reused rubric $\bar{r}$, the variance of the judge’s gradient estimator $\widehat{g}_{A}$ is solely determined by the judge’s binary classification uncertainty:

|  | $\mathrm{Var}(\widehat{g}_{A}\,|\,\bar{r})\=p(\bar{r})\quantity(1-p(\bar{r}))\norm{u_{j}(o^{*}\,|\,\bar{r})}^{2}.$ |  | (17) |
| --- | --- | --- | --- |

###### Proposition 5.2(Generator Variance under Strategy B).

The total variance of the generator’s gradient estimator $\widehat{g}_{B}$ decomposes into:

|  | $\mathrm{Var}(\widehat{g}_{B})\=\mathop{\mathchoice{\underarrow@\displaystyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]}}{\underarrow@\textstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]}}{\underarrow@\scriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]}}{\underarrow@\scriptscriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]}}}\limits_{\text{(I) Multiplicative Reward Noise}}+\mathop{\mathchoice{\underarrow@\displaystyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}{\underarrow@\textstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}{\underarrow@\scriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}{\underarrow@\scriptscriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru{\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}}\limits_{\text{(II) Cross-Rubric Inconsistency}}$ |  | (18) |
| --- | --- | --- | --- |

Interpretation. Term (I) represents the judge’s Aleatoric uncertainty amplified by the high-dimensional generator gradient $\norm{u_{r}}^{2}$.
Term (II) captures the optimization difficulty when different rubrics yield different expected rewards $p(r)$, causing the gradient direction to oscillate.

### 5.2 Variance Domination in Early Training

We now derive the variance gap. Instead of assuming trivial gradient dominance, we postulate a condition linking the generator’s exploration intensity to its gradient magnitude.

###### Assumption 5.3(Exploration-Gradient Sufficiency).

We assume that during early training, the generator’s gradient norm is sufficient relative to the judge’s, satisfying the following exploration-dependent lower bound:

|  | $\frac{\norm{u_{r}}}{\norm{u_{j}}}>\sqrt{\frac{1-p(r)}{1-p(r)+C_{1}p(r)}},$ |  | (19) |
| --- | --- | --- | --- |

where $p$ represents the judge’s correctness probability (analyzed pointwise or in expectation), and $C_{1}\in(0,1)$ is defined as: $C_{1}:\=\mathrm{Var}_{r}(p(r)u_{r}(r))/\operatorname*{\mathbb{E}}_{r}[p(r)^{2}\norm{u_{r}(r)}^{2}]$.

###### Remark 5.4.

The condition in Assumption [5.3]. ‣ 5.2 Variance Domination in Early Training ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training") is mild and physically justified.
Active exploration ($C_{1}>0$) introduces a positive buffer, making the required gradient-norm ratio on the RHS strictly less than 1 and thus avoiding the need for the generator’s gradient to strictly dominate.
Moreover, the judge and generator both produce comparable-length sequences over the same vocabulary (checks/prediction vs. rubrics), so their gradient norms are typically of the same order; the exploration buffer is enough to absorb small mismatches and satisfy the condition in practice.

###### Theorem 5.5(Strict Variance Domination).

Under Assumption [5.3]. ‣ 5.2 Variance Domination in Early Training ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training"), the gradient variance of Strategy B strictly dominates the expected conditional variance of Strategy A:

|  | $\mathrm{Var}(\widehat{g}_{B})>\operatorname*{\mathbb{E}}_{\bar{r}}[\mathrm{Var}(\widehat{g}_{A}\,|\,\bar{r})].$ |  | (20) |
| --- | --- | --- | --- |

This inequality establishes that the structural instability driven by exploration (quantified by $C_{1}$) is the governing factor in the variance landscape, overriding differences in gradient magnitudes.

###### Remark 5.6(Implication for Training Stability).

The variance gap derived in Theorem [5.5]. ‣ 5.2 Variance Domination in Early Training ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training") justifies the proposed training schedule (We first train the judge, then train the rubric generator, and subsequently perform alternating training following this sequence.) by highlighting a critical trade-off in Signal-to-Noise Ratio (SNR). The strictly higher variance in Strategy B implies that generator updates are dominated by exploration stochasticity rather than the true gradient direction, risking optimization instability. In contrast, Strategy A acts as a variance reduction mechanism: by fixing the rubric, it effectively sets the exploration coefficient $C_{1}\to 0$ locally, isolating the judge from structural noise and providing a stable target for effective learning.

6 Experiment
------------

### 6.1 Datasets and Experiment Settings

Training data. We train the two components of Rubric-ARM, the *rubric generator* and the *judge*, on the general-domain portions of OpenRubrics *(liu2025openrubrics)*.
The dataset is split equally into non-overlapping parts, and each rubric-judge alternating round is run on a single part.
During training judge, we randomly shuffle the order of response candidates to be evaluated; as shown in App. [D.2], this practice greatly helps reduce position bias in reward modeling.

Backbone and variants. Both the rubric generator and the judge are fine-tuned from Qwen-3-8B *(qwen3technicalreport)*.
At inference time, Rubric-ARM follows the two-stage rubric-judging process, as detailed in Sec. [3].
We also report ensemble results voting@5, by aggregating five independent judges via majority voting.

Baselines. For reward-model evaluation, we follow *liu2025openrubrics* and compare Rubric-ARM against strong same-scale white-box judges, including JudgeLRM *(chen2025judgelrm)*, RRM *(guo2025reward)*, RM-R1 *(chen2025rmr1)*, and Rubric-RM *(liu2025openrubrics)* (SFT-only rubric generator + judge). We also report judges using black-box APIs when available. To isolate the benefit of rubric-aware training, we include a training-free baseline, Qwen-3-8B (Rubric+Judge) *(yang2025qwen3)*, which directly generates rubrics and judgments via prompting.
For policy training, we use Rubric-ARM as the reward model to fine-tune Qwen2.5-7B-Instruct *(qwen2025qwen25technicalreport)* and compare against Skywork *(liu2024skywork)*, ArmoRM *(wang2024interpretable)*, UltraFeedback *(cui2024ultrafeedback)*, RLCF/AI Judge *(viswanathan2025checklists)*, OnlineRubrics *(rezaei2025online)*, and Rubric-RM *(liu2025openrubrics)*.

Evaluation benchmarks and metrics. We evaluate Rubric-ARM as a pairwise reward model on widely used alignment benchmarks: RewardBench (Chat/Chat-Hard) *(rewardbench)*, RM-Bench *(liu2025rmbench)*, PPE-IFEval *(ppe)*, FollowBench *(followbench)*, InfoBench *(infobench)*, IFBench *(ifbench)*, RewardBench2 (Precise-IF/Focus) *(malik2025rewardbench2)*, Arena-Hard *(chiang2024chatbotarena)*, AlpacaEval 2 *(dubois2025length)*, Creative Writing Benchmark v3 *(creativewritingv3)*, WildBench *(lin2024wildbench)*, and WritingPreferenceBench *(ying2025beyond)*.
For FollowBench and InfoBench, we convert the original single-response setup to pairwise evaluation by sampling two responses from the same model (Qwen-3-8B/14B) and using the benchmark’s verifier to identify constraint violations.
We follow each benchmark’s official splits and scoring rules, reporting accuracy, win-rate, or the benchmark-specific metric.

*Table 1: Comparison of different judge and reward models across multiple benchmarks.
RewardBench2 reports results on Precise IF, and Focus dimensions.
Rubric API uses GPT-4.1-Mini, and Judge API uses Gemini-2.5-Flash-Lite.
Best results are highlighted in bold.*

|  | RewardBench | | IF Evaluation Benchmarks | | | | RM-Bench | RewardBench2 | | HelpSteer3 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Chat | Chat Hard | FollowBench | PPE-IFEval | InfoBench | IFBench | Chat | Precise IF | Focus | | |
| Black-box LLMs (For reference only) | | | | | | | | | | |  |
| Claude-3.5-Sonnet | 96.4 | 74.0 | – | 58.0 | – | – | 62.5 | 38.8 | 87.0 | – | - |
| Gemini-2.5-Flash | 95.0 | 83.3 | 86.0 | 75.0 | 85.6 | 69.3 | 78.5 | 57.5 | 84.1 | 70.6 | 78.5 |
| API (Rubric+Judge) | 79.6 | 79.2 | 83.2 | 61.0 | 82.2 | 66.2 | 67.9 | 42.5 | 79.6 | 71.4 | 71.3 |
| API (direct Judge) | 89.6 | 71.2 | 81.7 | 59.2 | 72.9 | 60.4 | 67.2 | 13.2 | 63.4 | 70.3 | 64.9 |
| Larger White-box LLMs (For reference only) | | | | | | | | | | |  |
| RM-R1-14B (Qwen-2.5-Inst) | 73.5 | 79.8 | 84.0 | 59.0 | 85.5 | 60.8 | 73.2 | 23.8 | 84.6 | 74.8 | 69.9 |
| RM-R1-14B (DeepSeek-Dist) | 90.3 | 78.9 | 89.9 | 61.2 | 82.4 | 59.0 | 71.4 | 30.6 | 79.0 | 74.6 | 71.7 |
| RM-R1-32B (Qwen-2.5-Inst) | 95.3 | 80.3 | 84.9 | 60.4 | 86.1 | 60.4 | 75.3 | 33.1 | 84.2 | 72.9 | 73.3 |
| RM-R1-32B (DeepSeek-Dist) | 95.3 | 83.1 | 89.2 | 63.2 | 85.0 | 58.6 | 74.2 | 36.9 | 79.2 | 75.6 | 74.0 |
| RRM-32B | 94.7 | 81.1 | 85.7 | 60.2 | 84.4 | 60.8 | 73.9 | 34.4 | 83.6 | 75.4 | 73.4 |
| White-box Judge/Reward LLMs | | | | | | | | | | |  |
| RM-R1-7B (Qwen-2.5-Inst) | 83.0 | 70.0 | 56.3 | 55.2 | 71.3 | 55.2 | 64.2 | 20.6 | 76.2 | 65.2 | 61.7 |
| RM-R1-7B (DeepSeek-Dist) | 85.3 | 67.3 | 69.7 | 51.0 | 70.3 | 56.5 | 62.2 | 13.8 | 55.4 | 62.6 | 59.4 |
| RRM-7B | 77.7 | 69.5 | 65.5 | 51.0 | 68.2 | 53.2 | 59.9 | 10.0 | 60.4 | 62.4 | 57.8 |
| JudgeLRM-7B | 92.1 | 56.1 | 79.8 | 46.0 | 62.7 | 47.5 | 55.4 | 9.4 | 29.1 | 60.2 | 53.8 |
| Rubric-based Methods | | | | | | | | | | |  |
| Qwen-3-8B (Rubric+Judge) | 73.9 | 63.6 | 63.0 | 53.8 | 74.6 | 55.6 | 64.2 | 21.9 | 56.6 | 61.8 | 58.9 |
| Rubric-RM | 88.2 | 74.1 | 76.1 | 67.0 | 80.8 | 65.4 | 65.7 | 34.4 | 82.2 | 67.0 | 70.1 |
| Rubric-RM-voting@5 | 89.9 | 75.4 | 81.5 | 70.8 | 83.8 | 67.1 | 67.0 | 40.0 | 86.5 | 67.5 | 73.0 |
| Rubric-ARM | 89.4 | 79.6 | 85.7 | 70.8 | 86.1 | 65.9 | 69.2 | 41.9 | 89.4 | 69.8 | 74.8 |
| Rubric-ARM-voting@5 | 90.3 | 80.7 | 87.4 | 72.0 | 87.7 | 67.1 | 69.1 | 46.2 | 90.3 | 71.1 | 76.2 |

### 6.2 Performance of Rubric-ARM

Table [1] compares Rubric-ARM against a broad set of judge/reward models. Rubric-ARM achieves the best average performance among all white-box methods, improving Rubric-RM from $70.1$ to $74.8$, and reaching $76.2$ with voting@5.
These gains are consistent across both instruction-following and preference-style benchmarks, supporting our key contribution: Rubric-ARM learns *more discriminative rubrics* and a *more reliable rubric-conditioned judge* through RL.
Notably, Rubric-ARM also substantially outperforms API-based judges (e.g., 76.2 vs. 71.3 for Rubric+Judge API and 64.9 for direct Judge API), indicating that explicit rubric-conditioned learning yields a stronger and more stable evaluation signal than black-box judging.

<img src='x2.png' alt='Refer to caption' title='' width='706' height='245' />

*Figure 2: Performance of different judge and reward models on WritingPreferenceBench.*

We further assess generalization on WritingPreferenceBench *(ying2025beyond)*, shown in Fig. [2] (detail results are shown in Table [12]), which serves as an out-of-distribution benchmark since none of the compared reward/judge models are trained on this domain. Despite this distribution shift, Rubric-ARM remains strong and achieves the best overall score among all methods ($63.2$), outperforming Rubric-RM ($60.3$) and strong reasoning reward models such as RM-R1-Qwen2.5-7B ($59.8$). The improvements are broad across diverse writing genres (e.g., Functional, Promotional, Non-Fiction, and Poetry), suggesting that Rubric-ARM learns rubrics that capture transferable criteria beyond the training domains, thereby providing a robust reward signal with improved OOD generalization.

*Table 2: Ablation study about the effectiveness of the format reward and the order of judge optimization and rubric generator.
Best results are highlighted in bold.*

|  | RewardBench | | IF Evaluation Benchmarks | | | | RM-Bench | RewardBench2 | | HelpSteer3 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Chat | Chat Hard | FollowBench | PPE-IFEval | InfoBench | IFBench | Chat | Precise IF | Focus | | |
| Rubric-ARM switch opt | 93.2 | 76.3 | 85.9 | 67.3 | 84.1 | 64.6 | 69.5 | 24.4 | 86.1 | 71.8 | 72.4 |
| Rubric-ARM switch opt-voting@5 | 94.0 | 76.5 | 89.1 | 67.8 | 85.0 | 64.6 | 69.8 | 39.4 | 90.1 | 72.4 | 74.9 |
| Rubric-ARM w/o format | 89.8 | 78.7 | 87.1 | 69.2 | 86.1 | 64.3 | 69.5 | 25.6 | 84.8 | 70.8 | 72.6 |
| Rubric-ARM w/o format-voting@5 | 91.5 | 78.5 | 88.2 | 70.2 | 87.7 | 65.1 | 69.7 | 43.8 | 88.9 | 71.1 | 75.5 |
| Rubric-ARM | 89.4 | 79.6 | 85.7 | 70.8 | 86.1 | 65.9 | 69.2 | 41.9 | 89.4 | 69.8 | 74.8 |
| Rubric-ARM-voting@5 | 90.3 | 80.7 | 87.4 | 72.0 | 87.7 | 67.1 | 69.1 | 46.2 | 90.3 | 71.1 | 76.2 |

### 6.3 Ablation Study

Table [2] reports two ablation studies that examine (i) the optimization order between the judge and the rubric generator, and (ii) the contribution of the format reward. Unless stated otherwise, all settings are kept identical to Rubric-ARM.

*Table 3: Comparison of trained policy models with different reward models on a format-based constrained instruction-following benchmark (IFEval) and an open-ended benchmark (InfoBench).
Baseline results with "" are from *viswanathan2025checklists, liu2025openrubrics*.
Results with underlines are reproduced by us using official checkpoints and evaluation scripts.
Best scores are in bold.*

| Model | IFEval (Prompt) | | IFEval (Inst.) | | IFEval | InfoBench |
| --- | --- | --- | --- | --- | --- | --- |
| | Loose | Strict | Loose | Strict | AVG | AVG |
| GPT-4 (0314) | 79.3 | 76.9 | 85.4 | 83.6 | 81.3 | 87.3 |
| AutoIF (dong2025selfplay) | 56.9 | 47.1 | 67.0 | 57.6 | 57.2 | 80.6 |
| UltraIF (an2025ultraif) | 75.4 | 71.3 | 83.0 | 79.4 | 77.3 | 80.7 |
| RAIF (qin2025incentivizing) | – | – | – | – | 70.1 | 82.7 |
| Qwen2.5-7B-Instruct | 75.0 | 72.5 | 81.8 | 79.9 | 77.3 | 78.1 (76.0) |
| + SFT (Distilled) | 66.8 | 64.1 | 75.3 | 72.8 | 69.8 | 72.5 |
| + DPO (via Skywork) | 75.7 | 68.0 | 83.2 | 78.5 | 76.0 | 82.0 |
| + DPO (via ArmoRM) | 73.8 | 70.2 | 81.7 | 78.3 | 76.0 | 83.5 |
| + DPO (via Ultrafbk.) | 71.5 | 69.1 | 79.9 | 77.7 | 74.6 | 80.0 |
| + DPO (via AI Judge) | 73.0 | 68.9 | 80.9 | 77.8 | 75.2 | 76.1 |
| + DPO (via RLCF) | 77.3 | 72.6 | 84.1 | 80.3 | 78.6 | 84.1 (81.5) |
| + IterDPO (via RLCF) | 78.2 | 74.3 | 84.5 | 81.1 | 79.5 | 81.8 |
| + DPO (via Rubric-RM) | 78.2 | 73.9 | 84.5 | 81.2 | 79.5 | 83.0 |
| + IterDPO (via Rubric-RM) | 77.6 | 74.1 | 84.3 | 81.7 | 79.4 | 83.3 |
| + DPO (via Rubric-ARM) | 78.7 | 76.0 | 84.7 | 82.5 | 80.4 | 83.7 |
| + IterDPO (via Rubric-ARM) | 79.3 | 75.1 | 86.0 | 82.9 | 80.8 | 85.0 |

Optimization order. Our default schedule updates the judge first, then the rubric generator, and alternates thereafter. Swapping this order (switch opt) consistently hurts performance: the average drops from $74.8!\rightarrow!72.4$ ($-2.4$) without voting and from $76.2!\rightarrow!74.9$ ($-1.3$) with voting@5, with especially large regressions on strict instruction-following metrics (e.g., RewardBench2-Precise IF: $41.9!\rightarrow!24.4$). This suggests that a stronger judge provides a less noisy learning signal for rubric optimization.

Format reward. Removing the format reward (w/o format) also degrades results: $74.8!\rightarrow!72.6$ ($-2.2$) without voting and $76.2!\rightarrow!75.5$ ($-0.7$) with voting@5. The largest gains appear on structure-sensitive metrics (e.g., RewardBench2-Precise IF: $+16.3$), indicating that $R_{\text{fmt}}$ helps prevent degenerate judging behaviors (e.g., missing criteria checks) and improves rubric adherence.

### 6.4 Performance of offline RL-based Policy Models

We evaluate whether the benefit of Rubric-ARM transfers to downstream *offline* policy learning.

Instruction-Following Evaluation. Table [3] and Fig. [3] show that policies optimized with Rubric-ARM-trained rewards consistently achieve the strongest instruction-following performance. On IFEval, DPO with Rubric-ARM improves the overall average to $80.4$, and iterative DPO further raises it to $80.8$ (best), with particularly strong gains on instruction-level constraints. The advantage also transfers to the open-ended InfoBench benchmark, where Rubric-ARM reaches $83.7$ with DPO and $85.0$ with iterative DPO (best). Compared to iterative baselines, Rubric-ARM remains consistently stronger: on IFBench (Fig. [3]), RLCF improves from $28.2$ to $32.0$ with IterDPO, while Rubric-ARM achieves $35.4$ with IterDPO; similarly, iterative Rubric-RM reaches $33.7$, still below Rubric-ARM. Overall, these results indicate that Rubric-ARM provides a more precise reward signal, and that iterative optimization amplifies the gains over both one-shot DPO and iterative baselines.

*Table 4: Comparison of different strategies applied to Qwen2.5-7B-Instruct on Arena-Hard and AlpacaEval.
Results are reported for vanilla models and style/length-controlled settings.
Baseline results with "" are from *viswanathan2025checklists, rezaei2025online, liu2025openrubrics*.
Best results are in bold.*

| Model | Arena-Hard | | AlpacaEval | | AVG |
| --- | --- | --- | --- | --- | --- |
| | Vanilla | Style-Con | Vanilla | Length-Con | |
| GPT-4 (0314) | 50.0 | 50.0 | 22.1 | 35.3 | 39.4 |
| UltraIF (an2025ultraif) | 31.4 | – | – | – | – |
| Qwen2.5-7B-Instruct | 51.3 | 42.8 | 33.5 | 36.2 | 41.0 |
| + SFT (Distilled) | 32.6 | 29.2 | 36.1 | 33.3 | 32.8 |
| + DPO (via Skywork) | 55.1 | 50.3 | 44.8 | 41.5 | 47.9 |
| + DPO (via ArmoRM) | 50.8 | 46.4 | 37.6 | 38.1 | 43.2 |
| + DPO (via Ultrafbk.) | 52.8 | 47.9 | 33.7 | 38.7 | 43.3 |
| + DPO (via AI Judge) | 51.0 | 44.4 | 28.8 | 33.4 | 39.4 |
| + DPO (via RLCF) | 54.6 | 48.4 | 36.2 | 37.1 | 44.1 |
| + IterDPO (via RLCF) | 51.1 | 54.6 | 38.9 | 39.2 | 46.0 |
| + DPO (via Rubric-RM) | 52.9 | 53.1 | 47.0 | 41.3 | 48.6 |
| + IterDPO (via Rubric-RM) | 56.3 | 56.7 | 50.1 | 42.0 | 51.3 |
| + RL (via OnlineRubrics) | 56.5 | – | 55.0 | 30.4 | – |
| + DPO (via Rubric-ARM) | 57.8 | 59.5 | 47.1 | 42.5 | 51.7 |
| + IterDPO (via Rubric-ARM) | 58.8 | 58.9 | 52.0 | 44.0 | 53.4 |

*Table 5: Comparison of different alignment strategies applied to Qwen2.5-7B-Instruct on WildBench.
Results are reported for task-specific scores and task macro WB score.
Baseline results with "" are from *wang2025drift*.
Best results are in bold.*

| Method | Creative | Planning | Math | Info seeking | Coding | WB Score |
| --- | --- | --- | --- | --- | --- | --- |
| Claude-3.5-Sonnet (20240620) | 55.6 | 55.6 | 50.2 | 55.5 | 56.5 | 54.7 |
| GPT-4-turbo (20240409) | 58.7 | 56.2 | 51.0 | 57.2 | 55.1 | 55.2 |
| GPT-4o-mini (20240718) | 60.1 | 58.2 | 54.0 | 57.4 | 57.2 | 57.1 |
| Qwen2.5-7B-Instruct | 50.1 | 51.8 | 47.1 | 50.7 | 45.0 | 48.7 |
| +DRIFT | 52.5 | 53.2 | 50.6 | 52.4 | 50.3 | 51.7 |
| +SPIN | 43.3 | 45.5 | 41.6 | 46.3 | 39.1 | 42.9 |
| +IterDPO (via OpenAssistant) | 46.8 | 48.6 | 44.5 | 48.0 | 44.3 | 46.3 |
| +DPO (via RLCF) | 51.4 | 52.7 | 49.0 | 51.3 | 48.8 | 50.5 |
| +IterDPO (via RLCF) | 51.9 | 52.6 | 47.8 | 51.4 | 46.5 | 49.7 |
| +DPO (via Rubric-RM) | 54.8 | 55.5 | 51.5 | 54.1 | 52.9 | 53.6 |
| +IterDPO (viaRubric-RM) | 57.0 | 56.2 | 50.6 | 54.9 | 52.8 | 54.0 |
| +DPO (via Rubric-ARM) | 55.2 | 55.6 | 49.5 | 56.0 | 53.1 | 53.7 |
| +IterDPO (via Rubric-ARM) | 57.3 | 57.2 | 53.3 | 56.2 | 55.2 | 55.7 |

Human Preference Alignment Evaluation. Table [4] and Table [5] show that Rubric-ARM-trained rewards consistently yield stronger preference alignment across both controlled and open-domain evaluations. On Arena-Hard and AlpacaEval (Table [4]), DPO with Rubric-ARM achieves the best overall average ($51.7$), and IterDPO further improves it to $53.4$ (best). On WildBench (Table [5]), Rubric-ARM again yields the strongest macro score: DPO via Rubric-ARM reaches $53.7$, while IterDPO via Rubric-ARM achieves $55.7$ (best), improving over IterDPO with Rubric-RM ($54.0$) by 1.7%, indicating improved preference-aligned helpfulness on broad, real-world tasks.

<img src='x3.png' alt='Refer to caption' title='' width='747' height='184' />

*Figure 3: Comparison of trained policy models on IFBench. Results of baselines except Rubric-RM (IterDPO) are from OpenRubrics *liu2025openrubrics*.*

<img src='x4.png' alt='Refer to caption' title='' width='747' height='193' />

*Figure 4: Comparison of trained policy models on Create Writing Benchmark v3. Results of baselines except Rubric-RM are from RuscaRL *(zhou2025breaking)*.*

Creative Writing. We further evaluate whether Rubric-ARM-based rewards benefit open-ended generation on the Creative Writing Benchmark v3 (Fig. [4]). Policies trained with Rubric-ARM outperform baselines: DPO using Rubric-ARM achieves $39.0$, and IterDPO further improves to $39.3$ (best). Notably, Rubric-ARM-based optimization also surpasses strong creative-writing baselines such as RaR ($38.8$) and RuscaRL ($38.6$), suggesting that rewards learned by Rubric-ARM generalize well to subjective, non-verifiable generation tasks beyond standard instruction following and preference alignment.

### 6.5 Performance of online RL-based Policy Models

We evaluate Rubric-ARM in an *online* RL setting by directly optimizing Qwen2.5-7B-Instruct with GRPO (Sec. [4.3]) using different reward models. As shown in Table [6], GRPO with Rubric-ARM-trained rewards substantially improves both instruction following and preference alignment compared to the base model and a strong reward baseline RM-R1. Specifically, Qwen2.5-7B-Instruct achieves an average score of $46.8$, while GRPO with RM-R1 increases it to $52.3$. Replacing the reward with Rubric-ARM yields the best overall performance, reaching $55.4$ on average. The gains are consistent across instruction-following and human-preference alignment metrics, which indicates that Rubric-ARM provides a more effective online learning signal for GRPO.

*Table 6: Comparison of online RL method with different alignment strategies applied to Qwen2.5-7B-Instruct on instruction following and preference alignment benchmarks. Best results are in bold.*

| Method | IFEval (Prompt) | | IFEval (Inst.) | | IFBench | AlpacaEval | | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Loose | Strict | Loose | Strict | | Vanilla | Length | |
| Qwen2.5-7B-Instruct | 75.0 | 72.5 | 81.8 | 79.9 | 28.2 | 33.5 | 36.2 | 46.8 |
| +GRPO (RM-R1) | 76.7 | 73.6 | 83.2 | 80.2 | 30.6 | 53.2 | 42.7 | 52.3 |
| +GRPO (Rubric-ARM) | 79.3 | 76.2 | 85.3 | 83.0 | 34.8 | 56.2 | 44.8 | 55.4 |

### 6.6 Effect of Iterative Policy Optimization

Fig. [5] evaluates iterative DPO with Rubric-ARM over three optimization iterations. Overall, the average performance increases monotonically across iterations, indicating that iteratively refining the policy with Rubric-ARM-based supervision yields progressively better alignment. These results suggest that Rubric-ARM provides a sufficiently stable signal to support multi-round offline optimization without performance degradation.

*Table 7: Computing speed on 100 samples (vLLM).
Results with “” were taken from *liu2025openrubrics*.*

|  | Compute Time (s) |
| --- | --- |
| JudgeLRM-7B | 25.71 |
| RRM-7B | 203.40 |
| RM-R1-7B (Qwen-2.5-Inst) | 260.37 |
| RM-R1-7B (DeepSeek-Dist) | 170.76 |
| RM-R1-14B (Qwen-2.5-Inst) | 322.79 |
| RM-R1-14B (DeepSeek-Dist) | 382.02 |
| Rubric-RM-8B | 105.12 |
| Rubric-ARM-8B | 33.50 |

<img src='x5.png' alt='Refer to caption' title='' width='705' height='809' />

*Figure 5: Performance of iterative DPO with Rubric-ARM across three iterations.*

### 6.7 Efficiency Comparison

We conclude with an inference-cost analysis and case studies. Table [7] reports wall-clock time on 100 RewardBench2 prompts. Despite using two Qwen-3-8B modules (rubric generator + judge), Rubric-ARM runs in 33.50s, faster than most reasoning-based and rubric-based baselines. While JudgeLRM is slightly faster, it does not provide the explicit, interpretable rubric-conditioned signals that Rubric-ARM is designed for downstream policy optimization. Overall, our rubric-judge design replaces long chain-of-thought with short rubric generation and lightweight judging, yielding strong efficiency. Rubric-ARM is also faster than Rubric-RM, which typically generates longer rubric lists and incurs higher overhead.

*Table 8: Case study with error highlighting. Baselines either lose in the reasoning process, or favor the longer response but miss the explicit constraint about thumb war, while Rubric-ARM enforces hard rules before evaluating principles.*

| Case Study on RewardBench Chat Hard | |
| --- | --- |
| Prompt | Wars involve armed conflicts between nations, states, groups, or individuals, resulting in the loss of life, property damage, displacement of populations, and numerous other types of suffering. Is a *thumb war* violent? I care about both physical and psychological violence. |
| Resp A (snippet) | “War is inherently violent …direct physical violence …psychological violence …” (*“war”; not “thumb war”*) |
| Resp B (snippet) | “ …not considered a violent activity but rather a fun and harmless pastime …friendly competition …” |
| Label | Resp B. |
| RRM-7B | “ …Assistant 1 …doesn’t directly answer …But wait …the instruction doesn’t specify only thumb war …it’s more about war in general …I’m a bit confused …(No Decision)” |
| JudgeLRM | “Assistant 1 provided a detailed and accurate response …Assistant 2’s answer is inaccurate …(Choose A)” |
| Rubric-ARM | |
| Rubric | “1. The response must directly address whether a thumb war constitutes violence. [Hard Rule] 2. The response must consider both physical and psychological aspects of violence. [Hard Rule] 3. The response should accurately define and distinguish relevant concepts to ensure clarity. [Principle] 4. The response should present a balanced analysis by acknowledging multiple perspectives or contexts. [Principle] 5. The response should support its conclusions with logical reasoning or evidence. [Principle] 6. The response should maintain a respectful and empathetic tone. [Principle] ” |
| Judge | “The response must directly address whether a thumb war constitutes violence …Response A fails …not addressing thumb wars at all. Response B satisfies both …. (Choose B)” |

### 6.8 Case Study

We qualitatively analyze failures of baseline reward models on challenging examples. Table [8] shows a RewardBench Chat-Hard instance about *thumb war*: reasoning-based models (e.g., RRM-7B and JudgeLRM) are distracted by “war” and incorrectly prefer an armed-conflict response. In contrast, Rubric-ARM generates and enforces a rubric with an explicit hard rule about *thumb war*, leading to the correct preference. We provide additional IFBench examples in App. [D.3], where Rubric-ARM reliably extracts hard constraints and judges correctly while Rubric-RM fails.

7 Conclusion
------------

We present OpenRubrics, a large-scale dataset and framework for scalable and high-quality rubric generation.
By decomposing evaluation into hard rules and principles through Contrastive Rubric Generation (CRG) and applying preference–label consistency filtering, we constructed interpretable and discriminative rubric signals that better align with human judgment.
Our rubric-based reward model, Rubric-ARM, delivers an average 6.8% improvement across diverse benchmarks and further boosts policy performance by 1.1%–6.5% when used as a reward in offline reinforcement learning.
These results position rubrics-as-rewards as a practical foundation for transparent and generalizable LLM alignment. For future work, we will extend rubric generation to more open-ended tasks and leverage rubrics as intermediate supervision within RLHF pipelines.

References
----------

Appendix A Details for Group Relative Policy Optimization (GRPO)
----------------------------------------------------------------

GRPO *(shao2024deepseekmath)* is an actor-only policy optimization method that reduces variance by using the *within-prompt* average reward as a baseline. Concretely, for each prompt $q$, GRPO samples a group of responses $O\={o_{1},o_{2},\ldots,o_{G}}$ from the old policy $\pi_{\theta_{\text{old}}}(\cdot\mid q)$, computes a group-normalized advantage $\widehat{A}_{i,t}$ for each token, and then performs a PPO-style clipped update. Following *yu2025dapo*, we upweight informative prompts using a larger clipping threshold $\varepsilon_{\text{high}}$.

|  | $\displaystyle\mathcal{J}_{\text{GRPO}}(\theta)\={}$ | $\displaystyle\operatorname*{\mathbb{E}}_{q\sim P(Q),O\sim\pi_{\theta_{\text{old}}}(\cdot\mid q)}\Bigg[\frac{1}{G}\tsum\slimits@_{i\=1}^{G}\frac{1}{|o_{i}|}\tsum\slimits@_{t\=1}^{|o_{i}|}\min\Big(\rho_{i,t}(\theta)\,\widehat{A}_{i,t},\;\operatorname{clip}\big(\rho_{i,t}(\theta),1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}}\big)\,\widehat{A}_{i,t}\Big)\;-\;\beta\,\mathbb{D}_{\mathrm{KL}}!\left[\pi_{\theta}\,\|\,\pi_{\mathrm{ref}}\right]\Bigg],$ |  |
| --- | --- | --- | --- |

where $\rho_{i,t}(\theta)\=\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$ is the token-level importance ratio.

Appendix B Detailed Theoretical Derivations
-------------------------------------------

In this section, we provide the complete proofs for the variance analysis presented in Section [5].

### B.1 Preliminaries

Recall the definitions:

* •

    Reward: $R(o)\=\mathbb{I}[o\=o]$.

* •

    Judge Correctness: $p(r)\=\pi_{j}(o^{*}\,|\,c,r)$.

* •

    Generator Score: $u_{r}(r)\=\frac{\partial}{\partial\theta_{r}}{\theta_{r}}\log\pi_{r}(r\,|\,x)$.

* •

    Judge Score: $u_{j}(o\,|\,r)\=\frac{\partial}{\partial\theta_{j}}\log\pi_{j}(o\,|\,c,r)$.

We utilize the vector form of the Law of Total Variance:

###### Lemma B.1.

For random vectors $X$ and $Y$, $\mathrm{Var}(Y)\=\operatorname*{\mathbb{E}}_{X}[\mathrm{Var}(Y\,|\,X)]+\mathrm{Var}_{X}(\operatorname*{\mathbb{E}}[Y\,|\,X])$.

### B.2 Proof of Proposition [5.1]. ‣ 5.1 Variance Decomposition ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training") (Strategy A)

###### Proof.

In Strategy A, the rubric $\bar{r}$ is fixed. The gradient estimator is $\widehat{g}_{A}\=R(o)u_{j}(o\,|\,\bar{r})$, where $o\sim\pi_{j}(\cdot\,|\,\bar{r})$.
Since $\bar{r}$ is fixed, $u_{j}(o\,|\,\bar{r})$ takes two values: $u_{j}(o^{*}\,|\,\bar{r})$ (when correct) and $u_{j}(\neg o^{*}\,|\,\bar{r})$ (when wrong).
Considering the term associated with the reward $R(o)$, the variable is a scaled Bernoulli. Conditioned on $\bar{r}$:

* •

    With probability $p(\bar{r})$, $o\=o^{*}$, so $\widehat{g}_{A}\=1\cdot u_{j}(o^{*}\,|\,\bar{r})$.

* •

    With probability $1-p(\bar{r})$, $o\neq o^{*}$, so $\widehat{g}_{A}\=0$ (since $R\=0$).

Let $v:\=u_{j}(o^{*}\,|\,\bar{r})$. The first moment is:

|  | $\operatorname*{\mathbb{E}}[\widehat{g}_{A}\,|\,\bar{r}]\=p(\bar{r})v+(1-p(\bar{r}))\cdot 0\=p(\bar{r})v.$ |  |
| --- | --- | --- |

The second moment is:

|  | $\operatorname*{\mathbb{E}}[\norm{\widehat{g}_{A}}^{2}\,|\,\bar{r}]\=p(\bar{r})\norm{v}^{2}+(1-p(\bar{r}))\cdot 0\=p(\bar{r})\norm{v}^{2}.$ |  |
| --- | --- | --- |

Thus, the variance is:

|  | $\displaystyle\mathrm{Var}(\widehat{g}_{A}\,|\,\bar{r})$ | $\displaystyle\=\operatorname*{\mathbb{E}}\norm{\widehat{g}_{A}}^{2}-\norm{\operatorname*{\mathbb{E}}[\widehat{g}_{A}]}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=p(\bar{r})\norm{v}^{2}-\norm{p(\bar{r})v}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=(p(\bar{r})-p(\bar{r})^{2})\norm{v}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=p(\bar{r})(1-p(\bar{r}))\norm{u_{j}(o^{*}\,|\,\bar{r})}^{2}.$ |  |
| --- | --- | --- | --- |

∎

### B.3 Proof of Proposition [5.2]. ‣ 5.1 Variance Decomposition ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training") (Strategy B)

###### Proof.

In Strategy B, we update $\theta_{r}$. The estimator is $\widehat{g}_{B}\=R(o)u_{r}(r)$, where $r\sim\pi_{r}$ and $o\sim\pi_{j}(\cdot\,|\,r)$.
We apply Lemma [B.1] conditioning on $r$.

Step 1: Conditional Variance (Inner Term). Conditioned on $r$, $u_{r}(r)$ is a constant vector. The randomness comes only from $R(o)$.

|  | $\displaystyle\mathrm{Var}(\widehat{g}_{B}\,|\,r)\=\mathrm{Var}_{o\,|\,r}\quantity(R(o)u_{r}(r))\=\norm{u_{r}(r)}^{2}\mathrm{Var}_{o\,|\,r}(R(o)).$ |  |
| --- | --- | --- |

Since $R(o)\,|\,r\sim\mathrm{Bernoulli}(p(r))$, its variance is $p(r)(1-p(r))$. Thus:

|  | $\mathrm{Var}(\widehat{g}_{B}\,|\,r)\=p(r)(1-p(r))\norm{u_{r}(r)}^{2}.$ |  |
| --- | --- | --- |

Step 2: Conditional Expectation (Outer Term).

|  | $\operatorname*{\mathbb{E}}[\widehat{g}_{B}\,|\,r]\=\operatorname*{\mathbb{E}}_{o\,|\,r}[R(o)]u_{r}(r)\=p(r)u_{r}(r).$ |  |
| --- | --- | --- |

Step 3: Total Variance Decomposition. By applying the Law of Total Variance (Lemma [B.1]), we express the total variance as the sum of the expected conditional variance and the variance of the conditional expectation:

|  | $\mathrm{Var}(\widehat{g}_{B})\=\operatorname*{\mathbb{E}}_{r}\quantity[\mathrm{Var}(\widehat{g}_{B}\,|\,r)]+\mathrm{Var}_{r}\quantity(\operatorname*{\mathbb{E}}[\widehat{g}_{B}\,|\,r]).$ |  |
| --- | --- | --- |

Substituting the results derived in Step 1 and Step 2 into the equation above yields the final decomposition:

|  | $\mathrm{Var}(\widehat{g}_{B})\=\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]+\mathrm{Var}_{r}\quantity(p(r)u_{r}(r)).$ |  |
| --- | --- | --- |

This concludes the proof.
∎

### B.4 Proof of Theorem [5.5]. ‣ 5.2 Variance Domination in Early Training ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training")

###### Proof.

We analyze the sign of the variance difference $\Delta\=\mathrm{Var}(\widehat{g}_{B})-\operatorname*{\mathbb{E}}_{\bar{r}}[\mathrm{Var}(\widehat{g}_{A}\,|\,\bar{r})]$.

1. Variance Difference Expansion. Substituting the expressions from Propositions [5.1]. ‣ 5.1 Variance Decomposition ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training") and [5.2]. ‣ 5.1 Variance Decomposition ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training"):

|  | $\displaystyle\Delta$ | $\displaystyle\=\mathop{\mathchoice{\underarrow@$\displaystyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]+\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}{\underarrow@$\textstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]+\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}{\underarrow@$\scriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]+\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}{\underarrow@$\scriptscriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{r}(r)}^{2}]+\mathrm{Var}_{r}\quantity(p(r)u_{r}(r))}}}\limits_{V_{B}}-\mathop{\mathchoice{\underarrow@$\displaystyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{j}(o^{*}\,|\,r)}^{2}]}}{\underarrow@$\textstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{j}(o^{*}\,|\,r)}^{2}]}}{\underarrow@$\scriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{j}(o^{*}\,|\,r)}^{2}]}}{\underarrow@$\scriptscriptstyle\bracelu\leaders{\hbox{$\bracemid$}}{\hfill}\bracemu\leaders{\hbox{$\bracemid$}}{\hfill}\braceru${\operatorname*{\mathbb{E}}_{r}\quantity[p(r)(1-p(r))\norm{u_{j}(o^{*}\,|\,r)}^{2}]}}}\limits_{V_{A}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\operatorname*{\mathbb{E}}_{r}\Big[p(r)(1-p(r))\quantity(\norm{u_{r}(r)}^{2}-\norm{u_{j}(o^{*}\,|\,r)}^{2})\Big]+\mathrm{Var}_{r}\quantity(p(r)u_{r}(r)).$ |  |
| --- | --- | --- | --- |

2. Incorporating the Exploration Coefficient. Using the definition of $C_{1}$ from Assumption [5.3]. ‣ 5.2 Variance Domination in Early Training ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training"), we substitute $\mathrm{Var}_{r}(p(r)u_{r}(r))\=C_{1}\operatorname*{\mathbb{E}}_{r}[p(r)^{2}\norm{u_{r}(r)}^{2}]$:

|  | $\displaystyle\Delta$ | $\displaystyle\=\operatorname*{\mathbb{E}}_{r}\Big[p(r)(1-p(r))\quantity(\norm{u_{r}(r)}^{2}-\norm{u_{j}(o^{*}\,|\,r)}^{2})+C_{1}p(r)^{2}\norm{u_{r}(r)}^{2}\Big].$ |  |
| --- | --- | --- | --- |

3. Verification of Positivity. To show $\Delta>0$, we analyze the term inside the expectation (the integrand). We split the expression into multiple lines to isolate the quadratic components:

|  | Integrand | $\displaystyle\=(p(r)-p(r)^{2})\norm{u_{r}(r)}^{2}-(p(r)-p(r)^{2})\norm{u_{j}(o^{*}\,|\,r)}^{2}+C_{1}p(r)^{2}\norm{u_{r}(r)}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=p(r)\Big[(1-p(r))\norm{u_{r}(r)}^{2}-(1-p(r))\norm{u_{j}(o^{*}\,|\,r)}^{2}+C_{1}p(r)\norm{u_{r}(r)}^{2}\Big]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=p(r)\Big[\norm{u_{r}(r)}^{2}(1-p(r)+C_{1}p(r))-\norm{u_{j}(o^{*}\,|\,r)}^{2}(1-p(r))\Big].$ |  |
| --- | --- | --- | --- |

We now invoke the inequality from Assumption [5.3]. ‣ 5.2 Variance Domination in Early Training ‣ 5 Theoretical Analysis ‣ Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training"):

|  | $\frac{\norm{u_{r}(r)}}{\norm{u_{j}(o^{*}\,|\,r)}}>\sqrt{\frac{1-p(r)}{1-p(r)+C_{1}p(r)}}.$ |  |
| --- | --- | --- |

Squaring both sides and rearranging:

|  | $\norm{u_{r}(r)}^{2}(1-p(r)+C_{1}p(r))>\norm{u_{j}(o^{*}\,|\,r)}^{2}(1-p(r)).$ |  |
| --- | --- | --- |

This implies that the term inside the square brackets is strictly positive. Since $p(r)\in(0,1)$, the entire integrand is strictly positive for any $r$. Therefore, the expectation is strictly positive:

|  | $\Delta>0\implies\mathrm{Var}(\widehat{g}_{B})>\operatorname*{\mathbb{E}}_{\bar{r}}[\mathrm{Var}(\widehat{g}_{A}\,|\,\bar{r})].$ |  |
| --- | --- | --- |

This concludes the proof.
∎

Appendix C Implementation Details
---------------------------------

Table [9] and Table [10] show the hyperparameters used in Rubric-ARM and policy model training. We implement the GRPO training based on ms-swift111[https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift "") library *(msswift)* and implement DPO and IterDPO based on LLaMA-Factory222<https://github.com/hiyouga/LlamaFactory> *(llamafactory)*. We totally conduct 3 iterations for Rubric-ARM alternating RL training. Additionally, the sampling parameters used in inference are summarized in Table [11]. We used the same sampling parameters as their official implementations and papers for baseline methods.

*Table 9: Hyper-parameters used in Rubric-ARM training.*

| Module | Parameter | Value | Module | Parameter | Value |
| --- | --- | --- | --- | --- | --- |
| Rubric Generator | #generations | 6 | Judge | #generations | 7 |
| | Cutoff Length | 512 | | Cutoff Length | 1024 |
| Batch Size | 288 | Batch Size | 224 |
| Optimizer | AdamW | Optimizer | AdamW |
| Learning Rate | 1e-6 | Learning Rate | 1e-6 |
| Temperature | 1.0 | Temperature | 1.0 |
| #iterations | 2 | #iterations | 2 |
| Epochs | 1 | Epochs | 1 |
| $\epsilon_{\text{high}}$ | 0.28 | $\epsilon_{\text{high}}$ | 0.28 |
| $\epsilon_{\text{low}}$ | 0.2 | $\epsilon_{\text{low}}$ | 0.2 |
|  | $\beta$ | 0.001 |  | $\beta$ | 0.001 |

*Table 10: Hyper-parameters used in policy model training.*

| Method | Parameter | Value | Method | Parameter | Value |
| --- | --- | --- | --- | --- | --- |
| DPO | Cutoff Length | 2048 | GRPO | #generations | 6 |
| | Batch Size | 64 | | Cutoff Length | 2048 |
| Optimizer | AdamW | Batch Size | 288 |
| Learning Rate | 8e-7 | Optimizer | AdamW |
| Epochs | 1 | Learning Rate | 5e-7 |
| beta | 0.1 | Temperature | 1.0 |
| SFT mixing weight | 0.2 | #iterations | 2 |
| / | / | Epochs | 1 |
| / | / | $\epsilon_{\text{high}}$ | 0.28 |
| / | / | $\epsilon_{\text{low}}$ | 0.2 |
|  | / | / |  | $\beta$ | 0.001 |

*Table 11: Sampling parameters used in Rubric-ARM inference.*

| Module | Parameter | Value | Module | Parameter | Value |
| --- | --- | --- | --- | --- | --- |
| Rubric Generator | Maximum Tokens | 1024 | Judge | Maximum Tokens | 4096 |
| | Temperature | 0.0 | | Temperature | 1.0 |
| Top-P | / | Top-P | 1.0 |
| Top-K | / | Top-K | -1 |
| Enable-thinking | False | Enable-thinking | False |

Appendix D Additional Experimental Results
------------------------------------------

### D.1 Performance on WritingPreferenceBench

We present the performance on WritingPreferenceBench in Table [12].

*Table 12: Comparison of different judge and reward models on WritingPreferenceBench.
Best results are highlighted in bold.*

|  | Func. | Promo. | Non-Fic. | Fiction | Funny | Poetry | Script | Role | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM as Judge (black-box model) | | | | | | | | | |
| Claude-4-Opus-thinking | 65.7 | 64.3 | 64.1 | 60.1 | 54.2 | 64.0 | 43.5 | 51.7 | 61.0 |
| OpenAI-o4-mini | 58.3 | 58.6 | 60.9 | 55.5 | 53.2 | 68.0 | 30.4 | 55.2 | 56.6 |
| Gemini-2.5-Flash | 59.1 | 57.7 | 62.5 | 59.8 | 52.2 | 56.0 | 34.8 | 51.7 | 57.5 |
| White-box Reward Models | | | | | | | | | |
| Skywork-Llama-3.1-8B | 53.6 | 56.3 | 60.6 | 49.0 | 52.2 | 56.0 | 65.2 | 41.4 | 53.1 |
| Skywork-Gemma-2-27B | 49.0 | 53.9 | 59.6 | 33.9 | 55.1 | 36.0 | 21.7 | 51.7 | 46.8 |
| RM-R1-DeepSeek-Qwen-7B | 62.5 | 55.1 | 59.2 | 55.4 | 58.0 | 56.0 | 65.2 | 41.4 | 57.4 |
| RM-R1-Qwen2.5-7B | 67.0 | 57.2 | 53.9 | 60.0 | 54.6 | 72.0 | 47.8 | 65.5 | 59.8 |
| RRM-7B | 50.0 | 35.3 | 50.0 | 49.5 | 38.5 | 36.4 | 45.5 | 53.8 | 44.7 |
| Rubric-based Models | | | | | | | | | |
| Rubric-RM | 58.3 | 58.5 | 57.9 | 58.3 | 58.0 | 76.0 | 47.8 | 55.2 | 60.3 |
| Rubric-ARM | 67.8 | 63.1 | 65.8 | 60.9 | 61.0 | 80.0 | 47.8 | 55.2 | 63.2 |

### D.2 Position Bias Analysis

In this section, we study position bias in pairwise judge and reward models, where the predicted preference may depend on the relative order of the two responses *(shi2025judging)*.
We evaluate three settings: (1) keeping the response order fixed as in the original dataset, (2) flipping the order for all instances, and (3) randomly flipping the order on a per-instance basis.
Table [13] reports results on RewardBench and the IF evaluation benchmarks.
Overall, baseline methods exhibit non-trivial position bias.
For RRM-7B, changing the order leads to a 46.2-point difference on PPE-IFEval (75.8 vs. 29.6).
Likewise, for RM-R1-7B (Qwen-2.5-Inst), flipping the order changes InfoBench by 11.9 points (81.8 vs. 69.9).
For RM-R1-7B (DeepSeek-Dist), the order sensitivity remains substantial, with a 9.9-point difference on InfoBench (78.3 vs. 68.4) and a 9.3-point difference on FollowBench (79.0 vs. 69.7).
In contrast, our Rubric-ARM remains consistently stable across different orderings, suggesting substantially reduced position bias and more robust evaluation.
This design choice is aligned with our RL training design, where we randomize the response order when collecting reward signals, which further mitigates position bias in downstream policy optimization.

*Table 13: Position bias analysis for different judge and reward models.Rubric-ARM shows much lower sensitivity to the ordering of response pairs.*

|  | RewardBench | | IF Evaluation Benchmarks | | | | Avg. Variation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | Chat | Chat Hard | FollowBench | PPE-IFEval | InfoBench | IFBench | |
| White-box Judge/Reward LLM: RRM-7B | | | | | | | |
| Mixed Ord | 77.7 | 69.5 | 65.5 | 51.0 | 68.2 | 53.2 |  |
| Fixed Ord-1 | 73.9 | 61.6 | 53.8 | 29.6 | 62.3 | 30.2 |  |
| Fixed Ord-2 | 82.1 | 72.1 | 64.7 | 75.8 | 74.2 | 74.2 |  |
| Variation | 8.2 | 10.5 | 11.7 | 46.2 | 11.9 | 44.0 | 22.08 |
| White-box Judge/Reward LLM: RM-R1-7B (Qwen-2.5-Inst) | | | | | | | |
| Mixed Ord | 83.0 | 70.0 | 56.3 | 55.2 | 71.3 | 55.2 |  |
| Fixed Ord-1 | 82.1 | 63.4 | 57.1 | 54.8 | 81.8 | 53.8 |  |
| Fixed Ord-2 | 82.4 | 71.1 | 56.3 | 50.4 | 69.9 | 54.1 |  |
| Variation | 0.9 | 7.7 | 0.8 | 4.8 | 11.9 | 1.4 | 4.58 |
| White-box Judge/Reward LLM: RM-R1-7B (DeepSeek-Dist) | | | | | | | |
| Mixed Ord | 85.3 | 67.3 | 69.7 | 51.0 | 70.3 | 56.5 |  |
| Fixed Ord-1 | 87.1 | 67.3 | 79.0 | 52.8 | 78.3 | 53.2 |  |
| Fixed Ord-2 | 82.7 | 69.5 | 70.6 | 54.7 | 68.4 | 60.6 |  |
| Variation | 4.4 | 2.2 | 9.3 | 3.7 | 9.9 | 7.4 | 6.15 |
| Rubric-based Method: Rubric-RM | | | | | | | |
| Mixed Ord | 88.2 | 74.1 | 76.1 | 67.0 | 80.8 | 65.4 |  |
| Fixed Ord-1 | 87.4 | 74.6 | 79.8 | 70.8 | 80.9 | 66.4 |  |
| Fixed Ord-2 | 88.7 | 73.5 | 75.6 | 67.2 | 78.5 | 64.4 |  |
| Variation | 1.3 | 1.1 | 4.2 | 3.8 | 2.4 | 2.0 | 2.47 |
| Rubric-based Method: Rubric-ARM (Ours) | | | | | | | |
| Mixed Ord | 89.4 | 79.6 | 85.7 | 70.8 | 86.1 | 65.9 |  |
| Fixed Ord-1 | 89.9 | 79.4 | 84.9 | 71.8 | 86.1 | 65.3 |  |
| Fixed Ord-2 | 88.4 | 80.3 | 85.7 | 71.0 | 87.9 | 66.9 |  |
| Variation | 1.5 | 0.9 | 0.8 | 1.0 | 1.8 | 1.6 | 1.27 |

### D.3 Additional Case Study

In this section
we compare Rubric-ARM with Rubric-RM, another rubric-based RM trained with SFT, on a randomly chosen example from IFBench. The case specifies keywords and paragraph length. Results are shown in Table [14].
In this IFBench example, which requires specific keywords and exactly two paragraphs, the baseline Rubric-RM suffers from a judging hallucination, incorrectly claiming that a valid response is split into three paragraphs. Rubric-ARM, on the contrary, accurately extracts these hard constraints and identifies the missing *open-source* keyword in the negative sample, while correctly verifying the structure of the positive one.

*Table 14: Case study with error highlighting. In the baseline Rubric-RM, the rubric output confused the judge and led to an incorrect prediction.*

| Case Study on IFBench | |
| --- | --- |
| Prompt | What is Nextcloud, why should I use it, and how does it compare to other cloud storage solutions? Please provide the response with a tone of excitement and enthusiasm. Your response should contain *exactly 2 paragraphs* and include the keywords ‘cloud storage’ and ‘open-source’. |
| Resp A (snippet) | “Nextcloud is an incredibly exciting and innovative cloud storage solution …As an open-source platform …Imagine having your very own private cloud …” (*2 paragraphs*) |
| Resp B (snippet) | “Nextcloud is an incredibly powerful and flexible solution …While services like Google Drive and Dropbox are great …” (*2 paragraphs; missing “open-source”*) |
| Label | Resp A. |
| Rubric-RM | |
| Rubric | 1. The response must define what Nextcloud is. [Hard Rule] 2. The response must explain why the user should use Nextcloud. [Hard Rule] 3. The response must compare Nextcloud to other cloud storage solutions. [Hard Rule] 4. The response must use a tone of excitement and enthusiasm. [Hard Rule] 5. The response must consist of exactly two paragraphs. [Hard Rule] 6. The response must include the keywords ‘cloud storage’ and ‘open-source’. [Hard Rule] 7. The response should present information in a clear, logical structure that guides the reader through definition, benefits, and comparison. [Principle] 8. The response should use vivid, engaging language to maintain reader interest. [Principle] 9. The response should employ precise and varied vocabulary to convey ideas effectively. [Principle] 10. The response should provide specific, relevant details to substantiate its claims. [Principle] 11. The response should maintain coherence and flow, ensuring each paragraph builds on the previous one. [Principle] 12. The response should balance conciseness with sufficient elaboration to cover all required points. [Principle] |
| Judge | “…Response A is split into three paragraphs, not exactly two …Response A fails the gatekeeper criterion (paragraph count) …(Choose B) |
| Rubric-ARM | |
| Rubric | 1. The response must address what Nextcloud is, why it should be used, and how it compares to other cloud storage solutions. [Hard Rule] 2. The response must maintain an enthusiastic and excited tone throughout. [Hard Rule] 3. The response must consist of exactly two paragraphs. [Hard Rule] 4. The response must include the keywords ‘cloud storage’ and ‘open-source’. [Hard Rule] 5. The response should present information in a clear and logical structure, grouping related ideas coherently. [Principle] 6. The response should use vivid and engaging language to sustain reader interest. [Principle] 7. The response should provide relevant comparisons or contrasts to contextualize the subject. [Principle] 8. The response should maintain conciseness and avoid unnecessary repetition. [Principle] |
| Judge | “…Response B mentions ‘cloud’ and ‘open’ but not the exact keyword ‘open-source,’ so fails. …Response A fulfills all hard rules, including the precise keywords …(Choose A)” |

Appendix E Prompts
------------------

We present the prompts we used in this section.
For baseline methods, we adopted the prompts from their official implementations and papers.
