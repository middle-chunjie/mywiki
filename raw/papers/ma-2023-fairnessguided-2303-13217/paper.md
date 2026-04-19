# Fairness-guided Few-shot Prompting for Large Language Models

Huan Ma $^{1,2*}$ , Changqing Zhang $^{2\dagger}$ , Yatao Bian $^{1}$ , Lemao Liu $^{1}$ , Zhirui Zhang $^{1}$ , Peilin Zhao $^{1}$ , Shu Zhang $^{1}$ , Huazhu Fu $^{3}$ , Qinghua Hu $^{2}$ , Bingzhe Wu $^{1\dagger}$

$^{1}$  AI Lab, Tencent, Shenzhen, China

$^{2}$  College of Intelligence and Computing, Tianjin University, Tianjin, China

$^{3}$  Institute of High Performance Computing, A*STAR, Singapore

$^{2}$  zhanchangqing@tju.edu.cn;  $^{1}$  bingzhewu@tencent.com

# Abstract

Large language models have demonstrated surprising ability to perform in-context learning, i.e., these models can be directly applied to solve numerous downstream tasks by conditioning on a prompt constructed by a few input-output examples. However, prior research has shown that in-context learning can suffer from high instability due to variations in training examples, example order, and prompt formats. Therefore, the construction of an appropriate prompt is essential for improving the performance of in-context learning. In this paper, we revisit this problem from the view of predictive bias. Specifically, we introduce a metric to evaluate the predictive bias of a fixed prompt against labels or a given attributes. Then we empirically show that prompts with higher bias always lead to unsatisfactory predictive quality. Based on this observation, we propose a novel search strategy based on the greedy search to identify the near-optimal prompt for improving the performance of in-context learning. We perform comprehensive experiments with state-of-the-art mainstream models such as GPT-3 on various downstream tasks. Our results indicate that our method can enhance the model's in-context learning performance in an effective and interpretable manner. Code is available at: https://github.com/MaHuanAAA.

# 1 Introduction

Large language models (LLMs), such as GPT-3 [1] and BLOOM [2], have demonstrated remarkable ability in performing in-context learning (ICL) on downstream tasks. ICL refers to the process of conditioning an LLM to solve various downstream tasks using prompts constructed from a few demonstration input-output pairs [3] (i.e., few-shot prompting). Despite its impressive performance, prior research has shown that ICL suffers from high instability due to variations in the choice of in-context demonstrations, demonstration order, and prompt formats [4, 5]. Therefore, constructing an appropriate prompt has been identified as a critical factor for improving the performance of ICL [6].

Previous research studies this problem typically from two directions: (1) prompt tuning in the embedding space [7, 8, 9, 10, 11] (2) prompt searching in the text space [4, 12, 13, 14, 15, 16]. The key idea of prompt tuning is to inject task-specific embedding into hidden layers and then tune these embeddings using gradient-based optimization [8, 15]. However, these methods require to modify the original inference process of the model, which is impractical for the case of black-box LM services such as GPT3 and ChatGPT [17]. Furthermore, prompt tuning introduces additional computational

and storage costs, which is typically expensive for LLM. A more feasible and efficient way is to optimize prompting via searching approximate demonstration samples and ordering in the original text space [4, 15]. Bunch of works are presented to constructs prompts from either "global" or "local" views. On the one hand, global-view based methods typically optimize the different elements of the prompt as a whole, with the aim of achieving superior performance. For example, one approach, as described in [14], constructs a search procedure that leverages the overall diversity of demonstrations. Another approach [4] attempts to optimize the ordering of the entire set of demonstrations to achieve better performance. In contrast to the global view, local-view based methods optimize each individual demonstration by designing different heuristic selection criteria such as prior work KATE [15]. These methods have achieved impressive improvements on a wide range of tasks. However, most of them still suffer from the following limitations: (1) Most of current research mainly focuses on searching prompts along a single dimension, such as example selection or order. However, the overall influence of various dimensions on the performance remains unclear. (2) These methods are typically based on heuristic criteria, and there is a gap between them and actual performance. A unified view that explains how these methods work is needed. (3) More importantly, existing methods optimize prompts globally or locally, which may lead to suboptimal performance.

In this paper, we revisit this problem from the perspective of predictive bias. We find a key insight that the quality of a given prompt depends on its inherent bias. Based on this insight, we propose a surrogate metric based on predictive bias for evaluating the quality of prompts. This metric allows us to evaluate a prompt in a single forward process without an additional development set. Specifically, we apply a given prompt to a "content-free" input and expect the model output an uniform predictive distribution (a content-free input contains no useful information). Therefore, we employ the uniformity of the predictive distribution to characterize the bias of a given prompt. This shares a similar idea to the prior work which uses this metric to calibrate the model output [18]. In contrast to this work which mainly focus on using this metric for calibration when the prompt is fixed, we further explore its usage in automatically searching an approximate prompt. Moreover, through extensive experiments, we empirically validate the correlation between the inherent bias of a given prompt and its quality measured by the average task performance on a given test set (see Fig. 2).

Moreover, this bias-based metric allows us to build prompting optimization techniques in a "local-to-global" manner. We present two novel strategies for efficiently searching high-quality prompts in a bias-guided way: (1) T-fair-Prompting (2) G-fair-Prompting. We focus on a general setting where a labeled set with size  $N$  is given. The goal of our strategies is to perform combinatorial optimization over this set to find near-optimal prompts (i.e., select demonstrations and their orders). Specifically, T-fair-Prompting uses an intuitive way that first computes the bias of each single demonstration (i.e., one-shot prompting) and then select the top-k fair demonstrations to form the final prompts. This strategy can be efficiently done with a complexity of  $O(N)$ . Note that T-fair-Prompting is based on the assumption that the optimal prompt is usually constructed from demonstrations with the smallest individual bias. However, this may not hold true in real situations and often leads to sub-optimal solutions. Therefore, we further introduce G-fair-Prompting to improve the search quality. G-fair-Prompting follows the normal procedure of the greedy search which finds the optimal solution by making locally optimal choices at each step. At each step of the algorithm, the selected demonstration is the one which makes the updated prompts achieve the best fairness score. This strategy trades off the quality of the search with the worst-case time complexity. By accepting a higher worst-case time complexity of  $O(N^2)$ , the search quality is significantly improved. Note that G-fair-Prompting works from a local to global perspective, wherein bias of individual samples are considered in the early stages while the later stage focuses on the reduction of global predictive bias.

To evaluate the effectiveness of our strategies, we conduct extensive experiments with current mainstream models, such as GPT-3 [1], on various downstream tasks. Our results indicate that our method can significantly enhance the model's in-context learning performance in an effective and interpretable manner. The overall contribution is summarized as follows:

- We introduce to use the predictive bias to assess the quality of a given prompt in an efficient and development set independent way and the empirical effectiveness of this metric is comprehensively validated.  
- Based on the above idea, we propose two efficient and effective strategies, namely, T-fair-Prompting and G-fair-Prompting to optimize the prompts.

(a) Selection

(b) Selection (cal)  
Figure 1: ICL suffers from high instability due to high variations in demonstrations selection and order, even when post calibration is performed.

(c) Permutation

(d) Permutation (cal)

- The effectiveness of these two strategies are validated on various LLMs ranging from GPT-series models to LMaMA family [19] released by Meta recently. Consistent relative improvements of over  $10\%$  have been observed over different downstream tasks in contrast to SOTA methods.

Relation to Calibration-before-use: Our paper shares a similar metric with cal-before-use [18] to assess the predictive bias of a given prompt. However, the prior approach aims to use this metric to calibrate the output, which can be still easily affected by the quality of the used prompt (more results can be found in Table 3). In contrast, our research aims to find a near-optimal prompt on the original space to improve the model's performance, without requiring any post-adjustment to the output of the model. Moreover, we have firstly empirically validated the connection between predictive bias and the final task performance as shown in Fig. 2, which has not been studied in [18]. Through experiments, we have discovered that, even without calibration, the prompt selected by our method can outperform a randomly selected prompt with calibration.

# 2 Related Work

In-context Learning Previous research, as cited in [1, 20], has demonstrated that Large Language Models can complete tasks with zero- or few-shot learning using in-context learning. LLMs perform well with an appropriate prompt. However, recent works [4, 18] have shown that the performance of LLMs is affected by the prompt used. Therefore, determining the optimal prompt is a crucial and fundamental research area.

Original space searching A more intuitive approach for determining the best prompt is to search in the original space by selecting or reordering the prompt sentences entered by users. The searching can be concluded in two perspective.  $\bullet$  Global view: A naive strategy is to enumerate all candidates to find the prompt that can achieve the best performance on validation set, but this strategy is computationally expensive since its complexity is  $\sum_{k=1}^{n} C_n^k k!$ . Zhang et al. [12] find that errors frequently fall into the same cluster, where each cluster contains similar questions, so they proposed a diversity-guided searching strategy to select diverse demonstrations. In addition to demonstrations selection, [4] have identified the impact of the prompt order on the results. They found the best sequence which yields the most diverse prediction results on the probing set by generating a probing set through LLMs. However, this method is also computationally expensive, and it may be difficult to ensure that the generated probing set is sufficiently balanced.  $\bullet$  Local view: Previous studies [13] show that reducing the model's uncertainty helps improve the model's performance, and [14] propose Active Prompting to select demonstrations according to the uncertainty of LLMs. KATE [15] selects the prompt based on the distance amongst embeddings, with the goal of selecting the closest example. However, this method ignores the influence of the order of the examples and requires access to sentence embeddings. [16] demonstrate that LLMs can be easily distracted by irrelevant context, accordingly they identify several approaches for filtering out irrelevant information in context.

In the realm of original space searching, most of the current methods tend to focus solely on the influence of a singular factor (highlighted above) on performance, utilizing heuristic metrics to select context demonstrations that perform well according to this criterion. While these investigations certainly bring benefits to the community, they lack a comprehensive consideration of both local and global perspectives. The method proposed in this paper offers a metric to select context demonstrations from the perspective of predictive bias, which naturally facilitates a transition from the local view to global view.

(a) AGNews (BLOOM 176B)

(b) AGNews (LLaMA 13B)

(c) AGNews (LLaMA 65B)

(d) TREC (BLOOM 176B)

(e) TREC (LLaMA 13B)  
Figure 2: Accuracy is highly consistency with fairness and greedy search can find a good prompt, where "Random" and "Oracle" indicate the average accuracy of all prompts and the upper-bound performance according to fairness.

(f) TREC (LLaMA 65B)

# 3 Revisiting the Sensitivity across Demonstrations

In this section, we will clarify the notations and the templates used in this paper. Then, we will demonstrate some brief empirical results to show how different demonstration construction factors (e.g., example selection and order) affect performance. We further introduce the definition of predictive bias/fairness of a given prompt and show its connection to the predictive performance on different downstream tasks.

# 3.1 Notations

We consider a training set consisting of  $N$  samples  $S = \{(x_{i},y_{i})\}_{i}^{N}$ , where  $x_{i}$  is the sentence and  $y_{i}\in \mathcal{V}$  is the label of the  $i^{th}$  training sample, and  $\mathcal{V}$  is the space of all labels for the task. We use a template  $\Gamma (\cdot)$  to transform these sentences and labels into natural language space (i.e., prompt construction). Take an instance from the AGNews dataset [21] for example, we have  $x_{i} = "Cubans\ Risking\ Life\ for\ Lure\ of\ America."$ ,  $y_{i} = "World"$ , and  $\Gamma (x_i,y_i)$  is "Article: Cubans Risking Life for Lure of America. Answer: World". We concatenate these demonstrations to form a prompt  $\rho$ , which by default is  $\rho = \Gamma (x_1,y_1)\oplus \dots \oplus \Gamma (x_n,y_n)$ . At test time, we append the prompt  $\rho$  with  $\tau = "\text{Article: <test sentence>}$ . Answer: "and feed it to a large language model  $\mathcal{M}$ . The predicted class is given by:

$$
\hat {y} = \arg \max  _ {y \in \mathcal {Y}} \hat {p} (y | \rho \oplus \tau), \quad \hat {p} (y | \rho \oplus \tau) = \frac {\mathcal {M} (y | \rho \oplus \tau)}{\sum_ {y \in \mathcal {Y}} \mathcal {M} (y | \rho \oplus \tau)}, \tag {1}
$$

where  $\mathcal{M}(y|\rho \oplus \tau)$  indicates the probability predicted by LLM, and the probability is normalized to fit the task. We denote the predictive distribution by  $\hat{P} (x)\coloneqq \{\hat{p} (y|\rho \oplus \tau)|y\in \mathcal{Y}\}$ . In this paper, we focus on evaluating the instability caused by demonstrations, and we fix the prompt template following prior work [18].

# 3.2 Stability of Few-shot Prompting

As demonstrated by prior research, the few-shot prompting technique is highly susceptible to a variety of factors, including the selection and order of demonstrations [4, 18]. In this study, we delve deeper into the stability of few-shot prompting, specifically focusing on the recently released LLaMA family by Meta [19]. Additionally, we evaluate the stability of LLaMA models calibrated using the current state-of-the-art method [12, 15].

To elucidate the impact of demonstration selection, we select four demonstrations for each different seed and randomly sample an order for each combination. Subsequently, we present the performance on AGNews in the form of a boxplot, which displays the data distribution based on a five-number

summary (minimum, first quartile [Q1], median, third quartile [Q3], and maximum). As depicted in Fig.1(a)(b), the accuracy demonstrates significant variability across various demonstrations.

To investigate the influence of permutations, we examine all possible permutations of four fixed demonstrations, resulting in 4! distinct candidates. Fig.1(c)(d) also reveals a high degree of variance. While post-calibration contributes to mitigating instability, it is essential to note that the model remains sensitive even after post-calibration. This finding underscores the importance of meticulous demonstration selection. In subsequent experiments, we discover that our approach can be employed to further enhance the performance of the calibrated model.

# 3.3 Predictive Bias of ICL

As demonstrated in the preceding discussion, the performance of ICL is significantly impacted by various factors such as demonstration, permutation, and selection (refer to Appendix A.4 for additional information). Consequently, devising an efficient method for constructing an appropriate prompt with near-optimal performance is a crucial step in deploying LLMs for diverse downstream tasks. As outlined in the introduction, numerous studies aim to optimize prompts in ICL. This paper further investigates this issue through the lens of predictive bias, which refers to the discrepancy between targeted classes. $^3$

To achieve this, we initially introduce an efficient technique to assess the inherent predictive bias of a given prompt, drawing inspiration from previous work [18]. We construct a training set-independent metric to measure predictive bias as follows: first, we merge the provided prompt with "semantic-free" test sample information (e.g., "[N/A]", denoted by  $\eta$ ) and obtain the LLM's predictive distribution for this sample. Ideally, the predictive distribution should closely resemble a uniform distribution, as the test sample lacks semantic information. In this paper, we employ entropy as a measure of predictive bias, defined as:

$$
\operatorname {f a i r} (\rho) = - \sum_ {y \in \mathcal {Y}} p (y | \rho \oplus \eta) \log p (y | \rho \oplus \eta) \tag {2}
$$

Previous studies have utilized this metric to calibrate the model's output. In this paper, we conduct a comprehensive examination of the relationship between predictive bias and overall performance. Specifically, in a scenario with four training samples (due to the time-consuming nature of enumerating all prompt cases for a larger number), we enumerate all possible combinations and permutations of demonstrations for various datasets and LLMs. Subsequently, we arrange all candidates in descending order based on fairness, where an "index 0" denotes the prompt with the highest fairness. We perform experiments using five different seeds, resulting in training sets comprising distinct demonstrations while maintaining the test samples with seed 0. Fig. 2 displays the results for different models, revealing a strong correlation between the model's performance and fairness score (i.e., fairer prompts yield better performance). The red star, referred to as the "Oracle" represents the optimal average performance, which consistently correlates with higher fairness. This observation prompts us to enhance the ICL performance by identifying the fairest prompt.

Nevertheless, discovering the fairest demonstration combination proves to be a formidable challenge, given the existence of  $\sum_{k=1}^{N} C_N^k k!$  distinct candidates. As the size of the training set increases, this task becomes intractable. In order to tackle this problem, we propose two efficient strategies for approximating the most suitable demonstrations in the subsequent section.

# 4 Fairest Prompt Search

Drawing upon the aforementioned observations, we propose two strategies aimed at identifying the most fair prompt, which have been empirically demonstrated to achieve superior performance. Let us consider a training set  $S$  comprising  $n$  samples; the goal of these search strategies is to select a subset of samples from the training set and construct the context in a specific order so as to optimize the fairness criterion in Eq. 2.

In an ideal scenario, we would consider the factors of demonstration selection and order permutation by examining  $\sum_{k=1}^{N} C_N^k k!$  distinct candidates, which enumerates all possible situations. Here,

Figure 3: Overview of Most-fair Prompting.

$k$  represents the number of demonstrations selected, and  $C$  signifies the combinatorial function. However, evaluating every candidate is infeasible, as demonstrated when  $N = 8$ , yielding over  $10^{6}$  candidates. In this paper, we introduce two search strategies to reduce computational cost: T-fair-Prompting and G-fair-Prompting. The T-fair-Prompting strategy decreases complexity from  $\Theta (\sum_{k = 1}^{N}C_N^k k!)$  to  $\Theta (N)$ , but its performance hinges on the selection of  $k$  and may be unstable when an unsuitable value of  $k$  is chosen. As a result, we propose an additional greedy search strategy, termed G-fair-Prompting, which lowers complexity to  $O(N^{2})$  and offers a superior approximation of the oracle solution. Fig. 8 visualizes the computational costs over different training set size.

# 4.1 T-fair-Prompting

The central idea of T-fair-Prompting is founded on the heuristic understanding that the fairest prompt usually consists of demonstration samples with reduced individual biases. Consequently, T-fair-Prompting constructs the prompt through a two-stage process. Initially, the prediction bias is assessed when the prompt is formulated using individual demonstrations. Subsequently, the top- $k$  fairest demonstrations are chosen and employed to prompt the LLM. It is important to note that fairer demonstrations are likely to be situated towards the end of the sequence, as the generation is more influenced by proximate demonstrations, in accordance with prior research [18]. A comprehensive description of the process is presented in Algorithm 1, while a visual representation can be found in Fig. 3. Specifically, when  $k$  is equivalent to the size of the training set, the method degrades to a search for the optimal order of demonstrations. Nevertheless, T-fair-Prompting is heavily reliant on the chosen value of  $k$ . More crucially, T-fair-Prompting addresses this issue through a purely local perspective, thereby neglecting considerations from a global standpoint, which typically results in sub-optimal outcomes. As a result, we subsequently introduce the G-fair-Prompting method, which operates in a local-to-global fashion, as described below.

# 4.2 G-fair-Prompting

The G-fair-Prompting algorithm adheres to the standard procedure of greedy search, which seeks the optimal solution by making locally optimal choices at each stage. In each step of the algorithm, the chosen demonstration is the one that allows the updated prompts to achieve the highest fairness score. This strategy balances the quality of the search with the worst-case time complexity. By accepting an increased worst-case time complexity of  $O(N^2)$ , the search quality is significantly enhanced. It is important to note that the G-fair-Prompting algorithm operates from a local to global perspective as shown by Algorithm. During the initial stages, the bias of individual samples is taken into account, while the later stages focus on reducing global predictive bias. Specifically, at each step, we insert a new demonstration  $\Gamma(x_i, y_i)$  from the remaining demonstration set  $S'$  (ensuring demonstrations are

not repeated) at the beginning of the current context  $\rho$  and select the demonstration that maximizes the fairness improvement. Formally, at step 9 in Algorithm 2, the inserted demonstration should satisfy the following criterion:

$$
\arg \max  _ {x _ {i} \in \mathcal {S} ^ {\prime}} \operatorname {f a i r} \left(\Gamma \left(x _ {i}, y _ {i}\right) \oplus \rho\right) \quad \text {s . t .} \operatorname {f a i r} \left(\Gamma \left(x _ {i}, y _ {i}\right) \oplus \rho\right) > \operatorname {f a i r} (\rho). \tag {3}
$$

# Algorithm 1 T-fair-Prompting

1: Given: training set  $S = \{(x_i, y_i)\}_i^N$ , pretrained LLM  $\mathcal{M}$ , transformation template  $\Gamma(\cdot)$ , and context-free input  $\eta$  
2: Initial prompt  $\rho$  
3: for  $(x_{i},y_{i})$  in  $S$  do  
4: Inference  $\hat{P} \gets \{\hat{p}(y|\Gamma(x_i, y_i) \oplus \eta)|y \in \mathcal{Y}\}$  via  $\mathcal{M}$  
5: Calculate the fair  $(\Gamma (x_i,y_i))$  according to Eq. 2  
6: end for  
7: Sort  $\text{fair}_{i=1,\dots,N}(\Gamma(x_i, y_i))$  in descending order  
8: for  $d$  in  $1, \dots, k$  do  
9: Insert the most  $d$  fair demonstration at the head of  $\rho$  
10: end for  
11: return  $\rho$

# Algorithm 2 G-fair-Prompting

1: Given: training set  $S = \{(x_i, y_i)\}_i^N$ , pretrained LLM  $\mathcal{M}$ , transformation template  $\Gamma(\cdot)$ , and context-free input  $\eta$  
2: Initial prompt  $\rho$  
3: while  $S$  is not null do  
4: for  $(x_{i},y_{i})$  in  $S$  do  
5:  $\rho_{\mathrm{tmp}}\gets \Gamma (x_i,y_i)\oplus \rho$  
6: Inference  $\hat{P} \gets \{\hat{p}(y|\rho_{\mathrm{tmp}} \oplus \eta)|y \in \mathcal{Y}\}$  via  $\mathcal{M}$  
7: Calculate the fair  $(\rho_{\mathrm{tmp}})$  according to Eq. 2  
8: end for  
9: Insert the demonstration that can improve fairness best and remove it from  $S$  
10: Stop searching when fairness can't be improved  
11: end while  
12: return  $\rho$

# 5 Experiments

# 5.1 Experimental Setup

Models. There are a large number of available LLMs (Appendix A.2) including open-source models and black-box cloud API. Recently, Meta has released their powerful pretrained LLMs, LLaMA. LLaMA models with 13B parameters can achieve comparable performance in contrast to BLOOM and GPT-3 with much larger model size. In this paper, we evaluate the effectiveness of our method on BLOOM (176B) and LLaMA models of different sizes. We have opted to employ LLaMA (65B) as a substitute for GPT-3 in our experiments, since oepnai strictly restricts the API access to certain areas.

Datasets. We conducted experiments on various text classification datasets [21], namely SST-2, AGNews, CoLA, TREC, and RTE. Furthermore, the maximum input length of LLaMA is 512, and the sentences in RTE are too long for LLaMA. The task descriptions and statistics are available in Table 1.

Table 1: Dataset descriptions.  

<table><tr><td>Corpus</td><td>Task</td><td>Classes</td><td>Domain</td><td>Total Cost1</td></tr><tr><td>SST-2</td><td>sentiment</td><td>2</td><td>movie reviews</td><td>over 60 GPU hours</td></tr><tr><td>TREC</td><td>QA/QC</td><td>6</td><td>open domain</td><td>over 220 GPU hours</td></tr><tr><td>AGNews</td><td>topic</td><td>4</td><td>news</td><td>over 250 GPU hours</td></tr><tr><td>CoLA</td><td>acceptability</td><td>2</td><td>misc.</td><td>over 160 GPU hours</td></tr><tr><td>RTE2</td><td>NLI</td><td>2</td><td>news, Wikipedia</td><td>over 110 GPU hours</td></tr></table>

1 Total Cost=Hours× GPUs. Hardware: BLOOM=A100, LLaMA=V100.  
2 Not applicable to LLaMA because of the maximum prompt token limit.

# 5.2 Results

We conducted experiments on different settings and reported the results of five runs. We compared our method with the diversity-guided searching strategy proposed by Zhang et al.[12] (Global view) and

Table 2: Accuracy for different prompting strategies (averaged on  $5_{0,\ldots ,4}$  different seeds, where Top-  $k$  and Greedy indicate T-fair-Prompting with  $k$  demonstrations and G-fair-Prompting respectively).  

<table><tr><td>Model</td><td>Dataset</td><td>Random</td><td>Diversity</td><td>Similarity</td><td>Top-2</td><td>Ours Top-4</td><td>Greedy</td></tr><tr><td rowspan="5">BLOOM (176B)</td><td>SST2</td><td>92.72.3</td><td>95.00.9</td><td>94.00.9</td><td>94.60.5</td><td>93.82.1</td><td>91.24.0</td></tr><tr><td>AGNews</td><td>73.95.9</td><td>70.210.1</td><td>74.83.8</td><td>75.42.2</td><td>74.82.3</td><td>79.61.4</td></tr><tr><td>TREC</td><td>47.914.6</td><td>46.08.7</td><td>31.43.1</td><td>55.413.3</td><td>39.219.3</td><td>66.82.5</td></tr><tr><td>RTE</td><td>62.44.2</td><td>69.21.9</td><td>67.23.5</td><td>55.61.0</td><td>57.61.9</td><td>63.02.1</td></tr><tr><td>CoLA</td><td>68.44.8</td><td>71.03.7</td><td>69.82.5</td><td>66.48.6</td><td>66.83.7</td><td>68.26.2</td></tr><tr><td rowspan="4">LLaMA (33B)</td><td>SST2</td><td>82.511.8</td><td>90.02.7</td><td>72.84.4</td><td>82.011.1</td><td>80.012.2</td><td>85.68.2</td></tr><tr><td>AGNews</td><td>75.25.0</td><td>75.05.1</td><td>75.02.4</td><td>73.23.9</td><td>69.84.4</td><td>76.44.6</td></tr><tr><td>TREC</td><td>68.111.1</td><td>68.24.7</td><td>60.63.4</td><td>71.411.1</td><td>57.817.3</td><td>80.25.3</td></tr><tr><td>CoLA</td><td>66.911.0</td><td>68.86.8</td><td>72.82.0</td><td>63.813.3</td><td>69.83.9</td><td>70.64.2</td></tr><tr><td rowspan="4">LLaMA (65B)</td><td>SST2</td><td>90.07.7</td><td>90.89.0</td><td>87.43.1</td><td>88.28.6</td><td>95.81.5</td><td>87.89.0</td></tr><tr><td>AGNews</td><td>76.85.0</td><td>78.23.1</td><td>78.21.8</td><td>77.03.4</td><td>76.24.9</td><td>76.04.0</td></tr><tr><td>TREC</td><td>63.614.2</td><td>65.210.9</td><td>64.05.5</td><td>65.813.0</td><td>57.419.9</td><td>74.012.2</td></tr><tr><td>CoLA</td><td>66.29.8</td><td>62.68.6</td><td>59.214.0</td><td>67.611.7</td><td>62.66.5</td><td>72.04.5</td></tr></table>

the similarity-guided searching strategy proposed by Liu et al.[15] (Local view). Note that methods based on local view are time-consuming since they require searching different demonstrations for every test example. Table 2 shows the performance of the different strategies, where "Random" indicates the average accuracy for enumerating all situations, "Diversity" and "Similarity" indicate demonstrations are selected according to diversity and similarity, respectively. For each dataset, we set the size of the training set to 4. "Diversity" and "Similarity" select 4 from 16 demonstrations, as they need more candidates. The baseline is expensive to compute since enumerating all candidates for 4 demonstrations in RTE on BLOOM will take more than 120 NVIDIA A100 GPU hours. We enumerate all candidates for the training set with 4 demonstrations on different models, as shown in Fig. 2. The results on models whose parameters less than 13B are shown in Table 5 (i.e., GPT2-XL (1.5B), LLaMA (7B), and LLaMA (13B)).

- G-fair-Prompting can reach a close approximation of enumeration. To evaluate whether the G-fair-Prompting (Greedy) method can approximate the best performance of enumerating all candidates, we marked the performance of G-fair-Prompting with a green star (representing the closest value to averaged accuracy of G-fair-Prompting on the line). We found that G-fair-Prompting can achieve a very close approximation to enumeration. As shown in Fig. 2, most prompts searched by G-fair-Prompting achieved a top  $20\%$  ranking, and on BLOOM (176B), G-fair-Prompting almost found the most fair prompt.

- G-fair-Prompting outperforms T-fair-Prompting. As shown in Table 2, although T-fair-Prompting achieves better performance compared with random selection, G-fair-Prompting consistently outperforms T-fair-Prompting. Furthermore, Top-2 significantly outperforms Top-4 in most cases (over  $5\%$ ), indicating that the number of demonstrations selected is crucial. Overall, the results demonstrate that G-fair-Prompting achieves satisfactory performance with only a slight additional cost.

- Compared with SOTA methods. We compared our methods with several State-of-the-Art (SOTA) methods, including diversity-guided and similarity-guided techniques. We observed that our greedy approach outperforms most of these SOTA methods in most situations, and the improvements of over  $10\%$  are observed on dataset TREC. The similarity-guided method, on the other hand, achieved the best performance on the topic classification task (AGNews).

Figure 4: BLOOM is not sensitive to CoLA.

This is because it searches for a unique prompt for every different test example based on the distance between the embeddings of the training samples and the test example. This strategy selects demonstrations with labels that are the same as the test samples, and Language Models (LLMs) tend to predict biased predictions toward the labels that always appear in the context. However, the similarity-guided method may prove inadequate when applied to other tasks. Specifically, the similarity-guided strategy exhibits lower performance compared to random selection in QC and acceptability tasks. Furthermore, the G-fair-Prompting approach may occasionally falter when the model's sensitivity to the task is not immediately evident, as observed in the acceptability task on BLOOM (depicted in Fig. 4). Note that the training set size of compared methods is  $4 \times$  larger than ours.

- Comparison with Calibration Method. Post-calibration [18], can enhance the accuracy of a given prompt in most cases. However, when the selected prompt is of poor quality, the performance may remain inadequate even after calibration. We compared the performance of G-fair-Prompting with random selection with calibration (averaged on all candidates), and found that G-fair-Prompting can outperform random selection with calibrated most situations. For example, on the topic classification task, G-fair-Prompting achieves the best performance on most models. Moreover, we find that post calibration can harm the performance of the model and it occurs significantly times, so it is worthwhile to reconsider the influence of manipulating the model's probability directly.

Table 3: Accuracy comparison after post calibration.  

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">BLOOM (176B)</td><td colspan="2">LLaMA (33B)</td><td colspan="2">LLaMA (65B)</td></tr><tr><td>Average</td><td>Worst</td><td>Average</td><td>Worst</td><td>Average</td><td>Worst</td></tr><tr><td rowspan="3">TREC</td><td>Random (cal)</td><td>66.89.0</td><td>57.2</td><td>69.26.2</td><td>59.4</td><td>74.69.7</td><td>66.2</td></tr><tr><td>Ours</td><td>66.82.5</td><td>64.0</td><td>80.25.3</td><td>75.0</td><td>74.012.2</td><td>50.0</td></tr><tr><td>Ours (cal)</td><td>77.01.1</td><td>75.0</td><td>76.65.1</td><td>70.0</td><td>72.812.6</td><td>48.0</td></tr><tr><td rowspan="3">AGNews</td><td>Random (cal)</td><td>73.06.6</td><td>61.8</td><td>71.95.0</td><td>64.0</td><td>78.24.7</td><td>71.6</td></tr><tr><td>Ours</td><td>79.61.4</td><td>77.0</td><td>76.44.6</td><td>69.0</td><td>76.04.0</td><td>71.0</td></tr><tr><td>Ours (cal)</td><td>77.41.4</td><td>76.0</td><td>76.04.4</td><td>68.0</td><td>76.43.6</td><td>70.0</td></tr><tr><td rowspan="3">CoLA</td><td>Random (cal)</td><td>68.55.5</td><td>61.2</td><td>67.85.1</td><td>63.6</td><td>54.012.4</td><td>42.4</td></tr><tr><td>Ours</td><td>68.26.2</td><td>57.0</td><td>70.64.2</td><td>64.0</td><td>72.04.5</td><td>66.0</td></tr><tr><td>Ours (cal)</td><td>68.05.2</td><td>58.0</td><td>70.43.8</td><td>65.0</td><td>72.04.5</td><td>66.0</td></tr></table>

Post calibration [18] can improve the accuracy of a certain prompt (in most cases), but when the selected prompt is very poor, the performance still very poor even after calibration. We conducted experiments (Table 3) to compare the performance of G-fair-Prompting and random selection with calibration ("Average" and "Worst" indicate averaged accuracy and worst performance on all permutations of training examples), and observed that G-fair-Prompting outperforms random selection with calibration in most case. For instance, on the CoLA, G-fair-Prompting exhibited superior performance on most models. Additionally, we find that post-calibration could negatively affect the model's performance in many scenarios while it sometimes can improve the performance significantly even on selected prompts, for example, an improvement by  $10\%$  is observed on BLOOM-TREC. Hence, it is crucial to reconsider the impact of directly manipulating the model's probability.

# 6 Conclusion

In this paper, we revisit the sensitivity of large language model across prompts, and analyse the issue from a predictive bias perspective. Accordingly, we employ a "content-free" strategy as a metric termed as fairness to evaluate the predictive bias of a fixed prompt and show that model's performance is highly consistency with fairness. Then, we propose two strategy to search the most fair prompt in the original space. We conduct extensive experiments on current famous LLMs, and validate the effectiveness of the proposed strategy. Moreover, in addition to fairness adopted in this paper, there would be more metrics for prompt searching in the future for different scenarios.

# References

[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, volume 33, pages 1877–1901, 2020.  
[2] Bloom: A 176b-parameter open-access multilingual language model. https://huggingface.co/bigscience/bloom.  
[3] Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. Language models as knowledge bases? arXiv preprint arXiv:1909.01066, 2019.  
[4] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In ACL, 2021.  
[5] Feng Nie, Meixi Chen, Zhirui Zhang, and Xu Cheng. Improving few-shot performance of language models via nearest neighbor calibration. arXiv preprint arXiv:2212.02216, 2022.  
[6] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1-35, 2023.  
[7] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation, 2021.  
[8] Xiao Liu, Kaixuan Ji, Yicheng Fu, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv preprint arXiv:2110.07602, 2021.  
[9] Karen Hambardzumyan, Hrant Khachatrian, and Jonathan May. Warp: Word-level adversarial reprogramming. arXiv preprint arXiv:2101.00121, 2021.  
[10] Guanghui Qin and Jason Eisner. Learning how to ask: Querying lms with mixtures of soft prompts. arXiv preprint arXiv:2104.06599, 2021.  
[11] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. arXiv preprint arXiv:2103.10385, 2021.  
[12] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022.  
[13] Claudio Gentile, Zhilei Wang, and Tong Zhang. Fast rates in pool-based batch active learning. arXiv preprint arXiv:2202.05448, 2022.  
[14] Shizhe Diao, Pengcheng Wang, Yong Lin, and Tong Zhang. Active prompting with chain-of-thought for large language models. arXiv preprint arXiv:2302.12246, 2023.  
[15] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804, 2021.  
[16] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. arXiv preprint arXiv:2302.00093, 2023.  
[17] https://openai.com/blog/chatgpt.  
[18] Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In ICML, 2021.  
[19] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.  
[20] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical Report, 2018.

[21] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.  
[22] Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.  
[23] https://huggingface.co/EleutherAI/gpt-neox-20b.

# A Appendix

# A.1 Pretrained Large Language Models

Neural autoregressive language model (LMs) are designed for next token prediction to predict the probability distribution over the next token after a sequence of tokens input, and pre-trained LMs show their superior performance since they are trained on various programming languages and a large-scale curated dataset. Training large natural LMs are very expansive and time-consuming process since they always have billions of parameters, which limits the development of LMs. Fortunately, many pre-trained LMs are open access or limited access, which promotes researchers to pool their time and makes the resources to collectively achieve a higher impact. EleutherAI makes the GPT-J [22] and GPT-Neox [23] public available on Hugging Face. GPT-3 [1] is limited access in OpenAI which can be used by researchers for a fee, and another large open-science open-access multilingual language model named Bloom [2] is provided by BigScience.

# A.2 Open Access Models

Table 4: Pretrained language models  

<table><tr><td>Model</td><td>Params</td><td>Provider</td><td>Access</td></tr><tr><td>GPT-2</td><td>124 M</td><td>Hugging Face</td><td>OPEN</td></tr><tr><td>GPT-Medium</td><td>335 M</td><td>Hugging Face</td><td>OPEN</td></tr><tr><td>GPT2-Large</td><td>774 M</td><td>Hugging Face</td><td>OPEN</td></tr><tr><td>GPT-XL</td><td>1.5 B</td><td>Hugging Face</td><td>OPEN</td></tr><tr><td>GPT-3 (ada)</td><td>350 M</td><td>OPENAI</td><td>LIMITED</td></tr><tr><td>GPT-3 (babbage)</td><td>1.3 B</td><td>OPENAI</td><td>LIMITED</td></tr><tr><td>GPT-3 (curie)</td><td>6.7 B</td><td>OPENAI</td><td>LIMITED</td></tr><tr><td>GPT-3 (davinci)</td><td>175 B</td><td>OPENAI</td><td>LIMITED</td></tr><tr><td>GPT-J</td><td>6 B</td><td>EleutherAI</td><td>OPEN</td></tr><tr><td>GPT-NeoX</td><td>20 B</td><td>EleutherAI</td><td>OPEN</td></tr><tr><td>Bloom</td><td>176 B</td><td>BigScience</td><td>OPEN</td></tr><tr><td rowspan="4">LLaMA</td><td>7 B</td><td>Meta</td><td>OPEN</td></tr><tr><td>13 B</td><td>Meta</td><td>OPEN</td></tr><tr><td>33 B</td><td>Meta</td><td>OPEN</td></tr><tr><td>65 B</td><td>Meta</td><td>OPEN</td></tr></table>

# A.3 Additional Figures on Different Settings

In addition to the Fig. 2, we show the performance on different models for enumerating all candidates, note that the shadow indicates the half value of standard deviation for clear presentation since the variance is very high for LLMs.

# A.4 Accuracy Varies with demonstrations

Accuracy Varies with Example Amount Demonstrations play an important role in imparting task-related information to language models through in-context learning. Then, the question arises - does a larger number of demonstrations necessarily equate to better performance? To answer this question, we evaluated performance in terms of accuracy by gradually increasing the number of demonstrations. We set  $\rho = \Gamma(x_1, y_1) \oplus \dots \oplus \Gamma(x_k, y_k)$ , where  $k = 1, \dots, n$ , and demonstrations are erased with  $k$  decreasing from  $n$  to 1. Intuitively, accuracy would vary highly across different numbers of demonstrations, and the phenomenon is observed in Fig. 6a. To our surprise, however, erasing some demonstrations can result in a better performance. Removing some demonstrations can perform better and sometimes GPT-3 achieves best accuracy when there is only a few demonstrations remaining. This highlights the importance of considering the appropriate number of demonstrations.

(a) AGNews (GPT2-XL 1.5B)

(b) TREC (GPT2-XL 1.5B)

(c) RTE (GPT2-XL 1.5B)

(d) AGNews (LLaMA 33B)

(e) TREC (LLaMA 33B)  
Figure 5: Accuracy is highly consistency with fairness and greedy search can find a good prompt, where "Random" and "Oracle" indicates the average accuracy of all prompts and the upper-bound performance according to fairness.

(f) SST-2 (LLaMA 33B)

(a) Varying amount of examples  
Figure 6: ICL suffers from high instability due to variations in example amount, example order, and example selection.

(b) Permutation

(c) Select different examples

Example Order The performance of a model is sensitive to the order of the demonstrations, as has been discussed in [4]. Even when the demonstrations are the same, different permutations of the demonstrations can result in vastly different outcomes. As there are  $n!$  possible permutations, we introducing a strategy of permuting the demonstrations by circularly shifting the index of the demonstrations. The demonstration can be represented as  $\rho = \Gamma(x_{k+1}, y_{k+1}) \oplus \dots \oplus \Gamma(x_n, y_n) \oplus \Gamma(x_1, y_1) \oplus \dots \oplus \Gamma(x_k, y_k)$ . As shown in Fig. 6b, the accuracy varies highly with permutation which is consistent with the observations in [4].

Example Selection In this paper, we find which demonstrations are selected is influence the model extremely. This scenario can be described as selecting  $k$  demonstrations in  $n$  training samples. In Fig. 6c, we only select one example for demonstration to ablate the impact of demonstrations order, and the accuracy also varies highly with different example selected. In this work, we only detail evaluate the proposed probing method on the erasing demonstrations and permutation, although our method improves by  $20\%$  in the setting of example selection on SST-2 (GPT2-XL), because selecting  $k$  demonstrations on a set with  $n$  training samples can't be regarded as  $k$ -shot learning in the strict sense.

# A.5 Relationship between with- and without-calibration

- G-fair-Prompting without post-calibration outperforms random demonstrations after post-calibration. Based on Table 2, it is apparent that G-fair-Prompting outperforms random selection prior to post-calibration. This leads to a natural question: do prompts with better performance before calibration also indicate better performance after calibration proposed by Zhao et al. [18]? To investigate the relationship between performance with- and without-calibration, we calculated the Pearson correlation coefficient between the accuracy with- and without-calibration Pearson  $(acc_{w/o}, acc_{with})$ . A positive coefficient value suggests that a prompt with high accuracy

Table 5: Accuracy for different prompting strategies (averaged on  ${5}_{0},\ldots ,4$  different seeds).  

<table><tr><td>Model</td><td>Dataset</td><td>Random</td><td>Diversity</td><td>Similarity</td><td>Top-2</td><td>Ours Top-4</td><td>Greedy</td></tr><tr><td rowspan="4">GPT2-XL (1.5B)</td><td>SST-2</td><td>61.16.1</td><td>-</td><td>-</td><td>60.811.4</td><td>65.88.7</td><td>74.212.0</td></tr><tr><td>AGNews</td><td>38.911.4</td><td>-</td><td>-</td><td>45.212.5</td><td>37.211.2</td><td>46.411.9</td></tr><tr><td>TREC</td><td>22.15.7</td><td>-</td><td>-</td><td>19.48.9</td><td>28.29.2</td><td>25.07.4</td></tr><tr><td>RTE</td><td>53.26.9</td><td>-</td><td>-</td><td>54.07.5</td><td>53.65.9</td><td>56.42.2</td></tr><tr><td rowspan="3">LLaMA (7B)</td><td>AGNews</td><td>64.510.0</td><td>66.49.1</td><td>-</td><td>66.011.7</td><td>69.25.5</td><td>63.85.7</td></tr><tr><td>TREC</td><td>49.510.4</td><td>51.49.6</td><td>-</td><td>48.410.5</td><td>38.615.2</td><td>61.34.8</td></tr><tr><td>CoLA</td><td>60.410.6</td><td>63.88.7</td><td>-</td><td>58.27.8</td><td>61.66.5</td><td>36.43.6</td></tr><tr><td rowspan="3">LLaMA (13B)</td><td>AGNews</td><td>72.27.7</td><td>78.43.5</td><td>-</td><td>73.69.0</td><td>74.24.3</td><td>75.22.8</td></tr><tr><td>TREC</td><td>46.416.5</td><td>48.016.0</td><td>-</td><td>51.016.6</td><td>39.223.3</td><td>61.412.1</td></tr><tr><td>CoLA</td><td>67.72.9</td><td>67.22.4</td><td>-</td><td>67.02.0</td><td>67.21.6</td><td>67.02.0</td></tr></table>

Figure 7: Illustration of accuracy relationship between with- and without calibration when Pearson is positive.

before calibration has a higher likelihood of achieving higher accuracy after calibration than other prompts. We take the topic classification task on LLaMA(65B) for illustration to show the relationship between with- and without calibration when Pearson is positive in Fig.7. Table 6 presents the Pearson correlation coefficient on accuracy of permutation and G-fair-Prompting after calibration. The majority of Pearson correlation coefficients were found to be positive, indicating that prompts with better performance before calibration have more potential to perform well after calibration. Furthermore, our results on the LLaMA family reveal that the larger the model, the stronger the correlation between performance with- and without-calibration. For instance, the value of the Pearson correlation coefficient increases from 0 to 0.7 as the model size increases.

Theorem A.1. Suppose the performance of the model under certain prompts with- and without calibration is positively correlated, i.e.,  $\text{Pearson}(acc_{w/o}, acc_{with}) > 0$ , if we can assure  $\mathbb{E}(acc_{w/o}^{Selected}) > \mathbb{E}(acc_{w/o}^{Random})$ , then we have  $\mathbb{E}(acc_{with}^{Selected}) > \mathbb{E}(acc_{with}^{Random})$ .

Table 6: Pearson's r between the with- and without-calibration.  

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">BLOOM
176B</td><td rowspan="2">7B</td><td colspan="3">LLaMA</td></tr><tr><td>13B</td><td>33B</td><td>65B</td></tr><tr><td>TREC</td><td>0.1274</td><td>0.1551</td><td>0.2959</td><td>0.3090</td><td>0.5151</td></tr><tr><td>AGNews</td><td>0.3875</td><td>-0.0471</td><td>0.3044</td><td>0.6953</td><td>0.7100</td></tr><tr><td>CoLA</td><td>0.4050</td><td>0.3592</td><td>0.5193</td><td>0.3611</td><td>0.8012</td></tr></table>

As analysed in Theorem A.1, if we can find a prompt with high accuracy before calibration, we have a higher likelihood of achieving higher accuracy after calibration than random selection. Our approach consistently identifies an appropriate prompt, as evidenced by the results in Table 2. Moreover, the

performance of the model exhibits a positive correlation with and without calibration under certain prompts, as illustrated in Table 6. Therefore, our method is more likely to enhance calibration performance.

# A.6 Complexity of different strategies

Figure 8: Computational cost. T-fair and G-fair indicate T-fair-Prompting and G-fair-Prompting respectively, and "w/c" indicates the worst case.

# A.7 Performance on Zero-shot and SOTA Classifiers

# Footnotes:

Page 0: *The project was conducted during the internship in AI Lab, Tencent † Corresponding author 
Page 4: This notion differs slightly from the concept of social bias, which concentrates on specific feature attributes rather than labels. Our approach can be naturally extended to mitigate social bias in various settings. 
