# Unified Demonstration Retriever for In-Context Learning

Xiaonan  $\mathsf{Li}^{1*}$ , Kai  $\mathsf{Lv}^{1*}$ , Hang Yan $^{1}$ , Tianyang Lin $^{1}$ , Yu Wei $^{2\dagger}$ , Yuan Ni $^{3}$ , Guotong Xie $^{3}$ , Xiaoling Wang $^{2}$ , Xipeng Qiu $^{1\dagger}$

$^{1}$  Shanghai Key Laboratory of Intelligent Information Processing, Fudan University

$^{1}$  School of Computer Science, Fudan University

$^{2}$ East China Normal University  $^{3}$  Pingan Health Tech

$^{1}\{liox20,klv21,hyan19,tylin20,xpqiu\} @fudan.edu.cn,$

wzhu@stu.ecnu.edu.cn, xlwang@cs.ecnu.edu.cn

$^{3}\{\mathrm{n iy u a n}442$  , xieguotong  $\} @$  pingan.com.cn

# Abstract

In-context learning is a new learning paradigm where a language model conditions on a few input-output pairs (demonstrations) and a test input, and directly outputs the prediction. It has been shown highly dependent on the provided demonstrations and thus promotes the research of demonstration retrieval: given a test input, relevant examples are retrieved from the training set to serve as informative demonstrations for in-context learning. While previous works focus on training task-specific retrievers for several tasks separately, these methods are often hard to transfer and scale on various tasks, and separately trained retrievers incur a lot of parameter storage and deployment cost. In this paper, we propose Unified Demonstration Retriever (UDR), a single model to retrieve demonstrations for a wide range of tasks. To train UDR, we cast various tasks' training signals into a unified listwise ranking formulation by language model's feedback. Then we propose a multi-task listwise ranking training framework, with an iterative mining strategy to find high-quality candidates, which can help UDR fully incorporate various tasks' signals. Experiments on  $30+$  tasks across 13 task families and multiple data domains show that UDR significantly outperforms baselines. Further analyses show the effectiveness of each proposed component and UDR's strong ability in various scenarios including different LMs  $(1.3\mathrm{B}\sim 175\mathrm{B})$ , unseen datasets, varying demonstration quantities, etc.

# 1 Introduction

Large language models have shown an impressive in-context learning ability for various Natural Language Processing (NLP) tasks (Brown et al., 2020; Dong et al., 2022). In-context learning (ICL) is a recent learning paradigm where a language

Figure 1: Demonstration retrieval: Given a test input  $x_{test}$ , relevant demonstrations are retrieved from the training set. Then the inference LM takes demonstrations and  $x_{test}$  as input and generates the output.

model (LM) learns a task by observing a few input-output pairs (demonstrations) and directly output the prediction of the given test input. Thus ICL can unify a wide range of NLP tasks through one language model's inference without parameter updates, which makes it a promising alternative to supervised fine-tuning (Devlin et al., 2019).

However, it has been shown that ICL's performance highly depends on the provided demonstrations (Liu et al., 2022; Zhang et al., 2022; Li and Qiu, 2023a). This promotes the research of demonstration retrieval for in-context learning (Liu et al., 2022; Rubin et al., 2022; Shi et al., 2022): As shown in Figure 1, given a test input, relevant examples are retrieved from an annotated training set, to serve as informative demonstrations for ICL.

There are about two lines of methods to retrieve demonstrations. One is to leverage off-the-shelf retrievers, e.g., BM25 (Robertson and Zaragoza, 2009) or Sentence-BERT (Reimers and Gurevych, 2019a). They can retrieve demonstrations that are textually or semantically similar to the test input and achieve empirical improvements. Thanks to their versatility, they can serve for extensive NLP tasks, but they are heuristic and sub-optimal since they are not guided by task supervision. Another line is to train a task-specific retriever by a specially designed task signal. Das et al. (2021) train the retriever for knowledge-based question answering,

based on the logic form's surface similarity. Hu et al. (2022) explore ICL on dialogue state tracking and design the similarity between dialogue's states as the retriever's training signal. Rubin et al. (2022) and Shi et al. (2022) leverage the LM's feedback to train demonstration retrievers for semantic parsing in English and cross-lingual scenarios, respectively. These task-specialized retrievers show better performance than the former, but they still face two challenges: 1. these explorations are limited to a small range of tasks and demonstrated separately on each task, e.g., semantic parsing or dialogue state tracking, which restricts systematic and compatible research on demonstration retrieval for ICL while ICL is a unified framework for extensive tasks. 2. it is costly for these methods to transfer and scale on various tasks and the reason is twofold: (i) they need to design a specialized training signal for each task. (ii) the number of retrievers will scale up with increasing tasks, which results in massive parameter storage and deployment costs.

To address these limitations, we explore learning various tasks' demonstration retrieval in a unified formulation and propose Unified Demonstration Retriever (UDR), a single multi-task model for demonstration retrieval of a wide range of tasks. To train UDR, we cast various tasks' training signals into a unified list-wise ranking formulation. For a training example from task  $\mathcal{T}$  we select a list of candidate examples from  $\mathcal{T}$ 's training set and rank them by LM's feedback. Then we propose a multitask list-wise ranking training framework, with an iterative mining strategy to find high-quality candidates. Specifically, we iteratively train the retriever to rank candidates and use itself to find high-quality positive candidates and hard negatives. Compared with the representative method for demonstration retrieval, EPR(Rubin et al., 2022), which trains the retriever by the binary label from LM's feedback and selects candidates in a manually limited range, our training framework can explore the entire dataset to get high-quality candidates and help UDR fully incorporate the LM's feedback through list-wise ranking training.

Experiments on  $30+$  tasks across 13 task families and multiple data domains show that UDR significantly outperforms baselines and further analyses show the effectiveness of each proposed component and UDR's strong ability under various scenarios including different LMs  $(1.3\mathrm{B}\sim 175\mathrm{B})$  ,unseen datasets, varying demonstrations quantities,

etc. We release the code and model checkpoint at https://github.com/KaiLv69/UDR.

# 2 Unified Demonstration Retriever

Provided a language model  $G$ , a training set  $\mathcal{D}_{\text{train}}$  and a test case  $x_{\text{test}}$ , demonstration retrieval aims to retrieve  $x_{\text{test}}$ 's relevant demonstrations from  $\mathcal{D}_{\text{train}}$  to help LM  $G$  decode the target output. Previous works (Das et al., 2021; Rubin et al., 2022; Shi et al., 2022) propose task-specialized methods for several tasks separately, but they are hard to transfer and scale on various tasks. In this work, we focus on learning various tasks' demonstration retrieval in a unified formulation and propose UDR, a single model for demonstration retrieval of a wide range of tasks, as shown in Figure 2. We introduce its architecture, training, and inference as follows.

# 2.1 Bi-encoder with Task Instruction

UDR is based on the prevailing bi-encoder architecture, dense passage retriever (DPR) (Karpukhin et al., 2020), which encodes the query example and candidate examples separately and then calculates their similarity. To distinguish examples from different tasks, UDR encodes the example together with its task instruction, which is a short piece of text related to the task objective. Taking CNN-/DailyMail (Hermann et al., 2015) as an example, its task instruction can be "Summarize the text". Given an example query  $x$  and a candidate demonstration  $z = \{x', y'\}$  from task  $T_i$ , UDR uses the query encoder  $E_q$  and demonstration encoder  $E_d$  to encode them respectively and calculates their similarity as:

$$
\operatorname {s i m} (x, z) = E _ {q} \left(I _ {i} \oplus x\right) ^ {\top} E _ {d} \left(I _ {i} \oplus z\right), \tag {1}
$$

where  $I_{i}$  is  $T_{i}$ 's task instruction and  $\oplus$  is the concatenation operator.  $E_{q}$  and  $E_{d}$  are two multilayer Transformer (Vaswani et al., 2017) encoders with "CLS" pooling and can be initialized with pre-trained models (Devlin et al., 2019).

Thus, we can not only get task-specific features by specifying the task instruction, but also retain the uniformity and parameter efficiency of ICL.

# 2.2 Learning from LM Feedback

To train the demonstration retriever, previous works (Das et al., 2021; Rubin et al., 2022; Hu et al., 2022) design task-specific training signals for several tasks separately, which makes their methods hard to transfer and scale on various tasks.

Figure 2: Illustration of UDR's inference for various tasks: Given a test input and its task's instruction, UDR can retrieve informative demonstrations from the corresponding datasets for ICL, where arrows and lines with various colors such as  $\rightarrow$  and  $\rightarrow$  indicate corresponding tasks' pipelines, respectively.

and hinders systematic and compatible research on demonstration retrieval. For UDR's training, we propose to cast various tasks' training signals into a unified list-wise ranking formulation. Then we introduce a multi-task list-wise ranking training framework, where we iteratively let the retriever itself to mine high-quality candidates and learn to rank them in turn, across various tasks, shown in Algorithm 1. We introduce the list-wise ranking training and iterative mining strategy as follows.

# 2.2.1 Ranking Candidates by LM

Given a training example  $(x,y)$  and its candidates  $Z = \{z_{i}\}_{i = 1}^{l}$ , we first rank these candidates as:

$$
r \left(z _ {j}\right) = r a n k \left(s \left(z _ {j}\right) \mid \left\{s \left(z _ {i}\right) \right\} _ {i = 1} ^ {l}\right) \tag {2}
$$

$$
s _ {g e n} \left(z _ {j}\right) = p _ {G} (y \mid z _ {j}, x), \tag {3}
$$

$$
s _ {c l s} \left(z _ {j}\right) = \frac {p _ {G} \left(y \mid z _ {j} , x\right)}{\sum_ {y ^ {\prime} \in Y} p _ {G} \left(y ^ {\prime} \mid z _ {j} , x\right)}, \tag {4}
$$

where  $s(z_{j}) = s_{gen}(z_{j})$  for generation tasks and  $s(z_{j}) = s_{cls}(z_{j})$  for classification and multi-choice tasks.  $p_{G}(\cdot |\cdot)$  is the LM  $G$ 's conditional likelihood.  $Y$  is the label space or choices of the classification or multi-choice task, respectively. For simplicity, we omit special tokens and classification tasks' verbalizers in the equations above.

First we use  $G$  to score each candidate (Rubin et al., 2022) and calculate  $s(z_{j})$  as the ground truth  $y$ 's likelihood conditioned on the candidates  $z_{j}$  and the query input  $x$ .  $s(z_{j})$  indicates the importance of  $z_{j}$  for  $G$  to encode  $x$  and generate the ground truth  $y$ . Then we rank  $Z$  according to  $\{s(z_{i})\}_{i = 1}^{l}$ . The more important  $z_{j}$  is for  $x$ , the higher  $z_{j}$ 's rank will be. Thus we unify various tasks' training signals into the same list-wise ranking formulation using LM's feedback, instead of designing task-specific objectives (Das et al., 2021; Hu et al., 2022).

# 2.2.2 Loss Function

With these candidates' ranks from  $G$ 's feedback, we propose to use the following loss function to inject the ranking signal into the retriever  $E$ , inspired by LambdaRank (Burges, 2010):

$$
\mathcal {L} _ {\text {r a n k}} = \sum_ {z _ {i}, z _ {j} \in Z} w * \log \left(1 + e ^ {\sin \left(x, z _ {j}\right) - \sin \left(x, z _ {i}\right)}\right) \tag {5}
$$

where  $w = \max \left(0, \frac{1}{r(z_i)} - \frac{1}{r(z_j)}\right)$ .

For those  $z_{i}$  and  $z_{j}$  where  $r(z_{i}) < r(z_{j})$ ,  $L_{rank}$  will draw  $\mathrm{sim}(x, z_{i})$  up and optimize the retriever towards  $\mathrm{sim}(x, z_{i}) > \mathrm{sim}(x, z_{j})$ . Additionally,  $w$  adjusts the weight for each pair of demonstrations and inject list-wise ranking information into  $\mathcal{L}_{rank}$ . When  $z_{i}$  has a much higher rank than  $z_{j}$ , e.g.,  $r(z_{i}) = 1$  and  $r(z_{j}) = 10$ ,  $w$  will be a high weight and strongly draw  $\mathrm{sim}(x, z_{i})$  up from  $\mathrm{sim}(x, z_{j})$ . Since we optimize the retriever on demonstration pairs under different  $w$ ,  $\mathcal{L}_{rank}$  can help UDR fully incorporate candidates' listwise ranking signals from  $G$ 's feedback for various tasks and learn to retrieve those helpful demonstrations.

To fully leverage the computation of the same batch, we also use the in-batch negative loss as:

$$
\mathcal {L} _ {i b} = - \log \frac {e ^ {\operatorname* {s i m} (x , z ^ {*})}}{\sum_ {z \in \mathbb {Z}} e ^ {\operatorname* {s i m} (x , z)}}, \tag {6}
$$

where  $z^{*}$  is the rank-1 candidate of  $x$  and  $\mathbb{Z}$  is all candidates ( $x$ 's or not  $x$ 's) in the batch. Each batch is sampled from the same task, and to alleviate the bias towards high-resource tasks, we sample each task according to the multinomial distribution with probabilities  $\{p(\mathcal{T}_i)\}_{i=1}^T$  as:

$$
p \left(\mathcal {T} _ {i}\right) = \frac {q _ {i} ^ {\alpha}}{\sum_ {j = 1} ^ {T} q _ {j} ^ {\alpha}} \text {w i t h} q _ {i} = \frac {\left| \mathcal {D} ^ {\mathcal {T} _ {i}} \right|}{\sum_ {j = 1} ^ {T} \left| \mathcal {D} ^ {\mathcal {T} _ {j}} \right|}, \tag {7}
$$

where  $\mathcal{D}^{\mathcal{T}_i}$  is the  $i$ th task's dataset.  $\alpha$  is a predefined hyper-parameter and we follow Conneau and Lample (2019) to set  $\alpha$  as 0.5.

The overall loss function of UDR is the integration of these two losses as follows,

$$
\mathcal {L} = \lambda * \mathcal {L} _ {\text {r a n k}} + (1 - \lambda) * \mathcal {L} _ {i b}, \tag {8}
$$

where  $\lambda$  is a pre-defined hyper-parameter.

# 2.2.3 Iterative Candidate Mining

The selection of candidates can be a key factor for retriever's training (Karpukhin et al., 2020; Xiong et al., 2021). It is desirable for UDR to take the entire training set as candidates to provide abundant ranking signals. However, it is infeasible since scoring all pairs of training examples is quadratic in  $|\mathcal{D}|$  and costly. Previous work (Rubin et al., 2022) selects those examples which have textually similar targets with  $x$ 's as candidates. However, it may bias the retriever to learn among candidates with highly similar targets. Meanwhile, it can probably miss important demonstrations. For instance, if an example  $z$  contains relevant logic with the query  $x$  but has a dissimilar target with  $x$ 's, the valuable  $z$  will not be selected as candidate to provide signal for the retriever. So, we propose an iterative mining strategy to select candidates by the retriever itself. Specifically, we iteratively train the retriever and use it to select candidates in turn. At each iteration, we update each training example's candidates as:

$$
Z ^ {*} = \operatorname {t o p} - K _ {z \in \mathcal {D}} \sin (x, z) \tag {9}
$$

where  $\mathcal{D}$  is the task's entire training set.

Then we will use LM  $G$  to score and rank  $Z^{*}$ . The new candidates in  $Z^{*}$  can be divided into two categories. If a new candidate  $z$  has a low score, it means that we find a hard-negative candidate that can provide crucial negative signal for the retriever. If the score of  $z$  is high and even higher than all old candidates, it means that we find a valuable positive candidate that can help the retriever learn to find informative demonstrations. Thus, with iterative mining, we can explore the entire dataset, find high-quality candidates and improve training progressively. Before the first iteration, the retriever is untrained, so we initialize candidates based on surface similarity, inspired by Rubin et al. (2022).

For computational efficiency, we first update candidates and score  $Z^{*}$  at each iteration, and then randomly sample  $l$  of  $Z^{*}$  and rank them at each training step. In summary, Algorithm 1 shows the UDR's overall training procedure.

# Algorithm 1 Multitask List-wise Ranking Training

Require: Bi-encoder  $E_{q}$  and  $E_{d}$ , language model  $G$ , Training sets of  $T$  tasks  $\{\mathcal{D}^{\mathcal{T}_i}\}_{i = 1}^T$

1: Initialize the bi-encoder.  
2: Initialize candidates of each training example.  
3: Score initialized candidates by  $G$ .  
4: for Each iteration do  
5: for Each training step,  $\mathcal{T}_i\sim p(\mathcal{T})$  do  
6: Sample a batch of examples.  
7: For each example, sample  $l$  examples  $z_{1\sim l}$  from its candidates and rank  $z_{1\sim l}$  by  $G$  's score.  
8: Update the bi-encoder's parameters by  $\mathcal{L}$  
9: end for  
10: Update candidates by new  $\mathrm{E}_q$  and  $\mathrm{E}_d$ .  
11: Score new candidates by  $G$ .  
12: end for

# 2.3 Inference

After training, we encode each task  $\mathcal{T}_i$ 's training set using  $E_{d}(p_{i}\oplus \cdot)$ . At the test stage, given a task  $\mathcal{T}_i$ 's input,  $x_{test}$ , we use  $E_{q}(p_{i}\oplus \cdot)$  to compute its encoding and then use FAISS (Johnson et al., 2021) to search over  $\mathcal{T}_i$ 's training set to find the most relevant demonstrations, ascendingly sorted by  $\mathrm{sim}(x_{test},\cdot),D = (z_1,z_2,\dots ,z_L)$ . For generation tasks, the number of final demonstrations,  $L$ , is determined by the LM  $G$ 's maximal input length  $C$ . Specifically,  $\sum_{i = 1}^{L}|z_i| + |x_{test}| + |y|\leq C$ , where  $|y|$  is the pre-defined maximal length of the generated target. For classification and multi-choice tasks, we observe that increasing  $L$  brings negligible performance improvement and thus we set  $L$  to a small value, 8. We conduct further analysis of the number of demonstrations in section 3.3.5. Finally, we use greedy decoding to get the result of  $G([z_{1};z_{2};\dots ;z_{L};x_{test}])$ . Notice that here  $D$  is ascendingly sorted by  $\mathrm{sim}(x_{test},\cdot)$  unless otherwise specified. Our analysis in section 3.3.4 shows that different orderings lead to similar performance. Thus we use the same ordering strategy with EPR (Rubin et al., 2022) for fair comparison.

# 3 Experiment

# 3.1 Experimental Settings

Dataset We train UDR on a wide range of NLP tasks, consisting of about 40 tasks across 13 task families and multiple data domains, including: Sentiment Classification: SST-2, SST-5 (Socher et al., 2013), Amazon (McAuley and Leskovec, 2013), Yelp (Zhang et al., 2015), MR (Pang and Lee, 2005)

and CR (Amplayo et al., 2022); Topic Classification: AGNews, Yahoo (Zhang et al., 2015), TREC (Voorhees and Tice, 2000) and DBPeida (Lehmann et al., 2015); Multi Choice: COPA (Roemmele et al., 2011), Cosmos QA (Huang et al., 2019), Commonsense Validation and Explanation (ComE and ComV) (Wang et al., 2019b); NLI: MNLI (Williams et al., 2018), SNLI (Bowman et al., 2015) and RTE (Bar-Haim et al., 2014); Subjectivity Classification: Subj (Pang and Lee, 2004); Linguistic Acceptibility: COLA; Semantic Parsing: BREAK (Wolfson et al., 2020), MTOP (Li et al., 2021) and SMCalFlow (Andreas et al., 2020); Text Summarization: CNN/DailyMail (Hermann et al., 2015), PubMed (Cohan et al., 2018) and Reddit (Kim et al., 2019); Commonsense Generation: CommonGen (Lin et al., 2020); Story Generation: Roc Story and Ending Generation (Mostafazadeh et al., 2016); Code Summarizaton: Go, Python, Java and PHP (Lu et al., 2021); Text Simplification: WikiAuto + Turk/ASSET (Jiang et al., 2020); Data to Text: DART (Nan et al., 2021) and E2E (Dušek et al., 2019). These tasks' input/output, statistics, split and evaluation metrics are in Appendix A.

Implementation Details We follow EPR (Rubin et al., 2022) to use GPT-Neo-2.7B (Black et al., 2021) as the scoring LM and the inference LM for most experiments in the paper unless otherwise specified. We also explore UDR's transferability across different inference LMs in section 3.3.2. Following EPR(Rubin et al., 2022), we initialize  $E_{q}$  and  $E_{d}$  as two separate "BERT-base-uncased" encoders (Devlin et al., 2019). We list the overall hyper-parameters and implementation details in Appendix B. On each task, we use one specific template for scoring and inference (see Appendix A). We evaluate UDR's performance when inference templates are different with the scoring template in Appendix C, and the results show that UDR has stable performance across varying inference templates, which reflects UDR's generality.

Model Comparison With the same inference LM, GPT-Neo-2.7B, we compare UDR with previous methods for demonstration retrieval by the downstream ICL performance, including: 1. Random: We randomly sample demonstrations from the corresponding task's training set. 2. BM25 (Robertson and Zaragoza, 2009): A prevailing sparse retriever. For each test input  $x_{test}$ , we use BM25 to retrieve examples with the most

similar input. 3. SBERT (Reimers and Gurevych, 2019b): We use the Sentence-BERT as the dense demonstration retriever. Specifically, we follow Rubin et al. (2022) to take "paraphrase-mpnet-base-v2" to encode the test input  $x_{test}$  and training set's inputs, and retrieve the examples with the most similar input as demonstrations. 4. Instructor (Su et al., 2022): Instructor is a recently proposed competitive text embedding model trained on 330 tasks with instructions. By providing the specialized instruction, it can serve for demonstration retrieval. For fair comparison, we conduct experiments on its released base-size model. 5. DR-Target: This baseline is inspired by previous works on generation tasks like dialogue state tracking, question answering and code generation (Hu et al., 2022; Das et al., 2021; Poesia et al., 2022), which design the task-specific target's similarity and use examples with similar targets to train the retriever. Here we use BM25 as the similarity function for each task's target output. Specifically, we use BM25 to find positive pairs with similar targets and use DPR (Karpukhin et al., 2020) for training. 6. EPR (Rubin et al., 2022): EPR is a recently proposed representative method for training demonstration retriever. It uses the language model to assign candidate examples with positive and negative labels and thus trains a task-specific demonstration retriever by DPR. For fair comparison, we train EPR on each task using the same hyper-parameters of UDR. Specially, we discuss EPR's candidate quantity in Appendix B.

Except that the performance of Random, BM25, SBERT and EPR on semantic parsing is from the previous paper (Rubin et al., 2022), other results are from our implementation since they are not explored previously.

# 3.2 Main Results

We show the performance comparison of classification tasks and generation tasks in Table 1 and Table 2, respectively. We can see that UDR outperforms baselines significantly on most tasks, which shows UDR's best overall demonstration retrieval ability on a wide range of NLP tasks. Specially, compared with DR-Target and EPR, UDR has better overall performance and this shows the effectiveness of our unification of various tasks' training signals. Meanwhile, compared with Instructor (Su et al., 2022), the text embedding model trained on 330 tasks' text pairs, UDR has an improvement

<table><tr><td rowspan="2">Retrieval Method</td><td colspan="6">Sentiment Classification</td><td colspan="5">Topic Classification</td></tr><tr><td>SST-2</td><td>SST-5</td><td>Amazon</td><td>Yelp</td><td>MR</td><td>CR</td><td>AGNews</td><td>TREC</td><td>DBPedia</td><td>Yahoo</td><td></td></tr><tr><td>Random</td><td>57.7</td><td>28.2</td><td>23.9</td><td>25.3</td><td>56.0</td><td>52.4</td><td>74.2</td><td>42.6</td><td>73.7</td><td>39.1</td><td></td></tr><tr><td>BM25</td><td>74.1</td><td>38.3</td><td>31.6</td><td>36.9</td><td>71.4</td><td>57.2</td><td>88.4</td><td>89.4</td><td>97.2</td><td>62.5</td><td></td></tr><tr><td>SBERT</td><td>84.3</td><td>40.0</td><td>33.4</td><td>36.0</td><td>79.0</td><td>61.3</td><td>88.3</td><td>89.4</td><td>96.7</td><td>58.4</td><td></td></tr><tr><td>Instructor</td><td>83.7</td><td>42.4</td><td>42.4</td><td>46.6</td><td>78.5</td><td>64.1</td><td>89.6</td><td>91.2</td><td>97.7</td><td>67.2</td><td></td></tr><tr><td>EPR</td><td>87.9</td><td>46.9</td><td>49.1</td><td>49.6</td><td>80.6</td><td>65.7</td><td>89.9</td><td>95.2</td><td>98.1</td><td>66.1</td><td></td></tr><tr><td>UDR</td><td>92.4</td><td>50.5</td><td>54.9</td><td>61.7</td><td>85.2</td><td>82.6</td><td>91.5</td><td>96.6</td><td>98.7</td><td>67.5</td><td></td></tr><tr><td rowspan="2">Retrieval Method</td><td colspan="4">Multi Choice</td><td colspan="3">NLI</td><td colspan="2">Other</td><td rowspan="2">Overall</td><td></td></tr><tr><td>COPA</td><td>Cosmos QA</td><td>ComE</td><td>ComV</td><td>MNLI</td><td>SNLI</td><td>RTE</td><td>Subj</td><td>COLA</td><td></td></tr><tr><td>Random</td><td>71.6</td><td>26.2</td><td>41.4</td><td>50.5</td><td>34.1</td><td>33.0</td><td>55.6</td><td>60.0</td><td>52.8</td><td>47.3</td><td></td></tr><tr><td>BM25</td><td>71.2</td><td>27.1</td><td>41.4</td><td>50.9</td><td>35.3</td><td>41.5</td><td>50.5</td><td>78.8</td><td>53.3</td><td>57.7</td><td></td></tr><tr><td>SBERT</td><td>72.4</td><td>27.3</td><td>41.1</td><td>50.3</td><td>38.0</td><td>42.0</td><td>49.8</td><td>88.7</td><td>56.3</td><td>61.6</td><td></td></tr><tr><td>Instructor</td><td>71.6</td><td>27.1</td><td>41.9</td><td>49.9</td><td>41.3</td><td>46.7</td><td>52.7</td><td>84.3</td><td>56.0</td><td>63.2</td><td></td></tr><tr><td>EPR</td><td>73.2</td><td>28.4</td><td>43.0</td><td>50.4</td><td>54.3</td><td>74.0</td><td>55.6</td><td>92.1</td><td>70.3</td><td>68.8</td><td></td></tr><tr><td>UDR</td><td>72.8</td><td>29.9</td><td>45.6</td><td>63.9</td><td>73.8</td><td>83.6</td><td>65.3</td><td>95.0</td><td>78.9</td><td>73.2</td><td></td></tr></table>

Table 1: Main results on classification and multi-choice tasks.  

<table><tr><td rowspan="2">Retrieval Method</td><td colspan="4">Semantic Parsing</td><td colspan="3">Text Summarizaiton</td><td>CommonGen</td><td colspan="2">Story Generation</td></tr><tr><td>BREAK</td><td>MTOP</td><td>SMCalFlow</td><td>CNN/DM</td><td>PubMed</td><td>Reddit</td><td>CommonGen</td><td>Roc Story</td><td>Roc Ending</td><td></td></tr><tr><td>Random</td><td>1.9</td><td>6.6</td><td>8.7</td><td>20.8</td><td>23.6</td><td>15.6</td><td>21.1</td><td>9.3</td><td>13.4</td><td></td></tr><tr><td>BM25</td><td>26.0</td><td>52.9</td><td>46.1</td><td>18.6</td><td>24.5</td><td>15.3</td><td>26.0</td><td>12.3</td><td>19.2</td><td></td></tr><tr><td>SBERT</td><td>22.4</td><td>48.6</td><td>43.1</td><td>19.2</td><td>25.2</td><td>15.4</td><td>25.7</td><td>12.2</td><td>19.1</td><td></td></tr><tr><td>Instructor</td><td>22.7</td><td>50.5</td><td>46.3</td><td>19.0</td><td>24.8</td><td>15.3</td><td>26.5</td><td>12.4</td><td>21.8</td><td></td></tr><tr><td>DR-Target</td><td>22.1</td><td>49.6</td><td>41.6</td><td>19.4</td><td>24.6</td><td>16.0</td><td>24.5</td><td>11.9</td><td>20.1</td><td></td></tr><tr><td>EPR</td><td>31.9</td><td>64.4</td><td>54.3</td><td>20.3</td><td>24.8</td><td>15.5</td><td>25.3</td><td>12.9</td><td>21.2</td><td></td></tr><tr><td>UDR</td><td>35.2</td><td>66.8</td><td>60.4</td><td>21.2</td><td>26.1</td><td>16.2</td><td>27.1</td><td>17.6</td><td>24.7</td><td></td></tr><tr><td rowspan="2">Retrieval Method</td><td colspan="4">Code Summarization</td><td colspan="3">Text Simplification</td><td colspan="2">Data to Text</td><td rowspan="2">Overall</td></tr><tr><td>Go</td><td>Python</td><td>Java</td><td>PHP</td><td>WikiAuto</td><td>Turk</td><td>ASSET</td><td>DART</td><td>E2E</td></tr><tr><td>Random</td><td>27.3</td><td>7.9</td><td>6.7</td><td>18.9</td><td>8.3</td><td>28.0</td><td>24.8</td><td>20.4</td><td>21.9</td><td>15.8</td></tr><tr><td>BM25</td><td>30.4</td><td>9.7</td><td>11.7</td><td>23.6</td><td>10.2</td><td>29.1</td><td>26.6</td><td>28.4</td><td>29.2</td><td>24.2</td></tr><tr><td>SBERT</td><td>28.3</td><td>13.7</td><td>15.1</td><td>22.0</td><td>9.5</td><td>29.1</td><td>26.7</td><td>27.9</td><td>24.2</td><td>23.7</td></tr><tr><td>Instructor</td><td>29.9</td><td>11.5</td><td>13.1</td><td>24.0</td><td>11.3</td><td>29.0</td><td>26.3</td><td>28.7</td><td>22.4</td><td>24.2</td></tr><tr><td>DR-Target</td><td>28.1</td><td>12.2</td><td>13.0</td><td>24.2</td><td>10.8</td><td>29.4</td><td>26.7</td><td>30.1</td><td>24.7</td><td>23.8</td></tr><tr><td>EPR</td><td>30.5</td><td>17.4</td><td>17.4</td><td>30.2</td><td>13.3</td><td>30.8</td><td>27.6</td><td>31.8</td><td>29.3</td><td>27.7</td></tr><tr><td>UDR</td><td>29.4</td><td>22.3</td><td>25.2</td><td>33.2</td><td>19.5</td><td>32.9</td><td>32.1</td><td>34.5</td><td>32.6</td><td>30.9</td></tr></table>

Table 2: Main results on generation tasks.

of 10 and 6.7 points for classification and generation tasks respectively with less training data. This straightly demonstrates that our proposed training framework can help UDR incorporate LM's feedback through a unified ranking formulation and better retrieve informative demonstrations.

Additionally, we find the random baseline shows the worst performance on most tasks and this reflects the necessity to retrieve high-quality relevant demonstrations. Meanwhile, EPR and UDR have better performance than other methods, which reflects the importance of LM's feedback. Among these datasets, we notice a different trend on text summarization datasets like CNN/DailyMail and Reddit, on which these methods have similar performance. We conjecture that the LM can already have the knowledge of summarization since there are a lot of “[Article, TL;DR, Abstract]” texts in its pre

training corpus (Radford et al., 2018), thus random demonstrations can well activate LM's summarization ability without example-specific information.

# 3.3 Analysis

# 3.3.1 Ablation Study

To evaluate the effect of UDR's each component, we conduct ablation study on SMCalFlow, SST-2 and Java code summarization, shown in Table 3. When removing list-wise ranking training, we use EPR's training strategy (Rubin et al., 2022). We can see that removing task instructions cause slight performance degradation, which indicates that they can help UDR distinguish examples from various tasks and thus get better task-specific features. Meanwhile, we can see that UDR has a slightly better performance than the single-task counterpart on SST-2 and Java. We suppose that is because

<table><tr><td></td><td>SMCalFlow</td><td>SST-2</td><td>Java</td><td>Avg</td></tr><tr><td>UDR</td><td>60.8</td><td>91.3</td><td>23.2</td><td>58.4</td></tr><tr><td>- w/o Task Prompt</td><td>60.1</td><td>90.8</td><td>21.9</td><td>57.6</td></tr><tr><td>- w/o MultiTask</td><td>60.9</td><td>91</td><td>22.9</td><td>58.3</td></tr><tr><td>- w/o Rank Loss</td><td>56.7</td><td>89.2</td><td>21.1</td><td>55.7</td></tr><tr><td>- w/o Self-Guided</td><td>59.5</td><td>90.2</td><td>19.7</td><td>56.5</td></tr></table>

Table 3: Ablation study of UDR's each component.  

<table><tr><td>Dataset</td><td colspan="3">SMCalFlow</td><td colspan="3">E2E</td></tr><tr><td>LMs / Methods</td><td>BM25</td><td>EPR</td><td>UDR</td><td>BM25</td><td>EPR</td><td>UDR</td></tr><tr><td>Text-Davinci-003</td><td>55.0</td><td>58.9</td><td>64.7</td><td>31.3</td><td>31.5</td><td>34.3</td></tr><tr><td>Code-Davinci-002</td><td>50.9</td><td>55.2</td><td>62.9</td><td>23.5</td><td>24.4</td><td>26.4</td></tr><tr><td>GPT-J</td><td>49.0</td><td>55.9</td><td>64.0</td><td>33.3</td><td>33.7</td><td>35.0</td></tr><tr><td>GPT-Neo-1.3B</td><td>44.8</td><td>52.9</td><td>59.5</td><td>29.9</td><td>29.7</td><td>31.9</td></tr><tr><td>GPT-Neo-2.7B</td><td>46.5</td><td>53.7</td><td>62.2</td><td>29.2</td><td>29.1</td><td>32.6</td></tr></table>

there are several relevant tasks in UDR's training tasks and our multi-task ranking unification can help UDR fully share these tasks' knowledge. The performance of single-task UDR still outperforms EPR significantly and this straightly reflects that our training components, i.e., list-wise ranking formulation and iterative candidate mining strategy, can 1. help UDR better incorporate LM's feedback than EPR 2. serve as a competitive universal training method for a task-specific retriever. Removing list-wise ranking training and iterative candidate mining both cause performance degradation, which straightly indicates their effectiveness.

# 3.3.2 Transferability across Different LMs

In this section, we evaluate UDR's transferability across different inference LMs on SMCalFlow and E2E. Specifically, we compare BM25, EPR and UDR on inference LMs with different sizes, including: GPT-Neo-1.3B (Black et al., 2021), GPT-J (6B) (Wang and Komatsuzaki, 2021), Code-Davinci-002 (175B) (Chen et al., 2021) and Text-Davinci-003 (175B) (Brown et al., 2020; Ouyang et al., 2022) and we show the result in Table 4. When comparing UDR with baselines, the trends are similar with using GPT-Neo-2.7B (the scoring LM) as inference LM. UDR outperforms BM25 and EPR significantly and it shows UDR's strong transferability across different inference LMs. Meanwhile, we find that UDR with larger inference LM can improve performance such as Text-Davinci-003 on SMCalFlow and GPT-J on E2E, which shows UDR's potential utility in the

Table 4: Results on 1000 randomly sampled test examples across different inference LMs.  

<table><tr><td></td><td>Twitter</td><td>QNLI</td><td>Ruby</td><td>JavaScript</td></tr><tr><td>BM25</td><td>50.0</td><td>54.1</td><td>9.2</td><td>12.7</td></tr><tr><td>SBERT</td><td>51.6</td><td>53.7</td><td>8.7</td><td>15.9</td></tr><tr><td>UDR</td><td>56.8</td><td>74.4</td><td>19.6</td><td>21.6</td></tr></table>

Table 5: The performance of UDR on unseen datasets.

future where more competitive large-scale LM is built. When we demonstrate the example-specific demonstration transferability across different inference LMs in this paper, Li and Qiu (2023a) show that task-level demonstrations also exhibit such transferability. We leave the analysis of the transferability of ICL's demonstrations across different LMs as future work.

# 3.3.3 Performance on Unseen Datasets

In this section we explore UDR's zero-shot transferability and evaluate it on unseen datasets including: 1. Twitter sentiment classification (Naji, 2012) 2. question-answering NLI (QNLI) (Wang et al., 2019a) 3. Ruby and JavaScript code summarization (Lu et al., 2021). These domains or programming languages (Twitter, NLI on QA, Ruby and Javascript) are never seen during UDR's training and thus can straightly reflect UDR's zero-shot transferability. We compare UDR with two powerful universal retrievers, BM25 and SBERT, and show the result in Table 5. We can see UDR significantly outperforms BM25 and SBERT on these unseen datasets by about 10 points on average, which shows that the learned ranking knowledge inside UDR can be well transferred and generalized to unseen datasets.

# 3.3.4 The Order of Demonstrations

Previous work (Lu et al., 2022) has revealed that ICL is sensitive to demonstrations' order when using random examples. Specifically, the same randomly sampled demonstrations with different orders can lead to the performance between random guess and near state-of-the-art. Here we explore the effect of ordering on example-specific demonstrations retrieved by UDR. We compare 3 demonstrations' orders: 1. random, for this setting, we run experiments with 10 different random seeds and report the best and worst performance. 2. descending sorted by UDR's score, i.e., the demonstration which has the highest similarity with  $x_{test}$  is put at the beginning of LM's input. 3. ascending sorted by UDR's score, opposite to "2". The result is shown in Table 6. We

<table><tr><td></td><td>SST-2</td><td>TREC</td><td>Reddit</td><td>CommonGen</td></tr><tr><td>Random-OrderBest</td><td>92.5</td><td>96.6</td><td>16.8</td><td>27.5</td></tr><tr><td>Random-OrderWorst</td><td>92.0</td><td>96.2</td><td>16.2</td><td>26.6</td></tr><tr><td>Descending-Order</td><td>92.2</td><td>96.6</td><td>16.2</td><td>27.0</td></tr><tr><td>Ascending-Order</td><td>92.4</td><td>96.6</td><td>16.3</td><td>27.3</td></tr></table>

Table 6: The effect of different demonstration orders.

observe a different phenomenon from that in previous work (Lu et al., 2022). In general, The performance of UDR's demonstrations with different orders is more stable than previously investigated random examples. Across these tasks, different orders' performance gap is within 1 point, and it is far less than the performance fluctuation of up to tens points when using random examples (Lu et al., 2022). This indicates that high-quality demonstrations are less sensitive to the ordering and stabilize in-context learning, which is consistent with the analysis in previous work (Chen et al., 2022; Li and Qiu, 2023a).

# 3.3.5 The Impact of Demonstration Quantity

We compare UDR with BM25 and EPR under different amounts of demonstrations on two classification tasks: Yelp and RTE, and two generation tasks: WikiAuto and Java code summarization. We show results in Figure 3. We can see that UDR outperforms baselines consistently across varying amounts of demonstrations. Meanwhile, we can draw two conclusions from the results: 1. The number of demonstrations has a greater impact on generation tasks than classification tasks. Specifically, as the number of demonstrations increases, generation tasks' performance gets significant improvements while classification tasks' has slight or no improvements. 2. The quality of demonstrations can be more important than their quantity. In detail, UDR with the quota of 2 demonstrations still outperforms BM25 and EPR with 8 demonstrations. This also reflects the strong demonstration retrieval ability of UDR. Li and Qiu (2023b) observe the similar trends in the CoT-retrieval scenario, indicating that the relevance of the used reasoning paths is more important than their quantity.

# 4 Related Work

In this section, we introduce previous demonstration retrievers for in-context learning, and explain the difference between UDR and them. In general, there are two kinds of demonstration retrievers for ICL. One is to leverage off-the-shelf retrievers. For example, Liu et al. (2022) propose to use a fine-tuned BERT to encode examples and use a





Figure 3: The effect of demonstration quantity.

KNN-based method to retrieve semantically similar demonstrations to improve ICL. Agrawal et al. (2022) use BM25 to retrieve demonstrations for machine translation. Compared with them, UDR incorporates various tasks' supervision by unified LM's feedback and thus can better retrieve informative demonstrations. Another approach is to train a task-specific retriever by a designed task-specific signal. Das et al. (2021) explore demonstration retrieval for knowledge-based question answering and define the F1 score of logic forms as soft-label to train the retriever. Poesia et al. (2022) train a demonstration retriever for code generation, based on the edit distance of abstract syntax trees. Hu et al. (2022) define the similarity between dialogue states, and use it to train a demonstration retriever for dialogue state tracking. Rubin et al. (2022) propose Efficient Prompt Retriever (EPR) for semantic parsing, which is to use the language model to score examples, assign positive and negative labels for them and use DPR (Karpukhin et al., 2020) to train a demonstration retriever. Shi et al. (2022) explore demonstration retrieval for cross-lingual semantic parsing using a similar example scoring method with EPR. These task-specific methods serve for each task separately and are hard to transfer and scale on various tasks. For other tasks, it requires to redesign the similarity function or training signal. Compared with them, we introduce a unified training framework based on list-wise ranking and propose a single multi-task retriever UDR to serve for a wide range of tasks. Compared with EPR, besides UDR's versatility on various tasks, UDR can incorporate LM's feedback by ranking-based training in a more fine-grained way and receive more

crucial candidates' signals by the iterative mining strategy. Cheng et al. (2023) propose CLAIF to enhance the sentence embedder by the gigantic language model's feedback. Specifically, they use GPT-3 (Brown et al., 2020) to generate the data of sentence pairs and then score them by the output of GPT-3, which depends on the strong natural language understanding ability of GPT-3. Different from them, we leverage the conditional probability to measure the helpfulness of an example, which only needs a small language model, and is more efficient and environmental-friendly. Recently, Li and Qiu (2023b) propose MoT (Memory-of-Thought) to let the LLM self-improve in two stages: 1. Before test stage, the LLM generate reasoning paths and answers on an unlabeled dataset for itself, 2. At test stage, the LLM retrieves relevant reasoning paths (memory) to help itself answer the given test question. While MoT focuses on the scenario with unlabeled dataset and uses the LLM for retrieval, we train a small retriever by a LM's feedback from tasks' supervision and thus the proposed method is more lightweight. We leave demonstration retrieval with reasoning paths or unlabeled datasets as future work.

# 5 Conclusion

In this paper, we propose UDR, a single multi-task model for a wide range of tasks' demonstration retrieval. To train UDR, we cast various tasks' training into a unified list-wise ranking formulation by language model's feedback, and propose a multitask list-wise ranking training framework, with an iterative mining strategy to find high-quality candidates. Experiments on  $30+$  tasks show that UDR significantly outperforms baselines. Further analyses show the effectiveness of each proposed component and UDR's strong ability in various scenarios including different LMs  $(1.3\mathrm{B}\sim 175\mathrm{B})$  ,unseen datasets, varying demonstration quantities, etc.

# Limitations

We illustrate this paper's limitations from the following three aspects:

1) Limited by the computational resources, we only train UDR from the initialization of "BERT base uncased" following EPR (Rubin et al., 2022). We regard explorations based on other competitive pre-trained models like RoBERTa (Liu et al., 2019) and DeBERTa (He et al., 2021) as future work.

2) Most of current dense demonstration retrieval

ers, including UDR, are black-box models. Although they lead to significantly better performance than BM25, how they find informative demonstrations is still unknown. Therefore, a better understanding of the principle of informative demonstration's retrieval or an interpretable and transparent demonstration retriever may be the next stage of improving demonstration retrieval. Xu et al. (2023) propose a more explainable method, beyond-context learning, which first uses the language model to get training data's next word probability distribution, then assigns test instances with labels of their nearest neighbors with similar next word's probability distribution. We leave demonstration retrieval with better explainability as future work.

3) In the training stage we use LM to score candidates separately but in the inference stage LM is provided with a sequence of demonstrations. Although experimental results demonstrate UDR's effectiveness, we think it is a promising direction to model the dependence between different demonstrations and leave it to future work.

# 6 Acknowledgements

This work was supported by the National Natural Science Foundation of China (No. 62236004 and No. 62022027) and Shenzhen City's Science and Technology Plan Project (No. JSGG20210802153806021).

# References

Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. 2022. Incontext examples selection for machine translation. CoRR, abs/2212.02437.  
Reinald Kim Amplayo, Arthur Brazinskas, Yoshi Suhara, Xiaolan Wang, and Bing Liu. 2022. Beyond opinion mining: Summarizing opinions of customer reviews. In SIGIR '22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022, pages 3447-3450. ACM.  
Jacob Andreas, John Bufe, David Burkett, Charles Chen, Josh Clausman, Jean Crawford, Kate Crim, Jordan DeLoach, Leah Dorner, Jason Eisner, Hao Fang, Alan Guo, David Hall, Kristin Hayes, Kellie Hill, Diana Ho, Wendy Iwaszuk, Smriti Jha, Dan Klein, Jayant Krishnamurthy, Theo Lanman, Percy Liang, Christopher H. Lin, Ilya Lintsbakh, Andy McGovern, Aleksandr Nisnevich, Adam Pauls, Dmitrij Petters, Brent Read, Dan Roth, Subhro Roy, Jesse Rusak, Beth Short, Div Slomin, Ben Snyder,

Stephon Striplin, Yu Su, Zachary Tellman, Sam Thomson, Andrei Vorobev, Izabela Witoszko, Jason Andrew Wolfe, Abby Wray, Yuchen Zhang, and Alexander Zotov. 2020. Task-oriented dialogue as dataflow synthesis. Trans. Assoc. Comput. Linguistics, 8:556-571.  
Roy Bar-Haim, Ido Dagan, and Idan Szpektor. 2014. Benchmarking applied semantic inference: The PASCAL recognising textual entailment challenges. In Language, Culture, Computation. Computing - Theory and Technology - Essays Dedicated to Yaacov Choueka on the Occasion of His 75th Birthday, Part I, volume 8001 of Lecture Notes in Computer Science, pages 409-424. Springer.  
Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In *FAccT'21: 2021 ACM Conference on Fairness, Accountability, and Transparency*, Virtual Event / Toronto, Canada, March 3-10, 2021, pages 610-623. ACM.  
Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow.  
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, Lisbon, Portugal, September 17-21, 2015, pages 632-642. The Association for Computational Linguistics.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.  
Christopher J. C. Burges. 2010. From RankNet to LambdaRank to LambdaMART: An overview. Technical report, Microsoft Research.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power,

Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. CoRR, abs/2107.03374.  
Yanda Chen, Chen Zhao, Zhou Yu, Kathleen R. McKeown, and He He. 2022. On the relation between sensitivity and accuracy in in-context learning. CoRR, abs/2209.07661.  
Qinyuan Cheng, Xiaogui Yang, Tianxiang Sun, Linyang Li, and Xipeng Qiu. 2023. Improving contrastive learning of sentence embeddings from AI feedback. CoRR, abs/2305.01918.  
Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. 2018. A discourse-aware attention model for abstractive summarization of long documents. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 615-621, New Orleans, Louisiana. Association for Computational Linguistics.  
Alexis Conneau and Guillaume Lample. 2019. Crosslingual language model pretraining. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 7057-7067.  
Rajarshi Das, Manzil Zaheer, Dung Thai, Ameya Godbole, Ethan Perez, Jay Yoon Lee, Lizhen Tan, Lazaros Polymenakos, and Andrew McCallum. 2021. Case-based reasoning for natural language queries over knowledge bases. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 9594-9611. Association for Computational Linguistics.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171-4186. Association for Computational Linguistics.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. 2022. A survey for in-context learning.  
Ondrej Dusek, David M Howcroft, and Verena Rieser. 2019. Semantic Noise Matters for Neural Natural Language Generation. In Proceedings of the 12th International Conference on Natural Language Generation (INLG 2019), pages 421-426, Tokyo, Japan.  
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. Deberta: decoding-enhanced bert with disentangled attention. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.  
Karl Moritz Hermann, Tomás Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pages 1693-1701.  
Yushi Hu, Chia-Hsuan Lee, Tianbao Xie, Tao Yu, Noah A. Smith, and Mari Ostendorf. 2022. Incontext learning for few-shot dialogue state tracking. CoRR, abs/2203.08568.  
Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2019. Cosmos QA: Machine reading comprehension with contextual commonsense reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2391-2401, Hong Kong, China. Association for Computational Linguistics.  
Chao Jiang, Mounica Maddela, Wuwei Lan, Yang Zhong, and Wei Xu. 2020. Neural CRF model for sentence alignment in text simplification. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7943-7960, Online. Association for Computational Linguistics.  
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-scale similarity search with gpus. IEEE Trans. Big Data, 7(3):535-547.  
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 6769-6781. Association for Computational Linguistics.  
Byeongchang Kim, Hyunwoo Kim, and Gunhee Kim. 2019. Abstractive summarization of Reddit posts

with multi-level memory networks. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2519-2531, Minneapolis, Minnesota. Association for Computational Linguistics.  
Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef, Soren Auer, and Christian Bizer. 2015. Dbpedia - A large-scale, multilingual knowledge base extracted from wikipedia. Semantic Web, 6(2):167-195.  
Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. 2021. MTOP: A comprehensive multilingual task-oriented semantic parsing benchmark. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2950-2962, Online. Association for Computational Linguistics.  
Xiaonan Li and Xipeng Qiu. 2023a. Finding supporting examples for in-context learning. CoRR, abs/2302.13539.  
Xiaonan Li and Xipeng Qiu. 2023b. Mot: Pre-thinking and recalling enable chatgpt to self-improve with memory-of-thoughts.  
Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. 2020. CommonGen: A constrained text generation challenge for generative commonsense reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1823-1840, Online. Association for Computational Linguistics.  
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What makes good in-context examples for gpt-3? In Proceedings of Deep Learning Inside Out: The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, DeeLIO@ACL 2022, Dublin, Ireland and Online, May 27, 2022, pages 100–114. Association for Computational Linguistics.  
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. CoRR, abs/1907.11692.  
Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.  
Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin B. Clement,

Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Li-dong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. 2021. Codexglue: A machine learning benchmark dataset for code understanding and generation. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.  
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086-8098, Dublin, Ireland. Association for Computational Linguistics.  
Julian J. McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: understanding rating dimensions with review text. In Seventh ACM Conference on Recommender Systems, RecSys '13, Hong Kong, China, October 12-16, 2013, pages 165-172. ACM.  
Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Noisy channel language model prompting for few-shot text classification. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 5316-5330. Association for Computational Linguistics.  
Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016. A corpus and cloze evaluation for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 839-849, San Diego, California. Association for Computational Linguistics.  
Ibrahim Naji. 2012. TSATC: Twitter Sentiment Analysis Training Corpus. In thinknook.  
Linyong Nan, Dragomir Radev, Rui Zhang, Amrit Rau, Abhinand Sivaprasad, Chiachun Hsieh, Xiangru Tang, Aadit Vyas, Neha Verma, Pranav Krishna, Yangxiaokang Liu, Nadia Irwanto, Jessica Pan, Faiaz Rahman, Ahmad Zaidi, Mutethia M tuma, Yasin Tarabar, Ankit Gupta, Tao Yu, Yi Chern Tan, Xi Victoria Lin, Caiming Xiong, Richard Socher, and Nazneen Fatema Rajani. 2021. DART: Open-domain structured data record to text generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 432-447, Online. Association for Computational Linguistics.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. CoRR, abs/2203.02155.  
Bo Pang and Lillian Lee. 2004. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics, 21-26 July, 2004, Barcelona, Spain, pages 271-278. ACL.  
Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL 2005, 43rd Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, 25-30 June 2005, University of Michigan, USA, pages 115-124. The Association for Computer Linguistics.  
Gabriel Poesia, Alex Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, and Sumit Gulwani. 2022. Synchronesh: Reliable code generation from pre-trained language models. In *The Tenth International Conference on Learning Representations*, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.  
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2018. Language models are unsupervised multitask learners.  
Nils Reimers and Iryna Gurevych. 2019a. SentenceBERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982-3992, Hong Kong, China. Association for Computational Linguistics.  
Nils Reimers and Iryna Gurevych. 2019b. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 3980-3990. Association for Computational Linguistics.  
Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr., 3(4):333-389.  
Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. 2011. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI Spring Symposium Series.

Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning to retrieve prompts for in-context learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 2655-2671. Association for Computational Linguistics.  
Peng Shi, Rui Zhang, He Bai, and Jimmy Lin. 2022. XRICL: cross-lingual retrieval-augmented in-context learning for cross-lingual text-to-sql semantic parsing. CoRR, abs/2210.13693.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt Seattle, Seattle, Washington, USA, A meeting of SIGDAT, a Special Interest Group of the ACL, pages 1631-1642. ACL.  
Hongjin Su, Weijia Shi, Jungo Kasai, Yushi Hu Yizhong Wang, Mari Ostendorf, Wen tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2022. One embedder, any task: Instruction-finetuned text embeddings.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.  
Ellen M. Voorhees and Dawn M. Tice. 2000. Building a question answering test collection. In SIGIR 2000: Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, July 24-28, 2000, Athens, Greece, pages 200-207. ACM.  
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019a. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.  
Ben Wang and Aran Komatsuzaki. 2021. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax.  
Cunxiang Wang, Shuailong Liang, Yue Zhang, Xiaonan Li, and Tian Gao. 2019b. Does it make sense? and why? A pilot study for sense making and explanation. In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4020-4026. Association for Computational Linguistics.

Adina Williams, Nikita Nangia, and Samuel R. Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers), pages 1112-1122. Association for Computational Linguistics.  
Tomer Wolfson, Mor Geva, Ankit Gupta, Yoav Goldberg, Matt Gardner, Daniel Deutch, and Jonathan Berant. 2020. Break it down: A question understanding benchmark. Trans. Assoc. Comput. Linguistics, 8:183-198.  
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.  
Benfeng Xu, Quan Wang, Zhendong Mao, Yajuan Lyu, Qiaoqiao She, and Yongdong Zhang. 2023. knn prompting: Beyond-context learning with calibration-free nearest neighbor inference.  
Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pages 649-657.  
Yiming Zhang, Shi Feng, and Chenhao Tan. 2022. Active example selection for in-context learning. CoRR, abs/2211.04486.

<table><tr><td>Template</td><td>MR</td><td>Yahoo</td><td>Subj</td></tr><tr><td>Original Template</td><td>85.2</td><td>67.5</td><td>95.0</td></tr><tr><td>Template 1</td><td>85.1</td><td>67.8</td><td>94.8</td></tr><tr><td>Template 2</td><td>85.7</td><td>67.1</td><td>94.8</td></tr><tr><td>Template 3</td><td>85.4</td><td>67.2</td><td>95.2</td></tr></table>

Table 7: UDR's performance under different inference templates. For MR, the original template and template 1, 2, 3 are "It was [Verbalizer]", "A [Verbalizer] One", "All in all [Verbalizer] .", "A [Verbalizer] one .", respectively. The verbalizers are ["great", "terrible"]. For Yahoo, the original template and template 1, 2, 3 are "Topic: [Verbalizer]", "Subject: [Verbalizer]", "This is about [Verbalizer] .", "It is about [Verbalizer] .", respectively. The verbalizers are ["Society & Culture", "Science & Mathematics, . . . ]. For Subj, the original template and template 1, 2, 3 are "It's [verbalizer] .", "This is [Verbalizer]", "It's all [Verbalizer] .", "Is it [Verbalizer] ?", respectively. The Verbalizers are ["subjective", "objective"]. These templates are from previous works (Min et al., 2022) and for more details please refer to Table 11.

# A Task Overview

We show each task's 1. input/output domain 2. statistics and evaluation metric 3. instruction, inference template and example cases in Table 9, 10 and 11, respectively. For the dataset which has publicly available test data, we use the test data for evaluation, like SST-2, SST-5, MTOP, etc. For the others like BREAK and SMCalFlow, we follow previous work (Rubin et al., 2022) and use the dev data for evaluation. For training efficiency, we manually limit the training examples of UDR. Specifically, for the classification task whose training set size is  $>30000$ , we randomly sample a 30000 subset for UDR's training. For the generation task whose training set size is  $>100000$ , we randomly sample a 100000 subset for UDR's training. In the pilot experiment, we find such a strategy will not cause significant performance degradation. At the inference stage, we use the full training set as demonstrations' pool. Restricted by computational resources, we randomly sample a test set of 3000 samples for evaluation on these tasks: Amazon, Yelp, AGNews DBPedia and Yahoo.

# B Implementation Details and Hyper-Parameters

We follow Rubin et al. (2022) to use GPT-Neo-2.7B (Black et al., 2021) as the scoring LM and the inference LM for most experiments in the paper unless otherwise specified. Following EPR (Rubin

<table><tr><td colspan="2">Hyper-parameters</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Warmup Steps</td><td>500</td></tr><tr><td>Learning Rate</td><td>1e-4</td></tr><tr><td>Batch Size</td><td>128</td></tr><tr><td>Loss Weight</td><td>0.8</td></tr><tr><td>Iteration Number</td><td>3</td></tr><tr><td>Scoring Candidates Num (K)</td><td>50</td></tr><tr><td>Training Candidates Num (l)</td><td>8</td></tr></table>

Table 8: Hyper-parameters.

et al., 2022) and DPR (Karpukhin et al., 2020), we initialize  $E_{q}$  and  $E_{d}$  as two separate "BERT base uncased" encoders (Devlin et al., 2019). Thus the total number of parameters of UDR is about 220M. We use 8 NVIDIA A100s-80GB to train UDR for up to 30 epochs before iteratively mining candidates. And then we train UDR for 10 epochs at each iteration. The whole training pipeline including scoring candidates takes about 8 days. In the pilot experiment, we select the number of training epochs through the average performance on validation set on single-task SST-2, TREC, MTOP, Java code summarization, WikiAuto and DART. We set the number of iterations as 3. We follow EPR (Rubin et al., 2022) to set learning rate and batch size as 1e-4 and 128 and we use AdamW (Loshchilov and Hutter, 2019) as the optimizer. We list the overall hyper-parameters in Table 8. On each task, we use one specific template for scoring and inference (see Table 11). For fair comparison, we train DR-Target, EPR and UDR under the same hyperparameter and report their average performance under three random seeds.

The initialization of UDR's candidates For classification and multi-choice tasks, we initialize candidates as those examples that have similar input with  $x$  by BM25. For generation tasks, similarly, we initialize candidates as those of similar targets with  $x$ 's, inspired by previous work (Rubin et al., 2022).

The Quantity of EPR's Candidates Since UDR's training needs to score iteratively mined candidates and thus has to score more candidates than EPR, we also run experiments on EPR with the same candidate quantities of UDR. But we find increasing the candidates of EPR instead slightly hurts its overall performance, which is consistent with its original paper (Rubin et al., 2022). Thus for EPR, we use the same number of candidates as its original paper.

# C Performance across varying inference templates

For UDR, we use one specific template when scoring candidates and here we evaluate UDR's transferability across different inference templates on MR, Yahoo and Subj. The results are shown in Table 7. We can see that the performance gap across various inference templates is smaller than 1 point and this reflects UDR's stability and transferability across different inference templates.

# D Potential Risk

Previous works have shown Large language models can have various kinds of bias (Bender et al., 2021). Since UDR is trained from the feedback of large language models, it can also contain such bias.

<table><tr><td>Task Family</td><td>Task</td><td>Input</td><td>Output</td></tr><tr><td rowspan="6">Sentiment Classification</td><td>SST-2</td><td>Short Movie Review</td><td>Sentiment Label</td></tr><tr><td>SST-5</td><td>Short Movie Review</td><td>Sentiment Label</td></tr><tr><td>Amazon</td><td>Amazon Product Review</td><td>Sentiment Label</td></tr><tr><td>Yelp</td><td>Yelp Review</td><td>Sentiment Label</td></tr><tr><td>MR</td><td>Movie Review</td><td>Sentiment Label</td></tr><tr><td>CR</td><td>Electronics Review</td><td>Sentiment Label</td></tr><tr><td rowspan="4">Topic Classification</td><td>AGNews</td><td>News Article</td><td>Topic Label</td></tr><tr><td>TREC</td><td>Question</td><td>Topic Label</td></tr><tr><td>DBPedia</td><td>Wikipedia Text</td><td>Topic Label</td></tr><tr><td>Yahoo</td><td>Question-answer Pair</td><td>Topic Label</td></tr><tr><td rowspan="4">Multi-Choice</td><td>COPA</td><td>Causal Reasoning Question</td><td>Effect/Cause</td></tr><tr><td>Cosmos QA</td><td>Causal Reasoning Question</td><td>Effect/Cause</td></tr><tr><td>ComV</td><td>Commonsense Hypotheses</td><td>Wrong Hypothesis</td></tr><tr><td>ComE</td><td>Wrong Hypothesis</td><td>Explanation</td></tr><tr><td rowspan="3">NLI</td><td>MNLI</td><td>Image-caption Sentence Pair</td><td>Entailment Label</td></tr><tr><td>SNLI</td><td>Cross-genre Sentence Pair</td><td>Entailment Label</td></tr><tr><td>RTE</td><td>Wikipedia/News Sentence Pair</td><td>Entailment Label</td></tr><tr><td>Subjective Classification</td><td>Subj</td><td>Movie Review</td><td>Subjectivity</td></tr><tr><td>Linguistic Acceptability</td><td>COLA</td><td>Linguistics Publication Sentence</td><td>Grammatical Label</td></tr><tr><td rowspan="3">Semantic Parsing</td><td>BREAK</td><td>Question</td><td>Question Decomposition</td></tr><tr><td>MTOP</td><td>User Utterance</td><td>TOP Representation</td></tr><tr><td>SMCalFlow</td><td>User Utterance</td><td>Dataflow Program</td></tr><tr><td rowspan="3">Text Summarization</td><td>CNN/DailyMail</td><td>News Article</td><td>Highlights</td></tr><tr><td>PubMed</td><td>Scientific Paper&#x27;s Introduction</td><td>Abstract</td></tr><tr><td>Reddit</td><td>Reddit Post</td><td>Summary</td></tr><tr><td>Commensense Generation</td><td>Commen Gen</td><td>Concepts</td><td>Coherent Sentence</td></tr><tr><td rowspan="2">Story Generation</td><td>Roc Story</td><td>Head of Story</td><td>Remaining Story</td></tr><tr><td>Roc Stroy Ending</td><td>Four-sentence Story</td><td>Story Ending</td></tr><tr><td rowspan="4">Code Summarization</td><td>Go</td><td>Go Code</td><td>Documentation</td></tr><tr><td>Python</td><td>Python Code</td><td>Documentation</td></tr><tr><td>Java</td><td>Java Code</td><td>Documentation</td></tr><tr><td>PHP</td><td>PHP Code</td><td>Documentation</td></tr><tr><td rowspan="3">Text Simplification</td><td>WikiAuto</td><td>Wikipedia Sentence</td><td>Simplified Sentence</td></tr><tr><td>WikiAuto-Turk</td><td>Wikipedia Sentence</td><td>Simplified Sentence</td></tr><tr><td>WikiAuto-ASSET</td><td>Wikipedia Sentence</td><td>Simplified Sentence</td></tr><tr><td rowspan="2">Data to Text</td><td>DART</td><td>Triple Set</td><td>Text</td></tr><tr><td>E2E</td><td>Key-value Pairs</td><td>Text</td></tr></table>

Table 9: The Input/Output Domains of Tasks.

<table><tr><td>Task Family</td><td>Task</td><td>Train</td><td>Dev</td><td>Test</td><td>Report Split</td><td>Metric</td></tr><tr><td rowspan="6">Sentiment Classification</td><td>SST-2</td><td>6911</td><td>873</td><td>1821</td><td>Test</td><td>Acc</td></tr><tr><td>SST-5</td><td>8534</td><td>1101</td><td>2210</td><td>Test</td><td>Acc</td></tr><tr><td>Amazon</td><td>30000</td><td>5000</td><td>3000</td><td>Test</td><td>Acc</td></tr><tr><td>Yelp</td><td>30000</td><td>-</td><td>3000</td><td>Test</td><td>Acc</td></tr><tr><td>MR</td><td>8662</td><td>-</td><td>2000</td><td>Test</td><td>Acc</td></tr><tr><td>CR</td><td>1772</td><td>-</td><td>1996</td><td>Test</td><td>Acc</td></tr><tr><td rowspan="4">Topic Classification</td><td>AGNews</td><td>29914</td><td>-</td><td>3000</td><td>Test</td><td>Acc</td></tr><tr><td>TREC</td><td>5381</td><td>-</td><td>500</td><td>Test</td><td>Acc</td></tr><tr><td>DBPedia</td><td>30000</td><td>-</td><td>3000</td><td>Test</td><td>Acc</td></tr><tr><td>Yahoo</td><td>29150</td><td>-</td><td>3000</td><td>Test</td><td>Acc</td></tr><tr><td rowspan="4">Multi-Choice</td><td>COPA</td><td>500</td><td>-</td><td>500</td><td>Test</td><td>Acc</td></tr><tr><td>Cosmos QA</td><td>18770</td><td>2603</td><td>6030</td><td>Dev</td><td>Acc</td></tr><tr><td>ComE</td><td>9996</td><td>997</td><td>1000</td><td>Test</td><td>Acc</td></tr><tr><td>ComV</td><td>9992</td><td>997</td><td>1000</td><td>Test</td><td>Acc</td></tr><tr><td rowspan="3">NLI</td><td>MNLI</td><td>263789</td><td>3000</td><td>9796</td><td>Dev</td><td>Acc</td></tr><tr><td>SNLI</td><td>131062</td><td>3272</td><td>3262</td><td>Test</td><td>Acc</td></tr><tr><td>RTE</td><td>2490</td><td>277</td><td>3000</td><td>Dev</td><td>Acc</td></tr><tr><td>Subjective Classification</td><td>Subj</td><td>8000</td><td>-</td><td>2000</td><td>Test</td><td>Acc</td></tr><tr><td>Linguistic Acceptability</td><td>COLA</td><td>8532</td><td>-</td><td>527</td><td>Test</td><td>Acc</td></tr><tr><td rowspan="3">Semantic Parsing</td><td>BREAK</td><td>44321</td><td>7760</td><td>8069</td><td>Dev</td><td>LF-EM</td></tr><tr><td>MTOP</td><td>15667</td><td>2235</td><td>4386</td><td>Test</td><td>EM</td></tr><tr><td>SMCalFlow</td><td>133584</td><td>14751</td><td>22012</td><td>Dev</td><td>EM</td></tr><tr><td rowspan="3">Text Summarization</td><td>CNN/DailyMail</td><td>155098</td><td>7512</td><td>6379</td><td>Test</td><td>Rouge-L</td></tr><tr><td>PubMed</td><td>56254</td><td>3187</td><td>3481</td><td>Test</td><td>Rouge-L</td></tr><tr><td>Reddit</td><td>37643</td><td>576</td><td>562</td><td>Test</td><td>Rouge-L</td></tr><tr><td>Commensense Generation</td><td>Commen Gen</td><td>67389</td><td>993</td><td>1497</td><td>Dev</td><td>BLEU-3</td></tr><tr><td rowspan="2">Story Generation</td><td>Roc Story</td><td>87526</td><td>9799</td><td>9799</td><td>Test</td><td>BLEU-1</td></tr><tr><td>Roc Stroy Ending</td><td>87906</td><td>9807</td><td>9807</td><td>Test</td><td>BLEU-1</td></tr><tr><td rowspan="4">Code Summarization</td><td>Go</td><td>167137</td><td>7320</td><td>8115</td><td>Test</td><td>BLEU-1</td></tr><tr><td>Python</td><td>250818</td><td>13841</td><td>14840</td><td>Test</td><td>BLEU-1</td></tr><tr><td>Java</td><td>164514</td><td>5172</td><td>10928</td><td>Test</td><td>BLEU-1</td></tr><tr><td>PHP</td><td>240851</td><td>12964</td><td>13998</td><td>Test</td><td>BLEU-1</td></tr><tr><td rowspan="3">Text Simplification</td><td>WikiAuto</td><td>481018</td><td>1999</td><td>403</td><td>Test</td><td>SARI</td></tr><tr><td>WikiAuto-Turk</td><td>-</td><td>1999</td><td>359</td><td>Test</td><td>SARI</td></tr><tr><td>WikiAuto-ASSET</td><td>-</td><td>1999</td><td>359</td><td>Test</td><td>SARI</td></tr><tr><td rowspan="2">Data to Text</td><td>DART</td><td>30123</td><td>2718</td><td>4159</td><td>Test</td><td>BLEU-4</td></tr><tr><td>E2E</td><td>12563</td><td>1483</td><td>1847</td><td>Test</td><td>BLEU-4</td></tr></table>

Table 10: The statistics, split and evaluation metrics of each dataset.

# Task Family: Sentiment Classification

Task:SST-2

Task Instruction: Sentiment of the sentence:

Inference Verbalizer: {great, terrible}

Inference Template:

Input:

A three-hour cinema master class.

It was terrible.

A pretensions - and disposable story - sink the movie.

It was great.

···

The movie's blatant derivativeness is one reason it's so lackluster.

It was

Output:

terrible.

Task:SST-5

Task Instruction: Sentiment of the sentence:

Inference Verbalizer: {great, good, okay, bad, terrible}

Inference Template: Same as SST-2

Task: Amazon

Task Instruction: Sentiment of the sentence:

Inference Verbalizer: {great, good, okay, bad, terrible}

Inference Template: Same as SST-2

Task: Yelp

Task Instruction: Sentiment of the sentence:

Inference Verbalizer: {great, good, okay, bad, terrible}

Inference Template: Same as SST-2

Task: MR

Task Instruction: Sentiment of the sentence:

Inference Verbalizer: {great, terrible}

Inference Template: Same as SST-2

Task: CR

Task Instruction: Sentiment of the sentence:

Inference Verbalizer: {great, terrible}

Inference Template: Same as SST-2

# Task Family: Topic Classification

Task: AGNews

Task Instruction: Topic of the text:

Inference Verbalizer: {World, Sports, Business, Technology}

Inference Template:

Input:

LONDON, Oct 26 (AFP) - World oil prices will be driven down over the next two years due to there being enough crude to meet soaring demand, Claude Mandil, executive director of the International Energy Agency (IEA), said here Tuesday.

Topic: Business.

WASHINGTON - This year's surge in energy prices is likely to have far less of an impact on the economy than the oil shocks of the 1970s, Federal Reserve Chairman Alan Greenspan said Friday. Greenspan predicted that the global economy will adjust to the recent surge in prices, which has seen oil topping  $\mathbb{W}\mathbb{S}50$  per barrel, by boosting energy exploration and production and by increasing fuel efficiency...

Topic: World.

.

Oil demand is rising faster than predicted this year as OPEC pumps more low-quality oil in a failed bid to reduce record prices, according to International Energy Agency, an adviser to 26 industrialized nations.

Topic:

Output:

Business.

Task: TREC

Task Instruction: Topic of the question:

Inference Verbalizer: {Description, Entity, Expression, Human, Location, Number}

Inference Template: Same as AGNews

Task: DBPedia

Task Instruction: Topic of the text:

Inference Verbalizer: {Company, Educational Institution, Artist, Athlete, Office Holder, Mean of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work}

Inference Template: Same as AGNews

Task: Yahoo

Task Instruction: Topic of the text:

Inference Verbalizer: {Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government}

Inference Template: Same as AGNews

Task Family: Multi-Choice

Task: COPA

Task Instruction: Answer the question based on the text.

Inference Template:

Input:

I scratched my skin. What happened as a result?

My itch went away.

misplaced my wallet. What happened as a result?

I retraced my steps.

···

I emptied my pockets. What happened as a result?

Output:

I retrieved a ticket stub.

Task: Cosmos QA

Task Instruction: Answer the question based on the text.

Inference Template: Same as COPA

Task: ComV

Task Instruction: Which statement of the two is against common sense?

Inference Template: Same as COPA

Task: ComE

Task Instruction: Select the most corresponding reason why this statement is against common sense.

Inference Template: Same as COPA

Task Family: NLI

Task: MNLI

Task Instruction: Recognizing textual entailment between these 2 texts.

Inference Verbalizer: {Entailment, Inconclusive, Contradiction}

Inference Template:

Input:

uh-huh exactly not what color you are how old you are what if your male or female that would be wonderful i guess it's kind of an ideal world though huh Based on that information, is the claim The world would be better if race and gender did not matter. People would get along much better "Entailment", "Contradiction", or "Inconclusive"?

Answer: Inconclusive.

uh-huh exactly not what color you are how old you are what if your male or female that would be wonderful i guess it's kind of an ideal world though huh Based on that information, is the claim The world would be better if race and gender did not matter. "Entailment", "Contradiction", or "Inconclusive"?

Answer: Entailment.

···

It's that kind of world. Based on that information, is the claim The world is getting better. "Entailment", "Contradiction", or "Inconclusive"?

Answer:

Output:

Inconclusive

Task:SNLI

Task Instruction: Recognizing textual entailment between these 2 texts.

Inference Verbalizer: {Entailment, Inconclusive, Contradiction}

Inference Template: Same as MNLI

Task: RTE

Task Instruction: Recognizing textual entailment between these 2 texts.

Inference Verbalizer: {True, False}

Inference Template: Same as MNLI

Task Family: Subjective Classification

Task: Subj

Task Instruction: Subjectivity of the sentence:

Inference Verbalizer: {subjective, objective}

Inference Template:

Input:

thirteen conversations about one thing lays out a narrative puzzle that interweaves individual stories , and , like amobius strip , elliptically loops back to where it began .

It's subjective.

a small gem of a movie that defies classification and is as thought-provoking as it is funny , scary and sad .

It's subjective.

smart and alert , thirteen conversations about one thing is a small gem .

It's

Output:

subjective

Task Family: Linguistic Acceptibility

Task: COLA

Task Instruction: The grammaticality of this sentence:

Inference Verbalizer: {not grammatical, grammatical}

Inference Template:

Input:

The sea monster drowned the sailors.

It is grammatical.

He rode out the storm.

It is grammatical.

The sailors rode the breeze clear of the rocks.

It is

Output:

grammatical

Task Family: Semantic Parsing

Task: BREAK

Task Instruction: Parse the sentence into logical form:

Inference Template:

Input:

Parse the sentence into logical form: what flights are available from pittsburgh to boston on saturday

1(#) return flights 2(#) return #1 from pittsburgh 3(#) return #2 to boston 4(#) return #3 on saturday 5(#) return #4 that are available

Parse the sentence into logical form: what flights are available wednesday afternoon from denver to san francisco

1(#) return flights 2(#) return #1 from denver 3(#) return #2 to san francisco 4(#) return #3 on wednesday afternoon 5(#) return #4 that are available

Parse the sentence into logical form: what flights are available tomorrow from denver to philadelphia 1#)

Output:

return flights ;return #1 from denver ;return #2 to philadelphia ;return #3 if available

Task: MTOP

Task Instruction: Parse the sentence into logical form:

Inference Template: Same as BREAK

Task:SMCalFlow

Task Instruction: Parse the sentence into logical form:

Inference Template: Same as BREAK

Task Family: Text Summarization

Task: CNN/DailyMail

Task Instruction: Summarize the text:

Inference Template:

Input:

Summarize the text: JERUSALEM (CNN) - Israel moved to defend itself in the face of international criticism Monday over its eviction of dozens of Palestinian families from a neighborhood of Jerusalem they have lived in for generations.

TL;DR: Israel incurs international criticism over eviction of Palestinian families. Two Jewish families moved in after evictions in East Jerusalem. Israeli spokesman says dispute is a legal one between private parties.

Summarize the text: (CNN)The International Criminal Court opened an inquiry into attacks in Palestinian territories, paving the way for possible war crimes investigation against Israelis.

TL;DR: An inquiry allows the court to review evidence and determine whether to file charges. The U.S. calls for negotiations between Palestinian, Israeli officials.

Summarize the text: (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories.

TL;DR:

Output:

Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June. Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis.

Task: PubMed

Task Instruction: Summarize the text:

Inference Template: Same as CNN/DailyMail

Task: Reddit

Task Instruction: Summarize the text:

Inference Template: Same as CNN/DailyMail

Task Family: Commensense Generation

Task: Commen Gen

Task Instruction: Generate a sentence using these concepts:

Inference Template:

Input:

Generate a sentence using these concepts: counter, pizza, restaurant

Generated sentence: Two men standing at counters assembling pizzas in a restaurant.

Generate a sentence using these concepts: counter, restaurant, stand

Generated sentence: A man stands behind the counter of a restaurant.

.

Generate a sentence using these concepts: field, look, stand

Generated sentence:

Output:

The player stood in the field looking at the batter.

Task Family: Story Generation

Task: Roc Story

Task Instruction: Beginning of the story:

Inference Template:

Input:

Beginning of the story: Taylor had been up all night memorizing lines for the play.

Rest of the story: She knew that most of the girls in her class would be auditioning too. She watched the other girls stumble over their lines. She took a deep breath before going up on stage with a smile. All her lines were delivered perfectly and she got the part.

Beginning of the story: Gabby was proud to be given the lead role in the school play.

Rest of the story: She had worked hard on her audition piece. She worked hard to memorize her lines for the play.

When the show opened, she stood on the stage and took it all in. She loved the feeling of performing.

$\therefore m = \frac{3}{11}$

Beginning of the story: Natalie had auditioned for the lead in the school play.

Rest of the story:

Output:

She won the part and was super excited. She rehearsed for weeks and weeks. On opening night, she acted her little heart out. The play was a huge success!

Task: Roc Story Ending

Task Instruction: An unfinished story:

Inference Template: Same as ROC Story

Task Family: Code Summarization

Task: Go

Task Instruction: Comment on the code.

Inference Template:

Input:

Comment on the code. Code:

```txt
func NewSTM(c * v3 Client, apply func (STM) error, so ... stmOption) (*v3.TxnResponse, error) {
    opts := & stmOptions {
        ctx : c. Ctx()
    }
    for _, f := range so {
        f(options)
    }
    if len(args).prefetch != 0 {
        f := apply apply = func (s STM) error {
            s. Get(args).prefetch...
        }
        return f(s)
    }
    return runSTM(mkSTM(c, opts), apply)
}
```

Comment: RunContainer runs a fake Docker container

Comment on the code. Code:

```autohotkey
func (s Subnet) EnsureDead() (err error) {
    defer errors. DeferredAnnotatef (& err, "", s)
    if s.doc. Life == Dead {
        return nil
    }
    ops := [ ] txn. Op {
        C: subnetsC, Id: s.doc. DocID, Upd
    }
}  
txnErr := s.st.db().RunTransactionOps)
if txnErr == nil {
    s.doc. Life = Dead return nil
}  
return onAbort(txnErr, subnetAliveErr)
```

Comment: EnsureDead sets the Life of the subnet to Dead if it s Alive . If the subnet is already Dead no error is returned . When the subnet is no longer Alive or already removed errNotAlive is returned .

···

Comment on the code. Code:

```go
func NewSTM(c * v3 Client, apply func (STM) error, so ... stmOption) (*v3.TxnResponse, error) {
    opts := & stmOptions{ctx : c.Ctx()}
    for _, f := range so{f(options)}
    if len(args).prefetch != 0 {
        f := apply apply = func(sSTM) error{s.Get(args).prefetch...return f(s)}
    }
    return runSTM(mkSTM(c, opts), apply)
}
```

Comment:

Output:

NewSTM initiates a new STM instance using serializable snapshot isolation by default.

Task: Python

Task Instruction: Comment on the code.

Inference Template: Same as Go

Task: Java

Task Instruction: Comment on the code.

Inference Template: Same as Go

Task: PHP

Task Instruction: Comment on the code.

Inference Template: Same as Go

Task Family: Text Simplification

Task: WikiAuto

Task Instruction: Simplify the text:

Inference Template:

Input:

Simplify the text: Stanton went on to write some of the most influential books, documents, and speeches of the women's rights movement.

Simplified text: Together they wrote speeches, articles, and books.

Simplify the text: When she was eighteen and without a university education , she began writing for the newspaper "

Exce I sior " , doing interviews and society columns .

Simplified text: She began writing for the newspaper " Exce lsior ", doing interviews and society columns .

.

Simplify the text: Together with James, she compiled crosswords for several newspapers and magazines, including People, and it was in 1978 that they launched their own publishing company.

Simplified text:

Output:

Together with James, she compiled crosswords. It was for several newspapers and magazines, including People. They launched their own publishing company. It was in 1978.

Task: WikiAuto-Turk

Task Instruction: Simplify the text:

Inference Template: Same as WikiAuto

Task: WikiAuto-ASSET

Task Instruction: Simplify the text:

Inference Template: Same as WikiAuto

Task Family: Data to Text

Task: DART

Task Instruction: Describe the table in natural language.

Inference Template:

Input:

Describe the table in natural language. Table: [Baywatch | NOTES | Episode: Red Wind], [Baywatch | ROLE | Kim], [[TABLE CONTEXT] | TITLE | Baywatch], [[TABLE CONTEXT] | [TITLE] | Bobbie Phillips]

Sentence: Bobbie Phillips appeared on the episode Red Wind in Baywatch as Kim.

Describe the table in natural language. Table: [Silk Stalkings | ROLE | Tessa Shaver], [[TABLE CONTEXT]] | [TITLE]

| Bobbie Phillips], [TABLECONTEXT] | TITLE | Silk Stalkings], [Silk Stalkings | NOTES | Episode: Goodtime Charlie]

Sentence: Actress Bobbie Phillips was casted as Tessa Shaver on the episode Goodtime Charlie of Silk Stalkings.

Describe the table in natural language. Table: [Hawaii Five-O | NOTES | Episode: The Flight of the Jewels], [[TABLE CONTEXT]] | [TITLE] | Jeff Daniels, [[TABLE CONTEXT]] | TITLE | Hawaii Five-O]

Sentence:

Output:

Jeff Daniels played in the Hawaii Five-O episode The Flight of the Jewels

Task: E2E

Task Instruction: Describe the table in natural language.

Inference Template: Same as DART

Table 11: The instructions, inference templates and example cases of tasks.

# Footnotes:

Page 0: *Equal Contribution † Corresponding Authors 
