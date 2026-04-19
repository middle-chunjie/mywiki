# Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents

Weiwei Sun $^{1*}$  Lingyong Yan $^{2}$  Xinyu Ma $^{2}$  Shuaiqiang Wang $^{2}$  Pengjie Ren $^{1}$  Zhumin Chen $^{1}$  Dawei Yin $^{2\dagger}$  Zhaochun Ren $^{3\dagger}$

<sup>1</sup>Shandong University, Qingdao, China <sup>2</sup>Baidu Inc., Beijing, China <sup>3</sup>Leiden University, Leiden, The Netherlands

{sunnweiwei,lingyongy,xinyuma2016,shqiang.wang}@gmail.com {renpengjie,chenzhumin}@sdu.edu.cn yindawei@acm.org z. ren@liacs.leidenuniv.nl

# Abstract

Large Language Models (LLMs) have demonstrated remarkable zero-shot generalization across various language-related tasks, including search engines. However, existing work utilizes the generative ability of LLMs for Information Retrieval (IR) rather than direct passage ranking. The discrepancy between the pretraining objectives of LLMs and the ranking objective poses another challenge. In this paper, we first investigate generative LLMs such as ChatGPT and GPT-4 for relevance ranking in IR. Surprisingly, our experiments reveal that properly instructed LLMs can deliver competitive, even superior results to state-of-the-art supervised methods on popular IR benchmarks. Furthermore, to address concerns about data contamination of LLMs, we collect a new test set called NovelEval, based on the latest knowledge and aiming to verify the model's ability to rank unknown knowledge. Finally, to improve efficiency in real-world applications, we delve into the potential for distilling the ranking capabilities of ChatGPT into small specialized models using a permutation distillation scheme. Our evaluation results turn out that a distilled 440M model outperforms a 3B supervised model on the BEIR benchmark. The code to reproduce our results is available at www.github.com/sunnweiwei/RankGPT.

# 1 Introduction

Large Language Models (LLMs), such as ChatGPT and GPT-4 (OpenAI, 2022, 2023), are revolutionizing natural language processing with strong zero-shot and few-shot generalization. By pretraining on large-scale text corpora and alignment fine-tuning to follow human instructions, LLMs have demonstrated their superior capabilities in language understanding, generation, interaction, and reasoning (Ouyang et al., 2022).

Average nDCG@10  
Figure 1: Average results of ChatGPT and GPT-4 (zero-shot) on passage re-ranking benchmarks (TREC, BEIR, and Mr.TyDi), compared with BM25 and previous best-supervised systems (SOTA sup., e.g., monoT5 (Nogueira et al., 2020)).

As one of the most successful AI applications, Information Retrieval (IR) systems satisfy user requirements through several pipelined sub-modules, such as passage retrieval and re-ranking (Lin et al., 2020). Most previous methods heavily rely on manual supervision signals, which require significant human effort and demonstrate weak generalizability (Campos et al., 2016; Izacard et al., 2022). Therefore, there is a growing interest in leveraging the zero-shot language understanding and reasoning capabilities of LLMs in the IR area. However, most existing approaches primarily focus on exploiting LLMs for content generation (e.g., query or passage) rather than relevance ranking for groups of passages (Yu et al., 2023; Microsoft, 2023).

Compared to the common generation settings, the objectives of relevance re-ranking vary significantly from those of LLMs: the re-ranking agents need to comprehend user requirements, globally compare, and rank the passages based on their relevance to queries. Therefore, leveraging the LLMs' capabilities for passage re-ranking remains a challenging and unanswered task.

To this end, we focus on the following questions:

(RQ1) How does ChatGPT perform on passage re-ranking tasks?  
- (RQ2) How can we imitate the ranking capabilities of ChatGPT in a smaller, specialized model?

To answer the first question, we investigate prompting ChatGPT with two existing strategies (Sachan et al., 2022; Liang et al., 2022). However, we observe that they have limited performance and heavily rely on the availability of the log-probability of model output. Thus, we propose an alternative instructional permutation generation approach, instructing the LLMs to directly output the permutations of a group of passages. In addition, we propose an effective sliding window strategy to address context length limitations. For a comprehensive evaluation of LLMs, we employ three well-established IR benchmarks: TREC (Craswell et al., 2020), BEIR (Thakur et al., 2021), and My.TyDi (Zhang et al., 2021). Furthermore, to assess the LLMs on unknown knowledge and address concerns of data contamination, we suggest collecting a continuously updated evaluation testbed and propose NovelEval, a new test set with 21 novel questions.

To answer the second question, we introduce a permutation distillation technique to imitate the passage ranking capabilities of ChatGPT in a smaller, specialized ranking model. Specifically, we randomly sample 10K queries from the MS MARCO training set, and each query is retrieved by BM25 with 20 candidate passages. On this basis, we distill the permutation predicted by ChatGPT into a student model using a RankNet-based distillation objective (Burges et al., 2005).

Our evaluation results demonstrate that GPT-4, equipped with zero-shot instructional permutation generation, surpasses supervised systems across nearly all datasets. Figure 1 illustrates that GPT-4 outperforms the previous state-of-the-art models by an average nDCG improvement of 2.7, 2.3, and 2.7 on TREC, BEIR, and My.TyDi, respectively. Furthermore, GPT-4 achieves state-of-the-art performance on the new NovelEval test set. Through our permutation distillation experiments, we observe that a 435M student model outperforms the previous state-of-the-art monoT5 (3B) model by an average nDCG improvement of 1.67 on BEIR. Additionally, the proposed distillation method demonstrates cost-efficiency benefits.

In summary, our contributions are tri-fold:

- We examine instructional methods for LLMs on passage re-ranking tasks and introduce a novel permutation generation approach; See Section 3 for details.  
- We comprehensively evaluate ChatGPT and GPT-4 on various passage re-ranking benchmarks, including a newly proposed NovelEval test set; See Section 5 for details.  
- We propose a distillation approach for learning specialized models with the permutation generated by ChatGPT; See Section 4 for details.

# 2 Related Work

# 2.1 Information Retrieval with LLMs

Recently, large language models (LLMs) have found increasing applications in information retrieval (Zhu et al., 2023). Several approaches have been proposed to utilize LLMs for passage retrieval. For example, SGPT (Muennighoff, 2022) generates text embeddings using GPT, generative document retrieval explores a differentiable search index (Tay et al., 2022; Cao et al., 2021; Sun et al., 2023), and HyDE (Gao et al., 2023; Wang et al., 2023a) generates pseudo-documents using GPT-3. In addition, LLMs have also been used for passage re-ranking tasks. UPR (Sachan et al., 2022) and SGPT-CE (Muennighoff, 2022) introduce instructional query generation methods, while HELM (Liang et al., 2022) utilizes instructional relevance generation. LLMs are also employed for training data generation. InPars (Bonifacio et al., 2022) generates pseudo-queries using GPT-3, and Promptagator (Dai et al., 2023) proposes a few-shot dense retrieval to leverage a few demonstrations from the target domain for pseudo-query generation. Furthermore, LLMs have been used for content generation (Yu et al., 2023) and web browsing (Nakano et al., 2021; Izacard et al., 2023; Microsoft, 2023). In this paper, we explore using ChatGPT and GPT-4 in passage re-ranking tasks, propose an instructional permutation generation method, and conduct a comprehensive evaluation of benchmarks from various domains, tasks, and languages. Recent work (Ma et al., 2023) concurrently investigated listwise passage re-ranking using LLMs. In comparison, our study provides a more comprehensive evaluation, incorporating a newly annotated dataset, and validates the proposed permutation distillation technique.

Figure 2: Three types of instructions for zero-shot passage re-ranking with LLMs. The gray and yellow blocks indicate the inputs and outputs of the model. (a) Query generation relies on the log probability of LLMs to generate the query based on the passage. (b) Relevance generation instructs LLMs to output relevance judgments. (c) Permutation generation generates a ranked list of a group of passages. See Appendix A for details.

# 2.2 LLMs Specialization

Despite their impressive capabilities, LLMs such as GPT-4 often come with high costs and lack of source availability. As a result, considerable research has explored ways to distill the capabilities of LLMs into specialized, custom models. For instance, Fu et al. (2023) and Magister et al. (2023) have successfully distilled the reasoning ability of LLMs into smaller models. Self-instruct (Wang et al., 2023b; Taori et al., 2023) propose iterative approaches to distill GPT-3 using their outputs. Additionally, Sachan et al. (2023) and Shi et al. (2023) utilize the generation probability of LLMs to improve retrieval systems. This paper presents a permutation distillation method that leverages ChatGPT as a teacher to obtain specialized re-ranking models. Our experiments demonstrate that even with a small amount of ChatGPT-generated data, the specialized model can outperform strong supervised systems.

# 3 Passage Re-Ranking with LLMs

Ranking is the core task in information retrieval applications, such as ad-hoc search (Lin et al., 2020; Fan et al., 2022), Web search (Zou et al., 2021), and open-domain question answering (Karpukhin et al., 2020). Modern IR systems generally employ a multi-stage pipeline where the retrieval stage focuses on retrieving a set of candidates from a large

Figure 3: Illustration of re-ranking 8 passages using sliding windows with a window size of 4 and a step size of 2. The blue color represents the first two windows, while the yellow color represents the last window. The sliding windows are applied in back-to-first order, meaning that the first 2 passages in the previous window will participate in re-ranking the next window.

corpus, and the re-ranking stage aims to re-rank this set to output a more precise list. Recent studies have explored LLMs for zero-shot re-ranking, such as instructional query generation or relevance generation (Sachan et al., 2022; Liang et al., 2022). However, existing methods have limited performance in re-ranking and heavily rely on the availability of the log probability of model output and thus cannot be applied to the latest LLMs such as GPT-4. Since ChatGPT and GPT-4 have a strong capacity for text understanding, instruction following, and reasoning, we introduce a novel instructional permutation generation method with a sliding window strategy to directly output a ranked list given a set of candidate passages. Figure 2 illustrates examples of three types of instructions; all the detailed instructions are included in Appendix A.

# 3.1 Instructional Permutation Generation

As illustrated in Figure 2 (c), our approach involves inputting a group of passages into the LLMs, each identified by a unique identifier (e.g., [1], [2], etc.). We then ask the LLMs to generate the permutation of passages in descending order based on their relevance to the query. The passages are ranked using the identifiers, in a format such as [2]  $> [3] > [1] > \text{etc}$ . The proposed method ranks passages directly without producing an intermediate relevance score.

# 3.2 Sliding Window Strategy

Due to the token limitations of LLMs, we can only rank a limited number of passages using the permutation generation approach. To overcome this constraint, we propose a sliding window strategy. Figure 3 illustrates an example of re-ranking 8 pas

sages using a sliding window. Suppose the first-stage retrieval model returns  $M$  passages. We rerank these passages in a back-to-first order using a sliding window. This strategy involves two hyperparameters: window size  $(w)$  and step size  $(s)$ . We first use the LLMs to rank the passages from the  $(M - w)$ -th to the  $M$ -th. Then, we slide the window in steps of  $s$  and re-rank the passages within the range from the  $(M - w - s)$ -th to the  $(M - s)$ -th. This process is repeated until all passages have been re-ranked.

# 4 Specialization by Permutation Distillation

Although ChatGPT and GPT-4 are highly capable, they are also too expensive to deploy in commercial search systems. Using GPT-4 to re-rank passages will greatly increase the latency of the search system. In addition, large language models suffer from the problem of unstable generation. Therefore, we argue that the capabilities of large language models are redundant for the re-ranking task. Thus, we can distill the re-ranking capability of large language models into a small model by specialization.

# 4.1 Permutation Distillation

In this paper, we present a novel permutation distillation method that aims to distill the passage reranking capability of ChatGPT into a specialized model. The key difference between our approach and previous distillation methods is that we directly use the model-generated permutation as the target, without introducing any inductive bias such as consistency-checking or log-probability manipulation (Bonifacio et al., 2022; Sachan et al., 2023). To achieve this, we sample 10,000 queries from MS MARCO and retrieve 20 candidate passages using BM25 for each query. The objective of distillation aims to reduce the differences between the permutation outputs of the student and ChatGPT.

# 4.2 Training Objective

Formally, suppose we have a query  $q$  and  $M$  passages  $(p_1, \ldots, p_M)$  retrieved by BM25 ( $M = 20$  in our implementation). ChatGPT with instructional permutation generation could produce the ranking results of the  $M$  passages, denoted as  $R = (r_1, \ldots, r_M)$ , where  $r_i \in [1, 2, \ldots, M]$  is the rank of the passage  $p_i$ . For example,  $r_i = 3$  means  $p_i$  ranks third among the  $M$  passages. Now we have a specialized model  $s_i = f_\theta(q, p_i)$  with

parameters  $\theta$  to calculate the relevance score  $s_i$  of paired  $(q, p_i)$  using a cross-encoder. Using the permutation  $R$  generated by ChatGPT, we consider RankNet loss (Burges et al., 2005) to optimize the student model:

$$
\mathcal {L} _ {\text {R a n k N e t}} = \sum_ {i = 1} ^ {M} \sum_ {j = 1} ^ {M} \mathbb {1} _ {r _ {i} <   r _ {j}} \log \left(1 + \exp \left(s _ {i} - s _ {j}\right)\right)
$$

RankNet is a pairwise loss that measures the correctness of relative passage orders. When using permutations generated by ChatGPT, we can construct  $M(M - 1) / 2$  pairs.

# 4.3 Specialized Model Architecture

Regarding the architecture of the specialized model, we consider two model structures: the BERT-like model and the GPT-like model.

# 4.3.1 BERT-like model.

We utilize a cross-encoder model (Nogueira and Cho, 2019) based on DeBERTa-large. It concatenates the query and passage with a [SEP] token and estimates relevance using the representation of the [CLS] token.

# 4.3.2 GPT-like model.

We utilize the LLaMA-7B (Touvron et al., 2023) with a zero-shot relevance generation instruction (see Appendix A). It classifies the query and passage as relevance or irrelevance by generating a relevance token. The relevance score is then defined as the generation probability of the relevance token.

Figure 5 illustrates the structure of the two types of specialized models.

# 5 Datasets

Our experiments are conducted on three benchmark datasets and one newly collected test set NovelEval.

# 5.1 Benchmark Datasets

The benchmark datasets include, TREC-DL (Craswell et al., 2020), BEIR (Thakur et al., 2021), and Mr.TyDi (Zhang et al., 2021).

TREC is a widely used benchmark dataset in IR research. We use the test sets of the 2019 and 2020 competitions: (i) TREC-DL19 contains 43 queries, (ii) TREC-DL20 contains 54 queries.

BEIR consists of diverse retrieval tasks and domains. We choose eight tasks in BEIR to evaluate the models: (i) Covid: retrieves scientific articles

for COVID-19 related questions. (ii) NFCorpus is a bio-medical IR data. (iii) Touche is an argument retrieval datasets. (iv) DBPedia retrieves entities from DBpedia corpus. (v) SciFact retrieves evidence for claims verification. (vi) Signal retrieves relevant tweets for a given news title. (vii) News retrieves relevant news articles for news headlines. (viii) Robust04 evaluates poorly performing topics.

Mr.TyDi is a multilingual passages retrieval dataset of ten low-resource languages: Arabic, Bengali, Finnish, Indonesian, Japanese, Korean, Russian, Swahili, Telugu, and Thai. We use the first 100 samples in the test set of each language.

# 5.2 A New Test Set - NovelEval

The questions in the current benchmark dataset are typically gathered years ago, which raises the issue that existing LLMs already possess knowledge of these questions (Yu et al., 2023). Furthermore, since many LLMs do not disclose information about their training data, there is a potential risk of contamination of the existing benchmark test set (OpenAI, 2023). However, re-ranking models are expected to possess the capability to comprehend, deduce, and rank knowledge that is inherently unknown to them. Therefore, we suggest constructing continuously updated IR test sets to ensure that the questions, passages to be ranked, and relevance annotations have not been learned by the latest LLMs for a fair evaluation.

As an initial effort, we built NovelEval-2306, a novel test set with 21 novel questions. This test set is constructed by gathering questions and passages from 4 domains that were published after the release of GPT-4. To ensure that GPT-4 did not possess prior knowledge of these questions, we presented them to both gpt-4-0314 and gpt-4-0613. For instance, question "Which film was the 2023 Palme d'Or winner?" pertains to the Cannes Film Festival that took place on May 27, 2023, rendering its answer inaccessible to most existing LLMs. Next, we searched 20 candidate passages for each question using Google search. The relevance of these passages was manually labeled as: 0 for not relevant, 1 for partially relevant, and 2 for relevant. See Appendix C for more details.

# 6 Experimental Results of LLMs

# 6.1 Implementation and Metrics

In benchmark datasets, we re-rank the top-100 passages retrieved by BM25 using pyserini<sup>1</sup> and use nDCG@{1, 5,10} as evaluation metrics. Since ChatGPT cannot manage 100 passages at a time, we use the sliding window strategy introduced in Section 3.2 with a window size of 20 and step size of 10. In NovelEval, we randomly shuffled the 20 candidate passages searched by Google and re-ranked them using ChatGPT and GPT-4 with permutation generation.

# 6.2 Results on Benchmarks

On benchmarks, we compare ChatGPT and GPT-4 with state-of-the-art supervised and unsupervised passage re-ranking methods. The supervised baselines include: monoBERT (Nogueira and Cho, 2019), monoT5 (Nogueira et al., 2020), mmarcoCE (Bonifacio et al., 2021), and Cohere Rerank  ${}^{2}$  . The unsupervised baselines include: UPR (Sachan et al., 2022), InPars (Bonifacio et al., 2022), and Promptator++ (Dai et al., 2023). See Appendix E for more details on implementing the baseline.

Table 1 presents the evaluation results obtained from the TREC and BEIR datasets. The following observations can be made: (i) GPT-4, when equipped with the permutation generation instruction, demonstrates superior performance on both datasets. Notably, GPT-4 achieves an average improvement of 2.7 and 2.3 in nDCG@10 on TREC and BEIR, respectively, compared to monoT5 (3B). (ii) ChatGPT also exhibits impressive results on the BEIR dataset, surpassing the majority of supervised baselines. (iii) On BEIR, we use only GPT-4 to re-rank the top-30 passages re-ranked by ChatGPT. The method achieves good results, while the cost is only 1/5 of that of only using GPT-4 for re-ranking.

Table 2 illustrates the results on Mr. TyDi of ten low-resource languages. Overall, GPT-4 outperforms the supervised system in most languages, demonstrating an average improvement of 2.65 nDCG over mmarcoCE. However, there are instances where GPT-4 performs worse than mmarcoCE, particularly in low-resource languages like Bengali, Telugu, and Thai. This may be attributed to the weaker language modeling ability of GPT-4

<table><tr><td>Method</td><td>DL19</td><td>DL20</td><td>Covid</td><td>NFCorpus</td><td>Touche</td><td>DBPedia</td><td>SciFact</td><td>Signal</td><td>News</td><td>Robust04</td><td>BEIR (Avg)</td></tr><tr><td>BM25</td><td>50.58</td><td>47.96</td><td>59.47</td><td>30.75</td><td>44.22</td><td>31.80</td><td>67.89</td><td>33.05</td><td>39.52</td><td>40.70</td><td>43.42</td></tr><tr><td colspan="12">Supervised</td></tr><tr><td>monoBERT (340M)</td><td>70.50</td><td>67.28</td><td>70.01</td><td>36.88</td><td>31.75</td><td>41.87</td><td>71.36</td><td>31.44</td><td>44.62</td><td>49.35</td><td>47.16</td></tr><tr><td>monoT5 (220M)</td><td>71.48</td><td>66.99</td><td>78.34</td><td>37.38</td><td>30.82</td><td>42.42</td><td>73.40</td><td>31.67</td><td>46.83</td><td>51.72</td><td>49.07</td></tr><tr><td>monoT5 (3B)</td><td>71.83</td><td>68.89</td><td>80.71</td><td>38.97</td><td>32.41</td><td>44.45</td><td>76.57</td><td>32.55</td><td>48.49</td><td>56.71</td><td>51.36</td></tr><tr><td>Cohere Rerank-v2</td><td>73.22</td><td>67.08</td><td>81.81</td><td>36.36</td><td>32.51</td><td>42.51</td><td>74.44</td><td>29.60</td><td>47.59</td><td>50.78</td><td>49.45</td></tr><tr><td colspan="12">Unsupervised</td></tr><tr><td>UPR (FLAN-T5-XL)</td><td>53.85</td><td>56.02</td><td>68.11</td><td>35.04</td><td>19.69</td><td>30.91</td><td>72.69</td><td>31.91</td><td>43.11</td><td>42.43</td><td>42.99</td></tr><tr><td>InPars (monoT5-3B)</td><td>-</td><td>66.12</td><td>78.35</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Promptagator++ (few-shot)</td><td>-</td><td>-</td><td>76.2</td><td>37.0</td><td>38.1</td><td>43.4</td><td>73.1</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="12">LLM API (Permutation generation)</td></tr><tr><td>gpt-3.5-turbo</td><td>65.80</td><td>62.91</td><td>76.67</td><td>35.62</td><td>36.18</td><td>44.47</td><td>70.43</td><td>32.12</td><td>48.85</td><td>50.62</td><td>49.37</td></tr><tr><td>gpt-4†</td><td>75.59</td><td>70.56</td><td>85.51</td><td>38.47</td><td>38.57</td><td>47.12</td><td>74.95</td><td>34.40</td><td>52.89</td><td>57.55</td><td>53.68</td></tr></table>

Table 1: Results (nDCG@10) on TREC and BEIR. Best performing unsupervised and overall system(s) are marked bold. All models except InPars and Promptagator++ re-rank the same BM25 top-100 passages. †On BEIR, we use gpt-4 to re-rank the top-30 passages re-ranked by gpt-3.5-turbo to reduce the cost of calling gpt-4 API.  

<table><tr><td>Method</td><td>BM25</td><td>mmarcoCE</td><td>gpt-3.5</td><td>+gpt-4</td></tr><tr><td>Arabic</td><td>39.19</td><td>68.18</td><td>71.00</td><td>72.56</td></tr><tr><td>Bengali</td><td>45.56</td><td>65.98</td><td>53.10</td><td>64.37</td></tr><tr><td>Finnish</td><td>29.91</td><td>54.15</td><td>56.48</td><td>62.29</td></tr><tr><td>Indonesian</td><td>51.79</td><td>69.94</td><td>68.45</td><td>75.47</td></tr><tr><td>Japanese</td><td>27.39</td><td>49.80</td><td>50.70</td><td>58.22</td></tr><tr><td>Korean</td><td>26.29</td><td>44.00</td><td>41.48</td><td>49.63</td></tr><tr><td>Russian</td><td>34.04</td><td>53.16</td><td>48.75</td><td>53.45</td></tr><tr><td>Swahili</td><td>45.15</td><td>60.31</td><td>62.38</td><td>67.67</td></tr><tr><td>Telugu</td><td>37.05</td><td>68.92</td><td>51.69</td><td>62.22</td></tr><tr><td>Thai</td><td>44.62</td><td>68.36</td><td>55.57</td><td>63.41</td></tr><tr><td>Avg</td><td>38.10</td><td>60.28</td><td>55.96</td><td>62.93</td></tr></table>

in these languages and the fact that text in low-resource languages tends to consume more tokens than English text, leading to the over-cropping of passages. Similar trends are observed with ChatGPT, which is on par with the supervised system in most languages, and consistently trails behind GPT-4 in all languages.

# 6.3 Results on NovelEval

Table 3 illustrates the evaluation results on our newly collected NovelEval, a test set containing 21 novel questions and 420 passages that GPT-4 had not learned. The results show that GPT-4 performs well on these questions, significantly outperforming the previous best-supervised method, monoT5 (3B). Additionally, ChatGPT achieves a performance level comparable to that of monoBERT. This outcome implies that LLMs possess the capability to effectively re-rank unfamiliar information.

Table 2: Results (nDCG@10) on Mr.TyDi.  

<table><tr><td>Method</td><td>nDCG@1</td><td>nDCG@5</td><td>nDCG@10</td></tr><tr><td>BM25</td><td>33.33</td><td>45.96</td><td>55.77</td></tr><tr><td>monoBERT (340M)</td><td>78.57</td><td>70.65</td><td>77.27</td></tr><tr><td>monoT5 (220M)</td><td>83.33</td><td>77.46</td><td>81.27</td></tr><tr><td>monoT5 (3B)</td><td>83.33</td><td>78.38</td><td>84.62</td></tr><tr><td>gpt-3.5-turbo</td><td>76.19</td><td>74.15</td><td>75.71</td></tr><tr><td>gpt-4</td><td>85.71</td><td>87.49</td><td>90.45</td></tr></table>

Table 3: Results on NovelEval.  

<table><tr><td>Method</td><td>DL19
nDCG@1/5/10</td><td>DL20
nDCG@1/5/10</td></tr><tr><td>curie-001</td><td>RG 39.53 / 40.02 / 41.53</td><td>41.98 / 34.80 / 34.91</td></tr><tr><td>curie-001</td><td>QG 50.78 / 50.77 / 49.76</td><td>50.00 / 48.36 / 48.73</td></tr><tr><td>curie-001</td><td>PG 66.67 / 56.79 / 54.21</td><td>59.57 / 55.20 / 52.17</td></tr><tr><td>davinci-003</td><td>RG 54.26 / 52.78 / 50.58</td><td>64.20 / 58.41 / 56.87</td></tr><tr><td>davinci-003</td><td>QG 37.60 / 44.73 / 45.37</td><td>51.25 / 47.46 / 45.93</td></tr><tr><td>davinci-003</td><td>PG 69.77 / 64.73 / 61.50</td><td>69.75 / 58.76 / 57.05</td></tr><tr><td>gpt-3.5</td><td>PG 82.17 / 71.15 / 65.80</td><td>79.32 / 66.76 / 62.91</td></tr><tr><td>gpt-4</td><td>PG 82.56 / 79.16 / 75.59</td><td>78.40 / 74.11 / 70.56</td></tr></table>

Table 4: Compare different instruction and API endpoint. Best performing system(s) are marked bold. PG, QG, RG denote permutation generation, query generation and relevance generation, respectively.

# 6.4 Compare with Different Instructions

We conduct a comparison with the proposed permutation generation (PG) with previous query generation (QG) (Sachan et al., 2022) and relevance generation (RG) (Liang et al., 2022) on TREC-DL19. An example of the three types of instructions is in Figure 2, and the detailed implementation is in Appendix B. We also compare four LLMs provided

<table><tr><td>Method</td><td>nDCG@1</td><td>nDCG@5</td><td>nDCG@10</td></tr><tr><td>BM25</td><td>54.26</td><td>52.78</td><td>50.58</td></tr><tr><td>gpt-3.5-turbo</td><td>82.17</td><td>71.15</td><td>65.80</td></tr><tr><td colspan="4">Initial passage order</td></tr><tr><td>(1) Random order</td><td>26.36</td><td>25.32</td><td>25.17</td></tr><tr><td>(2) Reverse order</td><td>36.43</td><td>31.79</td><td>32.77</td></tr><tr><td colspan="4">Number of re-ranking</td></tr><tr><td>(3) Re-rank 2 times</td><td>78.29</td><td>69.37</td><td>66.62</td></tr><tr><td>(4) Re-rank 3 times</td><td>78.29</td><td>69.74</td><td>66.97</td></tr><tr><td>(5) gpt-4 Rerank</td><td>80.23</td><td>76.70</td><td>73.64</td></tr></table>

Table 5: Ablation study on TREC-DL19. We use gpt-3.5-turbo with permutation generation with different configuration.

in the OpenAI  $\mathrm{API}^3$ : curie-001 - GPT-3 model with about 6.7 billion parameters (Brown et al., 2020); davinci-003 - GPT-3.5 model trained with RLHF and about 175 billion parameters (Ouyang et al., 2022); gpt-3.5-turbo - The underlying model of ChatGPT (OpenAI, 2022); gpt-4 - GPT-4 model (OpenAI, 2023).

The results are listed in Table 4. From the results, we can see that: (i) The proposed PG method outperforms both QG and RG methods in instructing LLMs to re-rank passages. We suggest two explanations: First, from the result that PG has significantly higher top-1 accuracy compared to other methods, we infer that LLMs can explicitly compare multiple passages with PG, allowing subtle differences between passages to be discerned. Second, LLMs gain a more comprehensive understanding of the query and passages by reading multiple passages with potentially complementary information, thus improving the model's ranking ability. (ii) With PG, ChatGPT performs comparably to GPT-4 on nDCG@1, but lags behind it on nDCG@10. The Davinci model (text-davinci-003) performs poorly compared to ChatGPT and GPT-4. This may be because of the generation stability of Davinci and ChatGPT trails that of GPT-4. We delve into the stability analysis of Davinci, ChatGPT, and GPT-4 in Appendix F.

# 6.5 Ablation Study on TREC

We conducted an ablation study on TREC to gain insights into the detailed configuration of permutation generation. Table 5 illustrates the results.

Initial Passage Order While our standard implementation utilizes the ranking result of BM25 as the initial order, we examined two alternative variants: random order (1) and reversed BM25 order (2). The results reveal that the model's performance is highly sensitive to the initial passage order. This could be because BM25 provides a relatively good starting passage order, enabling satisfactory results with only a single sliding window re-ranking.

Number of Re-Ranking Furthermore, we studied the influence of the number of sliding window passes. Models (3-4) in Table 5 show that re-ranking more times may improve nDCG@10, but it somehow hurts the ranking performance on top passages (e.g., nDCG@1 decreased by 3.88). Re-ranking the top 30 passages using GPT-4 showed notable accuracy improvements (see the model (5)). This provides an alternative method to combine ChatGPT and GPT-4 in passage re-ranking to reduce the high cost of using the GPT-4 model.

# 6.6 Results of LLMs beyond ChatGPT

We further test the capabilities of other LLMs beyond the OpenAI series on TREC DL-19. As shown in Table 6, we evaluate the top-20 BM25 passage re-ranking nDCG of proprietary LLMs from OpenAI, Cohere, Antropic, and Google, and three open-source LLMs. We see that: (i) Among the proprietary LLMs, GPT-4 exhibited the highest re-ranking performance. Cohere Re-rank also showed promising results; however, it should be noted that it is a supervised specialized model. In contrast, the proprietary models from Antropic and Google fell behind ChatGPT in terms of re-ranking effectiveness. (ii) As for the open-source LLMs, we observed a significant performance gap compared to ChatGPT. One possible reason for this discrepancy could be the complexity involved in generating permutations of 20 passages, which seems to pose a challenge for the existing open-source models.

We analyze the model's unexpected behavior in Appendix F, and the cost of API in Appendix H.

# 7 Experimental Results of Specialization

As mentioned in Section 4, we randomly sample 10K queries from the MSMARCO training set and employ the proposed permutation distillation to distill ChatGPT's predicted permutation into specialized re-ranking models. The specialized re-ranking models could be DeBERTa-v3-Large with a cross-encoder architecture or LLaMA-7B with relevance

Figure 4: Scaling experiment. The dashed line indicates the baseline methods: GPT-4, monoT5, monoBERT, and ChatGPT. The solid green line and solid gray line indicate the specialized Deberta models obtained by the proposed permutation distillation and by supervised learning on MS MARCO, respectively. This figure compares the models' performance on TREC and BEIR across varying model sizes (70M to 435M) and training data sizes (500 to 10K).

<table><tr><td>Method</td><td>ND1</td><td>ND5</td><td>ND10</td></tr><tr><td>OpenAI text-davinci-003</td><td>70.54</td><td>61.90</td><td>57.24</td></tr><tr><td>OpenAI gpt-3.5-turbo</td><td>75.58</td><td>66.19</td><td>60.89</td></tr><tr><td>OpenAI gpt-4</td><td>79.46</td><td>71.65</td><td>65.68</td></tr><tr><td>Cohere rerank-english-v2.0</td><td>79.46</td><td>71.56</td><td>64.78</td></tr><tr><td>Antropic claude-2</td><td>66.66</td><td>59.33</td><td>55.91</td></tr><tr><td>Antropic claude-instant-1</td><td>81.01</td><td>66.71</td><td>62.23</td></tr><tr><td>Google text-bison-001</td><td>69.77</td><td>64.46</td><td>58.67</td></tr><tr><td>Google bard-2023.10.21</td><td>81.01</td><td>65.57</td><td>60.11</td></tr><tr><td>Google flan-t5-xxl</td><td>52.71</td><td>51.63</td><td>50.26</td></tr><tr><td>Tsinghua ChatGLM-6B</td><td>54.26</td><td>52.77</td><td>50.58</td></tr><tr><td>LMSYS Vicuna-13B</td><td>54.26</td><td>51.55</td><td>49.08</td></tr></table>

Table 6: Results of different LLMs on re-ranking top-20 passages on DL-19. ND{1,5,10} denote nDCG@{1,5,10}, respectively.  

<table><tr><td>Label</td><td>Method</td><td>DL19</td><td>DL20</td><td>BEIR (Avg)</td></tr><tr><td>∅</td><td>BM25</td><td>50.58</td><td>47.96</td><td>43.42</td></tr><tr><td>∅</td><td>ChatGPT</td><td>65.80</td><td>62.91</td><td>49.37</td></tr><tr><td>MARCO</td><td>monoT5 (3B)</td><td>71.83</td><td>68.89</td><td>51.36</td></tr><tr><td>MARCO</td><td>DeBERTa-Large</td><td>68.89</td><td>61.38</td><td>42.64</td></tr><tr><td>MARCO</td><td>LLaMA-7B</td><td>69.24</td><td>58.97</td><td>47.71</td></tr><tr><td>ChatGPT</td><td>DeBERTa-Large</td><td>70.66</td><td>67.15</td><td>53.03</td></tr><tr><td>ChatGPT</td><td>LLaMA-7B</td><td>71.78</td><td>66.89</td><td>51.68</td></tr></table>

Table 7: Results (nDCG@10) of specialized models. Best performing specialized model(s) are marked bold. The label column denotes the relevance judgements used in model training, where MARCO denotes use MS MARCO annotation, ChatGPT denotes use the outputs of permutation generation instructed ChatGPT as labels. BEIR (Avg) denotes average nDCG on eight BEIR datasets, and the detailed results are at Table 13.

generation instructions. We also implemented the specialized model trained using the original MS MARCO labels (aka supervised learning) for com

parison4.

# 7.1 Results on Benchmarks

Table 7 lists the results of specialized models, and Table 13 includes the detailed results. Our findings can be summarized as follows: (i) Permutation distillation outperforms the supervised counterpart on both TREC and BEIR datasets, potentially because ChatGPT's relevance judgments are more comprehensive than MS MARCO labels (Arabzadeh et al., 2021). (ii) The specialized DeBERTa model outperforms previous state-of-the-art (SOTA) baselines, monoT5 (3B), on BEIR with an average nDCG of 53.03. This result highlights the potential of distilling LLMs for IR since it is significantly more cost-efficient. (iii) The distilled specialized model also surpasses ChatGPT, its teacher model, on both datasets. This is probably because the re-ranking stability of specialized models is better than ChatGPT. As shown in the stability analysis in Appendix F, ChatGPT is very unstable in generating permutations.

# 7.2 Analysis on Model Size and Data Size

In Figure 4, we present the re-ranking performance of specialized DeBERTa models obtained through permutation distillation and supervised learning of different model sizes (ranging from 70M to 435M) and training data sizes (ranging from 500 to 10K). Our findings indicate that the permutation-distilled models consistently outperform their supervised counterparts across all settings, particularly on the BEIR datasets. Notably, even with only 1K training queries, the permutation-distilled DeBERTa model

achieves superior performance compared to the previous state-of-the-art monoT5 (3B) model on BEIR. We also observe that increasing the number of model parameters yields a greater improvement in the ranking results than increasing the training data. Finally, we find that the performance of supervised models is unstable for different model sizes and data sizes. This may be due to the presence of noise in the MS MARCO labels, which leads to overfitting problems (Arabzadeh et al., 2021).

# 8 Conclusion

In this paper, we conduct a comprehensive study on passage re-ranking with LLMs. We introduce a novel permutation generation approach to fully explore the power of LLMs. Our experiments on three benchmarks have demonstrated the capability of ChatGPT and GPT-4 in passage re-ranking. To further validate LLMs on unfamiliar knowledge, we introduce a new test set called NovelEval. Additionally, we propose a permutation distillation method, which demonstrates superior effectiveness and efficiency compared to existing supervised approaches.

# Limitations

The limitations of this work include the main analysis for OpenAI ChatGPT and GPT-4, which are proprietary models that are not open-source. Although we also tested on open-source models such as FLAN-T5, ChatGLM-6B, and Vicuna-13B, the results still differ significantly from ChatGPT. How to further exploit the open-source models is a question worth exploring. Additionally, this study solely focuses on examining LLMs in the re-ranking task. Consequently, the upper bound of the ranking effect is contingent upon the recall of the initial passage retrieval. Our findings also indicate that the re-ranking effect of LLMs is highly sensitive to the initial order of passages, which is usually determined by the first-stage retrieval, such as BM25. Therefore, there is a need for further exploration into effectively utilizing LLMs to enhance the first-stage retrieval and improve the robustness of LLMs in relation to the initial passage retrieval.

# Ethics Statement

We acknowledge the importance of the ACM Code of Ethics and totally agree with it. We ensure that this work is compatible with the provided code,

in terms of publicly accessed datasets and models. Risks and harms of large language models include the generation of harmful, offensive, or biased content. These models are often prone to generating incorrect information, sometimes referred to as hallucinations. We do not expect the studied model to be an exception in this regard. The LLMs used in this paper were shown to suffer from bias, hallucination, and other problems. Therefore, we are not recommending the use of LLMs for ranking tasks with social implications, such as ranking job candidates or ranking products, because LLMs may exhibit racial bias, geographical bias, gender bias, etc., in the ranking results. In addition, the use of LLMs in critical decision-making sessions may pose unspecified risks. Finally, the distilled models are licensed under the terms of OpenAI because they use ChatGPT. The distilled LLaMA models are further licensed under the non-commercial license of LLaMA.

# Acknowledgements

This work was supported by the Natural Science Foundation of China (62272274, 61972234, 62072279, 62102234, 62202271), the Natural Science Foundation of Shandong Province (ZR2021QF129, ZR2022QF004), the Key Scientific and Technological Innovation Program of Shandong Province (2019JZZY010129), the Fundamental Research Funds of Shandong University, the China Scholarship Council under grant nr. 202206220085.

# References

Negar Arabzadeh, Alexandra Vtyurina, Xinyi Yan, and Charles L. A. Clarke. 2021. Shallow pooling for sparse labels. Information Retrieval Journal, 25:365 - 385.  
Luiz Henrique Bonifacio, Hugo Queiroz Abonizio, Marzieh Fadaee, and Rodrigo Nogueira. 2022. Inpars: Data augmentation for information retrieval using large language models. In SIGIR 2022.  
Luiz Henrique Bonifacio, Israel Campiotti, Roberto de Alencar Lotufo, and Rodrigo Nogueira. 2021. mmarco: A multilingual version of ms marco passage ranking dataset. ArXiv, abs/2108.13897.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens

Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In NeurIPS 2020.  
Sebastian Bruch, Xuanhui Wang, Michael Bendersky, and Marc Najork. 2019. An analysis of the softmax cross entropy loss for learning-to-rank with binary relevance. In SIGIR 2019.  
Christopher J. C. Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Gregory N. Hullender. 2005. Learning to rank using gradient descent. In ICML 2005.  
Daniel Fernando Campos, Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng, and Bhaskar Mitra. 2016. Ms marco: A human generated machine reading comprehension dataset. ArXiv, abs/1611.09268.  
Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Autoregressive entity retrieval. In ICLR 2021.  
Hyung Won Chung, Le Hou, S. Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Wei Yu, Vincent Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed Huai hsin Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. ArXiv, abs/2210.11416.  
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Fernando Campos, and Ellen M. Voorhees. 2020. Overview of the trec 2020 deep learning track. ArXiv, abs/2102.07662.  
Zhuyun Dai, Vincent Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, and Ming-Wei Chang. 2023. Promptagator: Few-shot dense retrieval from 8 examples. In ICLR 2023.  
Yixing Fan, Xiaohui Xie, Yinqiong Cai, Jia Chen, Xinyu Ma, Xiangsheng Li, Ruqing Zhang, and Jiafeng Guo. 2022. Pre-training methods in information retrieval. Foundations and Trends in Information Retrieval, 16:178-317.  
Yao Fu, Hao-Chun Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. In ICML 2023.  
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval without relevance labels. In ACL 2023.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Towards unsupervised dense information retrieval with contrastive learning. TMLR.  
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane A. Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023. Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1-43.  
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Yu Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense passage retrieval for open-domain question answering. In EMNLP 2020.  
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher R'e, Diana Acosta-Navas, Drew A. Hudson, E. Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel J. Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan S. Kim, Neel Guha, Niladri S. Chatterji, O. Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas F. Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2022. Holistic evaluation of language models. ArXiv, abs/2211.09110.  
Jimmy J. Lin, Rodrigo Nogueira, and Andrew Yates. 2020. Pretrained transformers for text ranking: Bert and beyond. In WSDM 2020.  
Xueguang Ma, Xinyu Crystina Zhang, Ronak Pradeep, and Jimmy Lin. 2023. Zero-shot listwise document reranking with a large language model. ArXiv, abs/2305.02156.  
Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. 2023. Teaching small language models to reason. In ACL 2023.  
Microsoft. 2023. Confirmed: the new bing runs on openai's gpt-4. https://blogs.bing.com/search/march_2023/ Confirmed-the-new-Bing-runs-on-OpenAI% E2%80%99s-GPT-4.  
Niklas Muennighoff. 2022. Sgt: Gpt sentence embeddings for semantic search. ArXiv, abs/2202.08904.  
Reiichiro Nakano, Jacob Hilton, S. Arun Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin

Chess, and John Schulman. 2021. Webgpt: Browser-assisted question-answering with human feedback. ArXiv, abs/2112.09332.  
Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with bert. ArXiv, abs/1901.04085.  
Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-to-sequence model. In *Findings of EMNLP* 2020.  
OpenAI. 2022. Introducing chatgpt. https://openai.com/blog/chatgpt.  
OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774.  
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. 2022. Training language models to follow instructions with human feedback. In NeurIPS 2022.  
Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen tau Yih, Joëlle Pineau, and Luke Zettlemoyer. 2022. Improving passage retrieval with zero-shot question generation. In EMNLP 2022.  
Devendra Singh Sachan, Mike Lewis, Dani Yogatama, Luke Zettlemoyer, Joëlle Pineau, and Manzil Zaheer. 2023. Questions are all you need to train a dense passage retriever. TACL, page 600-616.  
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. 2023. Replug: Retrieval-augmented black-box language models. ArXiv, abs/2301.12652.  
Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, M. de Rijke, and Zhaochun Ren. 2023. Learning to tokenize for generative retrieval. In NeruIPS 2023.  
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca.  
Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer memory as a differentiable search index. In NeurIPS 2022.  
Nandan Thakur, Nils Reimers, Andreas Ruckl'e, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. In NeurIPS 2021.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aur'elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. *ArXiv*, abs/2302.13971.  
Liang Wang, Nan Yang, and Furu Wei. 2023a. Query2doc: Query expansion with large language models. ArXiv, abs/2303.07678.  
Xuanhui Wang, Cheng Li, Nadav Golbandi, Michael Bendersky, and Marc Najork. 2018. The lambdaloss framework for ranking metric optimization. In CIKM 2018.  
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023b. Self-instruct: Aligning language model with self generated instructions. In ACL 2023.  
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2023. Generate rather than retrieve: Large language models are strong context generators. In ICLR 2023.  
Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy J. Lin. 2021. Mr. tydi: A multi-lingual benchmark for dense retrieval. In MRL.  
Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, and Ji rong Wen. 2023. Large language models for information retrieval: A survey. ArXiv, abs/2308.07107.  
Lixin Zou, Shengqiang Zhang, Hengyi Cai, Dehong Ma, Suqi Cheng, Daiting Shi, Shuaiqiang Wang, Zhicong Cheng, and Dawei Yin. 2021. Pre-trained language model based ranking in baidu search. In KDD 2021.

# A Instructions

# A.1 Query Generation Instruction

The query generation instruction (Sachan et al., 2022) uses the log-probability of the query.

Please write a question based on this passage.

Passage:{{passage}}

Question: {{query}}

# A.2 Relevance Generation Instruction (few-shot)

Following HELM (Liang et al., 2022), the relevance generation instruction use 4 in-context examples.

Given a passage and a query, predict whether the passage includes an answer to the query by producing either 'Yes' or 'No'.

Passage: Its 25 drops per ml, you guys are all wrong. If it is water, the standard was changed 15 - 20 years ago to make 20 drops  $= 1\mathrm{mL}$ . The viscosity of most things is temperature dependent, so this would be at room temperature. Hope this helps.

Query: how many eye drops per ml

Does the passage answer the query?

Answer: Yes

Passage: RE: How many eyedrops are there in a  $10\mathrm{ml}$  bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day. In the past other pharmacies have given me 3 10-ml bottles for 100 days.E: How many eyedrops are there in a  $10\mathrm{ml}$  bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.

Query: how many eye drops per ml

Does the passage answer the query?

Answer: No

Passage: : You can transfer money to your checking account from other Wells Fargo. accounts through Wells Fargo Mobile Banking with the mobile app, online, at any. Wells Fargo ATM, or at a Wells Fargo branch. 1 Money in — deposits.

Query: can you open a wells fargo account online

Does the passage answer the query?

Answer: No

Passage: You can open a Wells Fargo banking account from your home or even online. It is really easy to do, provided you have all of the appropriate documentation. Wells Fargo has so many bank account options that you will be sure to find one that works for you. They offer free checking accounts with free online banking.

Query: can you open a wells fargo account online

Does the passage answer the query?

Answer: Yes

Passage: {{passage}}

Query:{{query}}

Does the passage answer the query?

Answer:

# A.3 Relevance Generation Instruction (zero-shot)

This instruction is used to train LLaMA-7B specialized models.

Given a passage and a query, predict whether the passage includes an answer to the query by producing either 'Yes' or 'No'.

Passage: {{passage}}

Query: {{query}}

Does the passage answer the query?

Answer:

# A.4 Permutation Generation Instruction (Text)

Permutation generation (text) is used for text-davinci-003.

This is RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.

The following are  $\{\{\mathsf{num}\}\}$  passages, each indicated by number identifier []. I can rank them based on their relevance to query: {{query}}

[1] {{passage_1}}  
[2] {{passage_2}}

(more passages) ...

The search query is: {{query}}

I will rank the  $\{\{\mathsf{num}\}\}$  passages above based on their relevance to the search query. The passages will be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be  $[ ] > [ ] > \text{etc}$ , e.g.,  $[1] > [2] > \text{etc}$ .

The ranking results of the  $\{\{\mathsf{num}\}\}$  passages (only identifiers) is:

# A.5 Permutation Generation Instruction (Chat)

Permutation generation instruction (chat) is used for gpt-3.5-turbo and gpt-4.

# system:

You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.

# user:

I will provide you with {{num}} passages, each indicated by number identifier []. Rank them based on their relevance to query: {{query}}.

# assistant:

Okay, please provide the passages.

# user:

[1] {{passage_1}}

# assistant:

Received passage [1]

# user:

[2] {{passage_2}}

# assistant:

Received passage [2]

(more passages) ...

# user

Search Query: {{query}}.

Rank the  $\{\{\mathsf{num}\}\}$  passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be  $[ ] > [ ]$ , e.g.,  $[1] > [2]$ . Only response the ranking results, do not say any word or explain.

# B Instructional Methods on LLMs as Rernaker

This paper focuses on re-ranking task, given  $M$  passages for a query  $q$ , the re-ranking aims to use an agent  $f(\cdot)$  to output their ranking results  $\mathbf{R} = (r_1, \dots, r_M)$ , where  $r_i \in [1, 2, \dots, M]$  denotes the rank of  $p_i$ . This paper studies using the LLMs as  $f(\cdot)$ .

# B.1 Instructional Query Generation

Query generation has been studied in Sachan et al. (2022); Muennighoff (2022), in which the relevance between a query and a passage is measured by the log-probability of the model to generate the query based on the passage. Figure 2 (a) shows an example of instructional query generation.

Formally, given query  $q$  and a passage  $p_i$ , their relevance score  $s_i$  is calculated as:

$$
s _ {i} = \frac {1}{| q |} \sum_ {t} \log p \left(q _ {t} \mid q _ {<   t}, p _ {i}, \mathcal {I} _ {\text {q u e r y}}\right) \tag {1}
$$

where  $|q|$  denotes the number of tokens in  $q$ ,  $q_{t}$  denotes the  $t$ -th token of  $q$ , and  $\mathcal{I}_{\mathrm{query}}$  denotes the instructions, referring to Figure 2 (a). The passages are then ranked based on relevance score  $s_i$ .

# B.2 Instructional Relevance Generation

Relevance generation is employed in HELM (Liang et al., 2022). Figure 2 (b) shows an example of instructional relevance generation, in which LLMs are instructed to output "Yes" if the query and passage

are relevant or "No" if they are irrelevant. The relevance score  $s_i$  is measured by the probability of LLMs generating the word 'Yes' or 'No':

$$
s _ {i} = \left\{ \begin{array}{l l} 1 + p (\text {Y e s}), & \text {i f o u t p u t i s Y e s} \\ 1 - p (\text {N o}), & \text {i f o u t p u t i s N o} \end{array} \right. \tag {2}
$$

where  $p(\mathrm{Yes} / \mathrm{No})$  denotes the probability of LLMs generating Yes or No, and the relevance score is normalized into the range [0, 2].

The above two methods rely on the log probability of LLM, which is often unavailable for LLM API. For example, at the time of writing, OpenAI's ChatCompletion API does not provide the log-probability of generation<sup>5</sup>.

# B.3 Instructional Permutation Generation

The proposed instructional permutation generation is a listwise approach, which directly assigns each passage  $p_i$  a unique ranking identifier  $a_i$  (e.g., [1], [2]) and places it at the beginning of  $p_i$ :  $p_i' = \operatorname{Concat}(a_i, p_i)$ . Subsequently, a generative LLM is instructed to generate a permutation of these identifiers:  $\mathbf{Perm} = f(q, p_1', \dots, p_M')$ , where the permutation  $\mathbf{Perm}$  indicates the rank of the identifiers  $a_i$  (e.g., [1], [2]). We then simply map the identifiers  $a_i$  to the passages  $p_i$  to obtain the ranking of the passages.

<table><tr><td>Domain</td><td>Question</td><td>Reference Answer</td></tr><tr><td>Sport</td><td>What is Messi&#x27;s annual income after transferring to Miami?</td><td>$50M-$60M</td></tr><tr><td>Sport</td><td>How many goals did Haaland scored in the 2023 Champions League Final?</td><td>0</td></tr><tr><td>Sport</td><td>Where did Benzema go after leaving Real Madrid?</td><td>Saudi Arabia</td></tr><tr><td>Sport</td><td>Where was the 2023 Premier League FA Cup Final held?</td><td>Wembley Stadium</td></tr><tr><td>Sport</td><td>Who won 2023 Laureus World Sportsman Of The Year Award?</td><td>Lionel Messi</td></tr><tr><td>Sport</td><td>Who wins NBA Finals 2023?</td><td>Denver Nuggets</td></tr><tr><td>Tech</td><td>What is the screen resolution of vision pro?</td><td>4K with one eye</td></tr><tr><td>Tech</td><td>What is the name of the combined Deepmind and Google Brain?</td><td>Google DeepMind</td></tr><tr><td>Tech</td><td>How much video memory does the DGX GH200 have?</td><td>144TB</td></tr><tr><td>Tech</td><td>What are the new features of PyTorch 2?</td><td>faster, low memory, dynamic shapes</td></tr><tr><td>Tech</td><td>Who will be the CEO of Twitter after Elon Musk is no longer the CEO?</td><td>Linda Yaccarino</td></tr><tr><td>Tech</td><td>What are the best papers of CVPR 2023?</td><td>Visual Programming: Compositional [...]</td></tr><tr><td>Movie</td><td>Who sang the theme song of Transformers Rise of the Beasts?</td><td>Notorious B.I.G</td></tr><tr><td>Movie</td><td>Who is the villain in The Flash?</td><td>Eobard Thawne/Professor Zoom</td></tr><tr><td>Movie</td><td>How many different Spider-Men are there in Across the Spider-Verse?</td><td>280 variations</td></tr><tr><td>Movie</td><td>Who does Momoa play in Fast X?</td><td>Dante</td></tr><tr><td>Movie</td><td>The Little Mermaid first week box office?</td><td>$163.8 million worldwide</td></tr><tr><td>Movie</td><td>Which film was the 2023 Palme d&#x27;Or winner?</td><td>Anatomy of a Fall</td></tr><tr><td>Other</td><td>Where will Blackpink&#x27;s 2023 world tour concert in France be held?</td><td>the Stade de France</td></tr><tr><td>Other</td><td>What is the release date of song Middle Ground?</td><td>May 19, 2023</td></tr><tr><td>Other</td><td>Where did the G7 Summit 2023 take place?</td><td>Hiroshima</td></tr></table>

Table 8: Questions and reference answers on NovelEval-2306.

# C NovelEval-2306

Table 8 lists the collected 21 questions. These questions come from four domains and include hot topics from the past few months. For each question, we used Google search to obtain 20 passages. When using Google search, in order to avoid all pages containing the answer, we used not only the question itself as a search query, but also the entities that appear in the question as an alternative search query to obtain some pages that are relevant but do not contain the answer. For example, for the first question "What is Messi's annual income after transferring to Miami?", we used "Messi" and "Messi transferring" as search queries to get some pages that do not contain the answer. When searching, we collected the highest-ranking web pages, news, and used a paragraph or paragraphs from the web pages related to the search term as candidate passages. Table 9 shows the statistical information of the data. All of the LLMs (including

gpt-4-0314 and gpt-4-0613) we tested achieved  $0\%$  question-answering accuracy on the obtained test set.

We searched for 20 candidate passages for each question using Google search. These passages were manually labeled for relevance by a group of annotators, including the authors and their highly educated colleagues. To ensure consistency, the annotation process was repeated twice. Each passage was assigned a relevance score: 0 for not relevant, 1 for partially relevant, and 2 for relevant. When evaluating the latest LLMs, we found that all non-retrieval-augmented models tested achieved  $0\%$  accuracy in answering the questions on the test set. This test set provides a reasonable evaluation of the latest LLMs at the moment. Since LLMs may be continuously trained on new data, the proposed test set should be continuously updated to counteract the contamination of the test set by LLMs.

<table><tr><td>Number of questions</td><td>21</td></tr><tr><td>Number of passages</td><td>420</td></tr><tr><td>Number of relevance annotation</td><td>420</td></tr><tr><td>Average number words of passage</td><td>149</td></tr><tr><td>Number of score 0</td><td>290</td></tr><tr><td>Number of score 1</td><td>40</td></tr><tr><td>Number of score 2</td><td>90</td></tr></table>

Table 9: Data Statistics of NovelEval.

# D Implementation Details

# D.1 Training Configuration

We use DeBERTa-V3-base, which concatenates the query and passage with a [SEP] token and utilizes the representation of the [CLS] token. To generate candidate passages, we randomly sample 10k queries and use BM25 to retrieve 20 passages for each query. We then re-rank the candidate passages using the gpt-3.5-turbo API with permutation generation instructions, at a cost of approximately $40. During training, we employ a batch size of 32 and utilize the AdamW optimizer with a constant learning rate of  $5 \times 10^{-5}$ . The model is trained for two epochs. Additionally, we implement models using the original MS MARCO labels for comparison.

The LLaMA-7B model is optimized with the AdamW optimizer, a constant learning rate of  $5 \times 10^{-5}$  and with mixed precision of bf16 and Deepspeed Zero3 strategy. All the experiments are conducted on 8 A100-40G GPUs.

Figure 5 illustrates the detailed model architecture of BERT-like and GPT-like specialized models.

(a) BERT-like Specialized Model

(b) GPT-like Specialized Model  
Figure 5: Model architecture of BERT-like and GPT-like specialized models.

# D.2 Training Objective

Using the permutation generated by ChatGPT, we consider the following losses to optimize the student model:

Listwise Cross-Entropy (CE) (Bruch et al., 2019). Listwise CE is the wide-use loss for passage ranking, which considers only one positive passage and defines the list-wise softmax cross-entropy on all candidate's passages:

$$
\mathcal {L} _ {\mathrm {L i s t w i s e \_ C E}} = - \sum_ {i = 1} ^ {M} \mathbb {1} _ {r _ {i} = 1} \log (\frac {\exp (s _ {i})}{\sum_ {j} \exp (s _ {j})})
$$

where  $\mathbb{1}$  is the indicator function.

RankNet (Burges et al., 2005). RankNet is a pairwise loss that measures the correctness of relative passage orders:

$$
\mathcal {L} _ {\text {R a n k N e t}} = \sum_ {i = 1} ^ {M} \sum_ {j = 1} ^ {M} \mathbb {1} _ {r _ {i} <   r _ {j}} \log \left(1 + \exp \left(s _ {i} - s _ {j}\right)\right)
$$

when using permutation generated by ChatGPT, we can construct  $M(M - 1) / 2$  pairs.

LambdaLoss (Wang et al., 2018). The LambdaLoss further accounts for the nDCG gains of the model ranks. LambdaLoss uses the student model's rank, denoted as  $\pi = (\pi_1, \dots, \pi_M)$ , where  $\pi_i$  is the model predicted rank of  $p_i$  with a similar definition with ChatGPT rank  $R$ . The loss function is defined as:

$$
\mathcal {L} _ {\mathrm {L a m b d a}} = \sum_ {r _ {i} <   r _ {j}} \Delta \mathrm {N D C G} \log_ {2} (1 + \exp (s _ {i} - s _ {j}))
$$

in which  $\Delta$  NDCG is the delta of NDCG which could be computed as  $\Delta$  NDCG =  $|G_i - G_j||\frac{1}{D(\pi_i)} -\frac{1}{D(\pi_j)}|$ , where  $D(\pi_i)$  and  $D(\pi_j)$  are the position discount functions and  $G_{i}$  and  $G_{j}$  are the gain functions used in NDCG (Wang et al., 2018).

Pointwise Binary Cross-Entropy (BCE). We also include the Pointwise BCE as the baseline loss for supervised methods, which is calculated based on each query-document pair independently:

$$
\mathcal {L} _ {\mathrm {B C E}} = - \sum_ {i = 1} ^ {M} \mathbb {1} _ {r _ {i} = 1} \log \sigma (s _ {i}) + \mathbb {1} _ {r _ {i} \neq 1} \log \sigma (1 - s _ {i})
$$

where  $\sigma (x) = \frac{1}{1 + \exp(-x)}$  is the logistic function.

# E Baselines Details

We include state-of-the-art supervised and unsupervised passage re-ranking methods for comparison. The supervised baselines are:

- monoBERT (Nogueira and Cho, 2019): A cross-encoder re-ranker based on BERT-large, trained on MS MARCO.  
- monoT5 (Nogueira et al., 2020): A sequence-to-sequence re-ranker that uses T5 to calculate the relevance score $^6$ .  
- Cohere Rerank: A passage reranking API rerank-english-v2.0 developed by Cohere<sup>7</sup>. Cohere does not provide details on the structure and training method of the model.  
- mmarcoCE (Bonifacio et al., 2021): A 12-layer mMiniLM-v2 cross-encoder model trained on mmarco, a translated version of MS MARCO. mmarcoCE serves as a baseline for Mr.TyDi.

# The unsupervised baselines are:

- UPR (Sachan et al., 2022): Unsupervised passage ranking with instructional query generation. Due to its superior performance, we use the FLAN-T5-XL (Chung et al., 2022) as the LLM of UPR.  
- InPars (Bonifacio et al., 2022): monoT5-3B trained on pseudo data generated by GPT-3.  
- Promptagator++ (Dai et al., 2023): A 110M cross-encoder re-ranker trained on pseudo queries generated by FALN 137B.

<table><tr><td>Method</td><td>Repetition↓</td><td>Missing↓</td><td>Rejection</td><td>RBO↑</td></tr><tr><td>text-davinci-003</td><td>0</td><td>280</td><td>0</td><td>72.30</td></tr><tr><td>gpt-3.5-turbo</td><td>14</td><td>153</td><td>7</td><td>81.49</td></tr><tr><td>gpt-4</td><td>0</td><td>1</td><td>11</td><td>82.08</td></tr></table>

Table 10: Analysis of model stability on TREC. Repetition refers to the number of times the model generates duplicate passage identifiers. Missing refers to the number of missing passage identifiers in model output. Rejection refers to the number of times the model rejects to perform the ranking. RBO, i.e., rank biased overlap, refers to the consistency of the model in ranking the same group of passages twice.

# F Model Behavior Analysis

In the permutation generation method, the ranking of passages is determined by the list of model-output passage identifiers. However, we have observed that the models do not always produce the desired output, as evidenced by occasional duplicates or missing identifiers in the generated text. In Table 10, we present quantitative results of unexpected model behavior observed during experiments with the GPT models.

Repetition. The repetition metric measures the occurrence of duplicate passage identifiers generated by the model. The results indicate that ChatGPT produced 14 duplicate passage identifiers during re-ranking 97 queries on two TREC datasets, whereas text-davinci-003 and GPT-4 did not exhibit any duplicates.

Missing. We conducted a count of the number of times the model failed to include all passages in the re-ranked permutation output<sup>9</sup>. Our findings revealed that text-davinci-003 has the highest number of missing passages, totaling 280 instances. ChatGPT also misses a considerable number of passages, occurring 153 times. On the other hand, GPT-4 demonstrates greater stability, with only one missing passage in total. These results suggest that GPT-4 has higher reliability in generating permutations, which is critical for effective ranking.

Rejection. We have observed instances where the model refuses to re-rank passages, as evidenced by responses such as "None of the provided passages is directly relevant to the query ..." To quantify this behavior, we count the number of times this occurred and find that GPT-4 rejects ranking the most frequently, followed by ChatGPT, while the Davinci model never refused to rank. This finding suggests that chat LLMs tend to be more adaptable compared to completion LLMs, and may exhibit more subjective responses. Note that we do not explicitly prohibit the models from rejecting ranking in the instructions, as we find that it does not significantly impact the overall ranking performance.

RBO. The sliding windows strategy involves re-ranking the top-ranked passages from the previous window in the next window. The models are expected to produce consistent rankings in two windows for the same group of passages. To measure the consistency of the model's rankings, we use RBO (rank biased overlap $^{10}$ ), which calculates the similarity between the two ranking results. The findings turn out that ChatGPT and GPT-4 are more consistent in ranking passages compared to the Davinci model. GPT-4 also slightly outperforms ChatGPT in terms of the RBO metric.

# G Analysis on Hyperparameters of Sliding Window

To analyze the influence of parameters of the sliding window strategy, we adjust the window size and set the step size to half of the window size. The main motivation for this setup is to keep the expected

<table><tr><td>API</td><td>Instruction</td><td>Tokens</td><td>Requests</td><td>USD</td></tr><tr><td>text-curie-001</td><td>Relevance generation</td><td>52,970</td><td>100</td><td>0.106</td></tr><tr><td>text-curie-001</td><td>Query generation</td><td>10,954</td><td>100</td><td>0.022</td></tr><tr><td>text-davinci-003</td><td>Query generation</td><td>11,269</td><td>100</td><td>0.225</td></tr><tr><td>text-davinci-003</td><td>Permutation generation</td><td>17,370</td><td>10</td><td>0.347</td></tr><tr><td>gpt-3.5-turbo</td><td>Permutation generation</td><td>19,960</td><td>10</td><td>0.040</td></tr><tr><td>gpt-4</td><td>Permutation generation</td><td>19,890</td><td>10</td><td>0.596</td></tr><tr><td>-rerank top-30</td><td>Permutation generation</td><td>3,271</td><td>1</td><td>0.098</td></tr></table>

Table 11: Average token cost, number API request, and $USD per query on TREC.  

<table><tr><td>Window size</td><td>Step size</td><td>nDCG@1</td><td>nDCG@5</td><td>nDCG@10</td></tr><tr><td>20</td><td>10</td><td>75.58</td><td>70.50</td><td>67.05</td></tr><tr><td>40</td><td>20</td><td>78.30</td><td>71.32</td><td>65.51</td></tr><tr><td>60</td><td>30</td><td>75.97</td><td>69.23</td><td>65.03</td></tr><tr><td>80</td><td>40</td><td>72.09</td><td>70.59</td><td>65.57</td></tr></table>

Table 12: Analysis on Hyperparameters of Sliding Window on TREC-DL19.

overhead of the method (number of tokens required for computation) low; i.e., most tokens in this setup are used for PG only twice. The experimental results are shown in Table 12<sup>11</sup>. The results show that the effect varies over a certain range of arrivals for different values of window size: window size=20 performs best in terms of nDCG@10, while window size=40 performs best in terms of nDCG@5 and nDCG@1. We speculate that a larger window size will increase the model's ranking horizon but will also present challenges in processing long contexts and large numbers of items.

# H API Cost

In Table 11, we provide details on the average token cost, API request times, and USD cost per query. In terms of average token cost, the relevance generation method is the most expensive, as it requires 4 in-context demonstrations. On the other hand, the permutation generation method incurs higher token costs compared to the query generation method, as it involves the repeated processing of passages in sliding windows. Regarding the number of requests, the permutation generation method requires 10 requests for sliding windows, while other methods require 100 requests for re-ranking 100 passages. In terms of average USD cost, GPT-4 is the most expensive, with a cost of  $0.596 per query. However, using GPT-4 for re-ranking the top-30 passages can result in significant cost savings, with a cost of$ 0.098 per query for GPT-4 usage, while still achieving good results. As a result, we only utilize GPT-4 for re-ranking the top 30 passages of ChatGPT on BEIR and Mr.TyDi. The total cost of our experiments with GPT-4 amounts to $556.

Since the experiments with ChatGPT and GPT-4 are conducted using the OpenAI API, the running time is contingent on the OpenAI service, e.g., API latency. Besides, the running time can also vary across different API versions and network environments. In our testing conditions, the average latency for API calls for gpt-3.5-turbo and gpt-4 was around 1.1 seconds and 3.2 seconds, respectively. Our proposed sliding window-based permutation generation approach requires 10 API calls per query to re-rank 100 passages. Consequently, the average running time per query is 11 seconds for gpt-3.5-turbo and 32 seconds for gpt-4.

# I Results of Specialized Models

Table 13 lists the detailed results of specialized models on TREC and BEIR.

<table><tr><td>Method</td><td>DL19</td><td>DL20</td><td>Covid</td><td>NFCorpus</td><td>Touche</td><td>DBPedia</td><td>SciFact</td><td>Signal</td><td>News</td><td>Robust04</td><td>BEIR (Avg)</td></tr><tr><td>BM25</td><td>50.58</td><td>47.96</td><td>59.47</td><td>30.75</td><td>44.22</td><td>31.80</td><td>67.89</td><td>33.05</td><td>39.52</td><td>40.70</td><td>43.42</td></tr><tr><td colspan="12">Supervised train on MS MRACO</td></tr><tr><td>monoBERT (340M)</td><td>70.50</td><td>67.28</td><td>70.01</td><td>36.88</td><td>31.75</td><td>41.87</td><td>71.36</td><td>31.44</td><td>44.62</td><td>49.35</td><td>47.16</td></tr><tr><td>monoT5 (220M)</td><td>71.48</td><td>66.99</td><td>78.34</td><td>37.38</td><td>30.82</td><td>42.42</td><td>73.40</td><td>31.67</td><td>46.83</td><td>51.72</td><td>49.07</td></tr><tr><td>monoT5 (3B)</td><td>71.83</td><td>68.89</td><td>80.71</td><td>38.97</td><td>32.41</td><td>44.45</td><td>76.57</td><td>32.55</td><td>48.49</td><td>56.71</td><td>51.36</td></tr><tr><td>Cohere Rerank-v2</td><td>73.22</td><td>67.08</td><td>81.81</td><td>36.36</td><td>32.51</td><td>42.51</td><td>74.44</td><td>29.60</td><td>47.59</td><td>50.78</td><td>49.45</td></tr><tr><td colspan="12">Unsupervised instructional permutation generation</td></tr><tr><td>ChatGPT</td><td>65.80</td><td>62.91</td><td>76.67</td><td>35.62</td><td>36.18</td><td>44.47</td><td>70.43</td><td>32.12</td><td>48.85</td><td>50.62</td><td>49.37</td></tr><tr><td>GPT-4</td><td>75.59</td><td>70.56</td><td>85.51</td><td>38.47</td><td>38.57</td><td>47.12</td><td>74.95</td><td>34.40</td><td>52.89</td><td>57.55</td><td>53.68</td></tr><tr><td>Specialized Models</td><td colspan="11">train on MARCO labels or ChatGPT predicted permutations</td></tr><tr><td>MARCO Pointwise BCE</td><td>65.57</td><td>56.72</td><td>70.82</td><td>33.10</td><td>17.08</td><td>32.28</td><td>55.37</td><td>19.30</td><td>41.52</td><td>46.00</td><td>39.43</td></tr><tr><td>MARCO Listwise CE</td><td>65.99</td><td>57.97</td><td>66.31</td><td>32.61</td><td>20.15</td><td>30.79</td><td>37.57</td><td>18.09</td><td>38.11</td><td>39.93</td><td>35.45</td></tr><tr><td>MARCO RankNet</td><td>66.34</td><td>58.51</td><td>70.29</td><td>34.23</td><td>20.27</td><td>29.62</td><td>49.01</td><td>23.22</td><td>39.82</td><td>43.87</td><td>38.79</td></tr><tr><td>MARCO LambdaLoss</td><td>64.82</td><td>56.16</td><td>72.86</td><td>34.20</td><td>19.51</td><td>32.55</td><td>51.88</td><td>26.22</td><td>42.47</td><td>45.28</td><td>40.62</td></tr><tr><td>ChatGPT Listwise CE</td><td>65.39</td><td>58.80</td><td>76.29</td><td>35.73</td><td>38.19</td><td>40.24</td><td>64.49</td><td>31.37</td><td>47.61</td><td>48.00</td><td>47.74</td></tr><tr><td>ChatGPT RankNet</td><td>65.75</td><td>59.34</td><td>81.26</td><td>36.57</td><td>39.03</td><td>42.10</td><td>68.77</td><td>31.55</td><td>52.54</td><td>52.44</td><td>50.53</td></tr><tr><td>ChatGPT LambdaLoss</td><td>67.17</td><td>60.56</td><td>80.63</td><td>36.74</td><td>36.73</td><td>43.75</td><td>68.21</td><td>32.58</td><td>49.00</td><td>50.51</td><td>49.77</td></tr><tr><td>deberta-v3-xsmall (70M)</td><td>64.75</td><td>55.07</td><td>78.21</td><td>35.95</td><td>35.42</td><td>41.37</td><td>67.86</td><td>30.04</td><td>47.68</td><td>49.91</td><td>48.31</td></tr><tr><td>deberta-v3-small (142M)</td><td>67.85</td><td>58.84</td><td>78.88</td><td>36.55</td><td>36.16</td><td>40.99</td><td>66.66</td><td>30.29</td><td>49.17</td><td>49.73</td><td>48.55</td></tr><tr><td>deberta-v3-base (184M)</td><td>70.28</td><td>62.52</td><td>80.81</td><td>36.15</td><td>37.25</td><td>44.06</td><td>71.70</td><td>32.45</td><td>50.84</td><td>51.33</td><td>50.57</td></tr><tr><td>deberta-v3-large (435M)</td><td>70.66</td><td>67.15</td><td>84.64</td><td>38.48</td><td>39.27</td><td>47.36</td><td>74.18</td><td>32.53</td><td>51.19</td><td>56.55</td><td>53.03</td></tr><tr><td>deberta-v3-large 5K</td><td>70.93</td><td>64.32</td><td>84.43</td><td>38.66</td><td>40.72</td><td>46.28</td><td>73.88</td><td>31.93</td><td>52.24</td><td>55.89</td><td>53.00</td></tr><tr><td>deberta-v3-large 3K</td><td>70.79</td><td>63.91</td><td>84.21</td><td>38.73</td><td>39.83</td><td>45.74</td><td>74.41</td><td>31.92</td><td>52.29</td><td>57.42</td><td>53.07</td></tr><tr><td>deberta-v3-large 1K</td><td>69.90</td><td>64.81</td><td>83.38</td><td>38.94</td><td>36.65</td><td>44.46</td><td>71.96</td><td>30.19</td><td>50.73</td><td>53.74</td><td>51.26</td></tr><tr><td>deberta-v3-large 500</td><td>69.71</td><td>62.00</td><td>83.54</td><td>37.23</td><td>33.68</td><td>44.56</td><td>70.48</td><td>28.70</td><td>45.64</td><td>42.67</td><td>48.31</td></tr><tr><td>deberta-v3-large label 10K</td><td>66.61</td><td>57.26</td><td>74.36</td><td>33.94</td><td>18.09</td><td>34.95</td><td>35.35</td><td>21.38</td><td>39.00</td><td>44.94</td><td>37.75</td></tr><tr><td>deberta-v3-large label 5K</td><td>68.98</td><td>61.38</td><td>80.73</td><td>35.68</td><td>20.48</td><td>37.34</td><td>54.63</td><td>24.25</td><td>36.94</td><td>51.13</td><td>42.64</td></tr><tr><td>deberta-v3-large label 3K</td><td>67.41</td><td>60.42</td><td>79.82</td><td>35.49</td><td>24.54</td><td>37.39</td><td>47.31</td><td>23.29</td><td>39.87</td><td>50.65</td><td>42.29</td></tr><tr><td>deberta-v3-large label 1K</td><td>65.55</td><td>60.93</td><td>77.70</td><td>33.29</td><td>23.36</td><td>36.38</td><td>31.10</td><td>21.71</td><td>34.28</td><td>38.31</td><td>37.01</td></tr><tr><td>deberta-v3-large label 500</td><td>60.59</td><td>54.45</td><td>76.20</td><td>32.93</td><td>19.66</td><td>31.54</td><td>45.66</td><td>13.99</td><td>33.48</td><td>44.49</td><td>37.24</td></tr><tr><td>deberta-v3-large monoT5-3B</td><td>73.05</td><td>68.82</td><td>84.78</td><td>38.55</td><td>34.43</td><td>43.61</td><td>75.45</td><td>30.75</td><td>49.85</td><td>56.80</td><td>51.78</td></tr><tr><td>deberta-v3-large chatgpt+label</td><td>72.42</td><td>67.30</td><td>85.96</td><td>38.75</td><td>35.06</td><td>45.43</td><td>71.81</td><td>28.52</td><td>45.91</td><td>55.57</td><td>50.88</td></tr><tr><td>deberta-v3-base label 10k</td><td>65.66</td><td>59.84</td><td>71.63</td><td>34.65</td><td>16.53</td><td>32.59</td><td>34.65</td><td>22.64</td><td>37.60</td><td>44.02</td><td>36.79</td></tr><tr><td>deberta-v3-small label 10k</td><td>63.63</td><td>52.83</td><td>68.17</td><td>30.48</td><td>18.12</td><td>31.72</td><td>33.62</td><td>18.02</td><td>34.57</td><td>36.09</td><td>33.85</td></tr><tr><td>deberta-v3-xsmall label 10k</td><td>60.89</td><td>51.15</td><td>63.58</td><td>28.67</td><td>14.87</td><td>27.12</td><td>20.60</td><td>18.97</td><td>32.61</td><td>32.67</td><td>29.89</td></tr><tr><td>llama-7b</td><td>71.33</td><td>66.06</td><td>78.23</td><td>37.60</td><td>34.87</td><td>45.46</td><td>76.13</td><td>34.17</td><td>51.79</td><td>55.22</td><td>51.68</td></tr><tr><td>vicuna-7b</td><td>71.80</td><td>66.89</td><td>78.32</td><td>36.87</td><td>31.81</td><td>45.40</td><td>74.23</td><td>34.28</td><td>51.13</td><td>52.91</td><td>50.62</td></tr><tr><td>llama-7b 10k label</td><td>65.22</td><td>56.85</td><td>75.36</td><td>36.24</td><td>20.88</td><td>37.34</td><td>69.04</td><td>25.22</td><td>41.21</td><td>49.21</td><td>44.31</td></tr><tr><td>llama-7b 5k label</td><td>69.24</td><td>58.97</td><td>80.49</td><td>37.55</td><td>28.23</td><td>39.66</td><td>71.79</td><td>26.04</td><td>44.09</td><td>53.83</td><td>47.71</td></tr></table>

Table 13: Results (nDCG@10) on TREC and BEIR. Best performing specialized and overall system(s) are marked bold. The specialized models are fine-tined on sampled queries using relevance judgements from MARCO or ChatGPT.

# Footnotes:

Page 0: *Work done during an internship at Baidu. † Corresponding authors. 
Page 4: <https://github.com/castorini/pyserini> 2https://txt.cohere.com/erank/ 
Page 6: 3https://platform.openai.com/docs/api-reference 
Page 7: 4Note that all models are trained using the RankNet loss for a fair comparison. 
Page 14: <sup>5</sup>https://platform.openai.com/docs/api-reference/chat/create 
Page 16: <sup>6</sup>https://huggingface.co/castorini/monot5-3b-msmarco-10k <sup>7</sup>https://cohere.com/erank <sup>8</sup>https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 
Page 17: In our implementation, we append the missing passages in their original order at the end of the re-ranked passages. 10https://github.com/changyaochen/rbo 
Page 18: <sup>11</sup>Note that the results are obtained using gpt-3.5-turbo-16k API for managing long context. 
