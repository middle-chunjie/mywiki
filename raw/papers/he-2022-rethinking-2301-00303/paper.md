# Rethinking with Retrieval: Faithful Large Language Model Inference

Hangfeng He†* Hongming Zhang‡ Dan Roth§

†University of Rochester

†Tencent AI Lab, Seattle

$^{\S}$ University of Pennsylvania

hanfeng.he@rochester.edu,hongmzhang@global.tencent.com

danroth@seas.upenn.edu

# Abstract

Despite the success of large language models (LLMs) in various natural language processing (NLP) tasks, the stored knowledge in these models may inevitably be incomplete, out-of-date, or incorrect. This motivates the need to utilize external knowledge to assist LLMs. Unfortunately, current methods for incorporating external knowledge often require additional training or fine-tuning, which can be costly and may not be feasible for LLMs. To address this issue, we propose a novel post-processing approach, rethinking with retrieval (RR), which retrieves relevant external knowledge based on the decomposed reasoning steps obtained from the chain-of-thought (CoT) prompting. This lightweight approach does not require additional training or fine-tuning and is not limited by the input length of LLMs. We evaluate the effectiveness of RR through extensive experiments with GPT-3 on three complex reasoning tasks: commonsense reasoning, temporal reasoning, and tabular reasoning. Our results show that RR can produce more faithful explanations and improve the performance of LLMs.<sup>1</sup>

# 1 Introduction

Large language models (LLMs) have shown exceptional performance across various tasks through in-context learning without task-specific training or fine-tuning (Brown et al., 2020; Chowdhery et al., 2022; Zhang et al., 2022; Ouyang et al., 2022). Recent progress in prompting (Wei et al., 2022; Zhou et al., 2022; Kojima et al., 2022) and decoding (Wang et al., 2022) has made it feasible for LLMs to tackle tasks that demand complex reasoning.

Figure 1: An overview of three approaches for using LLMs: (a) Standard prompting for generating a prediction in response to a query. (b) Chain-of-thought prompting for generating both an explanation and a prediction in response to a query. (c) Rethinking with retrieval, our proposed approach for using the decomposed reasoning steps obtained from chain-of-thought prompting to retrieve relevant external knowledge for LLMs, leading to more faithful explanations and improved predictions in response to a query.

However, the knowledge stored in LLMs might inevitably be incomplete, out-of-date, or incorrect. As a result, external sources of knowledge, such as Wikipedia, may be essential for the successful deployment of LLMs for real-world applications. Previously, people tried to utilize knowledge for smaller language models (LMs), such as T5 (Raffel et al., 2020), BERT (Devlin et al., 2019), and RoBERTa (Liu et al., 2019). However, these methods often require additional training or fine-tuning, which can be costly and thus impractical for LLMs.

In this paper, we present a post-processing approach called rethinking with retrieval (RR) for utilizing external knowledge in LLMs. Our method begins by using the chain-of-thought (CoT) prompting method (Wei et al., 2022) to generate a diverse set of reasoning paths, as described in Wang et al. (2022). We then use each reasoning step in those paths to retrieve relevant external knowledge, which enables RR to provide

more faithful explanations and more accurate predictions, as illustrated in Figure 1.

We evaluate the effectiveness of our proposed method, RR, on three complex reasoning tasks: commonsense reasoning, temporal reasoning, and tabular reasoning, using GPT-3 175B (Brown et al., 2020) and different external knowledge sources: Wikipedia, Wikidata (Vrandecic and Krötzsch, 2014), WordNet (Miller, 1995), and Conceptnet (Speer et al., 2017). The results demonstrate that RR consistently outperforms all baselines on all three tasks without requiring additional training or fine-tuning, indicating the superiority of our approach in leveraging external knowledge to enhance the performance of LLMs.

# 2 Related Work

Enhancing LMs through retrieval. Retrieval-enhanced LMs have received significant attention as a means of improving performance through the incorporation of external knowledge. For example, the k-most similar training contexts can be retrieved to improve the estimation of the next word distribution in both the training stage (Borgeaud et al., 2021) and the inference stage (Khandelwal et al., 2020). Furthermore, search query generators have been adopted to generate search queries for search engines to retrieve relevant documents (Komeili et al., 2022; Shuster et al., 2022; Thoppilan et al., 2022). Other approaches have utilized retrieved documents as the additional context in generation tasks (Joshi et al., 2020; Guu et al., 2020; Lewis et al., 2020). Nakano et al. (2021) instead use human feedback in a text-based web-browsing environment. Among these previous works, Khandelwal et al. (2020) is most closely related to our approach. However, they focus on improving local inference by using the nearest neighbor datastore constructed from training data, whereas we focus on conducting faithful inference using external knowledge. In contrast to other aforementioned approaches, which require training or fine-tuning to incorporate retrieved knowledge, we propose a post-processing method for leveraging retrieved knowledge without additional training or fine-tuning.

Incorporating external knowledge into LMs. Significant effort has been devoted to leveraging external knowledge to improve the reasoning ability of LMs. Previous work has incorporated external knowledge sources such as WordNet (Miller,

1995) and ConceptNet (Speer et al., 2017) to enhance LMs for tabular reasoning tasks (Neeraja et al., 2021; Varun et al., 2022). Explicit rules have also been added to inputs to improve reasoning ability over implicit knowledge (Talmor et al., 2020). In addition, explicit knowledge from Wikidata (Vrandecic and Krötzsch, 2014) and implicit knowledge in LLMs have been integrated into a transformer (Vaswani et al., 2017) for visual question answering (Gui et al., 2021). Nye et al. (2021) instead introduces a symbolic reasoning module to improve coherence and consistency in LLMs. Among these previous works, Nye et al. (2021) is the most relevant to our approach. Still, they focus on incorporating logical constraints to improve coherence and consistency, whereas we aim to improve the faithfulness of explanations through the use of external knowledge. In contrast to other aforementioned approaches that incorporate external knowledge before generation and require additional training or fine-tuning, our proposal leverages external knowledge in a post-processing manner to enhance LMs without additional training or fine-tuning.

Uncovering latent Knowledge in LLMs. There has been a line of work exploring the knowledge hidden within LLMs for reasoning. This has included the use of careful prompting to encourage LLMs to generate explanations in the reasoning process, such as through chain of thought prompting in few-shot (Wei et al., 2022) or zero-shot (Kojima et al., 2022) learning, or through the use of scratchpads for intermediate computation (Nye et al., 2022). In addition, various methods based on sampling a diverse set of reasoning paths in LLMs have been proposed, including training verifiers to judge the correctness of model completions (Cobbe et al., 2021), calibrating model predictions based on the reliability of the explanations (Ye and Durrett, 2022), and promoting self-consistency over diverse reasoning paths (Wang et al., 2022). Zelikman et al. (2022) instead iteratively bootstrap the ability of LLMs to generate high-quality rationales from a few initial examples. Liu et al. (2022) further propose generating knowledge from LLMs, which is then used as additional input to improve commonsense reasoning. In contrast to this line of work, our proposal focuses on leveraging external knowledge to enhance LLMs, while they aim to explore the knowledge hidden within LLMs.

# 3 Rethinking with Retrieval

LLMs have been shown to generate incorrect supporting facts from time to time, even when they accurately capture the perspective needed to answer a question. This phenomenon highlights intrinsic issues in the way LLMs store and retrieve knowledge, including (1) the presence of out-of-date, incorrect, or missing relevant knowledge in the pre-training corpus; (2) incorrect memorization of relevant knowledge during pre-training; and (3) incorrect retrieval of relevant knowledge during the inference stage. To address these issues, we propose the use of RR, which leverages external knowledge through the retrieval of relevant information based on decomposed reasoning steps.

Overview. Given a query  $Q$ , we utilize chain-of-thought prompting to generate a diverse set of reasoning paths  $R_{1}, R_{2}, \dots, R_{N}$ , where each reasoning path  $R_{i}$  consists of an explanation  $E_{i}$  followed by a prediction  $P_{i}$ . After that, we retrieve relevant knowledge  $K_{1}, \dots, K_{M}$  from a suitable knowledge base  $\mathcal{KB}$  to support the explanation in each reasoning path, and select the prediction  $\hat{P}$  that is most faithful to this knowledge. To better illustrate our proposal, we use "Did Aristotle use a laptop?" as a running example in this work.

Chain-of-thought prompting. In contrast to standard prompting, CoT prompting (Wei et al., 2022) includes demonstrations of step-by-step reasoning examples in the prompt to produce a series of short sentences that capture the reasoning process. For instance, given the question "Did Aristotle use a laptop?", CoT prompting aims to generate the complete reasoning path "Aristotle died in 322 BC. The first laptop was invented in 1980. Thus, Aristotle did not use a laptop. So the answer is no." rather than simply outputs "No." Empirical results show that CoT prompting significantly improves the performance of LLMs on many multi-step reasoning tasks. Therefore, we adopt CoT prompting to obtain both explanation  $E$  and prediction  $P$  for the query  $Q$ .

Sampling diverse reasoning paths. Similar to Wang et al. (2022), we sample a diverse set of reasoning paths  $R_{1}, R_{2}, \dots, R_{N}$  rather than only considering the greedy path as in Wei et al. (2022). For the question "Did Aristotle use a laptop?", the potential reasoning paths can be as follows:

$(R_{1})$  Aristotle died in 2000. The first laptop was

invented in 1980. Thus, Aristotle used a laptop. So the answer is yes.

$(R_{2})$  Aristotle died in 322BC. The first laptop was invented in 2000. Thus, Aristotle did not use a laptop. So the answer is no.  
$(R_{3})$  Aristotle died in 322BC. The first laptop was invented in 1980. Thus, Aristotle did not use a laptop. So the answer is no.

Knowledge retrieval. Different knowledge bases can be used to address different tasks. For example, to address the question "Did Aristotle use a laptop?", we can use Wikipedia as the external knowledge base  $\mathcal{KB}$ . Information retrieval techniques can be applied to retrieve the relevant knowledge  $K_{1},\dots K_{M}$  from Wikipedia based on the decomposed reasoning steps. Ideally, we would obtain the following two paragraphs from Wikipedia for this question:

$(K_{1})$  Aristotle (384-322 BC) was a Greek philosopher and polymath during the Classical period in Ancient Greece. ...  
$(K_{2})$  The Epson HX-20, the first laptop computer, was invented in 1980. ...

Faithful inference. The faithfulness of each reasoning path  $R_{i}$  can be estimated using a function  $f_{\mathcal{KB}}(R_i)$ , which is based on relevant knowledge  $K_{1}, \dots, K_{M}$  retrieved from the knowledge base  $\mathcal{KB}$ . The final prediction is obtained through the application of the following inference procedure:

$$
\hat {P} = \underset {P _ {i} \in \left\{P _ {1}, \dots , P _ {N} \right\}} {\arg \max } \sum_ {i = 1} ^ {N} \mathbb {1} \left(P _ {i} = P\right) f _ {\mathcal {K B}} \left(R _ {i}\right), \tag {1}
$$

where  $P_{i}$  denotes the corresponding prediction in the reasoning path  $R_{i}$ . This inference procedure is designed to identify the most faithful prediction  $\hat{P}$  to the knowledge base among all predictions in the  $N$  reasoning paths. For instance, in the running example, given reasoning paths  $R_{1}, R_{2}, R_{3}$  and the retrieved knowledge  $K_{1}, K_{2}$ , the above inference procedure would output the prediction "So the answer is no.", as it is supported by both  $R_{2}$  and  $R_{3}$  and has a higher faithfulness score compared to the prediction "So the answer is yes.", which is only supported by  $R_{1}$ .

# 4 Experiments

In this section, we present the evaluation of our proposed method, RR, on three complex reasoning tasks: commonsense reasoning, temporal reasoning, and tabular reasoning.

# 4.1 Baselines

We compare with the following baselines.

Zero-shot/few-shot prompting. In our experiments, we consider GPT-3 with standard zero-shot/few-shot prompting as baselines, following the approach described in Brown et al. (2020), in which zero or few in-context exemplars of input-output pairs are provided in the prompt.

Chain-of-thought prompting. In addition to the standard zero-shot/few-shot prompting, we also consider GPT-3 with the CoT prompting proposed in (Wei et al., 2022) as a baseline in our experiments. This approach involves feeding LLMs step-by-step reasoning examples instead of standard input-output examples.

Self-consistency. In addition, we also consider self-consistency (Wang et al., 2022) as a baseline in our experiments. This approach, proposed as an alternative to the naive greedy decoding used in CoT prompting (Wei et al., 2022), involves sampling a diverse set of reasoning paths and selecting the most consistent answer by marginalizing the sampled paths.

# 4.2 Commonsense Reasoning

Dataset description. For commonsense reasoning, we consider the StrategyQA dataset (Geva et al., 2021), which includes questions that require implicit reasoning strategies. For example, the question "Did Aristotle use a laptop?" requires implicit decomposition into reasoning steps, while the question "Was Aristotle alive when the laptop was invented?" explicitly specifies the reasoning process. The StrategyQA dataset includes 2,290 training examples, each consisting of a question (Q), a yes/no answer (A), a decomposition (D), evidence paragraphs (E), and supporting facts (F). On average, each question requires about 2.93 reasoning steps and 2.33 evidence paragraphs. In addition, a development set is constructed by randomly sampling  $10\%$  of the training examples (i.e., 229 examples). The answer distribution is roughly balanced, with approximately  $47\%$  "yes" questions in both the training and development

sets. Unless otherwise specified, the models are evaluated on the development set<sup>3</sup> for StrategyQA.

Implementation details. In this part, we utilize Wikipedia as the external knowledge base  $\mathcal{KB}$ . For each sentence in the explanation of every reasoning path, we first apply BM25 (Robertson et al., 2009) to retrieve the top 10 most relevant paragraphs from Wikipedia. In particular, we use the re-implementation of the sparse retrieval  $\mathrm{BM25^4}$  in Karpukhin et al. (2020) from Pyserini (Lin et al., 2021). Subsequently, we use the pretrained MPNet model (Song et al., 2020) to select the most similar paragraph based on the cosine similarity between the sentence embeddings of the retrieved paragraph and the sentence. We then employ a pre-trained natural language inference (NLI) model (Nie et al., 2020) to obtain the entailment and contradiction scores for the sentence, treating the most similar paragraph as the premise. The faithfulness of each reasoning path is then calculated using  $f_{\mathcal{KB}}(\cdot)$  based on the entailment scores, contradiction scores, and MPNet similarities of all sentences in the explanation of the reasoning path. The final prediction for each question is obtained through faithful inference (Equation 1). More details about  $f_{\mathcal{KB}}(\cdot)$  can be found in Appendix A.2.

# 4.3 Temporal Reasoning

Dataset description. In this experiment, we use the TempQuestions dataset (Jia et al., 2018) to investigate temporal reasoning. This dataset includes 1,271 temporal questions that are divided into four classes: explicit temporal, implicit temporal, temporal answer, and ordinal constraints. The questions are paired with their answers from Freebase (Bollacker et al., 2008). To examine the most challenging aspect of temporal reasoning, we focus on the set of implicit temporal questions, which contain implicit temporal expressions, including free-text temporal expressions. For example, the question "who was governor of oregon when shanghai noon was released?" is an implicit temporal question. To facilitate our analysis, we only consider questions with a single answer, resulting in a total of 175 examples. Of these ex

<table><tr><td></td><td>Methods</td><td>Commonsense</td><td>Temporal</td><td>Tabular</td></tr><tr><td rowspan="5">GPT-3</td><td>Zero-shot prompting</td><td>58.08</td><td>28.40</td><td>82.00</td></tr><tr><td>Few-shot prompting</td><td>63.32</td><td>29.59</td><td>83.08</td></tr><tr><td>Chain-of-thought prompting</td><td>65.94</td><td>33.14</td><td>83.33</td></tr><tr><td>Self-consistency</td><td>73.36</td><td>37.28</td><td>84.00</td></tr><tr><td>Rethinking with retrieval</td><td>77.73</td><td>39.05</td><td>84.83</td></tr></table>

Table 1: Performance of different methods using GPT-3 on three reasoning tasks.

amples, the first 6 are used for prompting, and the remaining 169 are used for evaluation.

Implementation details. In this part, we utilize Wikidata (Vrandecic and Krötzsch, 2014) as the external knowledge base  $\mathcal{KB}$ , as it is the largest publicly available knowledge graph, and the data from Freebase has been migrated to Wikidata. To incorporate this knowledge into our system, we apply an entity linking system<sup>5</sup> to each sentence in the explanation of each reasoning path to identify the corresponding Wikidata pages for all entities in the sentence. Next, we extract all temporal relations from these relevant Wikidata pages and use templates to convert these temporal relations into sentences. This step generates a set of relevant knowledge sentences for each sentence in the explanation of each reasoning path. The final prediction is then obtained by applying the procedure described in Section 4.2, in which the retrieved paragraphs are replaced with the relevant knowledge sentences from the current part.

# 4.4 Tabular Reasoning

Dataset description. We consider the INFOTABS dataset (Gupta et al., 2020) for tabular reasoning, which consists of 23,738 human-written textual hypotheses based on premises in the form of tables extracted from 2,540 unique Wikipedia info-boxes. We focus on the development set, which includes 1,800 hypotheses based on 200 tables, and only consider entailed and contradictory hypotheses as it is tricky to write CoT demonstrations for neutral hypotheses. This results in a total of 1,200 hypotheses based on 200 tables for evaluation, with an equal number of entailed and contradictory hypotheses.

Implementation details. In this part, we utilize WordNet (Miller, 1995) and ConceptNet (Speer

et al., 2017) as external knowledge bases. To convert tables into textual premises, we follow the same technique as in Varun et al. (2022). For each premise-hypothesis pair, we follow the procedure outlined in Varun et al. (2022) to retrieve relevant word relation triples that connect the premise and hypothesis words, such as "married"  $\xleftarrow{\text{RelatedTo}}$  "spouse". These triples are then converted into sentences using some simple templates. The resulting sentences, along with the textual premises from the tables, serve as relevant knowledge for each sentence in the explanation of each reasoning path. To obtain the final prediction, the procedure described in Section 4.2 is applied, whereby the retrieved paragraphs in Section 4.2 are replaced with the relevant knowledge from the current part.

# 4.5 Evaluation

Experimental settings. In all experiments, we utilize GPT-3 text-davinci-002 unless otherwise stated. The maximum number of tokens for generation during completion is set to 256. For zero-shot, few-shot, and chain-of-thought prompting, the temperature is fixed at 0. For self-consistency and rethinking with retrieval, we randomly sample 10 outputs<sup>6</sup> with temperature 0.7. Detailed prompts can be found in Appendix A.1. We evaluate the performance of different methods on commonsense and tabular reasoning using accuracy, and on temporal reasoning using the exact match metric as defined in Rajpurkar et al. (2016).

Results. As shown in Table 1, our proposed method, rethinking with retrieval, consistently outperforms all baselines on all three reasoning tasks without requiring additional training or finetuning. The results highlight the effectiveness of our approach in leveraging external knowledge to improve the performance of LLMs.

<table><tr><td></td><td>Wrong supporting facts</td><td>Wrong inference</td></tr><tr><td>Question</td><td>Was Lil Jon&#x27;s top ranked Billboard song a collaboration with a member of The Lox?</td><td>Would the top of Mount Fuji stick out of the Sea of Japan?</td></tr><tr><td>Answer</td><td>No</td><td>Yes</td></tr><tr><td>Facts</td><td>Lil Jon&#x27;s highest ranked billboard song was Yeah; Yeah was a collaboration between Lil Jon, Usher, and Ludacris; The Lox is a rap trio consisting of: Styles P, Sheek Louch, and Jadakiss.</td><td>The average depth of the Sea of Japan is 5,748 feet (1,752 metres) and its maximum depth is 12,276 feet (3,742 metres); Mount Fuji is 3,776.24 metres (12,389.2 ft) tall.</td></tr><tr><td>GPT-3</td><td>Lil Jon&#x27;s top ranked Billboard song was &quot;Get Low&quot; with the East Side Boyz. The Lox is not a member of the East Side Boyz. Thus, Lil Jon&#x27;s top ranked Billboard song was not a collaboration with a member of The Lox. So the answer is no.</td><td>Mount Fuji is 3,776 meters tall. The Sea of Japan is about 3,741 meters deep. Thus, the top of Mount Fuji would not stick out of the Sea of Japan. So the answer is no.</td></tr></table>

# 5 Analysis

In this section, we perform a thorough analysis to gain a deeper understanding of RR.

# 5.1 Limitations of LLMs in Reasoning

In this subsection, we present an analysis of GPT-3 with CoT prompting on the StrategyQA dataset. Upon closer examination of the outputs of GPT-3, we observed that it can provide reasonable explanations and correct predictions for a number of questions. For example, when given the question "Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?", GPT-3 produced the following output:

The Albany in New York has a population of about 98,000. The Albany in Georgia has a population of about 77,000. Thus, the Albany in New York is more populous than the Albany in Georgia. So the answer is no.

The above output consists of three components: (1) supporting facts (in cyan) that are based on a particular perspective, (2) chaining arguments (in orange), and (3) a prediction (in green). Components (1) and (2) contribute to the explanation. Overall, the output exhibits a high level of quality. However, we also observed that GPT-3 may occasionally produce incorrect supporting facts for its explanations or make incorrect inferences for its

Table 2: Examples of incorrect outputs from GPT-3 with CoT prompting.  

<table><tr><td>Retrieval</td><td>Commonsense</td><td>Tabular</td></tr><tr><td>Query-based</td><td>73.36</td><td>36.69</td></tr><tr><td>Decomposition-based</td><td>77.73</td><td>39.05</td></tr></table>

Table 3: Comparison of query-based and decomposition-based retrieval on commonsense and tabular reasoning.

predictions, despite generally being able to identify suitable perspectives.

Wrong supporting facts. As shown in Table 2, GPT-3 provides the incorrect supporting fact for Lil Jon's top-ranked Billboard song, stating that it was "Get Low" instead of the correct answer, "Yeah". However, it does have the correct perspective on how to answer the question, "Was Lil Jon's top ranked Billboard song a collaboration with a member of The Lox?"

Wrong inference. As shown in Table 2, GPT-3 makes an incorrect inference, stating that the top of Mount Fuji "would not stick out" of the Sea of Japan, rather than the correct answer, "would stick out". However, it does provide correct supporting facts based on the appropriate perspective for the question, "Would the top of Mount Fuji stick out of the Sea of Japan?"

# 5.2 Ablation Study

Importance of decomposition-based retrieval. In our proposed method, we retrieve relevant ex

<table><tr><td>Knowledge</td><td>Tabular</td></tr><tr><td>External</td><td>79.92</td></tr><tr><td>Background</td><td>84.75</td></tr><tr><td>Background + External</td><td>84.83</td></tr></table>

Table 4: Performance of RR with different types of knowledge on tabular reasoning: external only, background only, and a combination of both. External knowledge refers to WordNet and ConceptNet, while background knowledge refers to the tables.

ternal knowledge based on the decomposed reasoning steps rather than the original query. To further investigate the impact of this choice, we conducted additional experiments in which we used the original query for knowledge retrieval while keeping other aspects of our method unchanged. As shown in Table 3, the results for these experiments are poor for both commonsense and temporal reasoning, indicating the importance of using decomposition-based retrieval in our approach.

The impact of different types of knowledge. For tabular reasoning, we use both external knowledge (WordNet and ConceptNet) and background knowledge (tables) in our experiments. In this section, we further examine the effect of different types of knowledge on the performance of our proposed method. As shown in Table 4, the additional improvement gained by incorporating Wikidata and ConceptNet in addition to tables is limited, indicating that GPT-3 already captures many word-level relations in these external knowledge sources. In addition, the observed significant improvement in tabular reasoning from using tables alone suggests that our proposed method can also effectively leverage background knowledge.

# 5.3 Variations of the Proposed Approach

Basic approach: Weighting outputs. In Section 3, we present a basic version of our proposal for taking advantage of external knowledge. Our basic approach involves weighting outputs as individual units and using a voting mechanism to select the best-supported prediction. We can also directly choose the best-supported output, which includes both an explanation and a prediction, without using voting. For example, in the running example of "Did Aristotle use a laptop?" (see more in Section 3), the third reasoning path  $R_{3}$  is the output most supported by the knowledge para

graphs  $K_{1}$  and  $K_{2}$ .

Variant I: Fact selection. The first variant of our approach involves selecting facts from the outputs of LLMs based on external knowledge. For example, consider the running example of "Did Aristotle use a laptop?", where we only have access to the first two reasoning paths,  $R_{1}$  and  $R_{2}$ . In this case, the first sentence in  $R_{2}$  and the second sentence in  $R_{1}$  are supported by knowledge  $K_{1}$  and  $K_{2}$ , respectively. Therefore, the first variant would output the first sentence in  $R_{2}$  and the second sentence in  $R_{1}$  as the supporting facts.

Variant II: Fact generation. The second variant of our approach involves generating facts based on both the outputs of LLMs and external knowledge. For example, consider the running example of "Did Aristotle use a laptop?", where we only have access to the first reasoning path  $R_{1}$ . The second sentence in  $R_{1}$  is supported by the second knowledge paragraph  $K_{2}$ . However, the first sentence is not supported by any evidence paragraphs. We can generate questions about the first sentence, such as "When did Aristotle die?" and use the first knowledge paragraph  $K_{1}$  to generate a new fact: "Aristotle died in 322BC". As a result, the second variant would output the generated fact "Aristotle died in 322 BC" and the second sentence in  $R_{1}$  as the supporting facts.

Inference with supporting facts. For the two variants of our approach, we only have the supporting facts and need to perform a final inference step to obtain the corresponding prediction. One option for this inference is to use LLMs, but they can be costly (Brown et al., 2020) or difficult to use (Zhang et al., 2022). An alternative is to use an off-the-shelf model for inference with supporting facts, such as UnifiedQA (Khashabi et al., 2020, 2022). As discussed in Appendix A.5, UnifiedQA is more robust to noisy supporting facts than GPT-3. We thus use the second version of UnifiedQA, UnifiedQA-v2 (Khashabi et al., 2022), for the final step of inference.

Experimental settings. In this part, we focus on commonsense reasoning and use the evidence paragraphs provided in StrategyQA as the relevant knowledge, rather than the retrieved paragraphs discussed in Section 4.2. To evaluate the quality of the explanations, we adopt the best metric for factual consistency evaluation in Honovich

(a) Accuracy of predictions

(b) Faithfulness of explanations  
Figure 2: The effect of LM size on the performance of our proposed method (Variant II) and CoT prompting. We use various sizes of OPT models, with the exception of the 175B model, which is GPT-3.

<table><tr><td>Methods</td><td>Accuracy (%)</td><td>Faithfulness (%)</td></tr><tr><td>CoT prompting</td><td>65.94</td><td>38.73</td></tr><tr><td>Basic (w/o voting)</td><td>76.86</td><td>50.02</td></tr><tr><td>Variant I</td><td>78.60</td><td>54.11</td></tr><tr><td>Variant II</td><td>78.60</td><td>54.54</td></tr></table>

Table 5: Comparison of various variations of RR and the CoT prompting baseline on StrategyQA using evidence paragraphs.

et al. (2022). For simplicity, we use the pre-trained NLI model released by Nie et al. (2020) to compute the NLI-based metric, rather than fine-tuning T5-11B (Raffel et al., 2020) ourselves. The implementation details of the two variants can be found in Appendix A.4.

Results. Table 5 illustrates that the fact selection and fact generation variants of our proposal improve the faithfulness of the supporting facts in explanations, leading to increased prediction accuracy compared to the basic approach without voting. Across all variations of our proposal, we observe significant improvements in both prediction accuracy and the faithfulness of explanations when compared to the CoT prompting baseline.

The incorporation of a voting mechanism leads to an increased prediction accuracy of  $79.91\%$  for the basic approach. Comparison with the performance (i.e.,  $77.73\%$ ) of the same approach using retrieved paragraphs rather than evidence paragraphs in Table 1 demonstrates that retrieved paragraphs are also effective for our proposal, as both significantly outperform the voting baseline, self-consistency (i.e.,  $73.36\%$ ), as shown in Table 1.

It is noteworthy that UnifiedQA performs poorly on StrategyQA, achieving an accuracy of only  $58.95\%$ . However, when provided with gold supporting facts in StrategyQA, UnifiedQA demonstrates excellent performance with an accuracy of  $90.83\%$ . This suggests that UnifiedQA is suitable for last-step inference, but not effective for answering questions in StrategyQA.

# 5.4 Impact of the Size of LMs

In this subsection, we examine the effect of the size of LMs on the performance of our proposed method, specifically in the context of the fact generation variant. We compare the performance of our method using various sizes of OPT models (Zhang et al., 2022) in addition to GPT-3 (175B) using the same experimental setup as in Section 5.3. As shown in Figure 2, our proposed method (Variant II) consistently outperforms CoT prompting in terms of both prediction accuracy and the faithfulness of explanations, even when using smaller LMs.

# 6 Conclusion

In conclusion, the proposed approach is a promising solution for utilizing external knowledge to assist LLMs. Unlike traditional methods, RR does not require additional training or fine-tuning, making it a lightweight and feasible option for LLMs. Through extensive experiments on three reasoning tasks using GPT-3, we have shown that RR is able to produce more faithful explanations and improve the performance of LLMs. In the future, we plan to investigate various variations of RR to enhance its effectiveness and efficiency in augmenting LLMs with external knowledge.

# References

Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pages 1247-1250.  
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2021. Improving language models by retrieving from trillions of tokens. arXiv preprint arXiv:2112.04426.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.  
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.  
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.  
Ido Dagan, Oren Glickman, and Bernardo Magnini. 2005. The pascal recognising textual entailment challenge. In *Machine learning challenges* workshop, pages 177-190. Springer.  
Daniel Deutsch, Tania Bedrax-Weiss, and Dan Roth. 2021. Towards question-answering as an automatic metric for evaluating the content quality of a summary. Transactions of the Association for Computational Linguistics, 9:774-789.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019

Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186.  
Alexander R Fabbri, Chien-Sheng Wu, Wenhao Liu, and Caiming Xiong. 2021. Qafacteval: Improved qa-based factual consistency evaluation for summarization. arXiv preprint arXiv:2112.08542.  
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346-361.  
Liangke Gui, Borui Wang, Qiuyuan Huang, Alex Hauptmann, Yonatan Bisk, and Jianfeng Gao. 2021. Kat: A knowledge augmented transformer for vision-and-language. arXiv preprint arXiv:2112.08614.  
Vivek Gupta, Maitrey Mehta, Pegah Nokhiz, and Vivek Srikumar. 2020. Infotabs: Inference on tables as semi-structured data. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2309-2324.  
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International Conference on Machine Learning, pages 3929-3938. PMLR.  
Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. 2022. True: Reevaluating factual consistency evaluation. In Proceedings of the Second DialDoc Workshop on Document-grounded Dialogue and Conversational Question Answering, pages 161-175.  
Or Honovich, Leshem Choshen, Roee Aharoni, Ella Neeman, Idan Szpektor, and Omri Abend. 2021. Q2:: Evaluating factual consistency in knowledge-grounded dialogues via question generation and question answering. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7856-7870.

Zhen Jia, Abdalghani Abujabal, Rishiraj Saha Roy, Jannik Strötgen, and Gerhard Weikum. 2018. Tempquestions: A benchmark for temporal question answering. In *Companion Proceedings of the The Web Conference* 2018, pages 1057-1062.  
Mandar Joshi, Kenton Lee, Yi Luan, and Kristina Toutanova. 2020. Contextualized representations using textual encyclopedic knowledge. arXiv preprint arXiv:2004.12006.  
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769-6781.  
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations.  
Daniel Khashabi, Yeganeh Kordi, and Hannaneh Hajishirzi. 2022. Unifiedqa-v2: Stronger generalization via broader cross-format training. arXiv preprint arXiv:2202.12359.  
Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. 2020. Unifiedqa: Crossing format boundaries with a single qa system. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1896-1907.  
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916.  
Mojtaba Komeili, Kurt Shuster, and Jason Weston. 2022. Internet-augmented dialogue generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8460-8478.  
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Roktaschel, et al. 2020. Retrievalaugmented generation for knowledge-intensive

nlp tasks. Advances in Neural Information Processing Systems, 33:9459-9474.  
Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Zheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021), pages 2356-2362.  
Jiacheng Liu, Alisa Liu, Ximing Lu, Sean Welleck, Peter West, Ronan Le Bras, Yejin Choi, and Hannaneh Hajishirzi. 2022. Generated knowledge prompting for commonsense reasoning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3154-3169.  
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.  
George A Miller. 1995. Wordnet: a lexical database for english. Communications of the ACM, 38(11):39-41.  
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.  
J Neeraja, Vivek Gupta, and Vivek Srikumar. 2021. Incorporating external knowledge to enhance tabular reasoning. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2799-2809.  
Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. Adversarial nli: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4885-4901.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. 2022. Show your work: Scratchpads for intermediate computation with language models. In Deep Learning for Code Workshop.  
Maxwell Nye, Michael Tessler, Josh Tenenbaum, and Brenden M Lake. 2021. Improving coherence and consistency in neural sequence models with dual-system, neuro-symbolic reasoning. Advances in Neural Information Processing Systems, 34:25192-25204.  
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21:1-67.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392.  
Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333-389.  
Kurt Shuster, Mojtaba Komeili, Leonard Adolphs, Stephen Roller, Arthur Szlam, and Jason Weston. 2022. Language models that seek for knowledge: Modular search & generation for dialogue and prompt completion. arXiv preprint arXiv:2203.13224.  
Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. Mpnet: Masked and permuted pre-training for language understanding. Advances in Neural Information Processing Systems, 33:16857-16867.

Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Thirty-first AAAI conference on artificial intelligence.  
Alon Talmor, Oyvind Tafjord, Peter Clark, Yoav Goldberg, and Jonathan Berant. 2020. Leap-of-thought: Teaching pre-trained models to systematically reason over implicit knowledge. Advances in Neural Information Processing Systems, 33:20227-20237.  
Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, HengTze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239.  
Yerram Varun, Aayush Sharma, and Vivek Gupta. 2022. Trans-kblstm: An external knowledge enhanced transformer bilstm model for tabular reasoning. In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 62-78.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.  
Denny Vrandecic and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. Communications of the ACM, 57(10):78-85.  
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In

Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pages 38-45.  
Xi Ye and Greg Durrett. 2022. The unreliability of explanations in few-shot in-context learning. arXiv preprint arXiv:2205.03401.  
Eric Zelikman, Yuhuai Wu, and Noah D Goodman. 2022. Star: Bootstrapping reasoning with reasoning. arXiv preprint arXiv:2203.14465.  
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pretrained transformer language models. arXiv preprint arXiv:2205.01068.  
Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. 2022. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625.

# A Appendix

In this section, we provide additional details on our experimental setup. Further information can be found in our code.

# A.1 Detailed Prompts

We adopt the same CoT prompt for commonsense reasoning (i.e., StrategyQA) as those presented in Wei et al. (2022). The CoT prompt for temporal reasoning is provided in Table 6. For tabular reasoning, we adopt the method of Brown et al. (2020) for converting NLI into QA for RTE (Dagan et al., 2005), and randomly sample 6 examples from the training data to construct the prompt, as shown in Table 8. The few-shot prompt utilizes the same exemplars as the CoT prompt and does not involve CoT reasoning processes.

# A.2 Description of Faithfulness Functions

For a sentence  $s$ , we denote its MPNet similarity, entailment score, and contradiction score as  $M(s)$ ,  $E(s)$ , and  $C(s)$ , respectively. In our experiments, the corresponding thresholds for these scores are  $T_{m} = 0.5$ ,  $T_{e} = 0.6$ , and  $T_{c} = 0.99$ . Given the entailment scores, contradiction scores, and MPNet similarities of all supporting facts (denoted as  $S$ ) in the explanation of a reasoning path  $R$ , different faithfulness functions  $f_{\mathcal{KB}}(\cdot)$  can be adopted in different settings as follows:

(1)  $f_{\mathcal{KB}}(R) = \sum_{s\in S}[M(s)\times (M(s) > = T_m) +$ $E(s)\times (M(s) <   T_m) - C(s)]$  
(2)  $f_{\mathcal{KB}}(R) = \sum_{s\in S}[M(s) + E(s)]$  
(3)  $f_{\mathcal{KB}}(R) = \sum_{s\in S}[E(s)\times (E(s) > = T_e) - C(s)\times (C(s) > = T_c)]$

In Section 4, we employ function (1) for commonsense and tabular reasoning. For temporal reasoning, we use function (2) as the distinct nature of sentences converted from temporal relations leads to unreliable contradiction scores. In Sections 5.3-5.4, we use function (3) for commonsense reasoning with evidence paragraphs, as the high quality of the relevant knowledge negates the need for the complementary use of the MPNet similarity to improve the entailment score.

# A.3 Comparison of Retrieval Systems

For commonsense reasoning, we utilized different retrieval systems in Karpukhin et al. (2020) to retrieve relevant paragraphs from Wikipedia. The

performance of BM25, DPR, and BM25+DPR were  $77.73\%$ ,  $58.52\%$ , and  $77.29\%$ , respectively, indicating that BM25 is the best choice in our case.

# A.4 Implementation Details for the Two Variants of RR

Fact selection implementation details. In this work, we utilize the information present in the top-ranked output produced by our basic approach as a guide. To this end, we apply a greedy clustering algorithm to group the sentences from all outputs into distinct topic categories based on the cosine similarity of their MPNet sentence embeddings. For each fact in the top-ranked output of our basic approach, we identify the fact with the highest faithfulness within the same topic group and replace it in the output. The faithfulness of a fact is calculated using the  $f_{\mathcal{KB}}$  function by replacing the supporting facts with a single fact.

Fact generation implementation details. In this part, we generate questions for the named entities present in each fact of the top-ranked output produced by our basic approach, and retrieve the corresponding answers from the evidence paragraphs using UnifiedQA. We employ the question generation model described in Deutsch et al. (2021), which has been shown to be more extractive compared to other models as demonstrated in Fabbri et al. (2021). We adopt the question filtering approach proposed in Honovich et al. (2021) using an off-the-shelf extractive QA model (ktrapeznikov/albert-xlarge-v2-squadv2 from Hugging Face (Wolf et al., 2020)). We then use an off-the-shelf model (MarkS/bart-baseqa2d from Hugging Face) to convert the generated QA pairs into declarative sentences. We apply simple rules based on the entailment and contradiction scores of the selected facts from the fact selection variant and the generated declarative sentences to obtain the final generated facts.

# A.5 Comparison of Different Inference Methods with Supporting Facts

In our experiments, we utilize UnifiedQA for the final step of inference in both variants. However, it is worth noting that GPT-3 could also be used for this purpose. As shown in Table 7, we observe that UnifiedQA performs better at inference with generated facts, while GPT-3 with CoT prompting performs better with empty or gold facts. This suggests that UnifiedQA is more robust to noisy

Q: who was governor of minnesota when maathaad maathaadu mallige was released?

A: Maathaad Maathaadu Mallige was released on 24 August 2007. Tim Pawlenty served as the 39th governor of Minnesota from 2003 to 2011. Thus, Tim Pawlenty was governor of minnesota when maathaad maathaadu mallige was released. So the answer is Tim Pawlenty.

Q: who was us president during the costa rican civil war?

A: The Costa Rican civil war was a civil war in Costa Rica from 12 March to 24 April 1948. Harry S. Truman was the 33rd president of the United States, serving from 1945 to 1953. Thus, Harry S. Truman was us president during the costa rican civil war. So the answer is Harry S. Truman.

Q: who was governor of oregon when the collector was released?

A: The Collector premiered at the Cannes Film Festival on May 20, 1965. Mark Hatfield served as the 29th governor of Oregon from 1959 to 1967. Thus, Mark Hatfield was governor of oregon when the collector was released. So the answer is Mark Hatfield.

Q: who was governor of oregon when shanghai noon was released?

A: Shanghai Noon was released on May 26, 2000. John Kitzhaber served as the 35th governor of Oregon from 1995 to 2003. Thus, John Kitzhaber was governor of oregon when shanghai noon was released. So the answer is John Kitzhaber.

Q: who was us president when john andrew shulze was a teenager?

A: John Andrew Shulze was born on July 19, 1775. A teenager is someone who is between 13 and 19 years old. George Washington served as the first president of the United States from 1789 to 1797. Thus, George Washington was us president when john andrew shulze was a teenager. So the answer is George Washington.

Q: who was us president during the seventh coalition?

A: The War of the Seventh Coalition was from 20 March to 8 July 1815. James Madison served as the fourth president of the United States from 1809 to 1817. Thus, James Madison was us president during the seventh coalition. So the answer is James Madison.

Table 6: The CoT prompt for temporal reasoning.  

<table><tr><td></td><td>Methods</td><td>Accuracy (%)</td></tr><tr><td rowspan="3">Empty facts</td><td>GPT-3 (zero-shot)</td><td>58.08</td></tr><tr><td>GPT-3 (CoT)</td><td>65.94</td></tr><tr><td>UnifiedQA</td><td>58.95</td></tr><tr><td rowspan="3">Gold facts</td><td>GPT-3 (zero-shot)</td><td>81.66</td></tr><tr><td>GPT-3 (CoT)</td><td>91.70</td></tr><tr><td>UnifiedQA</td><td>90.83</td></tr><tr><td rowspan="3">Generated facts</td><td>GPT-3 (zero-shot)</td><td>69.87</td></tr><tr><td>GPT-3 (CoT)</td><td>76.42</td></tr><tr><td>UnifiedQA</td><td>78.60</td></tr></table>

Table 7: Comparison of different inference methods on empty, gold, and generated facts.

inputs compared to GPT-3. Additionally, both UnifiedQA and GPT-3 with CoT prompting significantly outperform GPT-3 with zero-shot prompting, indicating that the CoT prompting is also beneficial for the final step of inference.

Charles Sumner Tainter was Born on April 25, 1854 (1854-04-25) Watertown, Massachusetts, U.S.. Charles Sumner Tainter was Died on April 20, 1940 (1940-04-21) (aged 85) San Diego, California, U.S.. The Nationality of Charles Sumner Tainter are American. The Known for of Charles Sumner Tainter are Photophone, phonograph Father Of The Speaking Machine.

Question: Charles Sumner Tainter never left the state of Massachusetts. True or False?

Answer: Charles Sumner Tainter was died in San Diego, California, U.S.. California is a state. Thus, Charles Sumner Tainter has left the state of Massachusetts. So the answer is false.

The Region of Curitiba are South. The Elevation of Curitiba are  $934.6\mathrm{m}$  (3,066.3 ft). The Density of Curitiba are 4,062/km 2 (10,523/sq mi). The Metro density of Curitiba are 210.9/km 2 (546.2/sq mi).

Question: Curitiba is above sea level. True or False?

Answer: The elevation of Curitiba are  $934.6\mathrm{m}$  (3,066.3 ft). Elevation is a hypernym of level. Thus, Curitiba is above sea level. So the answer is true.

Charles (Prince of Wales) was Born on 14 November 1948 (1948-11-14) (age 70) Buckingham Palace, London, England. The Spouse of Charles (Prince of Wales) are Lady Diana Spencer (m. 1981; div. 1996), and Camilla Parker Bowles (m. 2005). The Issue of Charles (Prince of Wales) are Prince William, Duke of Cambridge, and Prince Harry, Duke of Sussex.

Question: Charles was born in 1948 and has been married twice. True or False?

Answer: Charles (Prince of Wales) was Born on 14 November 1948. The Spouse of Charles (Prince of Wales) are Lady Diana Spencer (m. 1981 ; div. 1996), and Camilla Parker Bowles (m. 2005). Married is related to spouse. Thus, Charles was born in 1948 and has been married twice. So the answer is true. The Born of Idris Elba are 6 September 1972 (age 46) Hackney, London, England. The Residence of Idris Elba are London. The Other names of Idris Elba are DJ Big Driis, Big Driis the Londoner, Big Driis, and 7 Dub. The Occupation of Idris Elba are Actor, producer, director, musician, and DJ.

Question: Idris Elba is an English entertainer. True or False?

Answer: The residence of Idris Elba is London. English is related to London. The occupation of Idris Elba are actor, producer, director, musician, and DJ. Actor is a hyponym of entertainer. Musician is a hyponym of entertainer. DJ is an entertainer. Thus, Idris Elba is an English entertainer. So the answer is true.

The Breed of Jean, the Vitagraph Dog are Scotch Collie. The Sex of Jean, the Vitagraph Dog are Female. The Born of Jean, the Vitagraph Dog are 1902 Eastport, Maine. The Years active of Jean, the Vitagraph Dog are 1909 - 1916.

Question: Jean, the Vitagraph Dog was a Golden Retriever which perform in circus. True or False?

Answer: The Breed of Jean, the Vitagraph Dog are Scotch Collie. Collie is a hyponym of dog. Retriever is a hyponym of dog. Thus, Jean, the Vitagraph Dog was not a Golden Retriever which perform in circus.

# So the answer is false.

The Studio of Hydrograd are Sphere Studios, North Hollywood, Los Angeles. The Genre of Hydrograd are Hard rock. The Label of Hydrograd are Roadrunner. The Producer of Hydrograd are Jay Ruston.

Question: Hydrograd is in the rap genre. True or False?

Answer: The Genre of Hydrograd are Hard rock. Rap is distinct from rock. Thus, Hydrograd is not in the rap genre. So the answer is false.

Table 8: The CoT prompt for tabular reasoning.

# Footnotes:

Page 0: *Part of this work was done while the author was at the University of Pennsylvania. 1Our code is publicly available at https://github. com/HornHehhf/RR. 
Page 2: Note that this is the basic version of faithful inference, and further variations can be found in Section 5.3. 
Page 3: As the annotations for the test set are not publicly available, we use the development set for evaluation. This allows us to perform a more comprehensive analysis. 4We also experimented with DPR and BM25+DPR, and found that BM25 outperformed these methods in our experiments. More details can be found in Appendix A.3. 
Page 4: For commonsense reasoning, we sample 9 outputs, as we have found that odd numbers of outputs tend to yield better voting performance for self-consistency on StrategyQA. 
