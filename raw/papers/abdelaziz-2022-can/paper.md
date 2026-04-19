# CAN MACHINES READ CODING MANUALS YET? - A BENCHMARK FOR BUILDING BETTER LANGUAGE MODELS FOR CODE UNDERSTANDING

Ibrahim Abdelaziz<sup>1</sup>, Julian Dolby<sup>1</sup>, Jamie McCusker<sup>2</sup>, and Kavitha Srinivas<sup>1</sup>

<sup>1</sup>IBM Research, T.J. Watson Research Center, Yorktown Heights, NY, USA, {ibrahim.abdelaziz1, kavitha.srinivas} @ibm.com, dolby@us.ibm.com

$^{2}$ Rensselaer Polytechnic Institute (RPI), Troy, NY, USA

mccusj2@rpi.edu

# ABSTRACT

Code understanding is an increasingly important application of Artificial Intelligence. A fundamental aspect of understanding code is understanding text about code, e.g., documentation and forum discussions. Pre-trained language models (e.g., BERT) are a popular approach for various NLP tasks, and there are now a variety of benchmarks, such as GLUE, to help improve the development of such models for natural language understanding. However, little is known about how well such models work on textual artifacts about code, and we are unaware of any systematic set of downstream tasks for such an evaluation. In this paper, we derive a set of benchmarks (BLANCA - Benchmarks for LANGUAGE models on Coding Artifacts) that assess code understanding based on tasks such as predicting the best answer to a question in a forum post, finding related forum posts, or predicting classes related in a hierarchy from class documentation. We evaluate the performance of current state-of-the-art language models on these tasks and show that there is a significant improvement on each task from fine tuning. We also show that multi-task training over BLANCA tasks helps build better language models for code understanding.

# 1 INTRODUCTION

Code understanding is an increasingly important application of AI, with over 100 papers targeting the area in the last year alone<sup>1</sup>. Much research in this area has focused on understanding code from abstract representations of the program such as Abstract Syntax Trees (ASTs) and program flow. However, there has been little emphasis in utilizing important semantics about code buried in textual artifacts, such as documentation or forum discussions. Extracting such information can significantly enrich code representations. As an example, Figure 1 shows a program where the classes GLM and SGDClassifier are being used. If one could enrich the representation of the two classes in the code with their key features from text, we would understand that both represent linear models, and hence both code snippets perform similar functions.

To enrich code with textual information, we need to be able to summarize textual information about classes and functions into vector representations. Pre-trained language models are an obvious choice, but we currently do not know how applicable they are to text about code, given the specialized language of the programming domain. We need a set of code-related downstream tasks to evaluate these models, just as GLUE [1] and SuperGLUE [2] have been used extensively to further language model development in the natural language understanding domain. CodeXGLUE [3] provides a suite of tasks but only a single task in it is related to textual code artifacts; it is translation of documentation about code from one natural language to another. To our knowledge, we know of no other tasks that focus on relations between textual artifacts about code. This paper attempts to fill this gap.

We have three goals in this paper: (a) design a suite of tasks we refer to as BLANCA (Benchmarks for LANguage models on Coding Artifacts) that can be used to train language models about the semantics of code, (b) evaluate whether existing models, fine-tuned for different aspects of natural language processing or different code oriented corpora, can perform well on these tasks, and (c) establish whether these tasks can be used to build better models for code understanding.

To construct these tasks, we relied on existing annotations in large public repositories such as GitHub (for code), StackOverflow, StackExchange and code documentation (for text about code). We exploited an integration of these sources in an open source dataset [4] to define the following five tasks focused on text about code:

- Forum Answer Ranking  $(R)$ . Some answers on forums have many votes or are selected as the best relative to others. Can language models predict the best answers?  
- Forum Link Prediction  $(L)$ . Users of forum posts often point to other similar posts, which reflect semantically related posts compared to random pairs. Can language models predict links?  
- Forum to Class Prediction  $(F)$ . Key features of classes or functions often get discussed in forum posts. Do language models discriminate related posts and class documentation from unrelated ones?  
- Class Hierarchy Distance Prediction  $(H)$ . Code is often organized into class hierarchies. Do embedding distances from language models reflect class distances in the hierarchy?  
- Class Usage Prediction  $(U)$ . Similar code is often used in similar ways. Are embedding distances smaller

```python
elif distr == 'binomial':
    model = SGDClassifier(loss='log',
                    penalty='elasticnet',
                    alpha= self.reg_lambda,
                    l1_ratio= self.alpha)
# fit-predict-score
model.fit(X_train, y_train)
y_test_hat = model.predict(X_test)
res[env]['score'] = model.score(X_test, y_test)
```

```txt
elif distr  $= =$  'poisson': model  $=$  sm.GLM(y_train, sm.add_constant(X_train), family  $\equiv$  sm.families.Poisson()) #fit-predict-score statsmodels_res  $=$  model.fit() y_test_hat  $=$  model.predict/statsmodels_res.params, exog  $\equiv$  sm.add_constant(X_test))
```

Figure 1: Usage in code

for documentation about classes that are used similarly, and larger for dissimilar ones?

We compare performance on these tasks for seven language models, chosen for differences in architecture, training tasks, and corpora, as outlined in Section 3. Our main findings are as follows:

- Out of the box, language models trained on general corpora perform reasonably well on most BLANCA tasks, compared to models trained on code specific corpora such as CodeBERT [5] or BERTOverflow [6], attesting to the generality of these models.  
- However, on every task, fine tuning on code specific models resulted in significant boost in performance, highlighting the usefulness of BLANCA tasks for building better language models.  
- Multi-task training produced better performance on many BLANCA tasks, suggesting the tasks do help models learn code semantics that transfers across tasks.

To aid further research in code understanding, the code, datasets and the fine-tuned models are publicly available<sup>2</sup> under an open source license (Eclipse for the code and Creative Common with Attribution for the data), and we hope they prove useful to the code understanding community to enrich representations of programs with textual information about classes and functions.

# 2 RELATED WORK

There have been numerous benchmarks for code summarization or generation of code from natural language, and hence they have focused on collecting code and textual documentation that characterize the code. For these tasks, most have used the approach of generating code and its associated documentation strings, e.g., [7], [8]. Similarly, code and corresponding textual documentation have been used for numerous tasks involving searching for code, e.g., [9], [10], or searching for posts given code, e.g., [11].

While such benchmarks are useful for joint embeddings of code and their associated text, they are restricted to tasks around code summarization, code generation, comment generation or code search; i.e., they do not directly help with the evaluation of language models for textual artifacts about code. Furthermore, most of the datasets in the literature do not correlate textual artifacts around code with code usage, with the exception of [12], which does link the generation of API sequence information from their usage in code to the problem of code summarization. The work in [13] connects code on GitHub to StackOverflow posts, but the latter dataset is not available. Again, their datasets are targeted to the task of code summarization, and code search respectively. Similarly StackOverflow posts have been used for tasks such as answer summarization [14], program repair [15] or generating code, e.g., [15]. Finding directly related or duplicate posts is a recent task and dataset proposed in [16], but there is no evaluation of any language model in that work.

Recently, [6] provided a BERTOverflow model for an in-domain representation of text about code. BERTOverflow is trained on 152 million StackOverflow questions over a BERT architecture, and has been fined tuned for software named entity recognition (e.g., finding mentions of operating systems in text). We use BERTOverflow as one of the models for the BLANCA tasks. There has also been work building structural language models from the abstract syntax trees, e.g., [17], which is clearly a related task, but the focus is once again on code. CodeXGLUE [3] provides a novel text-text benchmark which involves translation of documentation about code from one language to another, but that is arguably closer to natural language processing than code.

Thus, to the best of our knowledge, no work so far has examined how language models perform on a set of code related tasks for textual program artifacts, nor has there been much emphasis on building benchmarks to build better text representations for code understanding. BLANCA is built to address this gap.

# 3 MODELS

In choosing models for our experimentation, we needed language models to encode paragraphs in either class documentation or posts. We relied largely on the sentence transformers library [18], which provides a wide range of transformer models that have been fine-tuned for tasks such as information retrieval, paraphrase detection, and sentence similarity detection. These models have been shown to be effective in sentence and paragraph encoding style tasks. We also chose models with a different base, such as BERT [19], XLM-RoBERTa [20] and DistilBERT[21]. We also added a non-transformer style model (Google's Universal Sentence Encoder [22]), and models fine-tuned on StackOverflow posts (BERTOverflow [6]) and code documentation (CodeBERT [5]) to see if domain-specific training is helpful. We did not consider models, such as CuBERT [23], designed only for code, and not text about code. The reason is that cuBERT's vocabulary is based on programming language tokens for Java or Python, which is only partially useful for text about code. Table 1 shows the types of base models used in our evaluation, using the names from the sentence-transformers (SBERT $^3$ ) library.

<table><tr><td>Model name</td><td>Fine-tuning task</td></tr><tr><td>Universal Sentence Encoder‡</td><td>N/A</td></tr><tr><td>BERT-NLI†</td><td>Sentence similarity</td></tr><tr><td>DistilBERT-paraphrasing†</td><td>Paraphrase detection</td></tr><tr><td>xlm-r-paraphrase-v1†</td><td>Paraphrase detection</td></tr><tr><td>mmsmarco-DistilRoBERTa†</td><td>Information Retrieval</td></tr><tr><td>BERTOverflow†</td><td>StackOverflow/NER</td></tr><tr><td>CodeBERT-mlm†</td><td>NL-PL pairs in 6 languages</td></tr></table>

Table 1: Models used as baselines. Sources were tensorflow-hub, and SBERT. bert-base-nli-stsb-mean-tokens, distilroberta-base-paraphrase-v1, xlm-r-distilroberta-base-paraphrase-v1, msmarco-distilroberta-base-v2 and microsoft/codebert-baselm are their corresponding names in SBERT.

We also tested if fine-tuning on each task would enhance performance, to establish whether the tasks can be used to build a better language model. For fine-tuning, we started either with BERTOverflow or CodeBERT, with the assumption that an in-domain representation would provide some advantage. We also examined whether multi-task training would improve performance, to see if better models could be built from using a combination of BLANCA tasks.

# 4 TASKS

All our datasets describe code artifacts in Python, and are derived from Graph4Code, which links 1.3 million programs of Python code to associated posts and class-documentation [4]. For multi-task fine tuning, we report, for each task, the model with the best performance, and we outline its characteristics. In Section 4.6, we discuss more general findings for multi-task training. Performance on tasks is encoded as follows in tables: (1) Forum Answer Ranking (R), (2) Forum Link Prediction (L), (3) Forum Class Prediction (F), (4) Class Hierarchy Prediction (H), and (5) Class Usage Prediction (U). Table 2 lists each BLANCA task and the corresponding train/test data sizes.

# 4.0.1 Dataset Annotation Quality

Two of BLANCA tasks are based on manually curated datasets by millions of users such as ranking answers in StackOverflow forums (Forum Answer Ranking) and manually linking similar posts (Forum Link Prediction). These data are high quality, in the sense that they are crowd annotated by humans, which is how most gold standards get constructed. Class Hierarchy and Class Usage Prediction tasks are both based on objective properties of code artifacts (class hierarchy and similarities among classes in terms of their methods, respectively), so once again, the issues of data quality do not arise. The only task where we did not have explicit human labeling for every example is Forum to Class Prediction. In this task, we relied on heuristics to automatically label the data. Furthermore, to assure quality, we performed a manual evaluation of a sample with three human annotators (see Section 4.3 for details).

# 4.0.2 Hyperparameter Search for Finetuning

We started with the default parameters of our base models; CodeBERT and BERTOverflow. We also tried to use Population

Based Training from RayTune<sup>4</sup> to perform hyper-parameter search for the Forum Answer Ranking (R) and Forum Link Prediction (L) tasks. However, we did not get better performance compared to using the default parameters from the corresponding base models.

We describe below how we formulated each task, the dataset definition process and the performance of various language models on it.

# 4.1 Forum Answer Ranking (R)

# 4.1.1 Task Description

StackOverflow and StackExchange contain questions and answers. Accepted answers are manually annotated and most answers have a vote count. The core task here is to predict the best answer to each question, and order the answers by their popularity.

# 4.1.2 Dataset

We generated a dataset of 500K questions such that each question comes with at least three answers. The average number of answers per question in this dataset is 4.9 answers, and the average number of votes per question is 23.5 and per answer is 12.74. The train and test tasks were split 90-10, so the train set had 450,000 questions and test had 50,000 questions. To build the fine tuning model, we modeled this as a task similar to training on the Semantic Textual Similarity Benchmark (STSB) adopted by SBERT. Each answer was ranked according to popularity, and ties were broken by adding only one of the answers that were tied. The ranks were then converted to a score between 0 (worst rank) and 1 (best rank), with a cosine similarity loss, and an embedding similarity evaluator from the SBERT library. Fine tuning was performed on BERTOverflow and CodeBERT models, with the  $90\%$  of training data for training,  $10\%$  of the training data for validation, for 10 epochs.

# 4.1.3 Evaluation

To capture how well the embeddings of different language models identified the ranking of answers, we computed the cosine distances between the question embedding and the embedding of each of the answers, and ranked answers by nearest in cosine distance to furthest. We report standard information retrieval metrics of average Mean Reciprocal Rank (MRR) and average Normalized Discounted Cumulative Gain (NDCG) on this predicted ranking.

Table 3 shows that most language models do reasonably well on this task, which is not surprising because text in forum posts is mostly natural language. Surprisingly though, there is no benefit for the base BERTOverflow model that has been tuned on StackOverflow posts compared to the rest of non-finetuned models. However, fine-tuned BERTOverflow does much better, which is consistent with our hypothesis that it is possible to use these tasks for building better language models. Across many tasks, fine-tuning on BERTOverflow produced better performance than fine-tuning on CodeBERT, which suggests that forum discussions contain in most cases, the right mixture of

<table><tr><td></td><td>Data type</td><td>Train</td><td>Test</td></tr><tr><td>Forum Answer Ranking</td><td>Question-answer pairs</td><td>450,000</td><td>50,000</td></tr><tr><td>Forum Link Prediction</td><td>Question-Question pairs</td><td>23,516</td><td>5,854</td></tr><tr><td>Forum to Class Prediction</td><td>Question-Class pairs</td><td>11,488</td><td>1,275</td></tr><tr><td>Class Hierarchy Prediction</td><td>Class-Class pairs</td><td>16,215,400</td><td>1,801,716</td></tr><tr><td>Class Usage Prediction</td><td>Class-Class pairs</td><td>75,862</td><td>8,439</td></tr></table>

Table 2: BLANCA's tasks and datasets statistics  

<table><tr><td></td><td>MRR</td><td>NDCG</td></tr><tr><td>DistilBERT-paraphrasing</td><td>0.5937 (.001)</td><td>0.8393 (.001)</td></tr><tr><td>BERT-NLI</td><td>0.5972 (.001)</td><td>0.8407 (.001)</td></tr><tr><td>msmarco-DistilRoBERTa</td><td>0.5992 (.001)</td><td>0.8427 (.001)</td></tr><tr><td>xlm-r-paraphrase-v1</td><td>0.5977 (.001)</td><td>0.8411 (.001)</td></tr><tr><td>USE</td><td>0.6114 (.001)</td><td>0.8483 (.001)</td></tr><tr><td>BERTOverflow</td><td>0.5910 (.001)</td><td>0.8375 (.001)</td></tr><tr><td>CodeBERT</td><td>0.5926 (.001)</td><td>0.8375 (.001)</td></tr><tr><td>FT-BERTOverflow</td><td>0.6743 (.001)</td><td>0.8823 (.001)</td></tr><tr><td>FT-CodeBERT</td><td>0.6671 (.001)</td><td>0.8790 (.001)</td></tr><tr><td>RFLHU-BERTOverflow</td><td>0.6879 (.001)</td><td>0.8893 (.001)</td></tr></table>

Table 3: Performance of language models on forum answer ranking (R). The numbers in parentheses are the standard errors of the sample mean. FT represents fine tuning on R alone, RFLHU-BERTOverflow is the best multi-task training model.

explanations in natural language along with code. Moreover, the best performance was achieved with multi-task finetuning (RFLHU-BERTOverflow), which suggests that use of multiple BLANCA tasks builds better language models for textual code artifacts.

# 4.2 Forum Link Prediction (L)

# 4.2.1 Task Description

Forum posts with links to one another are usually related compared to unlinked posts; we investigate if language models place such related post pairs closer in vector space. We focus on embedding distance because it is a more direct metric for assessing the quality of the embedding rather than classification accuracy.

# 4.2.2 Dataset

For this task, we generated 23,516 pairs of posts for training (11,758 positive and 11,758 negative), 5,854 pairs (2,727 positive and 2,727 negative) for testing. Fine-tuning was set up as a classification task in SBERT, with the use of contrastive loss along with a binary classification evaluator from the SBERT library. All other training details were similar to the forum answer ranking task.

Relevant to this task, [16] recently introduced a similar benchmark for predicting relatedness in StackOverflow posts focused on Java code, as opposed to our dataset which is language agnostic. Their dataset contains 300K of linked pairs categorized into 1) duplicates: questions in StackOverflow marked by moderators as duplicates, 2) direct: explicitly linked posts, 3) indirectly or transitively connected posts through a direct or a duplicate link and 4) isolated or unlinked posts. Direct and isolated links are similar to our positive and negative examples. We evaluate all

<table><tr><td>Model</td><td>Linked</td><td>Unlinked</td><td>T</td></tr><tr><td>DistilBERT-paraphrasing</td><td>0.38</td><td>0.71</td><td>112.49</td></tr><tr><td>BERT-NLI</td><td>0.31</td><td>0.53</td><td>74.92</td></tr><tr><td>msmarco-DistilRoBERTa</td><td>0.34</td><td>0.74</td><td>110.42</td></tr><tr><td>xlm-r-paraphrase-v1</td><td>0.37</td><td>0.70</td><td>105.02</td></tr><tr><td>USE</td><td>0.34</td><td>0.74</td><td>142.04</td></tr><tr><td>BERTOverflow</td><td>0.20</td><td>0.31</td><td>59.52</td></tr><tr><td>CodeBERT</td><td>0.03</td><td>0.04</td><td>19.39</td></tr><tr><td>FT-BERTOverflow</td><td>0.09</td><td>0.52</td><td>180.42</td></tr><tr><td>FT-CodeBERT</td><td>0.08</td><td>0.50</td><td>147.21</td></tr><tr><td>RFLHU-BERTOverflow</td><td>0.08</td><td>0.58</td><td>198.10</td></tr></table>

Table 4: Cosine distance between linked and unlinked posts (L). FT represents fine-tuning on L alone.

our models' ability to differentiate these link types. Note that we did not use this data for fine-tuning a model which discriminates the different categories; but one might expect direct links and duplicates to be closer in embedding distance, and isolated links to be the furthest, with indirect links in the middle. Shirani et al. [16] did not evaluate this with any of the language models, so we examine whether these categories of relatedness of posts is reflected in embeddings of pre-trained models.

# 4.2.3 Evaluation

As shown in Table 4, all language models showed a statistically significant difference  $(p \leq .01)$  on independent sample t-tests between linked and unlinked posts. BERTOverflow with fine-tuning (both versions tuned on L only and RFLHU) performing the best in terms of pulling apart linked and unlinked posts. We note that the size of T value normalizes the distance between linked and unlinked posts by their variance; that is, the T value captures not only the average distance but also the separation between the two distributions. Our focus then is on the absolute value of that separation as provided by the T value. Figure 2 shows this visually. We note that BERTOverflow and CodeBERT as base models discriminated least between linked and unlinked posts, but fine-tuning clearly helped greatly. This is evident in the solid and dashed lines for RFLHU-BERTOverflow where it shows little overlap between linked and unlinked posts.

Figure 3 shows the results of a variety of language models for question relatedness variant of this task [16]. We ensured that none of Shirani et al. [16]'s test set examples were used in our training set. Across all models, directly related questions are closest in embedding space followed by indirectly related questions. Questions marked duplicate posts were similar to the indirect questions only in the RFLHU model, which seemed to be picking up relatedness in both indirect and duplicate questions. We note that questions marked duplicates in forums are

Figure 2: Linked versus unlinked pair distances for all models (L).

Figure 3: Direct, indirect, duplicate and isolated pair distances for all models.

only duplicates at a level of coding abstraction. For example, the two questions "How to return multiple objects from a Java method?" and "Java how to return two variables?" are a duplicate pair. Although the two questions talk about the same problem, the discussions and even the solutions are different. Therefore, cosine similarity between them is not as close as one would expect. Finally, isolated question pairs are the most distant compared to all other pairs across all models (all differences from isolated pairs to direct, indirect and duplicate pairs were statistically significant at the .01 level). Multi-task fine-tuning (RFLHU-BERTOverflow) clearly helped the best in getting semantically related posts closer and pulling apart the unrelated ones.

# 4.3 Forum to Class Prediction (F)

# 4.3.1 Task Description

Forum posts often describe specific code artifacts in text, where they discuss key features of a class or a function. A key question is whether a model can predict if a post about a class and documentation of the same class are related.

# 4.3.2 Dataset

In order to find posts that were more focused on discussions of a specific class or function's features, we queried an ElasticSearch index of posts with a query per class as in Graph4Code [4], insisting that the class and its package be both mentioned in the

<table><tr><td>Model</td><td>Related</td><td>Unrelated</td><td>T</td></tr><tr><td>DistilBERT-paraphrasing</td><td>0.55</td><td>0.68</td><td>16.61</td></tr><tr><td>BERT-NLI</td><td>0.45</td><td>0.60</td><td>14.73</td></tr><tr><td>msmarco-DistilRoBERTa</td><td>0.45</td><td>0.66</td><td>20.37</td></tr><tr><td>xlm-r-paraphrase-v1</td><td>0.53</td><td>0.67</td><td>17.15</td></tr><tr><td>USE</td><td>0.53</td><td>0.74</td><td>20.67</td></tr><tr><td>BERTOverflow</td><td>0.33</td><td>0.47</td><td>18.31</td></tr><tr><td>CodeBERT</td><td>0.06</td><td>0.09</td><td>12.23</td></tr><tr><td>FT-BERTOverflow</td><td>0.07</td><td>0.77</td><td>46.88</td></tr><tr><td>FT-CodeBERT</td><td>0.08</td><td>0.82</td><td>50.07</td></tr><tr><td>RFLHU-CodeBERT</td><td>0.11</td><td>0.66</td><td>53.98</td></tr></table>

Table 5: Cosine distance between documentation-post pairs (F) that are related and unrelated. FT represents fine-tuning on F alone.

question. These constituted our positive class-post examples. For negative examples, we chose hard negatives, requiring that both class name and its package not be mentioned anywhere within the question and its answers; but nevertheless the post matched either class or package names. To ensure the quality of this data, we asked 3 annotators to label a random sample of 100 examples; 50 positive and 50 negative. This manual inspection revealed that negatives were in fact negatives, in the sense that even if the class was mentioned, it was usually, from a different package, or very often from different programming languages (e.g. Java, Javascript, etc). The average hit and miss rates from the three annotators were  $96.7\%$  and  $3.3\%$ , respectively. In this task, we created 8,827 negative examples and 2,661 positives for training, and 980 negative examples and 295 positive examples for testing. Fine-tuning the model was analogous to the forum link prediction task.

# 4.3.3 Evaluation

As shown in Table 5, all language models showed a statistically significant difference  $(p\leq .01)$  on independent sample t-tests between positive and negative class-post examples. Again, fine-tuning helps improving the performance on this task significantly; e.g. single task tuning of FT-CodeBERT vs. CodeBERT and FT-BERTOverflow compared to BERTOverflow. Fine-tuning on multiple tasks, e.g. RFLHU-CodeBERT, gave better performance compared to the single-task tuned models, FT-CodeBERT and FT-BERTOverflow. As shown in Figure 4, the distance was greatly enhanced by fine-tuning e.g. BERTOverflow vs. fine-tuned RFLHU-BERTOverflow.

# 4.4 Class Hierarchy Distance Prediction (H)

# 4.4.1 Task Description

Semantically related classes tend to be linked by developers in a class hierarchy, so its reasonable to ask if neural embeddings of related classes cluster closer together in a class hierarchy. We structured this as a class distance prediction task, with class distances ranging from 1 to 10.

# 4.4.2 Dataset

We collected the documentation associated with 257,655 classes in Graph4Code [4], but many of these represent different names that resolve to the same class. We aliased the classes to its

Figure 4: Related and unrelated documentation-post pair distances for some fine-tuned and non-fine-tuned models (F).

Figure 5: Number of class-pairs by class distance.

canonical version by loading the class dynamically to obtain its runtime name, and added in classes that we could not load for some reason, which resulted in 90,464 classes.

To get classes related by distance, we created an undirected graph of class to superclass relations for every module, being careful not to add edges from any class to the class object. For each module graph, we computed distances between every pair of classes using an all pairs shortest paths algorithm. We eliminated pairs with distances greater than 10, and this resulted in a set of pairs that we split randomly such that 16,215,400 million pairs of classes were in train, and 1,801,716 million pairs were in test. For fine-tuning, we structured this similar to the forum ranking task, with distances translated to scores between 0 (least related) and 1 (most related), and we used cosine similarity loss, coupled with a embedding similarity evaluator from SBERT. Training on 16.2 million pairs was computationally expensive so we trained it on a random sample of 100,000 training examples, 10,000 of which was used for validation. Figure 5 shows the distribution of embedding distances for each class distance (1-10) to show the dataset characteristics.

# 4.4.3 Evaluation

Since this is a regression task, we evaluated the Pearson  $r$  correlation, which as shown in Table 6 varied from 0.17 (for BERTOverflow) to 0.34 (for Fine-tuned-BERTOverflow); all are statistically significant at  $p \leq 0.01$ . Regression for each model is shown in Figure 6. The improvement from fine-tuning for BERTOverflow showed that the task is useful for building better embeddings. Some other models showed reasonable per

Figure 6: Prediction of embedding distance from class distance (H) for all models. Standard error of regression was less than 0.0002 for all models.

<table><tr><td>Model</td><td>Pearson r</td></tr><tr><td>DistilBERT-paraphrasing</td><td>0.26</td></tr><tr><td>BERT-NLI</td><td>0.20</td></tr><tr><td>msmarco-DistilRoBERTa</td><td>0.23</td></tr><tr><td>xlm-r-paraphrase-v1</td><td>0.27</td></tr><tr><td>USE</td><td>0.28</td></tr><tr><td>BERTOverflow</td><td>0.17</td></tr><tr><td>CodeBERT</td><td>-0.01</td></tr><tr><td>FT-BERTOverflow</td><td>0.34</td></tr><tr><td>FT-CodeBERT</td><td>0.24</td></tr><tr><td>HU-BERTOverflow</td><td>0.29</td></tr></table>

Table 6: Correlation of class hierarchy (H) distance to embedding distance by model. FT represents fine-tuning on H alone.

formance with no tuning (xlm-r-paraphrase at 0.27, and USE at 0.28 respectively), so there is clearly a room to improve these different base models as well, but we leave that issue for future work, since our goal is more on task development rather than building better models.

# 4.5 Class Usage Prediction (U)

# 4.5.1 Task Description

GitHub contains millions of programs, where classes are used in code to achieve some purpose. Classes that are used in the same way; i.e., same set of methods get invoked on them, might be expected to be rated as more similar than classes that do not share any methods. We structured this as a similarity rating task.

# 4.5.2 Dataset

To construct this dataset, we used the Graph4Code knowledge graph [4] which has data flow graphs for 1.3 million GitHub programs. Dataflow tracks the flow of data through return values and parameters within a program. As an example, for the program snippet shown in Figure 1, dataflow would show that fit and predict calls occur on objects returned by calls to the constructors of SGDClassifier and GLM. In this example, SGDClassifier shares 2 methods (denoted as  $M$ ) with 1 class (denoted as  $C$ ), which in this instance is GLM. The classes are similar, in the sense that they both share the same methods in usage, but the degree of similarity is dependent on the number of shared methods ( $M$ ), and the number of classes that have the

Figure 7: Prediction of embedding distance by class usage (U) similarity for all models. Standard error of regression was less than  $1.0\mathrm{e - }4$  for all models.

same methods  $(C)$ . The smaller the  $C$ , the more likely it is that a pair is similar, and the larger the  $M$  the more likely it is that the pair is similar. To capture both dimensions of similarity into a single distance metric for learning, we defined an 'ideal' class pair in terms of our data - that is a vector with  $[max(M),min(C)]$ . We used the Euclidean distance of each class pair from this ideal vector as the dissimilarity metric. Given a pair, the task then is to predict if the classes were similar or distant based on their usage.

The train task contains 75,862 class pairs, and the test task contains 8,439 pairs, with an average distance of 312.21 for train pairs and an average distance of 312.12 for test pairs, suggesting the two had similar characteristics. The fine-tuning task was modeled the same as the class hierarchy prediction task.

# 4.5.3 Evaluation

We frame this task as a regression task and evaluate the Pearson  $r$  correlation once again for all models, which varied from 0.17 (for BERT-NLI) to 0.61 (for HU-BERTOverflow) as shown in Table 7; all results are statistically significant at  $p \leq 0.01$ . The improvement from fine-tuning for BERTOverflow (0.37 to 0.52) also shows the task is useful for building better embeddings. Using hierarchy task as well with HU-BERTOverflow further improved performance to 0.61. This was not the case though for CodeBERT models with and without fine-tuning where its performance dropped to 0.30 from 0.33. We also show in Figure 7 the effectiveness of usage distance as a predictor of cosine embedding distance.

# 4.6 Multi-Task Training

We focus our multi-task training discussion on BERTOverflow, because combining it with training produced the best performance consistently. As shown in Table 8 tasks that derived from code properties (usage (U) and hierarchy (H)) did not benefit from training on ranking (R), forum to class (F) or linked posts (L) tasks on BERTOverflow, which suggests that tasks derived from code properties require different features than those emphasized by RFL tasks. Tasks derived from code properties (HU) however helped RFL tasks, suggesting the importance of having a diversity of tasks for tuning. We were expecting and found class hierarchy training to help the usage task, since code that is closely-related in the type hierarchy tends to have similar

<table><tr><td>Model</td><td>Pearson r</td></tr><tr><td>DistilBERT-paraphrasing</td><td>0.35</td></tr><tr><td>BERT-NLI</td><td>0.17</td></tr><tr><td>msmarco-DistilRoBERTa</td><td>0.34</td></tr><tr><td>xlm-r-paraphrase-v1</td><td>0.36</td></tr><tr><td>USE</td><td>0.41</td></tr><tr><td>BERTOverflow</td><td>0.37</td></tr><tr><td>CodeBERT</td><td>0.33</td></tr><tr><td>FT-BERTOverflow</td><td>0.52</td></tr><tr><td>FT-CodeBERT</td><td>0.30</td></tr><tr><td>HU-BERTOverflow</td><td>0.61</td></tr></table>

Table 7: Correlation of class usage similarity (U) with embedding distance by model. FT represents fine-tuning on U alone.  

<table><tr><td>Model</td><td>R</td><td>F</td><td>L</td><td>H</td><td>U</td></tr><tr><td>RFLHU-BERTOverflow</td><td>0.69/0.89</td><td>46.49</td><td>198.10</td><td>0.15</td><td>0.27</td></tr><tr><td>RFLHU-CodeBERT</td><td>0.68/0.88</td><td>53.98</td><td>148.72</td><td>0.12</td><td>0.38</td></tr><tr><td>RFLH-BERTOverflow</td><td>0.68/0.89</td><td>47.56</td><td>188.73</td><td>0.17</td><td>0.14</td></tr><tr><td>RFLH-CodeBERT</td><td>0.67/0.88</td><td>49.04</td><td>141.97</td><td>0.10</td><td>0.14</td></tr><tr><td>RFL-BERTOverflow</td><td>0.68/0.88</td><td>48.81</td><td>197.63</td><td>0.13</td><td>0.25</td></tr><tr><td>RFL-CodeBERT</td><td>0.68/0.89</td><td>53.29</td><td>144.17</td><td>0.09</td><td>0.26</td></tr><tr><td>HU-BERTOverflow</td><td>0.59/0.84</td><td>15.89</td><td>68.43</td><td>0.29</td><td>0.61</td></tr><tr><td>HU-CodeBERT</td><td>0.61/0.85</td><td>12.95</td><td>45.50</td><td>0.05</td><td>0.41</td></tr></table>

Table 8: Answer Ranking (R) numbers are MRR/NDCG, Hierarchy (H) and Usage (U) tasks are correlation where as Linked Posts (L) and Forum to Doscstrings (F) are T-statistic.

usage due to the nature of classes; this was confirmed by our findings. We also expected usage analysis to help class hierarchy training, because we expected parameter types of methods to relate to the class hierarchy; this did not happen, perhaps due to the dynamically-typed nature of Python, where distinct types can share method names. We expect this to be different in typed languages such as Java, and we plan to investigate it in our future work.

# 5 CONCLUSIONS

In this paper, we presented BLANCA, a set of benchmarks to help further research in code understanding from textual manuals and posts about code. We used BLANCA tasks to show that one can build better language models for understanding code artifacts. We also used multi-task training to demonstrate better representations of classes and functions from these models. We hope these will be useful in enriching code representations with their textual semantics embedding in natural language artifacts of code.

# REFERENCES

[1] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multitask benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353-355, Brussels, Belgium, November 2018. Association for Computational

Linguistics. doi: 10.18653/v1/W18-5446. URL https://www.aclweb.org/anthology/W18-5446.  
[2] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. CoRR, abs / 1905.00537, 2019. URL http://arxiv.org/abs/1905.00537.  
[3] Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin B. Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. Codexglue: A machine learning benchmark dataset for code understanding and generation. CoRR, abs/2102.04664, 2021.  
[4] Ibrahim Abdelaziz, Julian Dolby, James P. McCusker, and Kavitha Srinivas. Graph4code: A machine interpretable knowledge graph for code. arXiv preprint arXiv:2002.09440, 2020.  
[5] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. Codebert: A pretrained model for programming and natural languages, 2020. URL http://arxiv.org/abs/2002.08155.cite arxiv:2002.08155Comment: Accepted to Findings of EMNLP 2020. 12 pages.  
[6] Jeniya Tabassum, Mounica Maddela, Wei Xu, and Alan Ritter. Code and named entity recognition in stackoverflow, 2020.  
[7] Alexander LeClair and Collin McMillan. Recommendations for datasets for source code summarization. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3931-3937, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1394. URL https://www.aclweb.org/anthology/N19-1394.  
[8] Dana Movshovitz-Attias and William W. Cohen. Natural language models for predicting programming comments. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 35–40, Sofia, Bulgaria, August 2013. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/P13-2007.  
[9] Hongyu Li, Seohyun Kim, and Satish Chandra. Neural code search evaluation dataset, 2019.  
[10] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search, 2019.  
[11] Luca Ponzanelli, Gabriele Bavota, Massimiliano Di Penta, Rocco Oliveto, and Michele Lanza. Mining stackoverflow to turn the ide into a self-confident programming prompter. In Proceedings of the 11th Working Conference on Mining Software Repositories, MSR 2014, page 102-111, New York, NY, USA, 2014. Association for Computing Machinery. ISBN 9781450328630. doi:

10.1145/2597073.2597077. URL https://doi.org/10. 1145/2597073.2597077.  
[12] Xing Hu, Ge Li, Xin Xia, David Lo, Shuai Lu, and Zhi Jin. Summarizing source code with transferred api knowledge. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18, pages 2269–2275. International Joint Conferences on Artificial Intelligence Organization, 7 2018. doi: 10.24963/ijcai.2018/314. URL https://doi.org/10.24963/ijcai.2018/314.  
[13] D. Yang, P. Martins, V. Saini, and C. Lopes. Stack overflow in github: Any snippets there? In 2017 IEEE/ACM 14th International Conference on Mining Software Repositories (MSR), pages 280-290, 2017.  
[14] Liang Cai, Haoye Wang, Bowen Xu, Qiao Huang, Xin Xia, David Lo, and Zhenchang Xing. Answerbot: An answer summary generation tool based on stack overflow. In Proceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE 2019, page 1134-1138, New York, NY, USA, 2019. Association for Computing Machinery. doi: 10.1145/3338906.3341186. URL https://doi.org/10. 1145/3338906.3341186.  
[15] X. Liu and H. Zhong. Mining stackoverflow for program repair. In 2018 IEEE 25th International Conference on Software Analysis, Evolution and Reengineering (SANER), pages 118-129, 2018.  
[16] Amirreza Shirani, Bowen Xu, David Lo, T. Solorio, and M. Alipour. Question relatedness on stack overflow: The task, dataset, and corpus-inspired models. *ArXiv*, abs/1905.01966, 2019.  
[17] Uri Alon, Roy Sadaka, Omer Levy, and Eran Yahav. Structural language models for any-code generation. CoRR, abs/ 1910.00577, 2019. URL http://arxiv.org/abs/1910.00577.  
[18] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL http://arxiv.org/abs/1908.10084.  
[19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://www.aclweb.org/anthology/N19-1423.  
[20] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. CoRR, abs/1911.02116, 2019. URL http://arxiv.org/abs/1911.02116.

[21] Victor Sanh, Lysandre Debut, Julien Chaumont, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter, 2020.  
[22] Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Brian Strope, and Ray Kurzweil. Universal sentence encoder for English. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 169-174, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-2029. URL https://wwwaclweb.org/anthology/D18-2029.  
[23] Aditya Kanade, Petros Maniatis, Gogul Balakrishnan, and Kensen Shi. Learning and evaluating contextual embedding of source code. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 12-18 July 2020, Proceedings of Machine Learning Research. PMLR, 2020.

# Footnotes:

Page 2: 4https://docs.sun.io/en/latest/tune/index.html 
