# CMU Advanced NLP Assignment 2: End-to-end NLP System Building

Large language models (LLMs) such as Llama2 have been shown effective for question-answering ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)), however, they are often limited by their knowledge in certain domains. A common technique here is to augment LLM's knowledge with documents that are relevant to the question. In this assignment, you will *develop a retrieval augmented generation system (RAG)* ([Lewis et al., 2021](https://arxiv.org/abs/2005.11401)) that's capable of answering questions about the [Language Technology Institute](https://lti.cs.cmu.edu) (LTI) and [Carnegie Mellon University](https://www.cmu.edu) (CMU).

```
Q: Who is offering the Advanced NLP course in Spring 2024?
A: Graham Neubig
```

So far in your machine learning classes, you may have experimented with standardized tasks and datasets that were easily accessible. However, in the real world, NLP practitioners often have to solve a problem from scratch (like this one!). This includes gathering and cleaning data, annotating your data, choosing a model, iterating on the model, and possibly going back to change your data. In this assignment, you'll get to experience this full process.

Please note that you'll be building your own system end-to-end for this assignment, and *there is no starter code*. You must collect your own data and develop a model of your choice on the data. We will be releasing the inputs for the test set a few days before the assignment deadline, and you will run your already-constructed system over this data and submit the results. We also ask you to follow several experimental best practices, and describe the result in your report.

The key checkpoints for this assignment are,

- [ ] [Understand the task specification](#task-retrieval-augmented-generation-rag)
- [ ] [Prepare your raw data](#preparing-raw-data)
- [ ] [Annotate data for model development](#annotating-data)
- [ ] [Develop a retrieval augmented generation system](#developing-your-rag-system)
- [ ] [Generating results](#generating-results)
- [ ] [Write a report](#writing-report)
- [ ] [Submit your work](#submission--grading)

All deliverables are due by **Thursday, February 29th**. This is a group assignment, see the assignment policies for this class.[^1]

## Task: Retrieval Augmented Generation (RAG)

You'll be working on the task of factual question-answering (QA). We will focus specifically on questions about various facts concerning LTI and CMU. Since existing QA systems might not have the necessary knowledge in this domain, you will need to augment each question with relevant documents. Given an input question, your system will first retrieve documents and use those documents to generate an answer.

### Data Format

**Input** (`questions.txt`): A text file containing one question per line.

**Output** (`system_output.txt`): A text file containing system generated answers. Each line contains a single answer string generated by your system for the corresponding question from `questions.txt`.

**Reference** (`reference_answers.txt`): A text file containing reference answers. Each line contains one or more reference answer strings for the corresponding question from `questions.txt`.

Read our [model and data policy](#model-and-data-policy) for this assignment.

## Preparing raw data

### Compiling a knowledge resource

For your test set and the RAG systems, you will first need to compile a knowledge resource of relevant documents. You are free to use any publicly available resource, but we *highly recommend* including the following,

+ Faculty @ LTI
    - List of faculty ([LTI faculty directory](https://lti.cs.cmu.edu/directory/all/154/1))
    - Research papers by LTI faculty and their metadata ([Semantic Scholar API](https://www.semanticscholar.org/product/api))
    - Teaching (see below)
+ Courses @ CMU
    - Courses offered by each department at CMU and their metadata such as instructors, locations, and credits. ([Schedule of Classes](https://enr-apps.as.cmu.edu/open/SOC/SOCServlet/completeSchedule))
    - Academic calendars for 2023-2024 and 2024-2025 ([CMU calendar](https://www.cmu.edu/hub/calendar/))
+ Academics @ LTI
    - Programs offered by LTI ([website](https://lti.cs.cmu.edu/learn))
    - Program handbooks for information on curriculum, requirements and staff ([PhD](https://lti.cs.cmu.edu/sites/default/files/PhD_Student_Handbook_2023-2024.pdf), [MLT](https://lti.cs.cmu.edu/sites/default/files/MLT%20Student%20Handbook%202023%20-%202024.pdf), [MIIS](https://lti.cs.cmu.edu/sites/default/files/MIIS%20Handbook_2023%20-%202024.pdf), [MCDS](https://lti.cs.cmu.edu/sites/default/files/MCDS%20Handbook%2023-24%20AY.pdf), [MSAII](https://msaii.cs.cmu.edu/sites/default/files/Handbook-MSAII-2022-2023.pdf))
+ Events @ CMU
    - Spring carnival and reunion weekend 2024 ([schedule](https://web.cvent.com/event/ab7f7aba-4e7c-4637-a1fc-dd1f608702c4/websitePage:645d57e4-75eb-4769-b2c0-f201a0bfc6ce?locale=en))
    - Commencement 2024 ([schedule](https://www.cmu.edu/commencement/schedule/index.html))
+ History @ SCS and CMU
    - School of Computer Science ([25 great things](https://www.cs.cmu.edu/scs25/25things), [history](https://www.cs.cmu.edu/scs25/history))
    - [CMU fact sheet](https://www.cmu.edu/about/cmu_fact_sheet_02.pdf) and [history](https://www.cmu.edu/about/history.html)
    - Buggy and it's history ([article](https://www.cmu.edu/news/stories/archives/2019/april/spring-carnival-buggy.html))
    - Athletics ([Tartans](https://athletics.cmu.edu/athletics/tartanfacts), [Scotty](https://athletics.cmu.edu/athletics/mascot/about), [Kiltie Band](https://athletics.cmu.edu/athletics/kiltieband/index))

### Collecting raw data

Your knowledge resource might include a mix of HTML pages, PDFs, and plain text documents. You will need to clean this data and convert it into a file format that suites your model development. Here are some tools that you could use,

+ For all things related to published research, you can use the [Semantic Scholar API](https://www.semanticscholar.org/product/api) to collect papers and their metadata.
+ To parse PDF documents into plain text, you can use [pypdf](https://github.com/py-pdf/pypdf) or [pdfplumber](https://github.com/jsvine/pdfplumber).
+ To process HTML pages, you can use [beautifulsoup4](https://pypi.org/project/beautifulsoup4/).

By the end of this step, you will have a collection of documents that will serve as the knowledge resource for your RAG system.

## Annotating data

Next, you will want to annotate question-answer pairs for two purposes: testing/analysis and training. Use the documents you compiled in the previous step to identify candidate questions for annotation. You will then use the same set of documents to identify answers for your questions.

### Test data

The testing (and analysis) data will be the data that you use to make sure that your system is working properly. In order to do so, you will want to annotate enough data so that you can get an accurate estimate of how your system is doing, and if any improvements to your system are having a positive impact. Some guidelines on this,

+ *Domain Relevance*: Your test data should be similar to the data that you will finally be tested on (questions about LTI and CMU). Use the knowledge resources mentioned above to curate your test set.
+ *Diversity*: Your test data should cover a wide range of questions about LTI and CMU.
+ *Size*: Your test data should be large enough to distinguish between good and bad models. If you want some guidelines about this, see the lecture on experimental design and human annotation.[^2]
+ *Quality*: Your test data should be of high quality. We recommend that you annotate it yourself and validate your annotations within your team.

To help you get started, here are some example questions,

+ Questions that could be answered by just prompting a LLM
    - When was Carnegie Mellon University founded?
+ Questions that can be better answered by augmenting LLM with relevant documents
    - Who is the president of CMU?
+ Questions that are likely answered only through augmentation
    - What courses are offered by Graham Neubig at CMU?
+ Questions that are sensitive to temporal signals
    - Who is teaching 11-711 in Spring 2024?

See [Vu et al., 2023](https://arxiv.org/abs/2310.03214) for ideas about questions to prompt LLMs. For questions with multiple valid answers, you can include multiple reference answers per line in `reference_answers.txt` (separated by a semicolon `;`). As long as your system generates one of the valid answers, it will be considered correct.

This test set will constitute `data/test/questions.txt` and `data/test/reference_answers.txt` in your [submission](#submission--grading).

### Training data

The choice of training data is a bit more flexible, and depends on your implementation. If you are fine-tuning a model, you could possibly:

+ Annotate it yourself manually through the same method as the test set.
+ Do some sort of automatic annotation and/or data augmentation.
+ Use existing datasets for transfer learning.

If you are using a LLM in a few-shot learning setting, you could possibly:

+ Annotate examples for the task using the same method as the test set.
+ Use existing datasets to identify examples for in-context learning.

This training set will constitute `data/train/questions.txt` and `data/train/reference_answers.txt` in your [submission](#submission--grading).

### Estimating your data quality

An important component of every data annotation effort is to estimate its quality. A standard approach is to measure inter-annotator agreement (IAA). To measure this, at least two members of your team should annotate a random subset of your test set. Compute IAA on this subset and report your findings.

## Developing your RAG system

Unlike assignment 1, there is no starter code for this assignment. You are *free to use any open-source model and library*, just make sure you provide due credit in your report. See our [model policy](#model-and-data-policy).

For your RAG system, you will need the following three components, 

1. Document & query embedder
2. Document retriever
3. Document reader (aka. question-answering system)

To get started, you can try langchain's RAG stack that utilizes GPT4All, Chroma and Llama2 ([langchain docs](https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa)).

Some additional resources that could be useful,

+ [11711 lecture notes](http://www.phontron.com/class/anlp2024/lectures/#retrieval-and-rag-feb-15)
+ [ACL 2023 tutorial on retrieval-augmented LMs](https://acl2023-retrieval-lm.github.io)
+ [llama-recipes](https://github.com/facebookresearch/llama-recipes/tree/main/demo_apps/RAG_Chatbot_example) for an example RAG chatbot with Llama2.
+ [Ollama](https://github.com/ollama/ollama) or [llama.cpp](https://github.com/ggerganov/llama.cpp) to run LLMs locally on your machine.

All the code for your data preprocessing, model development and evaluation will be a part of your GitHub repository (see [submission](#submission--grading) for details).

## Generating results

Finally, you will run your systems on our test set (questions only) and submit your results to us. This test set will be released **three days** before the assignment is due.

### Unseen test set

This test set will be curated by the course staff and will evaluate your system's ability to respond to a variety of questions about LTI and CMU. Because the goal of this assignment is not to perform hyperparameter optimization on this private test set, we ask you to not overfit to this test set. You are allowed to submit up to *three* output files (`system_outputs/system_output_{1,2,3}.txt`). We will use the best performing file for grading.

### Evaluation metrics

Your submissions will be evaluated on standard metrics, answer recall, exact match and F1. See section 6.1 of the [original SQuAD paper](https://arxiv.org/abs/1606.05250) for details. These metrics are token-based and measure the overlap between your system answer and the reference answer(s). Therefore, we recommend keeping your system generated responses as concise as possible.

## Writing report

We ask you to write a report detailing various aspects about your end-to-end system development (see the grading criteria below).

There will be a 7 page limit for the report, and there is no required template. However, we encourage you to use the [ACL template](https://github.com/acl-org/acl-style-files).

> [!IMPORTANT]
> Make sure you cite all your sources (open-source models, libraries, papers, blogs etc.,) in your report.

## Submission & Grading

### Submission

Submit all deliverables on Canvas. Your submission checklist is below,

- [ ] Your report.
- [ ] A link to your GitHub repository containing your code.[^3]
- [ ] A file listing contributions of each team member,
    - [ ] data annotation contributions from each team member (e.g. teammate A: instances 1-X; teammate B: instances X-Y, teammate C: instances Y-Z).
    - [ ] data collection (scraping, processing) and modeling contributions from each team member (e.g. teammate A: writing scripts to ..., implementing ...; teammate B:...; teammate C:...;)
- [ ] Testing and training data you annotated for this assignment.
- [ ] Your system outputs on our test set.

Your submission should be a zip file with the following structure (assuming the lowercase Andrew ID is ANDREWID). Make one submission per team.

```
ANDREWID/
├── report.pdf
├── github_url.txt
├── contributions.md
├── data/
│   ├── test/
│   │   ├── questions.txt
│   │   ├── reference_answers.txt
│   ├── train/
│   │   ├── questions.txt
│   │   ├── reference_answers.txt
├── system_outputs/
│   ├── system_output_1.txt
│   ├── system_output_2.txt (optional)
│   ├── system_output_3.txt (optional)
└── README.md
```

### Grading

The following points (max. 100 points) are derived from the results and your report. See course grading policy.[^4]

+ **Submit data** (15 points): submit testing/training data of your creation.
+ **Submit code** (15 points): submit your code for preprocessing and model development in the form of a GitHub repo. We may not necessarily run your code, but we will look at it. So please ensure that it contains up-to-date code with a README file outlining the steps to run it. Your repo 
+ **Results** (30 points): points based on your system's performance on our private test set. 10 points for non-trivial performance,[^5] plus up to 20 points based on level of performance relative to other submissions from the class.
+ **Report**: below points are awarded based on your report.
    + **Data creation** (10 points): clearly describe how you created your data. Please include the following details,
        - How did you compile your knowledge resource, and how did you decide which documents to include?
        - How did you extract raw data? What tools did you use?
        - What data was annotated for testing and training (what kind and how much)?
        - How did you decide what kind and how much data to annotate?
        - What sort of annotation interface did you use?
        - How did you estimate the quality of your annotations? (IAA)
        - For training data that you did not annotate, did you use any extra data and in what way?
    + **Model details** (10 points): clearly describe your model(s). Please include the following details,
        - What kind of methods (including baselines) did you try? Explain at least two variations (more is welcome). This can include which model you used, which data it was trained on, training strategy, etc.
        - What was your justification for trying these methods?
    + **Results** (10 points): report raw numbers from your experiments. Please include the following details,        
        - What was the result of each model that you tried on the testing data that you created?
        - Are the results statistically significant?
    + **Analysis** (10 points): perform quantitative/qualitative analysis and present your findings,
        - Perform a comparison of the outputs on a more fine-grained level than just holistic accuracy numbers, and report the results. For instance, how did your models perform across various types of questions?
        - Perform an analysis that evaluates the effectiveness of retrieve-and-augment strategy vs closed-book use of your models.
        - Show examples of outputs from at least two of the systems you created. Ideally, these examples could be representative of the quantitative differences that you found above.
 
## Model and Data Policy

To make the assignment accessible to everyone,

+ You are only allowed to use models that are also accessible through [HuggingFace](https://huggingface.co/models). This means you may *not* use closed models like OpenAI models, but you *can* opt to use a hosting service for an open model (such as the Hugging Face or Together APIs).
+ You are only allowed to include publicly available data in your knowledge resource, test data and training data.
+ You are welcome to use any open-source library to assist your data annotation and model development. Make sure you check the license and provide due credit.

If you have any questions about whether a model or data is allowed, please ask on Piazza.

## Acknowledgements

This assignment was based on the the [Fall 2023 version of this assignment](https://github.com/cmu-anlp/nlp-from-scratch-assignment-2023/tree/main).

## References

+ Lewis et al., 2021. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
+ Touvron et al., 2023. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288).
+ Vu et al., 2023. [FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation](https://arxiv.org/abs/2310.03214).



[^1]: See the [assignment policies](http://www.phontron.com/class/anlp2024/assignments/#assignment-policies) for this class, including submission information, late day policy and more.

[^2]: See the [lecture notes](http://www.phontron.com/class/anlp2024/lectures/#experimental-design-and-human-annotation-feb-13) on experimental design and human annotation for guidance on annotation, size of test/train data, and general experimental design.

[^3]: Create a private GitHub repo and give access to the TAs in charge of this assignment by the deadline. See piazza announcement post for our GitHub usernames.

[^4]: Grading policy: http://www.phontron.com/class/anlp2024/course_details/#grading

[^5]: In general, if your system is generating answers that are relevant to the question, it would be considered non-trivial. This could be achieved with a basic RAG system.
