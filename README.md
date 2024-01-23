# Model Editing Can Hurt General Abilities of Large Language Models
This repository will release the source code for the paper:
- [Model Editing Can Hurt General Abilities of Large Language Models](https://arxiv.org/pdf/2401.04700.pdf). <br>
  Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhe-Hua Ling, Kai-Wei Chang, Nanyun Peng <br>
  _Preprint_ <br>

## Overview
In response to the challenge of hallucinations in the output of LLM due to false or outdated knowledge, **model editing** has received a lot of attention due to its low resource consumption. Previous studies have proposed many effective methods and achieved good results in editing performance. However, these model editing methods often overlooking potential sideeffects on the general abilities of LLMs.

This paper analyze side effects by evaluating four popular editing methods on four LLMs across eight representative task categories.

<img src="https://github.com/JasonForJoy/Model-Editing-Hurt/blob/main/definition.png" width=80%>

## Datasets
The datasets are included in `data/`. There are three folders:
* `edited-data`: The data used to edit the model. In this article, we use the ZsRE dataset to edit the model.
* `task-data`: The data used in downstream tasks.
* `training-data`: The data used to train the MEND network. In this article, we use the ZsRE dataset to train the MEND network.

The whole data directory is as follows:
```bash
data/
    |__ edited-data 
        |__ zsre.json
    |__ BAKE_judge.json
        |__ test-dialogue
        |__ test-ClosedDomainQA.jsonl
        |__ test-NER.txt
        |__ test-NLI.tsv
        |__ test-OpenDomainQA.jsonl
        |__ test-reasoning.jsonl
        |__ test-SentimentAnalysis.tsv
        |__ test-summarization.json
    |__ training-data
        |__ zsre_mend_train.json
        |__ zsre_mend_eval.json
```
You can download these datasets here.
