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

## Prepare the environment

### Requirements

**Note: Please use Python 3.9+**
To get started, simply install conda and run:

```shell
git clone https://github.com/JasonForJoy/Model-Editing-Hurt.git
conda create -n EditHurt python=3.9.7
...
pip install -r requirements.txt
```

### Models
All models are putted in `hugging_cache/<model_name>` (model_name=gpt2-xl, gpt-j-6B, or llama-7b).
These could be changed in `hparams/<method_name>/`.

## Evaluation
Eight different downstream task evaluation metrics are as follows

- `Reasoning`: solve rate
- `Natural language inference (NLI) `: accuracy of two-way classification
- `Open-domain QA`: exact match(EM) with the reference answer after minor normalization
- `Closed-domain QA`: exact match(EM) score
- `Dialogue`:  select one best-matched response from four available candidates
- `Summarization`:  the average of ROUGE-1, ROUGE-2 and ROUGE-L
- `Named entity recognition (NER) `: entity-level F1-score
- `Sentiment analysis`:  accuracy of two-wayclassification

GPT-2 XL (1.5B), LLaMA-1 (7B), LLaMA-2 (7B), LLaMA-2 (13B) are used for editing.

- These model editing methods are used in our paper as follows:
  - [MEND](https://github.com/eric-mitchell/mend): Mitchell et al. Hypernetwork
  - [KN](https://github.com/Hunter-DDM/knowledge-neurons): Damai Dai et al. Locate then Edit
  - [ROME](https://github.com/kmeng01/rome): Kevin Meng et al. Locate and Edit
  - [MEMIT](https://github.com/kmeng01/memit): Kevin Meng et al. Locate and Edit
