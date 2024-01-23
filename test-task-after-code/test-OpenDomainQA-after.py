import os
import sys
import json
import regex
import string
import jsonlines
import numpy as np
from easyeditor import BaseEditor
from transformers import GPT2Tokenizer
from easyeditor import ROMEHyperParams
from easyeditor import MEMITHyperParams
from easyeditor import KNHyperParams
from easyeditor import MENDHyperParams

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

tokenizer = GPT2Tokenizer.from_pretrained('/disk1/hxxu/EasyEdit-main/EasyEdit-main/hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
flag = False

test_data = json.load(open(os.path.join("./data/edited-data", 'zsre_mend_eval_portability_gpt4.json'), 'r', encoding='utf-8'))
data = []
for i in range(3111):
    j = i % 1037
    data.append(test_data[j])

prompts = []
ground_truth = []
target_new = []
subject = []

mode = sys.argv[1]
method = sys.argv[2]
sample_begin = sys.argv[3]
sample_end = sys.argv[4]
sample_step = sys.argv[5]
sample_total = (int(sample_begin) - int(sample_end))//int(sample_step)

for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        prompts.append(data[i]['src'])
for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        ground_truth.append(data[i]['pred'])
for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        target_new.append(data[i]['alt'])
for i in range(int(sample_begin), int(sample_end), int(sample_step)):
        subject.append(data[i]['subject'])

if mode == "Batch-Single" and method == "MEND":
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )

if mode == "Batch-Single" and method == "MEMIT":
    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )

if mode == "Instance-Sequential" and method == "ROME":
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )

if mode == "Instance-Sequential" and method == "KN":
    hparams = KNHyperParams.from_hparams('./hparams/KN/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )

if mode == "Batch-Sequential" and method == "MEND":
    batch_size = sys.argv[6]
    prompts_small = []
    prompts_split = []
    for j in range(len(prompts)):
        prompts_small.append(prompts[j])
        if (j+1) % int(batch_size) == 0:
            prompts_split.append(prompts_small)
            prompts_small = []

    ground_truth_small = []
    ground_truth_split = []
    for j in range(len(ground_truth)):
        ground_truth_small.append(ground_truth[j])
        if (j+1) % int(batch_size) == 0:
            ground_truth_split.append(ground_truth_small)
            ground_truth_small = []

    target_new_small = []
    target_new_split = []
    for j in range(len(target_new)):
        target_new_small.append(target_new[j])
        if (j+1) % int(batch_size) == 0:
            target_new_split.append(target_new_small)
            target_new_small = []

    subject_small = []
    subject_split = []
    for j in range(len(subject)):
        subject_small.append(subject[j])
        if (j+1) % int(batch_size) == 0:
            subject_split.append(subject_small)
            subject_small = []

    hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    for k in range(len(prompts_split)):
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts_split[k],
            ground_truth=ground_truth_split[k],
            target_new=target_new_split[k],
            subject=subject_split[k],
            keep_original_weight=False
        )

if mode == "Batch-Sequential" and method == "MEMIT":
    batch_size = sys.argv[6]
    prompts_small = []
    prompts_split = []
    for j in range(len(prompts)):
        prompts_small.append(prompts[j])
        if (j+1) % int(batch_size) == 0:
            prompts_split.append(prompts_small)
            prompts_small = []

    ground_truth_small = []
    ground_truth_split = []
    for j in range(len(ground_truth)):
        ground_truth_small.append(ground_truth[j])
        if (j+1) % int(batch_size) == 0:
            ground_truth_split.append(ground_truth_small)
            ground_truth_small = []

    target_new_small = []
    target_new_split = []
    for j in range(len(target_new)):
        target_new_small.append(target_new[j])
        if (j+1) % int(batch_size) == 0:
            target_new_split.append(target_new_small)
            target_new_small = []

    subject_small = []
    subject_split = []
    for j in range(len(subject)):
        subject_small.append(subject[j])
        if (j+1) % int(batch_size) == 0:
            subject_split.append(subject_small)
            subject_small = []

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    for k in range(len(prompts_split)):
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts_split[k],
            ground_truth=ground_truth_split[k],
            target_new=target_new_split[k],
            subject=subject_split[k],
            keep_original_weight=False
        )

with open("./data/task-data/test-OpenDomainQA.jsonl", "r+", encoding="utf8") as f:
    exact_match_count = 0
    answer_lengths = []
    for data in jsonlines.Reader(f):
        if mode =="Batch-Sequential":
            edit_time = int(sample_total) // int(batch_size)
            result = open(f"./test-result/test-OpenDomainQA/result-OpenDomainQA-{mode}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
        else:
            result = open(f"./test-result/test-OpenDomainQA/result-OpenDomainQA-{mode}-{method}{sample_total}.txt", "a", encoding="utf8")
        question = data['question']
        document = data['output']
        answer = data['answer']
        generation_prompts = [f"Refer to the passage below and answer the following question. Passage: {document[0]} Question: {question}"]
        batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")

        post_edit_outputs = edited_model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=20)

        Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
        answer = data['answer']
        predict = Outputs[-1].split("Answer:")[-1].replace("\n", " ")
        predict = normalize_answer(predict)
        for per_answer in answer:
            result.write(str(normalize_answer(per_answer)) + " ")
        result.write("\t")
        result.write(f'The model predict is: {str(predict)}' + "\n")
        words = predict.split(" ")
        if ems(words[0], answer) or ems(words[-1], answer): 
            exact_match_count += 1
            continue
        for i in range(len(words)-1):
            output = words[i]
            if ems(output, answer): 
                exact_match_count += 1
                continue
            for j in range(i+1, len(words)):
                output = output + " " + words[j]
                if ems(output, answer): 
                    exact_match_count += 1
                    flag = True
                    break
            if flag:
                break
        answer_lengths.append(len(predict.split()))
        result.close()
    
    em = round(exact_match_count/3610, 4)
    lens = round(np.mean(answer_lengths), 4)
    
    if mode =="Batch-Sequential":
        edit_time = int(sample_total) // int(batch_size)
        result = open(f"./test-result/test-OpenDomainQA/result-OpenDomainQA-{mode}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
    else:
        result = open(f"./test-result/test-OpenDomainQA/result-OpenDomainQA-{mode}-{method}{sample_total}.txt", "a", encoding="utf8")
    result.write(str(em) + "\t")
    result.write(str(lens) + "\n")  
