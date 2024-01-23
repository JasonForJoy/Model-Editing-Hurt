import os
import sys
import json
from rouge import Rouge
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
from easyeditor import MEMITHyperParams
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from easyeditor import KNHyperParams
from easyeditor import MENDHyperParams


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

f = open('./data/task-data/test-summarization.json', 'r')
content = f.read()
corpus = json.loads(content)

summary = []
dialogue = []
for i in range(818):
    summary.append(corpus[i]['summary'])
for i in range(818):
    dialogue.append(corpus[i]['dialogue'])

tokenizer = GPT2Tokenizer.from_pretrained('/disk1/hxxu/EasyEdit-main/EasyEdit-main/hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

bleu_score_total = 0
rouge_score_total = 0
for i in range(len(dialogue)):
    if mode =="Batch-Sequential":
        edit_time = int(sample_total) // int(batch_size)
        result = open(f"./test-result/test-summarization/result-summarization-{mode}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
    else:
        result = open(f"./test-result/test-summarization/result-summarization-{mode}-{method}{sample_total}.txt", "a", encoding="utf8")
    generation_prompts = [f"{dialogue[i]}\nTL;DR:"]
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")

    post_edit_outputs = edited_model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=25)
        
    Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
    predict = Outputs[-1].split("DR:")[-1]
    predict = predict[0:-13]
    result.write(str(summary[i]) + "\t")
    result.write(str(predict) + "\t")

    if len(predict) <= 1:
        bleu_score = 0
        result.write(str(bleu_score) + "\t")
        bleu_score_total = bleu_score_total + bleu_score
        rouge_score = 0
        result.write(str(rouge_score) + "\n")
        rouge_score_total = rouge_score_total + rouge_score
        continue
    else:
        reference = []
        reference.append(summary[i].split())
        candidate = predict.split()
        bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        result.write(str(bleu_score) + "\t")
        bleu_score_total = bleu_score_total + bleu_score
        rouge = Rouge()
        score = rouge.get_scores(predict, summary[i])
        rouge_score = (score[0]['rouge-1']['f'] + score[0]['rouge-2']['f'] + score[0]['rouge-l']['f']) / 3
        result.write(str(rouge_score) + "\n")
        rouge_score_total = rouge_score_total + rouge_score
    result.close()

if mode =="Batch-Sequential":
    edit_time = int(sample_total) // int(batch_size)
    result = open(f"./test-result/test-summarization/result-summarization-{mode}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
else:
    result = open(f"./test-result/test-summarization/result-summarization-{mode}-{method}{sample_total}.txt", "a", encoding="utf8")
result.write(str(bleu_score_total / 818) + "\t")
result.write(str(rouge_score_total / 818) + "\n")
result.close()
