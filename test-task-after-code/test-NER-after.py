import os
import sys
import json
from easyeditor import BaseEditor
from transformers import GPT2Tokenizer
from easyeditor import ROMEHyperParams
from easyeditor import MEMITHyperParams
from easyeditor import MENDHyperParams
from easyeditor import KNHyperParams

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

test_data = json.load(open(os.path.join("./data/edited-data", 'zsre.json'), 'r', encoding='utf-8'))
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

valid_sentences = []
valid_labels = []
with open("./data/task-data/test-NER.txt") as f:
    sentence = []
    labels = []
    for line in f:
        line = line.strip()
        if line:
            word, pos, chunk, label = line.split()
            sentence.append(word)
            labels.append(label)
        else:
            valid_sentences.append(sentence)
            valid_labels.append(labels)
            sentence = []
            labels = []

per = []
org = []
loc = []
mis = []
count_per = 0
total_per = 0
count_loc = 0
total_loc = 0
count_org = 0
total_org = 0
count_misc = 0
total_misc = 0
for j in range(len(valid_sentences)):
    if mode =="Batch-Sequential":
        edit_time = int(sample_total) // int(batch_size)
        result = open(f"./test-result/test-NER/result-NER-{mode}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
    else:
        result = open(f"./test-result/test-NER/result-NER-{mode}-{method}{sample_total}.txt", "a", encoding="utf8")
    result.write("*******************************************************************" + "\n")
    sentence = ""
    per = []
    org = []
    loc = []
    misc = []
    for i in range(len(valid_sentences[j])):
        if valid_sentences[j][i] == "," or valid_sentences[j][i] == ".":
            sentence = sentence + valid_sentences[j][i]
        else:
            sentence = sentence + " " + valid_sentences[j][i]
    result.write(sentence + "\n")
    #person
    k = 0
    while k < len(valid_labels[j]):
        person = ""
        if valid_labels[j][k] == "B-PER":
            for m in range(k, len(valid_labels[j])):
                if valid_labels[j][m] == 'O':
                    break
            while k < m:
                person = person + " " + valid_sentences[j][k]
                k = k + 1
            per.append(person[1:])
            k = m
        k = k + 1
    #location
    k = 0
    while k < len(valid_labels[j]):
        location = ""
        if valid_labels[j][k] == "B-LOC":
            for m in range(k, len(valid_labels[j])):
                if valid_labels[j][m] == 'O':
                    break
            while k < m:
                location = location + " " + valid_sentences[j][k]
                k = k + 1
            loc.append(location[1:])
            k = m
        k = k + 1
    #organization
    k = 0
    while k < len(valid_labels[j]):
        organization = ""
        if valid_labels[j][k] == "B-ORG":
            for m in range(k, len(valid_labels[j])):
                if valid_labels[j][m] == 'O':
                    break
            while k < m:
                organization = organization + " " + valid_sentences[j][k]
                k = k + 1
            org.append(organization[1:])
            k = m
        k = k + 1
    #miscellaneous
    k = 0
    while k < len(valid_labels[j]):
        miscellaneous = ""
        if valid_labels[j][k] == "B-MISC":
            for m in range(k, len(valid_labels[j])):
                if valid_labels[j][m] == 'O':
                    break
            while k < m:
                miscellaneous = miscellaneous + " " + valid_sentences[j][k]
                k = k + 1
            misc.append(miscellaneous[1:])
            k = m
        k = k + 1

    #person
    result.write(f"person:")
    for m1 in range(len(per)):
        result.write(per[m1] + "\t")
    result.write("\n")
    generation_prompts = [f"Please identify Person Entity from the given text. Text: {sentence[1:]} Entity:"]
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
    predict = Outputs[-1].split("Entity:")[1]
    result.write(f"person predict:" + predict + "\n")
    if len(per) == 0 and len(predict) == 0:
        count_per = count_per + 1
        total_per = total_per + 1
    elif len(per) != 0:
        total_per = total_per + len(per)
    elif len(per) == 0 and len(per) != 0:
        total_per = per + 1
    for i in range(len(per)):
        if per[i] in predict:
            count_per = count_per + 1
    #location
    result.write(f"location:")
    for m2 in range(len(loc)):
        result.write(loc[m2] + "\t")
    result.write("\n")
    generation_prompts = [f"Please identify Location Entity from the given text. Text: {sentence[1:]} Entity:"]
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
    predict = Outputs[-1].split("Entity:")[1]
    result.write(f"location predict:" + predict + "\n")
    if len(loc) == 0 and len(predict) == 0:
        count_loc = count_loc + 1
        total_loc = total_loc + 1
    elif len(loc) != 0:
        total_loc = total_loc + len(loc)
    elif len(loc) == 0 and len(loc) != 0:
        total_loc = loc + 1
    for i in range(len(loc)):
        if loc[i] in predict:
            count_loc = count_loc + 1
    #organization
    result.write(f"organization:")
    for m3 in range(len(org)):
        result.write(org[m3] + "\t")
    result.write("\n")
    generation_prompts = [f"Please identify Organization Entity from the given text. Text: {sentence[1:]} Entity:"]
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
    predict = Outputs[-1].split("Entity:")[1]
    result.write(f"organization predict:" + predict + "\n")
    if len(org) == 0 and len(predict) == 0:
        count_org = count_org + 1
        total_org = total_org + 1
    elif len(org) != 0:
        total_org = total_org + len(org)
    elif len(org) == 0 and len(org) != 0:
        total_org = org + 1
    for i in range(len(org)):
        if org[i] in predict:
            count_org = count_org + 1
    #miscellaneous
    result.write(f"miscellaneous:")
    for m4 in range(len(misc)):
        result.write(misc[m4] + "\t")
    result.write("\n")
    generation_prompts = [f"Please identify Miscellaneous Entity from the given text. Text: {sentence[1:]} Entity:"]
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
    predict = Outputs[-1].split("Entity:")[1]
    result.write(f"miscellaneous predict:" + predict + "\n")
    if len(misc) == 0 and len(predict) == 0:
        count_misc = count_misc + 1
        total_misc = total_misc + 1
    elif len(misc) != 0:
        total_misc = total_misc + len(misc)
    elif len(misc) == 0 and len(predict) != 0:
        total_misc = total_misc + 1
    for i in range(len(misc)):
        if misc[i] in predict:
            count_misc = count_misc + 1
    
acc_per = count_per / total_per
acc_loc = count_loc / total_loc
acc_org = count_org / total_org
acc_misc = count_misc / total_misc
acc_total = (acc_per + acc_loc + acc_org + acc_misc) / 4
result.write(str(acc_per) + "\n")
result.write(str(acc_loc) + "\n")
result.write(str(acc_org) + "\n")
result.write(str(acc_misc) + "\n")
result.write(str(acc_total) + "\n")
