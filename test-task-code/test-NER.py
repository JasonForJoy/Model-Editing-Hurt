from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

valid_sentences = []
valid_labels = []
acc_per_total = 0
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
    result = open(f"./test-result/test-NER/result-NER-before-edit.txt", "a", encoding="utf8")
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
    result.write("\n")
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
    result.write("\n")
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
    result.write("\n")
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
    result.write("\n")

    #person
    result.write(f"person:")
    for m1 in range(len(per)):
        result.write(per[m1] + "\t")
    result.write("\n")
    generation_prompts = [f"Please identify Person Entity from the given text. Text: {sentence[1:]} Entity:"]
    model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
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
    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
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
    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
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
    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=10)
    Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
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
