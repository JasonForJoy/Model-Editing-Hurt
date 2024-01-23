import csv
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

with open('./data/task-data/test-NLI.tsv') as f:
    index = []
    sentence_1 = []
    sentence_2 = []
    label = []
    generation_prompts_list = []
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    others = 0
    tsvreader = csv.reader(f, delimiter='\t')
    for line in tsvreader:
        index.append(line[0])
        sentence_1.append(line[1])
        sentence_2.append(line[2])
        label.append(line[3])
    for i in range(1,len(index)):
        generation_prompts = [f"{sentence_1[i]} entails the {sentence_2[i]}. True or False? answer:"]
        generation_prompts_list.append(generation_prompts)
    for j in range(len(generation_prompts_list)):
        result = open(f"./test-result/test-NLI/result-NLI-before-edit.txt", "a", encoding="utf8")
        model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
        batch = tokenizer(generation_prompts_list[j], return_tensors='pt', padding="max_length")

        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=1)
        
        Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split("answer:")[-1]
        result.write(str(label[j+1]) + '\t')
        result.write(str(predict) + '\n')
        if ('true' in predict.lower()) or ('false' in predict.lower()):
            if 'not_entailment' in label[j+1].lower():
                if 'true' in predict.lower():
                    FP = FP + 1
                if 'false' in predict.lower():
                    FN = FN + 1
            else:
                if 'true' in predict.lower():
                    TP = TP + 1
                if 'false' in predict.lower():
                    TN = TN + 1
        else:
            others = others + 1
        result.close()

result = open(f"./test-result/test-NLI/result-NLI-before-edit.txt", "a", encoding="utf8")
if others == 2489:
    result.write("error" + '\n')
else:
    accuracy = (TP + FN)/(TP + FN + TN + FP)
    total_accuracy = (TP + FN)/(TP + FN + TN + FP + others)
    result.write(str(TP) + '\t')
    result.write(str(FN) + '\t')
    result.write(str(TN) + '\t')
    result.write(str(FP) + '\t')
    result.write(str(others) + '\n')
    result.write(str(accuracy) + '\t')
    result.write(str(total_accuracy) + '\n')
result.close()
