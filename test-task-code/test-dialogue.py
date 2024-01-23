from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

correct = 0
other = 0
for i in range(1,887):
    with open(f"./data/task-data/test-dialogue/dev_{i}.txt") as f:
        line = eval(f.read())
        answers = line["answers"]
        options = line["options"]
        article = line["article"]
        generation_prompts = [f"Q: {article} Which choice is correct? Answer Chioces: (A){options[0]}(B){options[1]}(C){options[2]}(D){options[3]} A: Among A through D, the answer is"]
        
        result = open(f"./test-result/test-dialogue/result-dialogue-before-edit.txt", "a", encoding="utf8")
        model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
        batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=1)
            
        Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split("the answer is")[-1]
        result.write(str(answers) + '\t')
        result.write(str(predict) + '\n')
        if ('A' in predict) or ('B' in predict) or ('C' in predict) or ('D' in predict):
            if (answers in predict):
                correct = correct + 1
        else:
            other = other + 1
        result.close()

    f.close()

result = open(f"./test-result/test-dialogue/result-dialogue-before-edit.txt", "a", encoding="utf8")
if other == 886:
    result.write("error" + '\n')
else:
    accuracy = correct / 886
    result.write(str(correct) + '\t')
    result.write(str(other) + '\t')
    result.write(str(accuracy) + '\n')
result.close()