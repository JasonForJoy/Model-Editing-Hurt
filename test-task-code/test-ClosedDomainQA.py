from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
import jsonlines

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

with open("./data/task-data/test-ClosedDomainQA.jsonl", "r+", encoding="utf8") as f:
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    others = 0
    for data in jsonlines.Reader(f):
        result = open(f"./test-result/test-ClosedDomainQA/result-ClosedDomainQA-before-edit.txt", "a", encoding="utf8")
        passage = data['passage']
        question = data['question']
        answers = data['answer']
        generation_prompts = [f"Please answer the given question based on the passage. The answer should be exact 'yes' or 'no'. passage: {passage} question: {question}. answer:"]
        model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
        batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")

        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=1)

        Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split("answer:")[1]
        result.write(str(answers) + '\t')
        result.write(str(predict) + '\n')
        if (predict.lower() == ' yes') or (predict.lower() == ' no'):
            if (predict.lower() == ' yes') and (answers):
                TP = TP + 1
            elif (predict.lower() == ' no') and (not(answers)):
                FN = FN + 1
            elif (predict.lower() == ' no') and (answers):
                TN = TN + 1
            elif (predict.lower() == ' yes') and (not(answers)):
                FP = FP + 1
        else:
            others = others + 1
        result.close()

result = open(f"./test-result/test-ClosedDomainQA/result-ClosedDomainQA-before-edit.txt", "a", encoding="utf8")
if others == 3267:
    result.write("error")
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