from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
import jsonlines

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

correct = 0
with open("./data/task-data/test-reasoning.jsonl", "r+", encoding="utf8") as f:
    for data in jsonlines.Reader(f):
        result = open(f"./test-result/test-reasoning/result-reasoning-before-edit.txt", "a", encoding="utf8")
        question = data['question']
        answers = data['answer']
        hint = answers.split("#### ")[0]
        generation_prompts = [f"Q: {question} A:Let's think step by step. {hint} Therefore, the answer (arabic numerals) is:"]
        answers = answers.split("#### ")[1]
        model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
        batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")

        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=5)

        Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split("Therefore, the answer (arabic numerals) is:")[1]
        result.write(f"The answers is:{str(answers)}" + "\n")
        result.write(f'The model predict is:{str(predict)}'+ "\n")
        if answers in predict:
            correct = correct + 1
        result.close()

    acc = correct / 1319
    result = open(f"./test-result/test-reasoning/result-reasoning-before-edit.txt", "a", encoding="utf8")
    result.write(str(acc) + "\n")
    result.close()
