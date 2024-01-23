import regex
import string
import numpy as np
import jsonlines
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

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

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
flag = False

with open("./data/task-data/test-OpenDomainQA.jsonl", "r+", encoding="utf8") as f:
    exact_match_count = 0
    answer_lengths = []
    for data in jsonlines.Reader(f):
        result = open(f"./test-result/test-OpenDomainQA/result-OpenDomainQA-before-edit.txt", "a", encoding="utf8")
        question = data['question']
        document = data['output']
        answer = data['answer']
        generation_prompts = [f"Refer to the passage below and answer the following question. Passage: {document[0]} Question: {question}"]
        model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
        batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")

        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=20)
        Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
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
    
    result = open(f"./test-result/test-OpenDomainQA/result-OpenDomainQA-before-edit.txt", "a", encoding="utf8")
    result.write(str(em) + "\t")
    result.write(str(lens))  
