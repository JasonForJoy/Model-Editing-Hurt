import json
from rouge import Rouge
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu

f = open('./data/task-data/test-summarization.json', 'r')
content = f.read()
corpus = json.loads(content)

summary = []
dialogue = []
for i in range(818):
    summary.append(corpus[i]['summary'])
for i in range(818):
    dialogue.append(corpus[i]['dialogue'])

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

bleu_score_total = 0
rouge_score_total = 0
for i in range(len(dialogue)):
    result = open(f"./test-result/test-summarization/result-summarization-before-edit.txt", "a", encoding="utf8")
    generation_prompts = [f"{dialogue[i]}\nTL;DR:"]
    model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")

    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=25)

    Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
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

result = open(f"./test-result/test-summarization/result-summarization-before-edit.txt", "a", encoding="utf8")
result.write(str(bleu_score_total / 818) + "\t")
result.write(str(rouge_score_total / 818) + "\n")
result.close()