import sys
import subprocess

task = sys.argv[1]
mode = sys.argv[2]
method = sys.argv[3]
sample_begin = sys.argv[4]
sample_end = sys.argv[5]
sample_step = sys.argv[6]

if task == "ClosedDomainQA":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-ClosedDomainQA-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-ClosedDomainQA-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "dialogue":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-dialogue-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-dialogue-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "NER":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-NER-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-NER-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "NLI":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-NLI-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-NLI-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "OpenDomainQA":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-OpenDomainQA-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-OpenDomainQA-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "reasoning":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-reasoning-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-reasoning-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "SentimentAnalysis":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-SentimentAnalysis-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-SentimentAnalysis-after.py", mode, method, sample_begin, sample_end, sample_step])

if task == "summarization":
    if mode == "Batch-Sequential":
        batch_size = sys.argv[7]
        process = subprocess.Popen(["python", "./test-task-after-code/test-summarization-after.py", mode, method, sample_begin, sample_end, sample_step, batch_size])
    else:
        process = subprocess.Popen(["python", "./test-task-after-code/test-summarization-after.py", mode, method, sample_begin, sample_end, sample_step])

process.wait()
print("Done")