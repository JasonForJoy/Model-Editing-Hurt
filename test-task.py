import sys
import subprocess

task = sys.argv[1]


if task == "ClosedDomainQA":
    process = subprocess.Popen(["python", "./test-task-code/test-ClosedDomainQA.py"])

if task == "dialogue":
    process = subprocess.Popen(["python", "./test-task-code/test-dialogue.py"])

if task == "NER":
    process = subprocess.Popen(["python", "./test-task-code/test-NER.py"])

if task == "NLI":
    process = subprocess.Popen(["python", "./test-task-code/test-NLI.py"])

if task == "OpenDomainQA":
    process = subprocess.Popen(["python", "./test-task-code/test-OpenDomainQA.py"])

if task == "reasoning":
    process = subprocess.Popen(["python", "./test-task-code/test-reasoning.py"])

if task == "SentimentAnalysis":
    process = subprocess.Popen(["python", "./test-task-code/test-SentimentAnalysis.py"])

if task == "summarization":
    process = subprocess.Popen(["python", "./test-task-code/test-summarization.py"])

process.wait()
print("Done")