import re
import os, json
from multiprocessing import Pool
from collections import defaultdict

instruction = "Instruct: Given a web search query, retrieve relevant passages\nQuery: "
output_path = ''

def handle(idx):
    lines = open('responses/{}.jsonl'.format(idx)).readlines()
    invalid_output = 0
    false_number = 0
    final = []
    for line in lines:
        js = json.loads(line.strip())
        generation = js["response"]
        generation = generation.split("</think>")[-1].strip()
        if str(generation) not in {"0", "1"}:
            invalid_output += 1
            continue
        if str(generation) == "0":
            final.append({
                'query': instruction + js['query'],
                'response': js['positive_document'],
                'rejected_response': [js["negative_document"]]
            })
        else:
            false_number += 1
    return final, invalid_output, false_number

all_invalid_output = 0
all_false_number = 0

files = [x for x in os.listdir('responses') if x.endswith('.jsonl')]
print(len(files))
with Pool(128) as p:
    results = p.map(handle, range(len(files)))

with open(output_path, 'w') as f:
    for final, io, fn in results:
        all_invalid_output += io
        all_false_number += fn

        for dic in final:
            f.write(json.dumps(dic, ensure_ascii=False) + "\n")

print(f"Invalid outputs: {all_invalid_output}")
print(f"Total false negatives: {all_false_number}")