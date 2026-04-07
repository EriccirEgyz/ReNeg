import re
import os, json
from multiprocessing import Pool
from collections import defaultdict

instruction = "Instruct: Given a web search query, retrieve relevant passages\nQuery: "
output_path = ''

def handle(idx):
    lines = open('responses/{}.jsonl'.format(idx)).readlines()
    no_json = 0
    incorrect_field = 0
    final = []
    for line in lines:
        js = json.loads(line.strip())
        generation = js["response"]
        generation = generation.split("</think>")[-1].strip()
        match = re.search(r"\{.*\}", generation, re.DOTALL)
        if not match:
            no_json += 1
            continue
        try:
            generation_json = json.loads(match.group(0))
        except json.JSONDecodeError:
            no_json += 1
            continue
        if "hardneg" not in generation_json:
            incorrect_field += 1
            continue
        final.append({
            'query': instruction + js['query'],
            'response': js['positive_document'],
            'rejected_response': [generation_json['hardneg']]
        })
    return final, no_json, incorrect_field

all_no_json = 0
all_incorrect_field = 0

files = [x for x in os.listdir('responses') if x.endswith('.jsonl')]
print(len(files))
with Pool(128) as p:
    results = p.map(handle, range(len(files)))

with open(output_path, 'w') as f:
    for final, nj, ic in results:
        all_no_json += nj
        all_incorrect_field += ic

        for dic in final:
            f.write(json.dumps(dic, ensure_ascii=False) + "\n")

print(f"Failed to extract JSON (no bracket match or parse failure): {all_no_json}")
print(f"JSON missing required field: {all_incorrect_field}")