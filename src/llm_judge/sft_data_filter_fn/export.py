import re
import os, json
from multiprocessing import Pool
from collections import defaultdict

instruction_p20 = '''
Given a web search query and its positive document, please generate a hard negative document for information retrieval.
The output must be a JSON object with a single key:
-"hardneg": a string, the hard negative document.

>>>
Please adhere to the following guidelines:
-The hard negative document may use similar keywords or topics as the user query or its positive document.
-The hard negative document should be plausible and accurate.
-The hard negative document should be diverse in topics, sources, styles and the levels of background knowledge required to understand it.

Ensure to generate only the JSON object, do not explain yourself or output anything else. Be creative!

# User query:
'''

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
            hardneg=js["negative_document"]
            hardneg_jsonstr = json.dumps(
                {"hardneg": hardneg},
                ensure_ascii=False
            )
            final.append({
                'query': instruction_p20 + js['query'] + '\n# Positive document:\n' + js['positive_document'],
                'response': hardneg_jsonstr
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