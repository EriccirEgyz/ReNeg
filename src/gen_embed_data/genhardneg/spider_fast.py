import multiprocessing
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import os
import random
import requests
from traceback import print_exc
from multiprocessing import Pool

proc_num, thread_num = 8, 256

save_dir = 'responses'
input_path = ""

TIMEOUT = 3600
MAX_GEN_TOKENS = 1024
ips_list = [x.strip() for x in open('ips.txt').readlines()]

# p20
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

def get_resp(prompt):
    for _ in range(5):
        try:
            url = random.choice(ips_list)
            request_data = {
                "model": 'default', 
                "messages": [
                    {'role': 'user', 'content': prompt},
                ],
                "temperature": 1.0, 
                "top_p": 0.95,
                "max_tokens": MAX_GEN_TOKENS,
                "stop": ['<|im_end|>', '<|endoftext|>'],
            }

            response = requests.post(
                url + '/v1/chat/completions',
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT,
            )

            return response.json()['choices'][0]['message']['content']
        except:
            print_exc()
            continue

def record_processed(data, proc_rank):
    target_path = f"{save_dir}/{proc_rank}.jsonl"
    with open(target_path, 'a+') as f:
        s = json.dumps(data, ensure_ascii=False)
        f.write(s + '\n')
    with open(f"{save_dir}/_id_{proc_rank}.txt", 'a+') as f:
        f.write(data['_id'] + '\n')

def thread_process_data(dq_tuple):
    proc_rank, data_queue = dq_tuple
    while data_queue:
        data = data_queue.popleft()
        prompt = instruction_p20 + data['query'] + '\n# Positive document:\n' + data['positive_document']
        response = get_resp(prompt)
        data.update({'response': response})
        record_processed(data, proc_rank)

def proc_process_file(dics):
    proc_rank, dics = dics
    dq = deque(dics)
    dq_tuple = (proc_rank, dq)
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(thread_process_data, dq_tuple) for _ in range(thread_num)]

os.makedirs(save_dir, exist_ok=True)
exist_ids_set = set()
dirs = os.listdir(save_dir)
for d in tqdm(dirs):
    if d.endswith('.txt') and d.startswith('_id_'):
        lines = open(f'{save_dir}/{d}').readlines()
        for line in lines:
            exist_ids_set.add(line.strip())

dics = []
lines = open(input_path, 'r').readlines()
for line in tqdm(lines):
    dic = json.loads(line.strip())
    if dic['_id'] in exist_ids_set:
        continue
    dics.append(dic)

dics_list = []
for i in range(proc_num):
    this_list = dics[i * len(dics) // proc_num:(i + 1) * len(dics) // proc_num]
    dics_list.append(
        (i, this_list)
    )

del dics

with multiprocessing.Pool(processes=proc_num) as pool:
    pool.map(proc_process_file, dics_list)