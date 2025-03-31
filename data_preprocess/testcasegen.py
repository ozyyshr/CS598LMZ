import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb



INSTRUCTION = """
You are a professional who writes {lang} test methods. Your task is to generate a {lang} function without any natural language descriptions.
"""

def extract_function_details(code_str):
    # Match the function definition line and extract the function name
    func_def_match = re.search(r'def\s+(\w+)\s*\(.*\):', code_str)
    if not func_def_match:
        return None, None

    func_name = func_def_match.group(1)

    # Extract the full function code starting from the matched 'def' line
    func_start_index = func_def_match.start()
    
    # Optionally, we can attempt to extract until the function logically ends
    # For simplicity, we'll take until the next non-indented line or end of string
    lines = code_str[func_start_index:].splitlines()
    func_lines = [lines[0]]

    for line in lines[1:]:
        if line.strip() == "":
            func_lines.append(line)
        elif line.startswith((' ', '\t')):
            func_lines.append(line)
        else:
            break

    func_code = "\n".join(func_lines)
    return func_name, func_code


def make_prefix(dp, split):

    prompt_template=open('./TestEval/prompt/template_base.txt').read()

    if split == 'test':
        func_name=dp['func_name']
        desc=dp['description']
        code=dp['python_solution']
    elif split == 'train':
        desc = dp['content'].split('**Example 1:**')[0]
        python_code = dp['python']
        func_name, code = extract_function_details(python_code)

    prompt=prompt_template.format(lang='python', program=code, description=desc, func_name=func_name)

    input_str = """<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + INSTRUCTION.format(lang='python')
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<think>
[thinking process]
</think>
<answer>
{
    "test_method": "...."
} 
</answer>. 
Note: Your answer should not include any quotation marks or descriptions outside the function definition.
"""

    input_str +=  prompt + """<|im_end|>
<|im_start|>assistant
Let me think step by step. 
<think>
"""

    return input_str


def load_dataset_testcasegen():

    original_leetcode = load_dataset('greengerong/leetcode')['train']
    with open('./TestEval/data/leetcode-py.jsonl', 'r') as f:
        testeval = [json.loads(line) for line in f]
    
    # filter out the testeval data from the original leetcode dataset
    testeval_ids = [x['task_num'] for x in testeval]
    original_leetcode = [x for x in original_leetcode if x['id'] not in testeval_ids]

    print(f"Filtered training data size: {len(original_leetcode)}")
    print(f"Test dataset size: {len(testeval)}")
    
    return original_leetcode, testeval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset', type=str, default='testcasegen')

    args = parser.parse_args()
    
    data_source = args.dataset
    
    train_data, test_data = load_dataset_testcasegen()

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    def make_map_fn(split):
        def process_fn_test(example, idx):
            question = make_prefix(example, split)
            solution = {
                "target": "",
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn_test
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    
    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

    lengths_list_test = []
    for d in test_dataset:
        lengths_list_test.append(len(d['prompt'][0]['content'].split()))
        
        
    print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list)}")
    print(f"Average length of test dataset: {sum(lengths_list_test) / len(lengths_list_test)}")
    
    local_dir = os.path.join(args.local_dir, args.dataset)
    hdfs_dir = os.path.join(args.hdfs_dir, args.dataset) if args.hdfs_dir is not None else None
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 