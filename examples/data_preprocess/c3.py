""" Preprocess dataset for knights and knaves logic task """

import os
import os.path as osp
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

def make_prefix(dp, template_type):
    quiz = dp['quiz']
    if template_type == 'base':
        prefix = f"""The user provides multiple dialogues and a question with several answer choices. The Assistant selects the best option for the question. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a choice question. After thinking, when you finally reach a conclusion, clearly state the choice No. within <answer> </answer> tags. For example, <answer>1</answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        raise ValueError
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='data/c3')
    parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    data_source = 'c3'

    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    train_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': osp.join(args.data_path, "train.jsonl")})
    test_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': osp.join(args.data_path, "test.jsonl")})
    # 900, 900
    # print(len(train_dataset), len(test_dataset))
    # {'quiz': '女：这么多活儿我一个人今天肯定干不完了。\n男：别担心，我开完会就来帮你。\n\n问题：男的在干什么?\n选项：1.开会；2.聊天儿；3.帮女的；4.干活儿。\n\n请回答哪个选项最能回答给出的问题。', 'solution': '1'}
    # print(test_dataset[0])

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "solution_text_format": example['solution'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.data_path, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.data_path, 'test.parquet'))