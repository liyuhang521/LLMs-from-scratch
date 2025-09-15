# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 使用传入的编码器对给定的文本进行编码
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        #将给定的文本编译成每一个id对应的列表
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 将整个id列表分割成多个长度为max_length的子列表
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            # 将分割的输入id存入临时的chunk列表中
            input_chunk = token_ids[i:i + max_length]
            # 将分割的目标id存入临时的chunk列表中
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # 将分割的临时输入id列表转化为tensor对象添加进输入id列表中
            self.input_ids.append(torch.tensor(input_chunk))
            # 将分割的临时目标id列表转化为tensor对象添加进目标id列表中
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 分词器使用gpt2分词器进行分词及转化为对应id列表
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 根据传入文本,分词器及窗口大小及步长创建数据加载器
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # 将创建好的数据集传入数据加载器中,batch_size代表一个批次的样本数量,例如id一共100行,如果batch_size=4,则每次取出4行数据进行训练
    # shuffle=True代表每次取出的样本都是随机的,False代表每次取出的样本是按顺序的
    # drop_last=True代表每次取出的样本数量是固定的,False代表每次取出的样本数量是可变的
    # num_workers=0代表数据加载器使用单线程进行数据加载
    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
