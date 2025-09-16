# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pkg.llms_from_scratch.ch02 import create_dataloader_v1

import os
import urllib.request

import pytest
import torch


@pytest.mark.parametrize("file_name", ["the-verdict.txt"])
def test_dataloader(tmp_path, file_name):

    if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    # 此处的值书本上写的是4
    context_length = 1024
    # 定义一个256列,50257行的矩阵 token嵌入层,内部装有随机的初始化值,表示初始权重,最终会优化每个列的权重值
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # 书本上得值代表的是创建一个4行256列的矩阵,此处是1024行256列的矩阵
    # 定义一个1024行,256列的矩阵 位置嵌入层,默认是有随机的初始值
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # 每8行数据作为一个批次
    batch_size = 8
    # 窗口大小是4
    max_length = 4
    # 根据上述参数构造数据加载器
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )
    # 每个批次遍历处理数据,一个批次是8行数据
    for batch in dataloader:
        # 此处取出的x和y分别是8行4列的数据,已经过debug验证,没有任何问题
        x, y = batch
        # 将取出的8行4列的x数据从嵌入层中取出这8行4列数据的每一个数据对应的256个数据表示的向量值,也就是一个8行4列256格的三维矩阵
        token_embeddings = token_embedding_layer(x)
        # torch.arange(x)表示生成一个一位数组,从0到x的一个一维数组,此处的x等于4,表示值0到3的
        # 此处使用的是1024行256列的矩阵,然后取出矩阵中的0到3行数据用来表示位置嵌入向量
        # 也就是说此处会取出一个4行256列的矩阵
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))
        # 然后用这个8行4列256格的三维矩阵的矩阵加上4行256列的矩阵,得到一个8行4列8行4列256格的记录了位置向量的三维矩阵
        # 实际上就是8行的每一行都是一个4行256列的矩阵,然后每个矩阵加上位置嵌入向量,
        # 也就是4行256列的矩阵,就是同维度向量加法就将位置信息添加到token嵌入向量中了
        input_embeddings = token_embeddings + pos_embeddings
        break

    input_embeddings.shape == torch.Size([8, 4, 256])
