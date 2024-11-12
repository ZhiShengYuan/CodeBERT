# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
from bleu import _bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """单个训练/测试样本类"""
    def __init__(self, idx, source, target):
        self.idx = idx        # 样本索引
        self.source = source  # 源文本
        self.target = target  # 目标文本

def read_examples(filename):
    """从文件读取样本"""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()# 去除行首尾的空白字符
            js = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=" ".join(js['nl'].split()),  # 将自然语言描述中的空白符标准化处理
                    target=" ".join(js["code"].split()),  # 将代码中的空白符标准化处理
                )
            )
    return examples


class InputFeatures(object):
    """一个样本的特征,包括样本ID、源ID和目标ID"""
    def __init__(self, example_id, source_ids, target_ids):
        self.example_id = example_id  # 样本ID
        self.source_ids = source_ids  # 源ID（源文本的token化ID序列）
        self.target_ids = target_ids  # 目标ID（目标文本的token化ID序列）

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    """将样本转换为特征,包括token化及填充"""
    features = []
    for example_index, example in enumerate(examples):
        # 源文本token化
        # 将源文本 example.source 进行 token 化，限制在 args.max_source_length - 5 的长度
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-5]
        ''' 在 token 列表的开头添加特定的标记：
            tokenizer.cls_token:表示序列的开始。
            "<encoder-decoder>"：用于指示编码器-解码器结构的标记。
            tokenizer.sep_token:分隔符标记。
            在 token 列表的末尾添加 "<mask0>" 和一个分隔符
            '''
        source_tokens = [tokenizer.cls_token, "<encoder-decoder>", tokenizer.sep_token] + source_tokens + ["<mask0>", tokenizer.sep_token]
        # 将source_tokens中的 tokens 转换为对应的 token IDs。模型通常需要数值输入，因此需要将文本 token 转换为 ID
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
        # 计算需要填充的长度。max_source_length 是最大允许的输入长度，len(source_ids) 是当前 token ID 列表的长度
        padding_length = args.max_source_length - len(source_ids)
        # 添加填充标记
        source_ids += [tokenizer.pad_token_id] * padding_length

        # 目标文本token化
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        # 在 token 列表的开头添加 "<mask0>" ，在末尾添加一个分隔符
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        # 打印部分样本以进行检查
        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))
                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

        # 添加样本特征到特征列表
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
            )
        )
    return features


def set_seed(seed=42):
    """设置随机种子以确保结果的可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # 命令行参数解析器
    parser = argparse.ArgumentParser()

    ## 必需参数
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="预训练模型路径")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="保存模型预测和检查点的输出目录")

    ## 其他参数
    # 设置训练、开发和测试数据的文件路径
    parser.add_argument("--train_filename", default=None, type=str, help="训练数据文件路径")
    parser.add_argument("--dev_filename", default=None, type=str, help="验证数据文件路径")
    parser.add_argument("--test_filename", default=None, type=str, help="测试数据文件路径")

    # 设置最大源序列和目标序列的长度
    parser.add_argument("--max_source_length", default=64, type=int, help="源序列的最大长度")
    parser.add_argument("--max_target_length", default=32, type=int, help="目标序列的最大长度")

    # 训练和评估相关参数
    parser.add_argument("--do_train", action='store_true', help="是否进行训练")
    parser.add_argument("--do_eval", action='store_true', help="是否进行验证集评估")
    parser.add_argument("--do_test", action='store_true', help="是否进行测试集评估")
    parser.add_argument("--no_cuda", action='store_true', help="是否禁用CUDA")

    # 训练的批次大小和学习率等参数
    parser.add_argument("--train_batch_size", default=8, type=int, help="训练时每批次的样本数量")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="评估时每批次的样本数量")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="初始学习率")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="训练的总轮数")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")

    # 打印参数
    args = parser.parse_args()

    # 设置日志格式和设备
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # 加载分词器
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    # 加载模型配置
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # 设置模型为decoder模式，意味着该模型将用于生成任务
    config.is_decoder = True 
    # 加载encoder模型
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    # 构建Seq2Seq模型
    model = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id)

    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    if args.n_gpu > 1:
        # 多GPU训练
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # 准备训练数据
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_target_ids)
        # 创建数据加载器
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        # 准备优化器和学习率调度器
        # no_decay 列表定义了不应用权重衰减（L2 正则化）的参数
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Adam优化器
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)、
        # 线性学习率调度器，随着训练的进行逐渐减小学习率
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(len(train_dataloader) * args.num_train_epochs * 0.1),
                                                    num_training_steps=len(train_dataloader) * args.num_train_epochs)

    
        # 开始训练
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        '''
        初始化一些变量：
        patience:用于早停的计数器。
        best_score:记录最佳得分。
        losses:记录每个批次的损失。
        dev_dataset:存储验证集的数据。
        '''
        patience, best_score, losses, dev_dataset = 0, 0, [], {}
        # 循环训练
        for epoch in range(args.num_train_epochs):
            for idx,batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                source_ids,target_ids = batch
                loss,_,_ = model(source_ids=source_ids,target_ids=target_ids)
                 # 多 GPU 和梯度累积处理
                if args.n_gpu > 1:
                    loss = loss.mean() # 多GPU时取平均
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                losses.append(loss.item())
                # 反向传播计算梯度
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    # 如果累积的批次数量达到了设定的步数，则更新模型参数 optimizer.step()，并清零梯度 optimizer.zero_grad()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    # 每 100 步记录一次当前损失
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
            if args.do_eval:
                # 使用验证集评估模型                   
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data = dev_dataset['dev_loss']
                else:
                    # 读取验证集数据并进行处理
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)   
                    eval_data = TensorDataset(all_source_ids,all_target_ids)   
                    dev_dataset['dev_loss' ]= eval_examples,eval_data
                # 创建评估数据加载器
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # 评估模型
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,target_ids = batch                  

                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids,target_ids=target_ids)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # 计算平均损失并输出评估结果    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   

                # 计算 BLEU
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long) 
                    eval_data = TensorDataset(all_source_ids)   
                    dev_dataset['dev_bleu'] = eval_examples,eval_data

                # 评估数据的采样器和数据加载器
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p = []
                # 遍历评估数据批次
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]
                    with torch.no_grad():
                        preds = model(source_ids) 
                    # 将ID转换为文本
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)

                model.train()

                # 存储预测结果和精确匹配（EM）
                predictions = []
                EM = []
                with open(args.output_dir + "/dev.output", 'w') as f, open(args.output_dir + "/dev.gold", 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(ref)
                        f.write(ref + '\n')
                        f1.write(gold.target + '\n')     
                        EM.append(ref.split() == gold.target.split()) 

                # 计算BLEU分数并记录结果
                dev_bleu = _bleu(args.output_dir + "/dev.gold", args.output_dir + "/dev.output") 
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  %s = %s " % ("EM", str(round(np.mean(EM) * 100, 2))))
                logger.info("  " + "*" * 20)    
                # 计算开发集的总得分 dev_score，该得分是 BLEU 分数与 EM 百分比的总和
                dev_score = dev_bleu + round(np.mean(EM) * 100, 2)
                # 如果当前的 dev_score 超过了之前记录的最佳得分 best_score，则更新最佳得分，并记录该信息
                if dev_score > best_score:
                    logger.info("  Best score:%s", dev_score)
                    logger.info("  " + "*" * 20)
                    best_score = dev_score

                # 保存最佳BLEU分数的模型检查点
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-score')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # 仅保存模型本身
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience = 0
                # 早停策略，以避免过拟合
                else:
                    patience += 1
                    if patience == -1:
                        break

    # 测试阶段
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-score/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))          

        # 加载测试数据
        eval_examples = read_examples(args.test_filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)   

        # 创建数据加载器
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval() 
        # 生成预测
        p = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]                  
            with torch.no_grad():
                preds = model(source_ids)   
                # 将ID转换为文本
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
        
        # 保存预测结果
        predictions = []
        with open(args.output_dir + "/predictions.txt", 'w') as f:
            for ref, gold in zip(p, eval_examples):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(ref + '\n') 


if __name__ == "__main__":
    main()

