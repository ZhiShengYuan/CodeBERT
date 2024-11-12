"""
对库中的模型进行微调，用于在文本文件上进行语言建模（GPT, GPT-2, BERT, RoBERTa）。
GPT 和 GPT-2 使用因果语言建模（CLM）损失进行微调，而 BERT 和 RoBERTa 使用掩码语言建模（MLM）损失进行微调。
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model import Model
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer
)

# 初始化日志记录器
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """单个训练/测试样本的特征表示。"""
    def __init__(self, input_tokens, input_ids, index, label):
        self.input_tokens = input_tokens  # 输入的词元列表
        self.input_ids = input_ids        # 输入的词元对应的ID列表
        self.index = index                # 样本的索引
        self.label = label                # 样本的标签

def convert_examples_to_features(js, tokenizer, args):
    """
    将样本转换为模型可接受的特征表示。
    
    参数:
        js (dict): 包含样本数据的字典，必须包含 'code', 'index', 'label' 键。
        tokenizer (RobertaTokenizer): 用于词元化的分词器。
        args (Namespace): 命令行参数，包含 block_size 等配置信息。
    
    返回:
        InputFeatures: 转换后的特征对象。
    """
    # 合并代码字符串中的多余空格
    code = ' '.join(js['code'].split())
    # 使用分词器对代码进行分词，并截断到指定长度
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 4]
    # 构建源输入的词元列表，包括特殊标记
    source_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    # 将词元转换为对应的ID
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # 计算需要填充的长度
    padding_length = args.block_size - len(source_ids)
    # 使用填充ID进行填充
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['index'], int(js['label']))

class TextDataset(Dataset):
    """自定义数据集类，用于加载和处理文本数据。"""
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []         # 存储所有样本的特征
        data = []                  # 临时存储原始数据
        # 打开并读取数据文件
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)  # 每行都是一个JSON对象
                data.append(js)
        # 将原始数据转换为特征表示
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        # 如果是训练数据，打印前3个样本的信息
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        # 按标签组织样本，便于后续采样
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)
    
    def __len__(self):
        """返回数据集的大小。"""
        return len(self.examples)

    def __getitem__(self, i):
        """
        根据索引返回一个样本，包括正样本和负样本。
        
        参数:
            i (int): 样本的索引。
        
        返回:
            tuple: (原始样本的输入ID, 正样本的输入ID, 负样本的输入ID, 标签)
        """
        label = self.examples[i].label      # 获取当前样本的标签
        index = self.examples[i].index      # 获取当前样本的索引
        labels = list(self.label_examples) # 获取所有标签列表
        labels.remove(label)                # 移除当前样本的标签，避免选择相同标签的负样本
        while True:
            # 随机选择一个与当前样本标签相同的正样本
            shuffle_example = random.sample(self.label_examples[label], 1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break
        # 随机选择一个不同标签的负样本
        n_example = random.sample(self.label_examples[random.sample(labels, 1)[0]], 1)[0]
        
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(p_example.input_ids),
            torch.tensor(n_example.input_ids),
            torch.tensor(label)
        )

def set_seed(seed=42):
    """
    设置随机种子以确保结果的可复现性。
    
    参数:
        seed (int): 随机种子值。
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer):
    """训练模型的函数。"""
    # 使用随机采样器
    train_sampler = RandomSampler(train_dataset)
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # 计算总训练步数
    args.max_steps = args.num_train_epochs * len(train_dataloader)

    # 准备优化器和学习率调度器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.max_steps * 0.1),
        num_training_steps=args.max_steps
    )

    # 训练开始的日志记录
    logger.info("***** 开始训练 *****")
    logger.info("  样本数量 = %d", len(train_dataset))
    logger.info("  训练轮数 = %d", args.num_train_epochs)
    logger.info("  每个GPU的即时批大小 = %d", args.train_batch_size // args.n_gpu)
    logger.info("  总训练批大小 = %d", args.train_batch_size)
    logger.info("  总优化步数 = %d", args.max_steps)

    losses, best_map = [], 0  # 初始化损失列表和最佳MAP值
    
    model.zero_grad()  # 梯度清零
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)     # 原始输入
            p_inputs = batch[1].to(args.device)   # 正样本输入
            n_inputs = batch[2].to(args.device)   # 负样本输入
            labels = batch[3].to(args.device)     # 标签
            model.train()                         # 设置模型为训练模式
            loss, vec = model(inputs, p_inputs, n_inputs, labels)  # 前向传播，计算损失和向量

            if args.n_gpu > 1:
                loss = loss.mean()  # 多GPU时取平均损失

            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪

            losses.append(loss.item())  # 记录损失

            # 每100步记录一次损失
            if (step + 1) % 100 == 0:
                logger.info("轮数 {} 步数 {} 损失 {}".format(epoch, step + 1, round(np.mean(losses[-100:]), 4)))

            optimizer.step()    # 更新参数
            optimizer.zero_grad()  # 清零梯度
            scheduler.step()    # 更新学习率

        # 每个epoch结束后进行评估
        results = evaluate(args, model, tokenizer, args.eval_data_file)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        # 如果当前MAP优于之前的最佳MAP，则保存模型
        if results['eval_map'] > best_map:
            best_map = results['eval_map']
            logger.info("  " + "*" * 20)
            logger.info("  最佳MAP:%s", round(best_map, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-map'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("保存模型检查点到 %s", output_dir)

def evaluate(args, model, tokenizer, data_file):
    """评估模型的函数。"""
    eval_dataset = TextDataset(tokenizer, args, data_file)  # 加载评估数据集
    eval_sampler = SequentialSampler(eval_dataset)          # 顺序采样
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4
    )
    
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # 评估开始的日志记录
    logger.info("***** 开始评估 *****")
    logger.info("  样本数量 = %d", len(eval_dataset))
    logger.info("  批大小 = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()  # 设置模型为评估模式
    vecs = [] 
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)     # 原始输入
        p_inputs = batch[1].to(args.device)   # 正样本输入
        n_inputs = batch[2].to(args.device)   # 负样本输入
        label = batch[3].to(args.device)      # 标签
        with torch.no_grad():                  # 禁用梯度计算
            lm_loss, vec = model(inputs, p_inputs, n_inputs, label)  # 前向传播
            eval_loss += lm_loss.mean().item()  # 累加损失
            vecs.append(vec.cpu().numpy())      # 收集向量
            labels.append(label.cpu().numpy())  # 收集标签
        nb_eval_steps += 1
    vecs = np.concatenate(vecs, 0)                # 合并所有向量
    labels = np.concatenate(labels, 0)            # 合并所有标签
    eval_loss = eval_loss / nb_eval_steps         # 计算平均损失
    perplexity = torch.tensor(eval_loss)          # 计算困惑度

    # 计算相似度分数
    scores = np.matmul(vecs, vecs.T)
    dic = {}
    for i in range(scores.shape[0]):
        scores[i, i] = -1000000  # 自身相似度设为极低值，避免影响结果
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]  # 对相似度排序，降序

    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1))
        MAP.append(sum(Avep) / dic[label])
          
    result = {
        "eval_loss": float(perplexity),
        "eval_map": float(np.mean(MAP))
    }

    return result

def main():
    """主函数，负责参数解析、模型训练和评估。"""
    parser = argparse.ArgumentParser()

    ## 必需参数
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="模型预测和检查点将被写入的输出目录。")

    ## 其他参数
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="输入训练数据文件（jsonl 格式）。")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="可选的输入评估数据文件，用于评估困惑度（jsonl 格式）。")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="可选的输入测试数据文件，用于评估困惑度（jsonl 格式）。")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="用于权重初始化的模型检查点。")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="词元化后可选的输入序列长度。")
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练。")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否在开发集上进行评估。")
    parser.add_argument("--do_test", action='store_true',
                        help="是否在测试集上进行评估。")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="训练时每个GPU/CPU的批大小。")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="评估时每个GPU/CPU的批大小。")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Adam 优化器的初始学习率。")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="权重衰减系数。")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam 优化器的 epsilon 值。")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="最大梯度范数。")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="总训练轮数。")
    parser.add_argument('--seed', type=int, default=42,
                        help="初始化的随机种子。")

    # 解析参数
    args = parser.parse_args()
    # 设置日志格式和级别
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    # 设置设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("设备: %s, GPU 数量: %s", device, args.n_gpu)
    
    # 设置随机种子
    set_seed(args.seed)

    # 加载分词器和模型配置
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 

    # 初始化自定义模型
    model = Model(model, config, tokenizer, args)
    logger.info("训练/评估参数 %s", args)

    # 将模型移动到指定设备
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  # 多GPU训练
                
    # 训练过程
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)
        
    # 评估过程
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** 评估结果 *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key] * 100 if "map" in key else result[key], 2)))
            
    # 测试过程
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** 测试结果 *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key] * 100 if "map" in key else result[key], 2)))

if __name__ == "__main__":
    main()
