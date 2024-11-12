

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
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from model import Model  # 导入自定义的Model类

logger = logging.getLogger(__name__)  # 创建日志记录器
cpu_cont = 16  # 定义使用的CPU数量

def get_example(item):
    """
    处理单个数据项，获取对应的代码片段并进行分词。
    
    参数:
        item: 包含url1, url2, label, tokenizer, args, cache, url_to_code的元组
    
    返回:
        转换后的特征对象
    """
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    if url1 in cache:
        code1 = cache[url1].copy()
    else:
        try:
            code = ' '.join(url_to_code[url1].split())
        except:
            code = ""
        code1 = tokenizer.tokenize(code)
    if url2 in cache:
        code2 = cache[url2].copy()
    else:
        try:
            code = ' '.join(url_to_code[url2].split())
        except:
            code = ""
        code2 = tokenizer.tokenize(code)
        
    return convert_examples_to_features(code1, code2, label, url1, url2, tokenizer, args, cache)

class InputFeatures(object):
    """一个用于存储单个训练/测试示例特征的类。"""
    def __init__(self, input_tokens, input_ids, label, url1, url2):
        self.input_tokens = input_tokens  # 输入的token列表
        self.input_ids = input_ids        # 对应的token ID列表
        self.label = label                # 标签
        self.url1 = url1                  # 第一个URL
        self.url2 = url2                  # 第二个URL

def convert_examples_to_features(code1_tokens, code2_tokens, label, url1, url2, tokenizer, args, cache):
    """
    将示例转换为token ID，并进行必要的填充。
    
    参数:
        code1_tokens: 第一个代码片段的token列表
        code2_tokens: 第二个代码片段的token列表
        label: 标签
        url1: 第一个URL
        url2: 第二个URL
        tokenizer: 分词器
        args: 命令行参数
        cache: 缓存字典
    
    返回:
        InputFeatures对象
    """
    # 截断token列表以适应block_size
    code1_tokens = code1_tokens[:args.block_size - 4]
    code1_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code1_tokens + [tokenizer.sep_token]
    code2_tokens = code2_tokens[:args.block_size - 4]
    code2_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code2_tokens + [tokenizer.sep_token]  
    
    # 将tokens转换为对应的ID
    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id] * padding_length
    
    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id] * padding_length
    
    # 合并两个代码片段的tokens和IDs
    source_tokens = code1_tokens + code2_tokens
    source_ids = code1_ids + code2_ids
    return InputFeatures(source_tokens, source_ids, label, url1, url2)

class TextDataset(Dataset):
    """
    自定义数据集类，用于加载和处理文本数据。
    """
    def __init__(self, tokenizer, args, file_path, pool=None):
        postfix = file_path.split('/')[-1].split('.txt')[0]  # 获取文件后缀
        self.examples = []
        index_filename = file_path
        logger.info("从索引文件创建特征: %s ", index_filename)
        url_to_code = {}
        # 加载data.jsonl文件，将URL映射到代码
        with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['func']

        data = []
        cache = {}
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                label = 0 if label == '0' else 1
                data.append((url1, url2, label, tokenizer, args, cache, url_to_code))
        # 如果是验证集，只使用10%的数据
        if 'valid' in postfix:
            data = random.sample(data, int(len(data) * 0.1))

        # 使用多进程处理数据
        self.examples = pool.map(get_example, tqdm(data, total=len(data)))
        # 如果是训练集，打印前3个示例
        if 'train' in postfix:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** 示例 ***")
                logger.info("索引: {}".format(idx))
                logger.info("标签: {}".format(example.label))
                logger.info("输入tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("输入IDs: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """
        获取指定索引的数据。
        
        返回:
            输入的token ID张量和标签张量
        """
        return torch.tensor(self.examples[item].input_ids), torch.tensor(self.examples[item].label)

def set_seed(seed=42):
    """
    设置随机种子，确保实验的可重复性。
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer, pool):
    """
    训练模型。
    
    参数:
        args: 命令行参数
        train_dataset: 训练数据集
        model: 模型
        tokenizer: 分词器
        pool: 多进程池
    """
    # 使用随机采样器和数据加载器
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = args.max_steps // 10

    # 准备优化器和学习率调度器（线性预热和衰减）
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.max_steps * 0.1),
                                                num_training_steps=args.max_steps)

    # 训练开始日志
    logger.info("***** 开始训练 *****")
    logger.info("  示例数量 = %d", len(train_dataset))
    logger.info("  训练轮数 = %d", args.num_train_epochs)
    logger.info("  每GPU的瞬时批大小 = %d", args.train_batch_size // args.n_gpu )
    logger.info("  总训练批大小 = %d", args.train_batch_size)
    logger.info("  总优化步骤 = %d", args.max_steps)

    losses, best_f1 = [], 0
    model.zero_grad()
 
    for epoch in range(args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)        
            labels = batch[1].to(args.device) 
            model.train()
            loss, logits = model(inputs, labels)
            
            if args.n_gpu > 1:
                loss = loss.mean()  # 在多GPU并行训练时取平均值

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            losses.append(loss.item())
            
            if (step + 1) % 100 == 0:
                logger.info("轮数 {} 步数 {} 平均损失 {}".format(epoch, step + 1, round(np.mean(losses[-100:]), 4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            
            # 定期保存模型并评估
            if len(losses) % args.save_steps == 0:
                results = evaluate(args, model, tokenizer, args.eval_data_file, pool)                 
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value, 4))    
                    
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  " + "*"*20)  
                    logger.info("  最佳F1:%s", round(best_f1, 4))
                    logger.info("  " + "*"*20)                          

                    checkpoint_prefix = 'checkpoint-best-f1'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_file = os.path.join(output_dir, 'model.bin') 
                    torch.save(model_to_save.state_dict(), output_file)
                    logger.info("保存模型检查点到 %s", output_file)

def evaluate(args, model, tokenizer, data_file, pool):
    """
    评估模型性能。
    
    参数:
        args: 命令行参数
        model: 模型
        tokenizer: 分词器
        data_file: 评估数据文件路径
        pool: 多进程池
    
    返回:
        包含评估指标的字典
    """
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, data_file, pool)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # 评估开始日志
    logger.info("***** 开始评估 *****")
    logger.info("  示例数量 = %d", len(eval_dataset))
    logger.info("  批大小 = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []  
    y_trues = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss, cos_sim = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(cos_sim.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits > 0.5  # 阈值判断预测结果
    
    # 计算评估指标
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),   
    }

    return result
                                                    
def main():
    """
    主函数，负责解析参数、设置环境、加载模型和分词器，以及执行训练、评估和测试。
    """
    parser = argparse.ArgumentParser()

    ## 必需参数
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="模型预测和检查点保存的输出目录。")

    ## 其他参数
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="训练数据文件（jsonl格式）。")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="可选的评估数据文件，用于评估困惑度（jsonl格式）。")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="可选的测试数据文件，用于评估困惑度（jsonl格式）。")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="用于权重初始化的模型检查点。")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="分词后可选的输入序列长度。")
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练。")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否在开发集上进行评估。")
    parser.add_argument("--do_test", action='store_true',
                        help="是否在测试集上进行评估。")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="每个GPU/CPU的训练批大小。")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="每个GPU/CPU的评估批大小。")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Adam优化器的初始学习率。")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="如果应用权重衰减，则使用的权重衰减率。")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam优化器的epsilon值。")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="梯度裁剪的最大范数。")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="总共进行的训练轮数。")
    parser.add_argument('--seed', type=int, default=42,
                        help="用于初始化的随机种子")
    
    pool = multiprocessing.Pool(cpu_cont)  # 创建多进程池
    
    # 解析命令行参数
    args = parser.parse_args()
    # 设置日志格式和级别
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO )
    # 设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("设备: %s, GPU数量: %s", device, args.n_gpu)
    
    # 设置随机种子
    set_seed(args.seed)

    # 构建模型和分词器
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 

    # 使用自定义的Model类包装预训练模型
    model = Model(model, config, tokenizer, args)
    logger.info("训练/评估参数 %s", args)

    model.to(args.device)  # 将模型移动到指定设备
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  # 多GPU并行处理
    
    # 开始训练
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, pool=pool)
        train(args, train_dataset, model, tokenizer, pool)
            
    # 开始评估
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))  # 加载最佳模型检查点     
        result = evaluate(args, model, tokenizer, args.eval_data_file, pool=pool)
        logger.info("***** 评估结果 *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key], 2)))
                
    # 开始测试
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))  # 加载最佳模型检查点       
        result = evaluate(args, model, tokenizer, args.test_data_file, pool=pool)
        logger.info("***** 测试结果 *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key], 2)))

if __name__ == "__main__":
    main()
