# 版权所有 (c) Microsoft Corporation。
# 根据 MIT 许可证授权。
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder          # 编码器模型，如BERT
        self.config = config            # 配置参数
        self.tokenizer = tokenizer      # 分词器
        self.args = args                # 其他参数

    def forward(self, input_ids=None, p_input_ids=None, n_input_ids=None, labels=None): 
        bs, _ = input_ids.size()  # 获取批次大小和序列长度
        # 将输入的正例和负例的input_ids在第0维（批次维）上拼接
        input_ids = torch.cat((input_ids, p_input_ids, n_input_ids), 0)
        
        # 使用编码器对拼接后的input_ids进行编码，attention_mask中1表示有效token
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        # 对编码输出进行掩码处理，忽略padding token（假设padding token id为1），然后求平均
        outputs = (outputs * input_ids.ne(1)[:,:,None]).sum(1) / input_ids.ne(1).sum(1)[:,None]
        # 对输出向量进行L2归一化
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        # 将拼接后的输出按原始批次大小拆分回三个部分
        outputs = outputs.split(bs, 0)
        
        # 计算正例与原始输入的点积相似度，并放大20倍
        prob_1 = (outputs[0] * outputs[1]).sum(-1) * 20
        # 计算负例与原始输入的点积相似度，并放大20倍
        prob_2 = (outputs[0] * outputs[2]).sum(-1) * 20
        # 将原始输出与正例输出拼接，用于计算与所有样本的相似度
        temp = torch.cat((outputs[0], outputs[1]), 0)
        # 将标签也进行拼接，便于后续的掩码操作
        temp_labels = torch.cat((labels, labels), 0)
        # 计算原始输出与所有拼接后的输出的相似度矩阵，并放大20倍
        prob_3 = torch.mm(outputs[0], temp.t()) * 20
        # 创建一个掩码，标记相同标签的位置
        mask = labels[:, None] == temp_labels[None, :]
        # 对相同标签的位置进行大幅度惩罚（-1e9），防止模型学习到这些相似度
        prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()
        
        # 将三种相似度拼接起来，并通过softmax得到概率分布
        prob = torch.softmax(torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)
        # 计算第一列（正例）的对数概率
        loss = torch.log(prob[:, 0] + 1e-10)
        # 取负对数概率的平均作为损失
        loss = -loss.mean()
        # 返回损失和原始输出的表示
        return loss, outputs[0]
