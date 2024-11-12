# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
class Seq2Seq(nn.Module):
    """
        构建序列到序列模型 (Sequence-to-Sequence).
        
        参数:

        * `encoder` - Seq2Seq模型的编码器,这里是roberta。
        * `decoder` - Seq2Seq模型的解码器,这里是 RoBERTa。
        * `config` - 编码器模型的配置。
        * `beam_size` - Beam Search中的beam大小。
        * `max_length` - Beam Search中目标序列的最大长度。
        * `sos_id` - Beam Search中目标序列的开始符号ID。
        * `eos_id` - Beam Search中目标序列的结束符号ID。
    """

    # 初始化，构建模型
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        # 调用父类 nn.Module 的构造函数，以确保模型能够正确初始化
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        # 注册一个下三角掩码矩阵，用于解码过程中的自注意力机制。这确保在生成序列时，模型只能关注当前和之前的 token，而无法看到未来的 token。
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )
        # 定义全连接层，用于映射隐藏状态到相同维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义语言模型头，将隐藏状态映射到词表维度，用于预测输出序列
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 将此层的权重初始化为编码器的词嵌入权重，以确保解码器的输出与编码器的输入在词汇表上是兼容的
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        # 定义LogSoftmax层，用于计算概率
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
        
    def forward(self, source_ids, target_ids=None):   
        # 如果目标序列不存在，直接进行生成
        if target_ids is None:
            return self.generate(source_ids)
        
        # 构建一个下三角掩码，确保模型在自注意力计算时只关注有效的输入 token
        mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        # 将 source_ids 和掩码传递给编码器，得到编码器的输出 encoder_output
        encoder_output = self.encoder(source_ids, attention_mask=mask, use_cache=True)
        ids = torch.cat((source_ids, target_ids), -1)
        # 为解码器构建掩码，确保解码器在自注意力计算时只关注当前和之前的 token
        mask = self.bias[:, source_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
        mask = mask & ids[:, None, :].ne(1)
        
        # 通过解码器处理目标序列
        out = self.decoder(target_ids, attention_mask=mask, past_key_values=encoder_output.past_key_values).last_hidden_state
        # 将解码器的输出通过语言模型头映射到词汇表大小的 logits
        lm_logits = self.lm_head(out)
        
        # 将预测位置前移，以实现n时刻预测n+1
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])
        
        """
        返回结果包括：
        loss: 总损失值。
        loss * active_loss.sum(): 有效 token 的总损失（加权损失）。
        active_loss.sum(): 有效 token 的数量。
        """
        outputs = loss, loss * active_loss.sum(), active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        # 构建掩码，用于编码器的自注意力机制
        mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        # 通过编码器处理输入序列，得到编码器的输出
        encoder_output = self.encoder(source_ids, attention_mask=mask, use_cache=True)     
        """
        初始化变量：
        preds 用于存储每个样本的预测结果。
        zero 是一个张量，填充了 0,用于后续的 padding。
        source_len 计算每个输入序列的有效长度，方便后续处理。
        """  
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        
        for i in range(source_ids.shape[0]):
            # 编码器的输出，针对当前样本的有效长度进行切片,以支持束搜索。
            context = [[x[i:i+1, :, :source_len[i]].repeat(self.beam_size, 1, 1, 1) for x in y] 
                       for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            # 束搜索的当前状态
            input_ids = beam.getCurrentState()
            # 当前样本的有效输入序列
            context_ids = source_ids[i:i+1, :source_len[i]].repeat(self.beam_size, 1)
            
            # 解码过程
            for _ in range(self.max_length): 
                if beam.done():
                    break
                # 构建掩码，用于解码器的自注意力机制
                ids = torch.cat((context_ids, input_ids), -1)
                mask = self.bias[:, context_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
                mask = mask & ids[:, None, :].ne(1)
                # 解码器预测下一个词
                out = self.decoder(input_ids, attention_mask=mask, past_key_values=context).last_hidden_state
                hidden_states = out[:, -1, :]
                # 得到输出概率
                out = self.lsm(self.lm_head(hidden_states)).data
                # 更新束搜索
                beam.advance(out)
                # 更新 input_ids，将当前束搜索的状态添加到输入中
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            
            # 从Beam Search中得到最佳预测序列
            hyp = beam.getHyp(beam.getFinal())
            # 将预测序列转换为目标 tokens
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))
        
        # 返回最终预测结果
        preds = torch.cat(preds, 0)
        return preds

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # 每条beam的分数，初始化为零
        self.scores = self.tt.FloatTensor(size).zero_()
        # 每个时间步的回溯指针，用于追踪生成序列的路径
        self.prevKs = []
        # 每个时间步的输出序列
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        self._eos = eos
        # 用于指示当前最高分的序列是否为结束 token
        self.eosTop = False
        # 记录已完成的序列及其得分
        self.finished = []
    
    # 获取当前状态
    def getCurrentState(self):
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch
    
    # 获取当前时间步的回溯指针
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    def advance(self, wordLk):
    # 给定上一步的每条beam的词概率 `wordLk`,计算并更新Beam Search。
    # 参数:
    # `wordLk` - 从上一步前进的词概率 (K , 词汇表大小)

        numWords = wordLk.size(1)  # 词汇表的大小

        # 如果不是第一步（即存在上一步的状态）
        if len(self.prevKs) > 0:
            # 将当前词概率与之前的得分相加，得到累积得分
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            # 不允许以EOS为结尾产生子节点
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:  # 如果前一个词是EOS
                    beamLk[i] = -1e20  # 设置该得分为极低值，防止其扩展
        else:
            # 第一层直接使用当前的词概率
            beamLk = wordLk[0]
        
        # 将 beamLk 展平为一维数组 flatBeamLk，以便于后续处理
        flatBeamLk = beamLk.view(-1)
        # 选取最佳的K个得分及其对应的索引
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        # 更新当前的得分
        self.scores = bestScores
        # 计算前一个时间步的索引
        prevK = bestScoresId // numWords
        # 保存前一个时间步的路径
        self.prevKs.append(prevK)
        # 保存选择的当前时间步的单词索引
        self.nextYs.append((bestScoresId - prevK * numWords))

        # 处理结束节点，记录EOS节点的得分
        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                # 将完成的序列加入到完成列表中
                self.finished.append((s, len(self.nextYs) - 1, i))

        # 检查当前步的最高分是否为EOS，如果是则标记搜索结束
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        """
        检查Beam Search是否完成,只有在EOS位于顶部并且完成的序列数达到beam大小时才认为搜索结束。
        """
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        """
        获取最终的序列结果,返回完成的beam中得分最高的结果。

        如果还没有完成的序列，则添加当前最高分的未完成序列。
        """
        # 如果没有完成的序列，则将最高分的未完成序列加入完成列表
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        # 将完成的序列按得分降序排序
        self.finished.sort(key=lambda a: -a[0])
        # 若完成的序列数不足beam大小，从未完成序列中补充
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            # 从未完成序列中选择得分最高的，补充到完成的序列列表中
            self.finished += unfinished[: self.size - len(self.finished)]
        return self.finished

    def getHyp(self, beam_res):
        """
        根据beam的结果逐步回溯构建完整的假设序列。

        参数:
        * `beam_res` - 最终的beam结果列表

        返回:
        * 构造的所有假设序列
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            # 遍历束结果
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]  # 回溯到前一个时间步的索引
            hyps.append(hyp[::-1])  # 将序列反转以获得正确顺序
        return hyps

    def buildTargetTokens(self, preds):
        """
        将预测的token序列构建成目标序列,遇到EOS时终止。

        参数:
        * `preds` - 预测的token序列列表

        返回:
        * 构造的目标token序列
        """
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break  # 遇到EOS终止
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
