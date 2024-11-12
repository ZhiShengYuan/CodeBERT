import collections
import math

def _get_ngrams(segment, max_order):
    """从输入文本片段中提取所有不超过最大阶数的n-gram。

    参数：
        segment: 输入的文本片段，按词切分为列表。
        max_order: 要提取的n-gram的最大长度。

    返回：
        包含所有n-gram及其出现次数的Counter对象。
    """
    ngram_counts = collections.Counter()  # 创建一个计数器对象
    for order in range(1, max_order + 1):  # 从1到max_order的n-gram
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])  # 提取当前order长度的n-gram
            ngram_counts[ngram] += 1  # 统计n-gram出现次数
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """计算翻译句子的BLEU分数。

    参数：
        reference_corpus: 参考翻译文本的列表，每个元素是一个参考翻译列表。
        translation_corpus: 机器翻译结果的列表，每个元素是一个翻译句子。
        max_order: 用于计算BLEU分数的最大n-gram阶数。
        smooth: 是否使用平滑方法（Lin等2004年提出）。

    返回：
        一个包含以下值的元组：
        - BLEU分数
        - 各阶n-gram精确度
        - n-gram精确度的几何平均值
        - 短句惩罚因子
    """
    matches_by_order = [0] * max_order  # 各阶n-gram的匹配数
    possible_matches_by_order = [0] * max_order  # 各阶n-gram的可能匹配数
    reference_length = 0  # 参考文本的总长度
    translation_length = 0  # 翻译文本的总长度

    # 遍历每个句子的参考文本和翻译
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)  # 最短参考句子长度
        translation_length += len(translation)  # 翻译句子长度

        # 合并所有参考句子的n-gram计数
        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)  # 翻译句子的n-gram计数

        # 计算n-gram的重叠部分
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]

        # 计算各阶可能的n-gram匹配数
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    # 计算各阶n-gram的精确度
    precisions = [0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0

    # 计算n-gram精确度的几何平均值
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    # 计算长度比率
    ratio = float(translation_length) / reference_length
    bp = 1. if ratio > 1.0 else math.exp(1 - 1. / ratio)  # 短句惩罚

    bleu = geo_mean * bp  # 计算最终BLEU分数

    return bleu, precisions, bp, ratio, translation_length, reference_length


def _bleu(ref_file, trans_file, subword_option=None):
    """从文件中读取参考文本和翻译文本，计算BLEU分数。

    参数：
        ref_file: 参考文本文件路径。
        trans_file: 翻译文本文件路径。
        subword_option: 子词处理选项（未使用）。

    返回：
        计算的BLEU分数（百分制）。
    """
    max_order = 4
    smooth = True

    # 读取参考文本
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename) as fh:
            reference_text.append(fh.readlines())
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference_list.append(reference.strip().split())
        per_segment_references.append(reference_list)

    # 读取翻译文本
    translations = []
    with open(trans_file) as fh:
        for line in fh:
            translations.append(line.strip().split())

    # 计算BLEU分数
    bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score, 2)  # 返回百分制BLEU分数
