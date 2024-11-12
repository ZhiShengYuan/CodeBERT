import collections
import math

def _get_ngrams(segment, max_order):
    """从输入片段中提取所有长度不超过给定最大值的n-grams。

    参数：
        segment: 要从中提取n-grams的文本片段。
        max_order: 返回的n-grams的最大长度。

    返回：
        一个Counter对象,包含所有长度不超过max_order的n-grams,以及每个n-gram在片段中出现的次数。
    """
    ngram_counts = collections.Counter()  # 用于存储n-grams及其计数
    for order in range(1, max_order + 1):  # 迭代从1到max_order的n-grams长度
        for i in range(0, len(segment) - order + 1):  # 计算每种长度的n-grams
            ngram = tuple(segment[i:i+order])  # 提取n-gram
            ngram_counts[ngram] += 1  # 更新n-gram计数
    return ngram_counts  # 返回所有n-grams的计数

def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """计算译文片段相对于一个或多个参考译文的BLEU分数。

    参数：
        reference_corpus: 每个翻译的参考译文列表，每个参考译文应该被标记化为一个词的列表。
        translation_corpus: 要评分的翻译列表，每个翻译应该被标记化为一个词的列表。
        max_order: 计算BLEU分数时使用的最大n-gram阶数。
        smooth: 是否应用Lin等人(2004年)提出的平滑处理,用于避免在某些情况下出现零分的问题。

    返回：
        一个包含BLEU分数、n-gram精确度、n-gram精确度的几何平均值和惩罚因子的元组。
    """
    matches_by_order = [0] * max_order  # 匹配的n-gram数量
    possible_matches_by_order = [0] * max_order  # 可能的n-gram数量
    reference_length = 0  # 参考译文的总长度
    translation_length = 0  # 翻译的总长度
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)  # 使用最短的参考译文长度
        translation_length += len(translation)  # 累加翻译的长度

        merged_ref_ngram_counts = collections.Counter()  # 合并所有参考译文的n-grams
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)  # 获取翻译的n-grams
        overlap = translation_ngram_counts & merged_ref_ngram_counts  # 计算重叠的n-grams
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]  # 更新匹配的n-gram数量
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1  # 计算可能的n-gram数量
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order  # 存储每个阶数的精确度
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))  # 平滑处理
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])  # 计算精确度
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)  # 计算精确度的对数和
        geo_mean = math.exp(p_log_sum)  # 计算精确度的几何平均值
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length  # 计算翻译长度与参考译文长度的比率

    if ratio > 1.0:
        bp = 1.  # 惩罚因子为1
    else:
        bp = math.exp(1 - 1. / ratio)  # 计算惩罚因子

    bleu = geo_mean * bp  # 计算BLEU分数

    return (bleu, precisions, bp, ratio, translation_length, reference_length)  # 返回结果

def _bleu(ref_file, trans_file, subword_option=None):
    max_order = 4  # 最大n-gram阶数
    smooth = True  # 是否使用平滑处理
    ref_files = [ref_file]  # 参考文件列表
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename) as fh:
            reference_text.append(fh.readlines())  # 读取参考文件内容
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference_list.append(reference.strip().split())  # 分词处理
        per_segment_references.append(reference_list)
    translations = []
    with open(trans_file) as fh:
        for line in fh:
            translations.append(line.strip().split())  # 分词处理翻译
    bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score, 2)  # 返回BLEU分数的百分制表示
