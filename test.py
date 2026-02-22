import config

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction 

from model import batch_greedy_decode, make_model
from data_loader import PrepareData

from torch.autograd import Variable
import torch.nn as nn

from beam_decoder import beam_search


# 特殊符号
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'
special_symbols = [PAD, BOS, EOS, UNK]

def get_next_output_filename():
    """获取下一个可用的输出文件名"""
    existing_files = glob(config.output_path+"out_*.txt")
    max_idx = 0
    for f in existing_files:
        try:
            idx = int(f.split("_")[1].split(".")[0])
            max_idx = max(max_idx, idx)
        except:
            continue
    return config.output_path+f"out_{max_idx+1}.txt"

def plot_bleu_scores(bleu_scores, filename='bleu_comparison.png'):
    """绘制BLEU分数对比图"""
    plt.figure(figsize=(10, 6))
    names = list(bleu_scores.keys())
    values = list(bleu_scores.values())
    
    bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.title('BLEU Scores Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

class LabelSmoothing(nn.Module):
    """标签平滑处理"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
class SimpleLossCompute:
    """
    简单的计算损失和进行参数反向传播更新训练的函数
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()

# 数据预处理
data = PrepareData(config.train_path, config.val_path,config.test_path)

# 准备字典映射
en_id2word = data.en_id2word_dict
cn_id2word = data.cn_id2word_dict

src_vocab = len(data.en_word2id_dict)
tgt_vocab = len(data.cn_word2id_dict)
#中英文字典长度
print("src_vocab %d" % src_vocab)
print("tgt_vocab %d" % tgt_vocab)

pad_id = data.en_word2id_dict[PAD]
bos_id = data.en_word2id_dict[BOS]
eos_id = data.en_word2id_dict[EOS]

criterion = LabelSmoothing(tgt_vocab, padding_idx = config.padding_idx, smoothing= 0.0)

# 初始化模型
model = make_model(
                    src_vocab, 
                    tgt_vocab, 
                    config.N_layers, 
                    config.d_model, 
                    config.d_ff,
                    config.h,
                    config.droupout
                )
model.load_state_dict(torch.load(config.model_path))


def run_test_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i , batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

    return total_loss / total_tokens


def id_to_sentence(id_list, id2word_dict, remove_symbols=True):
    """将id序列转换为句子"""
    # 确保输入是1D列表
    if isinstance(id_list[0], list):
        raise ValueError("输入必须是1D列表,检测到嵌套结构")
    
    if remove_symbols:
        pad_id = data.en_word2id_dict[PAD]
        bos_id = data.en_word2id_dict[BOS]
        eos_id = data.en_word2id_dict[EOS]
        filtered = [id for id in id_list if id not in {pad_id, bos_id, eos_id}]
    else:
        filtered = id_list
    return ' '.join([id2word_dict.get(id, UNK) for id in filtered])

def generate_translations(model, data_iter, en_id2word, cn_id2word, max_len=config.max_len,use_beam = config.use_beam):
    """生成翻译结果"""
    model.eval()
    sources = []        #原英文句子
    translations = []   #模型预测句子  
    references = []     #真实翻译句子
    

    # #获取原预测句子，不替换unk
    # with open(config.test_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         # 分割每行，中文部分在制表符(\t)后面
    #         parts = line.strip().split('\t')
    #         if len(parts) > 1:
    #             chinese_part = parts[1]
    #             # 将中文句子转换为列表中的字符串格式
    #             references.append([chinese_part])
    
    with torch.no_grad():
        for batch in data_iter:
            # 处理源语言
            src = batch.src.to(config.device)
            src_mask = (src != pad_id).unsqueeze(-2)

            # 转换原始英文句子（新增）
            src_ids = batch.src.cpu().numpy()
            for seq in src_ids:
                # 使用英文词典转换，保留特殊符号用于对齐
                #raw_src = id_to_sentence(seq, en_id2word, remove_symbols=False)  
                # 移除实际不需要的符号（可选）
                clean_src = id_to_sentence(seq, en_id2word, remove_symbols=True)
                sources.append(clean_src)

            # 生成翻译结果
            if use_beam:
                output, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, bos_id, eos_id,
                                               config.beam_size, config.device)
            else:
                output = batch_greedy_decode(
                    model, 
                    src, 
                    src_mask,
                    max_len=max_len,
                    start_symbol=data.cn_word2id_dict[BOS],
                    end_symbol=data.cn_word2id_dict[EOS]
                )
            # 转换预测结果
            if use_beam:
                # 取每个样本的最佳候选（假设output是(batch_size, beam_size, seq_len)）
                output =  [h[0] for h in output]  # 取第一个候选
            pred_ids = output
            # pred_ids = output
            for seq in pred_ids:
                translations.append(id_to_sentence(seq, cn_id2word))
            
            # 获取真实目标，存在unk替换
            trg_ids = batch.trg_y.cpu().numpy()
            for seq in trg_ids:
                references.append([id_to_sentence(seq, cn_id2word)])  # BLEU需要列表的列表
            
    return sources,translations, references

def evaluate_bleu(translations, references):
    """计算BLEU-1到BLEU-4分数"""
    refs = [[ref[0].split()] for ref in references]
    hyps = [trans.split() for trans in translations]
    smoothing = SmoothingFunction().method5
    
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1/n]*n + [0]*(4-n))  # n-gram权重
        bleu_scores[f'BLEU-{n}'] = corpus_bleu(
            refs, hyps,
            weights=weights,
            smoothing_function=smoothing
        )
    return bleu_scores


def evaluate_all_bleu_methods(translations, references):
    """计算所有平滑方法的BLEU-4分数"""
    refs = [[ref[0].split()] for ref in references]
    hyps = [trans.split() for trans in translations]
    
    # 获取所有平滑方法
    smoothing = SmoothingFunction()
    methods = [
        ('method0', smoothing.method0),
        ('method1', smoothing.method1),
        ('method2', smoothing.method2),
        ('method3', smoothing.method3),
        ('method4', smoothing.method4),
        ('method5', smoothing.method5),
        # ('method6', smoothing.method6),#method6不适用
        ('method7', smoothing.method7)
    ]
    
    bleu_scores = {}
    for name, method in methods:
        bleu_scores[name] = corpus_bleu(
            refs, hyps,
            weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4权重
            smoothing_function=method
        )
    return bleu_scores

def plot_all_bleu_scores(bleu_scores, filename='all_bleu_comparison.png'):
    """绘制所有平滑方法的BLEU-4分数对比图"""
    plt.figure(figsize=(12, 6))
    names = list(bleu_scores.keys())
    values = [bleu_scores[name] for name in names]
    
    # 创建柱状图
    bars = plt.bar(names, values, color=plt.cm.tab10(range(len(names))))
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.title('BLEU-4 Scores Comparison with Different Smoothing Methods')
    plt.xlabel('Smoothing Method')
    plt.ylabel('BLEU-4 Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.close()

def test(data, model, criterion):
    """
    测试函数，包含：
    1. 计算测试loss
    2. 生成翻译结果
    3. 写入文件
    4. 计算BLEU
    """
    model.eval()
    
    # 计算测试loss
    print('>>>>> Test')
    test_loss = run_test_epoch(data.test_data, model, SimpleLossCompute(model.generator, criterion, None))
    print(f'<<<<< Test loss: {test_loss:.4f}')
    
    
    # 生成翻译结果
    sources,translations, references = generate_translations(
        model, 
        data.test_data,
        en_id2word,
        cn_id2word,
        max_len=config.max_len
    )
    
    # 生成输出文件名
    output_path = get_next_output_filename()
    bleu_img_path = output_path.replace(".txt", "_bleu.png")

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        for src, trans, ref in zip(sources, translations, references):
            f.write(f"原文: {src}\n预测: {trans}\n真实: {ref[0]}\n\n")

    # 计算并展示BLEU分数
    bleu_scores = evaluate_bleu(translations, references)

    bleu4_scores =evaluate_all_bleu_methods(translations,references)
    plot_all_bleu_scores(bleu4_scores)

    for metric, score in bleu_scores.items():
        print(f"{metric}: {score:.4f}")
    
    # 绘制对比图
    plot_bleu_scores(bleu_scores, bleu_img_path)
    print(f"BLEU对比图已保存至: {bleu_img_path}")


# 运行测试
test(data, model, criterion)