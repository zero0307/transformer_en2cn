import json
import config
import numpy as np
from torch.autograd import Variable
import torch

# 特殊符号
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'
special_symbols = [PAD, BOS, EOS, UNK]

def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    
    #生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    #返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


def seq_padding(X, padding=0):
    """
    对一个batch批次(以单词id表示)的数据进行padding填充对齐长度
    """
    # 计算该批次数据各条数据句子长度
    L = [len(x) for x in X]
    # 获取该批次数据最大句子长度
    ML = max(L)
    # 对X中各条数据x进行遍历，如果长度短于该批次数据最大长度ML，则以padding id填充缺失长度ML-len(x)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

# 加载词表
def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab, {v: k for k, v in vocab.items()}

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        # 将输入与输出的单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(config.device).long()
        trg = torch.from_numpy(trg).to(config.device).long()
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()
    
    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class PrepareData:
    def __init__(self, train_file, val_file,test_file):
        # 读取单词表
        self.en_word2id_dict, self.en_id2word_dict= load_vocab(config.word2int_en_path)   #word2id和id2word
        self.cn_word2id_dict, self.cn_id2word_dict = load_vocab(config.word2int_cn_path)

        # 读取数据 并分词
        self.train_en,self.train_cn = self.read_data(train_file)
        self.val_en, self.val_cn = self.read_data(val_file)
        self.test_en, self.test_cn = self.read_data(test_file)
        # 划分batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, config.batch_size,shuffle=True)
        self.val_data = self.splitBatch(self.val_en, self.val_cn, config.batch_size,shuffle=True)
        self.test_data = self.splitBatch(self.test_en, self.test_cn, config.batch_size,shuffle=False)


    # 数据预处理
    def read_data(self,file_path):
        En_id = []
        Cn_id = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                en, cn = line.strip().split('\t')
                en_words = en.split()
                cn_words = cn.split()
                # 转换为索引序列并添加特殊符号
                en_ids = [self.en_word2id_dict.get(word, self.en_word2id_dict[UNK]) for word in en_words]
                en_ids = [self.en_word2id_dict[BOS]] + en_ids + [self.en_word2id_dict[EOS]]
                cn_ids = [self.cn_word2id_dict.get(word, self.cn_word2id_dict[UNK]) for word in cn_words]
                cn_ids = [self.cn_word2id_dict[BOS]] + cn_ids + [self.cn_word2id_dict[EOS]]
                En_id.append(en_ids)
                Cn_id.append(cn_ids)
        return En_id , Cn_id
        

    def splitBatch(self, en, cn, batch_size, shuffle):
        """
        将以单词id列表表示的翻译前(英文)数据和翻译后(中文)数据
        按照指定的batch_size进行划分
        如果shuffle参数为True,则会对这些batch数据顺序进行随机打乱
        """
        # 在按数据长度生成的各条数据下标列表[0, 1, ..., len(en)-1]中
        # 每隔指定长度(batch_size)取一个下标作为后续生成batch的起始下标
        idx_list = np.arange(0, len(en), batch_size)
        # 如果shuffle参数为True，则将这些各batch起始下标打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放各个batch批次的句子数据索引下标
        batch_indexs = []
        for idx in idx_list:
            # 注意，起始下标最大的那个batch可能会超出数据大小
            # 因此要限定其终止下标不能超过数据大小

            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        
        # 按各batch批次的句子数据索引下标，构建实际的单词id列表表示的各batch句子数据
        batches = []
        for batch_index in batch_indexs:
            # 按当前batch的各句子下标(数组批量索引)提取对应的单词id列表句子表示数据
            batch_en = [en[index] for index in batch_index]  
            batch_cn = [cn[index] for index in batch_index]
            # 对当前batch的各个句子都进行padding对齐长度
            # 维度为：batch数量×batch_size×每个batch最大句子长度
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            # 将当前batch的英文和中文数据添加到存放所有batch数据的列表中
            batches.append(Batch(batch_en, batch_cn))

        return batches
    

# data = PrepareData(config.train_path, config.val_path,config.test_path)