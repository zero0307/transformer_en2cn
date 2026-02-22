import config

import time
import torch
import torch.nn as nn
import numpy as np

from model import  make_model
from data_loader import PrepareData

from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import os


# 特殊符号
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'
special_symbols = [PAD, BOS, EOS, UNK]

print(config.device)

def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    
    #  生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class LabelSmoothing(nn.Module):
    """标签平滑处理"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')  #KL散度
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
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()
    
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i , batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens



def train(data, model, criterion, optimizer):
    """
    训练并保存模型，并绘制验证损失变化曲线，保存图像到loss_change文件夹
    """
    # 初始化模型在val集上的最优Loss为一个较大值
    best_val_loss = 1e5
    # 用于记录每个epoch的验证损失
    val_loss_history = []#存储验证损失绘图
    
    for epoch in range(config.epochs):
        # 模型训练
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()

        # 在val集上进行loss评估
        print('>>>>> Evaluate')
        val_loss = run_epoch(data.val_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % val_loss)
        
        # 记录当前epoch的验证损失
        val_loss_history.append(val_loss)
        
        # 如果当前epoch的模型在val集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), config.save_file)
            best_val_loss = val_loss
    
    # 修改后的绘图部分（替换原代码中的绘图部分）
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_loss_history)+1), 
            [v.cpu().numpy() if torch.is_tensor(v) else v for v in val_loss_history],  # 确保所有值都是CPU numpy数组
            'b-o', 
            label='Validation Loss')
    plt.title(f'Validation Loss History (Best: {min(val_loss_history):.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join('./transformer/loss_change', f'val_loss_epoch{len(val_loss_history)}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()  # 防止内存泄漏

    print(f' Validation loss plot saved to: {save_path}')
        


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

# 训练
print(">>>>>>> start train")
train_start = time.time()
criterion = LabelSmoothing(tgt_vocab, padding_idx = config.padding_idx, smoothing= 0.1)
optimizer = NoamOpt(model.src_embed[0].d_model, 1, 8000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

train(data, model, criterion, optimizer)
print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")