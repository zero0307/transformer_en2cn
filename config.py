import torch
# 初始化参数设置
batch_size =  32  # 每批次训练数据数量
epochs = 2  # 训练轮数
N_layers = 4  # transformer中堆叠的encoder和decoder block层数
h = 8  # multihead attention hidden个数
d_model = 512  # embedding维数
d_ff = 2048  # feed forward第一个全连接层维数
droupout = 0.1  # dropout比例
max_len = 64 # 最大句子长度
padding_idx = 0

# beam size for bleu
beam_size = 3
use_beam = True

train_path = './transformer/cmn-eng-simple/training.txt'  # 训练集数据文件
val_path = "./transformer/cmn-eng-simple/validation.txt"  # 验证集数据文件
test_path = "./transformer/cmn-eng-simple/testing.txt" #测试文件

save_file = './transformer/save/model.pt'  # 模型保存路径
model_path = './transformer/models/model5_8000.pt' #模型加载路径
output_path = './transformer/output/'

word2int_cn_path = "./transformer/cmn-eng-simple/word2int_cn.json"
word2int_en_path = "./transformer/cmn-eng-simple/word2int_en.json"



gpu_id = '0'
device_id = [0, 1]
# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')