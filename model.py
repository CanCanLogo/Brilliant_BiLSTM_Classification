import torch.nn as nn
# 定义LSTM网络
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # 定义遗忘门可训练参数
        # 权重（下同）
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # 偏置（下同）
        self.b_f = nn.Parameter(torch.randn(hidden_size))

        # 定义输入门Sigmoid层可训练参数
        self.W_i = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.randn(hidden_size))

        # 定义输入门tanh层可训练参数
        self.W_c = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_c = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.randn(hidden_size))

        # 定义输出门可训练参数
        self.W_o = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size))

        # 定义h2o可训练参数
        self.W_h = nn.Parameter(torch.randn(hidden_size, output_size))
        self.b_h = nn.Parameter(torch.randn(output_size))

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, cell_state):
        # 遗忘门
        f_gate = torch.sigmoid(torch.mm(input, self.W_f) + torch.mm(hidden_state, self.U_f) + self.b_f)

        # 输入门sigmoid层
        i_gate = torch.sigmoid(torch.mm(input, self.W_i) + torch.mm(hidden_state, self.U_i) + self.b_i)

        # 输入门tanh层
        c_tilde = torch.tanh(torch.mm(input, self.W_c) + torch.mm(hidden_state, self.U_c) + self.b_c)

        # 输出门
        o_gate = torch.sigmoid(torch.mm(input, self.W_o) + torch.mm(hidden_state, self.U_o) + self.b_o)

        # 得出新细胞状态
        cell_state = f_gate * cell_state + i_gate * c_tilde

        # 得出新隐藏层状态
        hidden_state = o_gate * torch.tanh(cell_state)

        # 得出output
        output = torch.mm(hidden_state, self.W_h) + self.b_h
        output = self.softmax(output)
        return output, hidden_state, cell_state

    # 初始化隐藏状态
    def initHidde(self):
        hidden_state = torch.zeros(1, self.hidden_size)
        cell_state = torch.zeros(1, self.hidden_size)
        return hidden_state, cell_state

class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=300, d_hid=128, nlayers=2, dropout=0.1, num_classes = 8, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)
        # self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选
        # self.classifier = nn.Linear(ntoken * d_hid * 2, num_classes)
        self.classifier = nn.Linear(d_hid * 2, num_classes)
        # 请自行设计分类器
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        x = self.lstm(x)[0]
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x)
        # x = x.reshape(-1, 50 * 80 * 2)# ntoken*nhid*2 (2 means bidirectional)
        x = self.classifier(x[:, -1, :])
        #------------------------------------------------------end------------------------------------------------------#
        return x