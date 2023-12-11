import torch
import torch.nn as nn


class MultiAttentionHead(nn.Module):
    def __init__(self, in_dims, attention_dims, n_head, dropout_rate):
        super().__init__()
        self.in_dims = in_dims
        self.n_head = n_head
        self.attention_dims = attention_dims
        self.attentionKey = nn.Linear(in_dims, attention_dims)
        self.attentionQuery = nn.Linear(in_dims, attention_dims)
        self.attentionValue = nn.Linear(in_dims, attention_dims)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        B, T, C = x.shape
        key = self.attentionKey(x)
        query = self.attentionQuery(x[:, -1:, :])
        value = self.attentionValue(x)

        key = key.view(B, T, self.n_head, self.attention_dims // self.n_head).transpose(1, 2)
        query = query.view(B, 1, self.n_head, self.attention_dims // self.n_head).transpose(1, 2)
        value = value.view(B, T, self.n_head, self.attention_dims // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout_rate if self.training else 0,
                                                             is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, 1, self.attention_dims)
        return y


class Block(nn.Module):
    def __init__(self, in_dims, att_dims, out_dims, n_heads, dropout_rate):
        super().__init__()
        self.multiAttentionHead = MultiAttentionHead(in_dims, att_dims, n_heads, dropout_rate)
        self.linear = nn.Linear(att_dims, 4 * out_dims)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * out_dims, out_dims)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.multiAttentionHead(x)
        x = self.relu(self.linear(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.input = configs.enc_in
        self.blockIn = configs.d_model

        self.fci1 = nn.Linear(self.input, self.blockIn)

        self.dropoutRate1 = 0.1
        self.dropout1 = nn.Dropout(self.dropoutRate1)

        self.dropoutAttention = 0.3
        self.attention = configs.d_model
        self.attentionHeads = configs.n_heads
        self.blockOut = configs.d_model

        self.blockTrade = Block(self.blockIn, self.attention, self.blockOut, self.attentionHeads, self.dropoutAttention)

        self.fc1 = nn.Linear(self.blockOut, self.blockOut)
        self.fc2 = nn.Linear(self.blockOut, self.blockOut)
        self.fc3 = nn.Linear(self.blockOut, self.blockOut)
        self.fc4 = nn.Linear(self.blockOut, 1)

        self.relu = nn.ReLU()

    def forward(self, trade):
        x = self.fci1(trade)
        x = self.relu(x)

        x = self.dropout1(x)

        x = self.blockTrade(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
