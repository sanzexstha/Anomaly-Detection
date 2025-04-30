import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class PointEmbedding(nn.Module):
  def __init__(self, args):
    super(PointEmbedding, self).__init__()
    self.linear = nn.Linear(args.n_vars, args.d_model, bias=False)

  def forward(self, x):
    return self.linear(x)


class LocalEmbedding(nn.Module):
  def __init__(self, args):
    super(LocalEmbedding, self).__init__()
    self.conv = nn.Conv1d(
      args.n_vars, args.d_model, args.k_size,
      padding="same", bias=False
    )

  def forward(self, x):
    x = torch.transpose(x, 1, 2)
    x = self.conv(x)
    x = torch.transpose(x, 1, 2)
    return x


class DataEmbedding2(nn.Module):
  def __init__(self, args):
    super(DataEmbedding2, self).__init__()
    if args.data_embed == "point":
      self.model = PointEmbedding(args)
    elif args.data_embed == "local":
      self.model = LocalEmbedding(args)
    else:
      ValueError("Expected 'point' or 'local', but got '{}'".format(args.data_embed))

  def forward(self, x):
      '''
      x : (batch size, window length, # vars)
      return : (batch size, window length, feature size)
      '''
      return self.model(x)

