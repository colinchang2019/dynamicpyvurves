import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import cfg
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

src_len = cfg.src_len  # length of source
tgt_len = cfg.tgt_len  # length of target

input_size = cfg.input_size
hidden_size = cfg.hidden_size
n_layers = cfg.n_layers
drop_rate = cfg.drop_rate
batch = cfg.batch

class PhysicalLSTM8(nn.Module):
    def __init__(self, input_dim=cfg.input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.lstm = nn.LSTM(input_dim, 256, 2, batch_first=True)
        self.fc1 = nn.Linear(256 + 512 * input_dim, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.conv1.weight.dtype)
        yd = x[:, -1, 5]
        yd = yd.reshape(yd.shape[0], -1)
        time = x[:, 0, 0]
        # x = x[:, :, [1, 2, 4, 5, 6]]
        # print(x[0])
        # print(yd[0])

        x_conv = self.relu(self.bn1(self.conv1(x)))
        x_conv = self.relu(self.bn2(self.conv2(x_conv)))
        # print(x_conv.shape)
        x_conv = x_conv.view(x_conv.size(0), -1)  # Flatten the tensor

        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]

        # print(x_conv.shape, x_lstm.shape)
        x_combined = torch.cat((x_conv, x_lstm), dim=1)
        # print(x_combined.shape)
        x = self.relu(self.bn3(self.fc1(x_combined)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x) * 1.0 + 1.5
        return x, yd, time
