#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class CharLSTM(nn.Module):
    def __init__(self):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        # self.h0 = torch.zeros(2, batch_size, 256).requires_grad_()
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x):
        x = self.embed(x)
        # if self.h0.size(1) == x.size(0):
        #     self.h0.data.zero_()
        #     # self.c0.data.zero_()
        # else:
        #     # resize hidden vars
        #     device = next(self.parameters()).device
        #     self.h0 = torch.zeros(2, x.size(0), 256).to(device).requires_grad_()
        x, hidden = self.lstm(x)
        x = self.drop(x)
        
        # x = x.contiguous().view(-1, 256)
        # x = x.contiguous().view(-1, 256)
        return self.out(x[:, -1, :])

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #
    #     initial_hidden = (weight.new(2, batch_size, 256).zero_(),
    #                       weight.new(2, batch_size, 256).zero_())
    #
    #     return initial_hidden
    
    
class RNN_FedShakespeare(nn.Module):
    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(RNN_FedShakespeare, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        # for name, param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #          nn.init.constant_(param, 0.0)
        #     elif 'weight_ih' in name:
        #          nn.init.kaiming_normal_(param)
        #     elif 'weight_hh' in name:
        #          nn.init.orthogonal_(param)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        # output = self.fc(final_hidden_state)
        # For fed_shakespeare
        output = self.fc(lstm_out[:, :])
        output = torch.transpose(output, 1, 2)
        return output