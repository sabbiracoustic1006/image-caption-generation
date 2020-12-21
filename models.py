#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:03:50 2020

@author: ratul
"""

import torch
import random
import torchvision
from torch import nn
from bpemb import BPEmb

class Encoder(nn.Module):
    def __init__(self, cnn='resnet18', num_features=512, hidden_size=128):
        super().__init__()
        cnn_model = getattr(torchvision.models, cnn)(pretrained=False)
        feat_ext_layers = list(cnn_model.children())[:-1]
        self.encoder = nn.Sequential(*feat_ext_layers)
        self.hidden_state = nn.Linear(num_features, hidden_size)
        self.cell_state = nn.Linear(num_features, hidden_size)
    
    def forward(self, x):
        out = self.encoder(x).flatten(1)
        hidden_state = self.hidden_state(out).unsqueeze(0)
        cell_state = self.cell_state(out).unsqueeze(0)
        return hidden_state, cell_state
    
class Decoder(nn.Module):
    def __init__(self, rnn_type='gru', input_size=300, hidden_size=128, dropout_p=0.25,
                 num_layers=1, vocab_size=10000):
        super().__init__()
        rnn = nn.GRU if rnn_type == 'gru' else nn.LSTM
        bpemb_bn = BPEmb(lang="en", vs=vocab_size, dim=300) #tensor(bpemb_zh.vectors)
        tensor = self.get_appended_embedding(bpemb_bn)
        self.embedding = nn.Embedding.from_pretrained(tensor, padding_idx=10000, freeze=False)
        self.rnn = rnn(input_size, hidden_size, num_layers, dropout=dropout_p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.classifier = nn.Linear(self.hidden_size, vocab_size, bias=True)
        
    def get_appended_embedding(self, bpemb_bn):
        tensor = torch.cat([torch.tensor(bpemb_bn.vectors),
                            torch.zeros(1,300)
                           ])
        return tensor
    
    def forward(self, input, hi):
        # input shape: (1, batch_size)
        # hi shape: (num_layers, batch_size , hidden_size)
        embedded_seq = self.embedding(input)
        # embedded_seq shape: (1, batch_size, input_size)
        out, ho = self.rnn(embedded_seq, hi)
        # out shape: (1, batch_size, hidden_size)
        # ho shape: (num_layers, batch_size, hidden_size)
        out = out.permute(1, 0, 2)
        # out shape: (batch_size, 1, hidden_size)
        pred = self.classifier(out)
        # pred shape: (batch_size, 1, vocab_size)
        return pred.squeeze(1), ho
    
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_size=300, hidden_size=256,
                 num_layers=1, dropout_p=0.25, rnn_type='lstm',teacher_forcing_ratio=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = Encoder()                               
        self.decoder = Decoder(vocab_size=vocab_size, input_size=embedding_size, num_layers=num_layers,  
                               hidden_size=hidden_size, dropout_p=dropout_p, rnn_type=rnn_type)
    def forward(self, img, target):
        target_len, batch_size = target.shape
        # src shape: (seq_len, batch_size)
        outputs = torch.zeros(target_len-1, batch_size, self.vocab_size).to(target.device)
        # outputs_shape: (target_seq_len-1, batch_size, vocab_size)
        hidden = self.encoder(img)
        # x shape: (seq_len, batch_size)
        x = target[:1]
        # x shape: (1, batch_size)
        for t in range(1, target_len):
            prediction, hidden = self.decoder(x, hidden)
            # prediction shape: (batch_size, vocab_size)    
            outputs[t-1] = prediction    
            predicted = prediction.argmax(1)
            # predicted_char shape: (batch_size)
            if self.training:
                x = target[t] if random.random() < self.teacher_forcing_ratio else predicted
            else:
                x = predicted
            # x shape: (batch_size)
            x = x.unsqueeze(0)
            # x shape: (1, batch_size)
        return outputs
    
    def greedyDecode(self, img, max_steps=50):
        hidden = self.encoder(img)
        # x shape: (seq_len, batch_size)
        x = torch.tensor([1]).view(1,1).to(img.device)
        # x shape: (1, batch_size)
        pred_sent = []
        for t in range(max_steps):
            prediction, hidden = self.decoder(x, hidden)
            # prediction shape: (batch_size, vocab_size)    
            predicted = prediction.argmax(1)
            # predicted_char shape: (batch_size)
            if predicted.item() == 2:
                break
            
            pred_sent.append(predicted.item())
            x = predicted
            # x shape: (batch_size)
            x = x.unsqueeze(0)
            # x shape: (1, batch_size)
        return pred_sent
    
    
def get_model(**kwargs):
    model = Seq2SeqModel(**kwargs)
    return model

if __name__ == '__main__':        
    # test encoder
    enc = Encoder()
    input = torch.rand(2,3,128,128)
    out = enc(input)
    print(out[0].shape)
    
    # test seq2seq model
    seq_model = Seq2SeqModel(vocab_size=10000)
    outs = seq_model(input, torch.rand(32,2).long())