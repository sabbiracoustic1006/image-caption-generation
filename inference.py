#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:05:24 2020

@author: ratul
"""

import json, torch, cv2, os
from dataset import get_dataloaders
from models import get_model
from bpemb import BPEmb
from matplotlib import pyplot as plt

train_dataloader, valid_dataloader = get_dataloaders(1)
bpe = BPEmb(lang='en', vs=10000, dim=300)

with open('config.json','r') as f:
    model_config = json.load(f)
    
model = get_model(**model_config) #Seq2SeqModel(dropout_p=0.25, hidden_size=256,num_layers=1)
model.cuda()    

state = torch.load('saved_models/test_run_again/state.pth')
model.load_state_dict(state['model_state_dict'])
model.eval()

with torch.no_grad():
    for step, (img, target) in enumerate(valid_dataloader):
        path = valid_dataloader.dataset.paths[step]
        pred_ids = model.greedyDecode(img.cuda())
        pred_sent = bpe.decode_ids(pred_ids)
        tgt_sent = bpe.decode_ids(target.view(-1).tolist())
        np_img = cv2.imread(path)[:,:,[2,1,0]]
        plt.imshow(np_img)
        plt.axis('off')
        plt.title(pred_sent)
        plt.savefig(f'predicted-captions/{os.path.basename(path)}_captioned.jpg')

        
        