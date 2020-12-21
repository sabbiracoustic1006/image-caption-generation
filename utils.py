#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:49:20 2020

@author: ratul
"""

import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,accuracy_score
from torch import nn
from bpemb import BPEmb
import nltk
from bnlp import NLTKTokenizer

tokenizer = NLTKTokenizer()
bpe_eng = BPEmb(lang='en',vs=10000,dim=300)


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

def SEED_EVERYTHING(seed_val=42):
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

def accuracy_non_zero(y_true,y_pred):
    y_true_non_zero = y_true[y_true!=805]
    y_pred_non_zero = y_pred[y_true!=805]
    return accuracy_score(y_true_non_zero,y_pred_non_zero)  

def calc_bleu_score(y_true, y_pred):
    # y_true shape (seq_len, batch_size)
    # y_pred shape (seq_len, batch_size)
    batch_size = y_true.shape[1]
    score = 0
    for j in range(batch_size):
        y_true_sample = y_true[:,j]
        y_pred_sample = y_pred[:,j]
        y_true_sample_important = y_true_sample[y_true_sample != 10000]
        y_pred_sample_important = y_pred_sample[y_true_sample != 10000]
        r = tokenizer.word_tokenize(bpe_eng.decode_ids(y_true_sample_important.tolist()))
        h = tokenizer.word_tokenize(bpe_eng.decode_ids(y_pred_sample_important.tolist()))
        score += nltk.translate.bleu_score.sentence_bleu([r],h)
        
    bleu = score/batch_size
    return bleu
    
def evaluate(model, valid_dataloader, device, epoch, key, loss_func): 
  model.eval() 
  loss_meter = AverageMeter()
  acc_meter = AverageMeter()
  pbar = tqdm(valid_dataloader, total=len(valid_dataloader))
  # file = open(f'sampled-texts/{key}/samples_{epoch+1}.txt','w')
  # file.close()
  for batch in pbar:
      b_input = batch[0].to(device)
      b_output = batch[1].to(device)
      
      with torch.no_grad():                  
          outputs = model(b_input, b_output)
          loss = loss_func(outputs.view(-1, 10000), b_output[1:].view(-1))
                                  
      loss_meter.update(loss.item())
      
      pred_labels = outputs.argmax(2).cpu().numpy()
      true_labels = batch[1][1:].numpy()
            
      acc = calc_bleu_score(true_labels, pred_labels)
     
      acc_meter.update(acc)
      if np.random.choice([0,1,2,3,4,5,6,7,8,9,10]) == 0:
           write_txt(true_labels, pred_labels, 
                     epoch=epoch, num_samples=15,
                     key=key)
      pbar.set_postfix({'loss':loss_meter.avg, 'bleu':acc_meter.avg})
  return loss_meter.avg, acc_meter.avg

def evaluate_on_test_set(model, valid_dataloader, device, test_sentences, test_labels, loss_func=nn.CrossEntropyLoss()): 
  model.eval() 
  loss_meter = AverageMeter()
  precision_meter = AverageMeter()
  recall_meter = AverageMeter()
  pbar = tqdm(valid_dataloader, total=len(valid_dataloader))
  all_labels = []
  all_true_labels = []
  probas = []
  for batch in pbar:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_labels = batch
      
      with torch.no_grad():                  
          logits = model(b_input_ids.permute(1,0))
          loss = loss_func(logits, b_labels)  
          
      loss_meter.update(loss.item())      
      pred_labels = logits.argmax(1).cpu().numpy()
      true_labels = b_labels.cpu().numpy()
      all_labels.extend(pred_labels.tolist())
      all_true_labels.extend(true_labels.tolist())
      probas.extend(logits.softmax(1).max(1).values)
      
      precision = precision_score(true_labels,pred_labels)
      recall = recall_score(true_labels, pred_labels)
      
      precision_meter.update(precision)
      recall_meter.update(recall)
      pbar.set_postfix({'loss':loss_meter.avg, 'precision':precision_meter.avg,
                          'recall':recall_meter.avg})
    
#  assert test_labels == all_true_labels
  with open('see_predictions.txt','w') as f:
      txts = []
      for sent, proba, pred, true, true_ in zip(test_sentences, probas, all_labels, test_labels, all_true_labels):
          txts.append(f'{sent} -- {proba} -- {pred} -- {true} -- {true_}')
      f.write('\n'.join(txts))
  return loss_meter.avg, precision_meter.avg, recall_meter.avg

#%%


# def prettify_text(text):
#     return text.replace(' "','').replace('" ','"')
def correct_pred_pos(pred_mat, seq_len):
    desc_wise_pred = pred_mat.argsort(descending=True)
    for pred in desc_wise_pred:
        if pred == 805:
            return pred
        else:
            if pred+1 < seq_len:
                return pred
            
    

def write_txt(y_true, y_pred, epoch=0,
              num_samples=10,pad_id=10000,
              key='test'):  
    
    with open(f'sampled-texts/{key}/samples_{epoch+1}.txt','a+') as f:
        batch_size = y_true.shape[1]
        for j in range(batch_size):
            y_true_sample = y_true[:,j]
            y_pred_sample = y_pred[:,j]
            y_true_sample_important = y_true_sample[y_true_sample != pad_id]
            y_pred_sample_important = y_pred_sample[y_true_sample != pad_id]
            r = bpe_eng.decode_ids(y_true_sample_important.tolist())
            h = bpe_eng.decode_ids(y_pred_sample_important.tolist())

            f.write(' --- '.join([h, r+'\n']))


    