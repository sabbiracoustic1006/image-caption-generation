#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:45:05 2020

@author: ratul
"""

import torch, argparse, os, json
from torch import nn
from dataset import get_dataloaders
from models import get_model 
from utils import SEED_EVERYTHING, AverageMeter, evaluate, calc_bleu_score
from tqdm import tqdm

def train(args):
    global img, tgt_caption
    SEED_EVERYTHING()
    batch_size = args.batch_size
    epochs = args.epochs
    device = torch.device(args.device)
    train_dataloader, valid_dataloader = get_dataloaders(batch_size)
    
    with open('config.json','r') as f:
        model_config = json.load(f)
    
    model = get_model(**model_config) #Seq2SeqModel(dropout_p=0.25, hidden_size=256,num_layers=1)
    model.to(device)
#    for param in model.encoder.parameters():
#        param.requires_grad = False
    print(model)
    print(model.decoder.embedding.weight.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=2, verbose=True, min_lr=1e-6, mode='max')
    
    if args.resume_from is not None:
        state = torch.load(args.resume_from)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
    
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    loss_func = nn.CrossEntropyLoss(ignore_index=args.padding_idx)
    best_bleu = 0
    for epoch_i in range(epochs):         
        loss_meter = AverageMeter()
        bleu_meter = AverageMeter()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        model.train() 
        for step, batch in enumerate(pbar):
            img = batch[0].to(device)
            tgt_caption = batch[1].to(device)
            
            optimizer.zero_grad()      
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(img, tgt_caption)
                    loss = loss_func(outputs.view(-1, args.padding_idx), tgt_caption[1:].view(-1))
                scaler.scale(loss).backward() 
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)      
                scaler.update()
            else:
                outputs = model(img, tgt_caption)
                loss = loss_func(outputs.view(-1, args.padding_idx), tgt_caption[1:].view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            pred_captions = outputs.argmax(2).cpu().numpy()
            true_captions = batch[1][1:].numpy()
            
            bleu = calc_bleu_score(true_captions, pred_captions)
           
            loss_meter.update(loss.item())
            bleu_meter.update(bleu)
            
            pbar.set_postfix({'loss':loss_meter.avg, 'bleu':bleu_meter.avg})
          
        valid_loss, valid_bleu = evaluate(model, valid_dataloader, device, epoch_i, args.key, loss_func)
        scheduler.step(valid_bleu)    
       
        if valid_bleu > best_bleu:
          print('validation bleu improved from %.4f to %.4f'%(best_bleu,valid_loss))
          print('saving model...')
          torch.save({'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict(),
                      'scheduler_state_dict':scheduler.state_dict()}, f'saved_models/{args.key}/state.pth')
          
          best_bleu = valid_bleu
    
        print(f'Epoch: {epoch_i+1}/{epochs}, train loss:{loss_meter.avg:.4f}, train bleu:{bleu_meter.avg:.4f}\nvalid loss: {valid_loss:.4f}, valid bleu: {valid_bleu:.4f}')
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', default='test_run_again', type=str, help='name of experiment')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs for training')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size used for training')
    parser.add_argument('--resume_from', default='saved_models/test_run/state.pth', help='resume from this ckpt')
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument('--padding_idx', default=10000, type=int, help='device to use for training')
    parser.add_argument('--attention', default=False, action='store_true', help='flag to indicate whether to use model with attention')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='flag to indicate whether to use mixed precision')
    args = parser.parse_args()
    os.makedirs(f'saved_models/{args.key}',exist_ok=True)
    os.makedirs(f'sampled-texts/{args.key}',exist_ok=True)
    train(args)
    
