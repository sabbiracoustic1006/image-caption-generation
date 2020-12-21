#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:51:09 2020

@author: ratul
"""


import json, random, cv2, torch
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt
from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torchvision import transforms as T

def get_transforms():
    train_transform = T.Compose([T.Resize((256,256)), T.RandomHorizontalFlip(0.5), 
                                 T.RandomAffine((5,30)), T.RandomCrop(224), 
                                 T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])    
    test_transform = T.Compose([T.Resize((224,224)), 
                                T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
    return train_transform, test_transform


def see_random_img(annot, data_root):
    annotations = annot['annotations']
    annotation = random.sample(annotations,1)[0]
    img_id = annotation['image_id']
    img_path = f'{data_root}/COCO_train2014_{img_id:012d}.jpg'
    img = cv2.imread(img_path)[:,:,[2,1,0]]
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.title(annotation['caption'])
    
def get_annotation_dict(annot):
    annotations = annot['annotations']
    dict_ = {}
    for annotation in annotations:
        img_id = annotation['image_id']
        dict_[f'/media/ratul/mydrive/image-caption-data/train2014/train2014/COCO_train2014_{img_id:012d}.jpg'] = annotation['caption'].strip().replace('\n','')
    return dict_

class CaptionDataset(Dataset):
    def __init__(self, paths, caption_dict, transform):
        self.paths = paths
        self.caption_dict = caption_dict
        self.bpe = BPEmb(lang="en", vs=10000, dim=300)
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            caption = self.caption_dict[path]
            img_tensor = self.transform(img)
            caption_label = torch.tensor(self.bpe.encode_ids_with_bos_eos(caption))
        except:
            img_tensor = -1
            caption_label = -1
        return img_tensor, caption_label
    
def collate_fn(batch):
    seq = torch.cat([el[0].unsqueeze(0) for el in batch if not isinstance(el[0],int)]).float()
    label = pad_sequence([el[1] for el in batch if not isinstance(el[0],int)],padding_value=10000).long()
    return seq,label

def get_dataloaders(batch_size=8):
    paths = glob('/media/ratul/mydrive/image-caption-data/train2014/train2014/*.jpg')
    with open('/media/ratul/mydrive/image-caption-data/annotations/captions_train2014.json','r') as f:
        content = json.load(f)
    caption_dict = get_annotation_dict(content) 
    train_paths, valid_paths = train_test_split(paths, random_state=0, test_size=0.1)
   
    train_transform, valid_transform = get_transforms()
    train_ds = CaptionDataset(train_paths, caption_dict, train_transform)
    valid_ds = CaptionDataset(valid_paths, caption_dict, valid_transform)
    
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    valid_dataloader = DataLoader(valid_ds, shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    data_root = '/media/ratul/mydrive/image-caption-data/train2014/train2014'
    paths = glob(f'{data_root}/*.jpg')
    with open('/media/ratul/mydrive/image-caption-data/annotations/captions_train2014.json','r') as f:
        content = json.load(f)
    print(len(paths))
    see_random_img(content, data_root)