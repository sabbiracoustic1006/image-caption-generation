# image-caption-generation
This repository contains code written using PyTorch library to train and infer from an image caption model. The model has two parts:
* Encoder
* Decoder

The Encoder is a Convolutional Neural Network to create a feature representation of the image and the Decoder is a LSTM that can be configured to decode the representation of the image to a sentence.

The steps for training the model is summarized in keypoints below:
1. Download COCO dataset
2. Prepare the dataset and the dataloader
3. Prepare the Model, optimizer, loss function and the learning rate scheduler
4. Write code for training loop
5. Save Model for the best evaluation results
6. Make inference using the model to see output

Now I will elaborate each step to train the model
## 1) Download COCO dataset
We can download the dataset using python with the code given below:
```markdown
import os
import tensorflow as tf

# Download caption annotation files
annotation_folder = './data/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                       cache_subdir=os.path.abspath('.'),
                                       origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                       extract = True)
  annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
  os.remove(annotation_zip)

# Download image files
image_folder = './data/val2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
  image_zip = tf.keras.utils.get_file('val2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/val2014.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip) + image_folder
  os.remove(image_zip)
else:
  PATH = os.path.abspath('.') + image_folder
```
We can also download the zip files directly from this [link](https://cocodataset.org/#download)

## 2) Prepare the dataset and the dataloader
The image and the captions for the corresponding image needs to be read from the disk and multiple samples has to passed to the model in a batch for efficient and effective training. PyTorch has given us Dataset and DataLoader class which can be used very easily to create our custom dataloader to train models even faster.

We can inspect random images with target captions using the code given below.
```markdown        
import json, cv2
from matplotlib import pyplot as plt

# function to see random image
def see_random_img(annot, data_root):
  annotations = annot['annotations']
  annotation = random.sample(annotations,1)[0]
  img_id = annotation['image_id']
  img_path = f'{data_root}/COCO_train2014_{img_id:012d}.jpg'
  img = cv2.imread(img_path)[:,:,[2,1,0]]
  plt.figure(figsize=(15,15))
  plt.imshow(img)
  plt.title(annotation['caption'])

# load the training json caption file
with open('/media/ratul/mydrive/image-caption-data/annotations/captions_train2014.json','r') as f:
  content = json.load(f)

# give the data root path
data_root = '/media/ratul/mydrive/image-caption-data/train2014/train2014'   
see_random_img(content, data_root)
```

The above can be found in the [dataset.py](https://github.com/sabbiracoustic1006/image-caption-generation/blob/main/dataset.py) file

We need to create a custom dataset by subclassing the Dataset class that can be imported from torch.utils.data
The code for the custom dataset class is given below which can be found in [dataset.py](https://github.com/sabbiracoustic1006/image-caption-generation/blob/main/dataset.py) file. We have to pass the paths of the images, a dictionary containing paths as keys and the target caption as values, and a torchvision transform in the constructor of the CaptionDataset class. We also need a tokenizer to tokenize the target sentence and convert it to a numerical sequence. For the tokenizer, vocabulary ids and embedding we use pretrained byte pair encoding embedding that can be found at this [link](https://nlp.h-its.org/bpemb/). For my solution, I have used a vocab size of 10000 with embedding dimension 300.

```markdown
# custom dataset for the dataloader
class CaptionDataset(Dataset):
    def __init__(self, paths, caption_dict, transform):
        self.paths = paths
        self.caption_dict = caption_dict
        self.bpe = BPEmb(lang="en", vs=10000, dim=300)
        self.transform = transform
    
    # dunder method for getting length of the instances of the class
    def __len__(self):
        return len(self.paths)
    
    # dunder method for getting samples corresponding to the index
    def __getitem__(self, idx):
        # load path
        path = self.paths[idx]
        # load image using PIL
        img = Image.open(path).convert('RGB')
        # load the caption
        caption = self.caption_dict[path]
        # convert the pil image using torchvision transform
        img_tensor = self.transform(img)
        # the target caption is tokenized and converted to target vocab ids with SOS and EOS
        caption_label = torch.tensor(self.bpe.encode_ids_with_bos_eos(caption))
        return img_tensor, caption_label

# function for padding the variable target caption sequence to max length sequence of a batch
def collate_fn(batch):
    seq = torch.cat([el[0].unsqueeze(0) for el in batch if not isinstance(el[0],int)]).float()
    label = pad_sequence([el[1] for el in batch if not isinstance(el[0],int)],padding_value=10000).long()
    return seq,label
    
# torchvision transforms for training and validation. Consists of resize, random horizontal flip with probability of 0.5, random rotation, random crop of 224x224
# training images are lightly augmented but the validation images or test images are not augmented at all
# the image net stats are used for normalizing the image tensor since, pretrained cnn is used as encoder
def get_transforms():
    train_transform = T.Compose([T.Resize((256,256)), T.RandomHorizontalFlip(0.5), 
                                 T.RandomAffine((5,30)), T.RandomCrop(224), 
                                 T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])    
    test_transform = T.Compose([T.Resize((224,224)), 
                                T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
    return train_transform, test_transform

# prepare the dataset instances of CaptionDataset class for training and validation
train_transform, valid_transform = get_transforms()
train_ds = CaptionDataset(train_paths, caption_dict, train_transform)
valid_ds = CaptionDataset(valid_paths, caption_dict, valid_transform)

# prepare the dataloader instances for training and validation
train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
valid_dataloader = DataLoader(valid_ds, shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
```

## 3) Prepare the Model, optimizer, loss function and the learning rate scheduler

I used CNN encoder with LSTM decoder. I used a pretrained resnet18 cnn model as my encoder and created two states for initial hidden and cell states of the lstm. The code for model encoder is given below.

```markdown
# CNN encoder
class Encoder(nn.Module):
    def __init__(self, cnn='resnet18', num_features=512, hidden_size=128):
        super().__init__()
        cnn_model = getattr(torchvision.models, cnn)(pretrained=True)
        feat_ext_layers = list(cnn_model.children())[:-1]
        self.encoder = nn.Sequential(*feat_ext_layers)
        self.hidden_state = nn.Linear(num_features, hidden_size)
        self.cell_state = nn.Linear(num_features, hidden_size)
    
    def forward(self, x):
        out = self.encoder(x).flatten(1)
        hidden_state = self.hidden_state(out).unsqueeze(0)
        cell_state = self.cell_state(out).unsqueeze(0)
        return hidden_state, cell_state
```

The decoder is a configurable LSTM decoder where multilayer LSTM can be used with hidden size and desired vocabulary size. For reducing hassle, I used pretrained subword embedding that can be found at this [link](https://nlp.h-its.org/bpemb/). The code for the decoder is given below.

```markdown
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
    
```
