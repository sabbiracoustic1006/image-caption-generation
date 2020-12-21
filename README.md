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

