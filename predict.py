import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import numpy as np
import PIL
from PIL import Image
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt
import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
import shutil, argparse
import json

parser = argparse.ArgumentParser(description='Predicting the flower class using the previous trained model')
parser.add_argument('--img', default='flowers/test/15/image_06369.jpg', help='Image path')
parser.add_argument('--checkpoint', default='checkpoint.pth', help='Model\'s Checkpoint')
parser.add_argument('--top_k', action='store', dest='topk', default=5, help='Top 5 prediction')

# Storing inputs
args = parser.parse_args()

#Loading the model
def load_model(filepath):
    checkpoint = torch.load(args.checkpoint)
    model = models.vgg11(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer_dict']
    criterion = checkpoint['criterion']
    
    return model

model = load_model('args.checkpoint')

#Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
def process_image(image):
    img = Image.open(image)
    
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    processed_img = img_transforms(img)
    
    return processed_img

# Function for showing the image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Predicting the class (or classes) of an image using a trained deep learning model.
def predict(image_path, model, topk=5):
    
    model.to('cpu')
    
    with torch.no_grad():
        img = process_image(image_path)
        
        img = img.type(torch.FloatTensor)
        
        img_tensor = img.unsqueeze_(0)
        
        model.eval()
        
        outputs = model.forward(img_tensor)
        
        predictions = torch.exp(outputs)
        
        top_ps, top_labels = predictions.topk(topk, dim=1)
        
        top_ps = top_ps.numpy()[0]
        
        top_labels = top_labels.numpy()[0]
        
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_labels = [idx_to_class[i] for i in top_labels]
    
    return top_ps, top_labels

#Opening the JSON file with categories
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Sanity checking: Displaying an image along with the top 5 classes
probs, classes = predict(args.img, model, 5)

flower_names = [cat_to_name[str(c)] for c in classes]

    # Set up plot
plt.figure(figsize = (6,10))
ax = plt.subplot(2,1,1)

    # Set up title
flower_num = args.img.split('/')[2]
title_ = cat_to_name[flower_num]

    # Plot flower
img = process_image(args.img)
imshow(img, ax, title = title_);

    # Plot probabilities
ax = plt.subplot(2,1,2)
ax.barh(np.arange(len(flower_names)), probs, align='center')
ax.set_yticks(np.arange(len(flower_names)))
ax.set_yticklabels(flower_names)
ax.invert_yaxis()
ax.set_xlabel('Probability')

plt.savefig('prediction2.png')
