# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:33:50 2019

@author: tungo
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim
from torchvision import transforms


def compute_content_loss(a_C, a_G):
    """
    Compute content loss of 2 tensors.
    
    Inputs:
    - a_C: features tensor for content image, with shape (N x C x H x W)
    - a_G: features tensor for generated image, with shape (N x C x H x W)
    
    Return:
    - tensor of content loss
    """
    n, c, h, w = a_C.shape
    return torch.sum((a_C - a_G)**2)/(4*n*c*h*w)


def gram_matrix(A):
    """
    Get Gram matrix representation of tensor.
    
    Input:
    - A: tensor of shape (N x C x H x W)
    
    Return:
    - tensor of shape ((NxC) x (NxC))
    """
    n, c, h, w = A.size()
    A = A.view(n*c, h*w)
    return torch.mm(A, A.t())

def style_loss_one_layer(a_S, a_G):
    """
    Compute style loss of 2 tensors.
    
    Inputs:
    - a_S: tensor of style features, with shape (N x C x H x W)
    - a_G: tensor of generated features, with shape (N x C x H x W)
    
    Return:
    - tensor of style loss
    """
    n, h, w, c = a_S.size()
    g_a_S = gram_matrix(a_S)
    g_a_G = gram_matrix(a_G)
    J = torch.sum((g_a_S - g_a_G)**2) / ((2*h*w*c)**2)
    return J

def load_image(img_name):
    """
    Get tensor of input image.
    
    Input:
    - image_name: full path string of image
    
    Return:
    - img: tensor of image
    """
    img = Image.open(img_name)
    transform = transforms.Compose([
        transforms.Resize((400, 600)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0)
    return img

def get_features(img, model, layers):
    """
    Get features of every model's layer.
    
    Input:
    - img: tensor of image
    - model: a VGG model
    - layers: a dictionary mapping model's layer names with our custom layer names
      For example:{
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
        }
      
    Return:
    - features: dictionary mapping custom layer names with tensor of features
    """
    x = img
    features = {}
    for name, layer in enumerate(model.features):
        x = layer(x)
        if str(name) in layers.keys():
            features[layers[str(name)]] = x
    return features

def train(model, content_img, style_img, generated_img, layers, style_layers, alpha=10, beta=40, learning_rate=0.01, epochs=5, device='cpu', print_every=10):
    """
    Training model and generate output image.
    
    Inputs:
    - model: VGG model
    - content_img: tensor of content image
    - style_img: tensor of content image
    - generated_img: tensor of un-trained generated image
    - layers: dictionary of layers we want to use
    - style_layers: dictionary of style layers' weights
    - alpha, beta, learning_rate, epochs: paramters
    - device: 'cpu' or 'cuda'. Default: 'cpu'
    - print_every: number of iterations to print the loss value
    
    Return:
    - generated_img: tensor of generated image
    """
    model = model.to(device)
    content_img = content_img.to(device)
    style_img = style_img.to(device)
    generated_img = generated_img.to(device)
    optimizer = optim.Adam([generated_img.requires_grad_()], lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        content_features = get_features(content_img, model, layers)
        generated_features = get_features(generated_img, model, layers)
        style_features = get_features(style_img, model, layers)
        content_loss = compute_content_loss(content_features['conv4_2'], generated_features['conv4_2'])
        style_loss = 0
        for layer in style_layers:
            a_S = style_features[layer]
            a_G = generated_features[layer]
            style_loss += style_layers[layer] * style_loss_one_layer(a_S, a_G)
        loss = alpha * content_loss + beta * style_loss
        loss.backward()
        optimizer.step()
        if epoch % print_every == 0:
            print('epoch: %d, total loss = %f, content loss = %f, style loss = %f' %(epoch, loss.item(), content_loss.item(), style_loss.item()))
    return generated_img

def imshow(tensor, title=None):
    """
    Show image from given tensor
    
    Inputs:
    - tensor: a tensor of PIL image
    - title: Custom title. Default: None
    """
    image = tensor.cpu().clone().detach().numpy().squeeze()
    image = image.transpose((1, 2, 0))
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    plt.imshow(image)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    
def imsave(tensor, name):
    """
    Save image.
    
    Inputs:
    - tensor: tensor of PIL image
    - name: target name
    """
    image = tensor.cpu().clone().detach().numpy().squeeze()
    image = image.transpose((1, 2, 0))
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    plt.imsave(name, image)