import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image


def model():
  vgg = models.vgg19(pretrained=True).features
  for param in vgg.parameters():
      param.requires_grad_(False)
 
  return vgg

def load_image(img_path, max_size=400, shape=None):
    
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

def im_convert(tensor):
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image



def get_features(image, model, layers=None):
    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  
                  '28': 'conv5_1'}
   
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    
    _,depth,height, width = tensor.size()
    tensor = tensor.view(depth, height * width)
    
    gram = torch.mm(tensor, tensor.t())
    
    return gram

def stylize(content,style,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content = content.to(device)
    style = style.to(device)
    vgg = model.to(device)
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)
    
    style_weights = {'conv1_1': 0.6,
                 'conv2_1': 0.5,
                 'conv3_1': 0.5,
                 'conv4_1': 0.8,
                 'conv5_1': 0.8}

    content_weight = 1  # alpha
    style_weight = 1e5  # beta
 
    optimizer = optim.Adam([target], lr=0.003)
    steps = 1  

    for ii in range(1, steps+1):

        target_features = get_features(target, vgg)

        content_loss =torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        style_loss = 0

        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape

            style_gram = style_grams[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)


        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return target







