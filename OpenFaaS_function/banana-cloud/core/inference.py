import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import cv2
import pickle
from torchsummary import summary

from PIL import Image

# import sys
# sys.path.append('/home/a.dimarino/Scrivania/inference')

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import torchvision
#from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as TF

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


import sys
#from models.statenet import StateNet
from alexnet import AlexNet
from resnet import resnet18, resnet34, resnet50
from .utility.pad import NewPad, Equalize
#from models.resnet import ResNet18, lr_schedule
#from models.resnet18Cifar10 import resnet18

def load_checkpoints():
    root = '/home/a.dimarino/Scrivania/FruitsEvaluationNet/checkpoints/best/'
    #root = '/home/a.dimarino/Scrivania/SiamMask/experiments/siammask_base/'
    #dir = dataset + '_BestModel'

    #path = os.path.join(root, 'resnet18.pt')

    path = os.path.join(root, 'best_checkpoint_alexnet_256b_256.pth')

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    # state_dict = torch.load(script_dir + '/state_dicts/'+arch+'.pt', map_location=device)
    # state_dict = torch.load(path)
    #net.load_state_dict(checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    print("\nLoading checkpoint complete")
    return checkpoint

net = AlexNet()

load_checkpoints()
net.eval()

train_transform = transforms.Compose([
    NewPad(),
    transforms.Resize((256,256)),
    #transforms.RandomCrop((227,227)),
    #Equalize(),
    transforms.ToTensor()
])

transform_normalize = transforms.Normalize(
    mean=[0.313, 0.300, 0.253],
    std=[0.343, 0.336, 0.315]
 )

#img = Image.open('/home/a.dimarino/Scrivania/FruitsEvaluationNet/tools/dataset/Banana_Dataset/cropped/4/b_25_29.jpg')

img = Image.open('/home/a.dimarino/Scrivania/FruitsEvaluationNet/crop/casco_crop_1.jpg')
#img = Image.open('/home/a.dimarino/Scrivania/FruitsEvaluationNet/crop/catt_ban_crop_0.jpg')
#img = Image.open('/home/a.dimarino/Scrivania/FruitsEvaluationNet/crop/banane_crop_0.jpg')

transformed_img = train_transform(img)
input = transform_normalize(transformed_img)
input = input.unsqueeze(0)


classes = ('underipe', 'barely ripe', 'ripe', 'very ripe', 'overipe', 'rotten')


#summary(net, (3,227,227))
outputs = net(input)

print('output size: ', outputs.size())
print('output: ', outputs.sum())
outputs = F.softmax(outputs, dim=1)
print('output: ', outputs.sum())
_, pred_label_idx = torch.max(outputs, 1)
# pred_label_idx = np.argmax(outputs, axis=1)

#pred_label_idx.squeeze_()
# print('Predicted: ', ' '.join('%s' % classes[predicted]))
print('classe: ', classes[pred_label_idx])
predicted_label = classes[pred_label_idx.item()]
print('predicted:', predicted_label)
# print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

def attribute_image_features(algorithm, input, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=pred_label_idx,
                                              **kwargs
                                             )

    return tensor_attributions


# saliency = Saliency(net)
# grads = saliency.attribute(input, target=pred_label_idx)
# grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
#
# ig = IntegratedGradients(net)
# attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
# attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
#
# # nt = NoiseTunnel(ig)
# # attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
# #                                       n_samples=100, stdevs=0.2)
# # attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
#
# dl = DeepLift(net)
# attr_dl = attribute_image_features(dl, input, baselines=input * 0)
# attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
#
# original_image = img
#
# fig1, _ = viz.visualize_image_attr(None, original_image,
#                       method="original_image", title="Original Image", use_pyplot=False)
# fig1.savefig('originalimage.png')
#
# fig2, _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
#                           show_colorbar=True, title="Overlayed Gradient Magnitudes", use_pyplot=False)
# fig2.savefig('ogm.png')
#
# fig3, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
#                           show_colorbar=True, title="Overlayed Integrated Gradients", use_pyplot=False)
# fig3.savefig('oig.png')
#
# # fig4, _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value",
# #                              outlier_perc=10, show_colorbar=True,
# #                              title="Overlayed Integrated Gradients \n with SmoothGrad Squared", use_pyplot=False)
# # fig4.savefig('oigss.png')
#
# fig5, _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True,
#                           title="Overlayed DeepLift", use_pyplot=False)
# fig5.savefig('od.png')


############### START ##############

# print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

integrated_gradients = IntegratedGradients(net)
attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)


default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

f1, _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)

f1.savefig('default_cmap.png')
print("default_cmap.png salvata")

noise_tunnel = NoiseTunnel(integrated_gradients)

print('blocco')

attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
print('blocco2')
f2, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)

print('blocco?')

f2.savefig('noise_tunnel.png')
print("noise_tunnel.png salvata")

torch.manual_seed(0)
np.random.seed(0)

gradient_shap = GradientShap(net)

#Defining baseline distribution of images
rand_img_dist = torch.cat([input * 0, input * 1])

attributions_gs = gradient_shap.attribute(input,
                                         n_samples=50,
                                         stdevs=0.0001,
                                         baselines=rand_img_dist,
                                         target=pred_label_idx)

f3, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                     ["original_image", "heat_map"],
                                     ["all", "absolute_value"],
                                     cmap=default_cmap,
                                     show_colorbar=True)

f3.savefig('figure3.png')
print("figure3.png salvata")

occlusion = Occlusion(net)

#print('input shape: ', input.shape)
transformed_img = transformed_img.unsqueeze(0)

attributions_occ = occlusion.attribute(input,
                                      strides = (3, 8, 8),
                                      target=pred_label_idx,
                                      sliding_window_shapes=(3,15, 15),
                                      baselines=0)

# print('oc_att shape: ', attributions_occ.shape)
# print('oc_att: ', attributions_occ)

if np.sum(attributions_occ.squeeze().cpu().detach().numpy().flatten()) == 0 :
    tmp = attributions_occ.squeeze().cpu().detach().numpy() + 1e-15
else:
    tmp = attributions_occ.squeeze().cpu().detach().numpy()     

##print('sum: ', np.sum(attributions_occ.squeeze().cpu().detach().numpy(), axis=2) )

f4, _ = viz.visualize_image_attr_multiple(np.transpose(tmp, (1,2,0)),
                                     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                     ["heat_map"],
                                     ["positive"],
                                     show_colorbar=True,
                                     outlier_perc=2,
                                    )

f4.savefig('occlusion.png')
print("occlusion.png salvata")

occlusion = Occlusion(net)

attributions_occ = occlusion.attribute(input,
                                      strides = (3, 50, 50),
                                      target=pred_label_idx,
                                      sliding_window_shapes=(3,60, 60),
                                      baselines=0)

if np.sum(attributions_occ.squeeze().cpu().detach().numpy().flatten()) == 0 :
    tmp = attributions_occ.squeeze().cpu().detach().numpy() + 1e-15
else:
    tmp = attributions_occ.squeeze().cpu().detach().numpy() 

f5, _ = viz.visualize_image_attr_multiple(np.transpose(tmp, (1,2,0)),
                                     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                     ["heat_map"],
                                     ["positive"],
                                     show_colorbar=True,
                                     outlier_perc=2,
                                    )

f5.savefig('occlusion2.png')
print('occlusion2.png salvata')

############ END ###############
