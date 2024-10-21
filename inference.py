from torch import nn, optim
import sys
sys.path.append('../..')
sys.path.append('./')
sys.path.append('.')
from dataset.Datasetvit import PairsDataset
import config
from model import RegressionModel,RegressionModel_Transformer
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import os
from utils import train_epoch, validation
from config import *
from tqdm import tqdm
import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig
import cv2 
import torchvision
from torchvision import models
resnet = models.resnet50(pretrained=True)


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermediate targetted layers. """
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda
  
	def get_gradients(self):
		return self.feature_extractor.gradients
    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        #fc = nn.Linear(in_features=2048, out_features= 6, bias=True)
        fc = nn.Linear(1254, 2)
        if self.cuda:
            output = output.cpu()
            output = fc(resnet(output)).cuda()
        else:
            output = fc(resnet(output))
        return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img
	input.requires_grad = True
	return input

def show_cam_on_image(img, mask,name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("/work/emmanuelle/2D_3D_Regressor/cam_{}.jpg".format(name), np.uint8(255 * cam))
 
 
class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam


#-------------Performance Metrics----------#

def mdrae(y_pred, true):
    return np.median(np.abs(true - y_pred))

def mean_bias_error(true, pred):
    bias_error = true - pred
    mbe_loss = np.mean(np.sum(bias_error) / true.size)
    return mbe_loss

def relative_absolute_error(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.sum(np.abs(true - pred))
    squared_error_den = np.sum(np.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss

def median_relative_absolute_error(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.abs(true - pred)
    squared_error_den = np.abs(true - true_mean)
    rae_loss = np.median(squared_error_num / squared_error_den)
    return rae_loss

# Instantiate training, validation, and test sets
train_set = PairsDataset(set_type='train',root = '/work/amirj/UKBiobank/')
val_set = PairsDataset(set_type='val',root = '/work/amirj/UKBiobank/')
test_set = PairsDataset(set_type='test',root = '/work/amirj/UKBiobank/')

# Define dataloader
train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True)
valid_loader = DataLoader(val_set, batch_size=config.valid_batch_size)
test_loader = DataLoader(test_set, batch_size=2)

PATH = "./checkpoints/all/epoch:{epoch}-loss_valid-points:{best_loss:.3}.pt"

model = RegressionModel_Transformer(input_dim=3, output_nodes=209*6, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
model.load_state_dict(torch.load(PATH))

for i,inp in enumerate(tqdm(test_loader)):
            
        inputs = inp['dxa_img'].to(device)
        targets = inp['coord']
        targets = targets.view(targets.size(0), -1)
        targets = targets.to(device)
    
        model.to(device)
        model.eval()
    
        """
        #visualise gradient activation maps
        grad_cam = GradCam(model , target_layer_names = ["layer4"], use_cuda=True)
        for img in inputs:
            img = np.float32(cv2.resize(np.uint8(img.detach().cpu().numpy()), (224, 224))) 
            input = preprocess_image(img)
            input.required_grad = True
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested index.
            target_index =None
            mask = grad_cam(inputs, target_index)
            show_cam_on_image(img, mask,i)
        """
        key_points = model(inputs).detach().cpu()
        
        img = inputs.cpu().numpy()
        targets = targets.reshape(-1,6).cpu()
        key_points = key_points.reshape(-1,6)

        key_points_coronal_left = key_points[:209,0]*img.shape[2]   
        key_points_coronal_middle = key_points[:209,1]*img.shape[2]                                  
        key_points_coronal_right = key_points[:209,2]*img.shape[2]  
        
        key_points_sagittal_left = key_points[209:209*2,3]*img.shape[2]   
        key_points_sagittal_middle = key_points[209:209*2,4]*img.shape[2]                                  
        key_points_sagittal_right = key_points[209:209*2,5]*img.shape[2]                                 
          
        targets_x = targets[209*i:209*(i+1),0]*img.shape[2]                                  
        targets_y = targets[209*(i+1):209*(i+2),1]*img.shape[3] 
        
        targets_coronal_left = targets[:209,0]*img.shape[2]   
        targets_coronal_middle = targets[:209,1]*img.shape[2]                                  
        targets_coronal_right = targets[:209,2]*img.shape[2]  
        
        targets_sagittal_left = targets[209:209*2,3]*img.shape[2]   
        targets_sagittal_middle = targets[209:209*2,4]*img.shape[2]                                  
        targets_sagittal_right = targets[209:209*2,5]*img.shape[2]    
        
        y = np.linspace(0,209,209)
    
        
        fig, axs = plt.subplots(figsize=(15,10),ncols=5, nrows=1)
        
    
        mse = mean_squared_error(targets_x, key_points_x)# take square root of mse
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_x, key_points_x)
        print(f'RMSE:{rmse:.3}')
        print("MAE",mean_absolute_error(targets_x, key_points_x))
        print("MAE Median",median_absolute_error(targets_x, key_points_x))
        print("Mean Bias Error",mean_bias_error(targets_x, key_points_x))

        print("Relative Abs Error",relative_absolute_error(targets_x, key_points_x))
        print("Relative Median Error",median_relative_absolute_error(key_points_x,targets_x))

        print("SAD:", np.sum(np.abs(targets_x - key_points_x)))
        print("SSD:", np.sum(np.square(targets_x - key_points_x)))
        print("correlation:", np.corrcoef(np.array((targets_x, key_points_x)))[0, 1])  
    
        axs[0].imshow(img[i,0,:,:],cmap='gray')
        axs[0].set_title("DXA")

        axs[1].set_title("DXA Pred Cor")
        axs[1].imshow(img[i,0,:,:],cmap='gray')
        axs[1].scatter(key_points_coronal_left, y,marker='.',c='blue',s=2)
        axs[1].scatter(key_points_coronal_middle, y,marker='.',c='red',s=2)
        axs[1].scatter(key_points_coronal_right, y,marker='.',c='magenta',s=2)

        axs[2].set_title("DXA GT Cor")
        axs[2].imshow(img[i,0,:,:],cmap='gray')
        axs[2].scatter(targets_coronal_left, y,marker='.',c='cyan',s=2)
        axs[2].scatter(targets_coronal_middle, y,marker='.',c='yellow',s=2)
        axs[2].scatter(targets_coronal_right, y,marker='.',c='brown',s=2)

        axs[3].set_title("DXA Pred Sag")
        axs[3].scatter(key_points_sagittal_left, y,marker='.',c='blue',s=2)
        axs[3].scatter(key_points_sagittal_middle, y,marker='.',c='red',s=2)
        axs[3].scatter(key_points_sagittal_right, y,marker='.',c='magenta',s=2)
        
        axs[4].set_title("DXA GT Sag")
        axs[4].scatter(targets_sagittal_left, y,marker='.',c='cyan',s=2)
        axs[4].scatter(targets_sagittal_middle, y,marker='.',c='yellow',s=2)
        axs[4].scatter(targets_sagittal_right, y,marker='.',c='brown',s=2)
    
        plt.show()
        os.makedirs("./result/",exist_ok=True)
        plt.savefig("./result/" + str(inp['dxa_filename'][0].split('/')[-1]) + '.png',dpi=1000)
        plt.close()
