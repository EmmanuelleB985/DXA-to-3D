import sys, os, glob
sys.path.append('../..')
sys.path.append('./')
sys.path.append('.')
from genericpath import isfile
from os.path import dirname,abspath,join, basename, isfile
from numpy.lib.npyio import savez_compressed
import torch
import pickle
import random
import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
import cv2
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt 
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig

# Contrast normalize UKBB bone scans
def ukbb_bone_normalize(curr_dxa, beta = 1, sigma = -3, p = 9.5):
    un = np.unique(curr_dxa[1:-20,:])
    curr_dxa_values = curr_dxa.ravel().copy()
    curr_dxa_values = np.delete(curr_dxa_values,np.argwhere(curr_dxa_values < un[1]))
    curr_dxa[curr_dxa < np.percentile(curr_dxa_values,p)] = np.percentile(curr_dxa_values,p)
    curr_dxa = pymat2gray(curr_dxa)
    curr_dxa = (1.0 / (1.0 + np.exp(-beta*(curr_dxa - 0.5) + sigma)))
    curr_dxa = pymat2gray(curr_dxa)
    return curr_dxa
def py2round(x):
    if x >= 0.0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)

def pymat2gray(x):
    lower_limit = np.min(x)
    upper_limit = np.max(x)

    if upper_limit==lower_limit:
        y = np.double(x)
    else:
        delta = 1 / (upper_limit -  lower_limit)
        y = (x-lower_limit) * delta
    return y


def ukbb_bone_reshape(curr_dxa, body_mask):
    target_height = 832
    target_width = 320

    x_remove = np.where(np.sum(body_mask,axis=0) == 0)
    y_remove = np.where(np.sum(body_mask,axis=1) == 0)
    curr_dxa = np.delete(curr_dxa, x_remove, axis=1)
    curr_dxa = np.delete(curr_dxa, y_remove, axis=0)
    new_width = int(py2round(target_height/curr_dxa.shape[0]*curr_dxa.shape[1]))
    int_height = curr_dxa.shape[0]
    int_width = curr_dxa.shape[1]
    curr_dxa = cv2.resize(curr_dxa, (new_width,target_height), interpolation=cv2.INTER_CUBIC)

    # Make sure everything is between 0 and 1, abd no <0 for seg
    curr_dxa[curr_dxa < 0] = 0
    curr_dxa[curr_dxa > 1] = 1

    # Add/Remove Left/Right
    add_or_remove_ind = list([0,0])
    add_or_remove_flag = 'none'
    if curr_dxa.shape[1] < target_width:
        to_add = target_width - curr_dxa.shape[1]
        add_left = py2round(to_add/2.0)
        add_right = int(to_add - add_left)
        add_ind = list(np.zeros(add_left,int))
        if add_right > 0:
            add_ind.extend(list(np.ones(add_right,int) * curr_dxa.shape[1]))
        curr_dxa =  np.insert(curr_dxa, add_ind, 0, axis=1)
        
        add_or_remove_ind = add_ind
        add_or_remove_flag = 'add'
    elif curr_dxa.shape[1] > target_width:
        to_remove = curr_dxa.shape[1] - target_width
        remove_left = py2round(to_remove/2.0)
        remove_right = int(to_remove - remove_left)
        remove_ind = list(range(remove_left))
        if remove_right > 0:
            remove_ind.extend(sorted(list(range(curr_dxa.shape[1]-1,curr_dxa.shape[1]-1-remove_right,-1))))

        add_or_remove_ind = remove_ind
        add_or_remove_flag = 'remove'

    # x_remove, y_remove
    x_remove = np.squeeze(x_remove)
    if x_remove.size == 0:
        x_remove = ''
    elif x_remove.size == 1:
        x_remove = str(x_remove)
    else:
        x_remove = ([str(i) for i in list(np.squeeze(x_remove))])
    y_remove = np.squeeze(y_remove)
    if y_remove.size == 0:
        y_remove = ''
    elif y_remove.size == 1:
        y_remove = str(y_remove)
    else:
        y_remove = ([str(i) for i in list(np.squeeze(y_remove))])
    return curr_dxa, add_or_remove_ind, add_or_remove_flag, int_height, int_width, x_remove, y_remove

def get_seqs(scan_obj, seq_names):
    out_img = []
    if isinstance(seq_names,str): seq_names = list(seq_names)
    for seq_name in seq_names:
        out_img.append(scan_obj[seq_name])
    return torch.cat(out_img,dim=1)

def resample_scans(scan_obj,sequences, resolution=2,transpose=False):
    scaling_factors = np.array(scan_obj['pixel_spacing'])/resolution

    for seq_name in sequences:
        if seq_name == 'pixel_spacing': scan_obj['pixel_spacing'] = [resolution]*2 
        else:
            scan  = torch.Tensor(scan_obj[seq_name])[None,None]
            if seq_name == 'bone' or seq_name == "tissue": 
                    scan_obj[seq_name] = F.interpolate(scan,
                                    scale_factor=list(np.repeat(scaling_factors,2)),
                                    recompute_scale_factor=False,
                                    align_corners=False,
                                    mode='bicubic')
            else:    
                scan_obj[seq_name] = F.interpolate(scan,
                                                scale_factor=list(scaling_factors),
                                                recompute_scale_factor=False,
                                                align_corners=False,
                                                mode='bicubic')
            
            if transpose:
                scan_obj[seq_name] = scan_obj[seq_name].permute(0,1,3,2)
    return scan_obj

def pad_to_size(scan_img : torch.Tensor, output_shape : tuple):
    ''' Pads or crops image to a given size'''
    if (scan_img.shape[1] != output_shape[1]) or (scan_img[2] != output_shape[2]):
        diff = (output_shape[1] - scan_img.shape[1], output_shape[2] - scan_img.shape[2])
        scan_img = F.pad(scan_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])
    return scan_img

def normalise_channels(scan_img : torch.Tensor, eps : float = 1e-5):
    scan_min = scan_img.flatten(start_dim=-2).min(dim=-1)[0][:,None,None]
    scan_max = scan_img.flatten(start_dim=-2).max(dim=-1)[0][:,None,None]
    return (scan_img-scan_min)/(scan_max-scan_min + eps)

def find_mid_cor_slice(mri):
    cor_slice_intensities = mri['fat_scan'].sum(axis=0).sum(axis=-1)
    mid_corr_slice = np.abs(np.cumsum(cor_slice_intensities)/cor_slice_intensities.sum() - 0.5).argmin()
    import pdb; pdb.set_trace()
    return mid_corr_slice


transform = T.Compose([T.ToTensor()])
res_transform = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')


class PairsDataset(Dataset):
    def __init__(self,
                 root : str,
                 set_type : str,
                 augment : bool = False,
                 mri_seqs : list = ['F','W'],
                 dxa_seqs : list = ['bone','tissue'],
                 skip_failed_samples : bool = False,
                 pad_scans = True):
        super().__init__()
        assert set_type in ['train', 'val', 'test', 'all']
        self.set_type = set_type
        self.mri_seqs = mri_seqs
        self.dxa_seqs = dxa_seqs
        self.pad_scans = pad_scans
        
        csv_path = './2D_3D_Regressor/dataset/' + set_type + '.csv'
        self.mri_root = './UKBiobank/Dixon-Mri-v2-stitched/' # path to MRI data 
        df = pd.read_csv(csv_path)
        self.data = df
        self.dxa_root  = './ukbb48/dxa-v2/bone/' # path to DXA data
        mri_list = []
        dxa_list = [] 
        
        for i in range(len(df['mri_filename'])):
            try:
                row = df.iloc[i]
                mri_filename = row['mri_filename']
                dxa_filename = row['dxa_filename']
                if mri_filename != '-' and dxa_filename != '-':
                    try: 
                        mri_list.append(mri_filename)
                        dxa_list.append(dxa_filename)
                    except: 
                        pass
            except:
                pass
        
        self.pairs =list(map(list,zip(dxa_list,mri_list)))
        self.augment = augment
        self.skip_failed_samples = skip_failed_samples
     

    def __len__(self):
        return len(self.pairs) 

    def __getitem__(self, idx):
        
     
        dxa_filename = self.dxa[idx].replace('.npy','')
        
        dxa_fp = self.dxa_root + '/' + dxa_filename + '.mat'
    
        
        row = self.data.iloc[idx].values
        
        DXA_points_sag = np.load("./xy_sagittal/" + dxa_filename + '.npy',allow_pickle=True)
        DXA_points_cor = np.load("./xy_coronal/" + dxa_filename + '.npy',allow_pickle=True)

        curr_name = self.dxa_root + dxa_filename + '.mat'
        
        curr_dxa = sio.loadmat(curr_name)['scan'].astype(np.float32)
        body_mask = sio.loadmat(curr_name.replace('bone','mask'))['scan'].astype(np.float32)

        
        # Contrast normalize UKBB bone scans
        curr_dxa = ukbb_bone_normalize(curr_dxa)


        # Reshape dxa to fit model (627 x 276 -> 830 x 276) -> 832 x 320 vs 416 x 128 
        curr_dxa, add_or_remove_ind, add_or_remove_flag, int_height, int_width, x_remove, y_remove = ukbb_bone_reshape(curr_dxa, body_mask)

        
        dxa_scan = {"bone":curr_dxa,"pixel_spacing":2.23} 
        
        
        sequences_dxa = ['pixel_spacing','bone','tissue']
        dxa_scan = resample_scans(dxa_scan,sequences = sequences_dxa, transpose=False)
        dxa_scan = normalise_channels(dxa_scan['bone'])
        dxa_scan = torch.squeeze(dxa_scan,dim=0)
        
        
        dxa = dxa_scan
        dxa_img = curr_dxa # selecting the bone scan
        
        #Select cropping window 224x224
        
        ymin = int(min(DXA_points_cor[:,1]))
        ymax = int(max(DXA_points_cor[:,1]))
        
        xmin = int(min(DXA_points_cor[:,0]))
        xmax = int(max(DXA_points_cor[:,0])) 
        
        y_cst = 224 - (ymax - ymin)    
        x_cst = 224 - (xmax - xmin)         
        
        im = dxa_img[int(ymin-y_cst/2):int(ymax+y_cst/2), int(xmin-x_cst/2):int(xmax+x_cst/2)]
        
    
        DXA_points_cor[:,:3] = DXA_points_cor[:,:3]*(xmax - xmin +  x_cst)/320
        DXA_points_cor[:,3:] = DXA_points_cor[:,3:]-(ymin +  abs(y_cst)/2)
        
        DXA_points_sag[:,:3] = DXA_points_cor[:,:3]*(xmax - xmin +  x_cst)/320
        DXA_points_sag[:,3:] = DXA_points_cor[:,3:]-(ymin +  abs(y_cst)/2)

        
        dx = Image.fromarray(im*255)
        dx = dx.convert('RGB')
        
        dxa = transform(dx)
        
        dx_trans = res_transform(dx, return_tensors='pt')
        dx_trans = dx_trans['pixel_values'].squeeze()
        
        # scaling wrt image dim  
        points[:,:3] = points[:,:3]/dxa.shape[1]                                  
        points[:,3:] = points[:,3:]/dxa.shape[2]  
        points = torch.tensor(points) 
        

        return_dict = {'dxa_img': dxa, 'dxa_trans':dx_trans,'dxa_filename': dxa_fp, 'coord':points}
        return return_dict
            
        
            
if __name__ == '__main__':
    
    D = PairsDataset(set_type='test',root = './UKBiobank/')
