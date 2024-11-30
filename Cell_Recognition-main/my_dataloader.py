from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms  # 导入数据变换模块
import random
from PIL import Image
import glob  # 导入用于模式匹配文件路径的模块
import skimage.io as io

class CellsDataset(Dataset):
    def __init__(self,img_root, gt_dmap_root, gt_dmap_all_root,gt_dots_root, class_indx, split_filepath=None, phase='train', fixed_size=-1, max_side=-1, max_scale=-1, return_padding=False):
        super(CellsDataset, self).__init__()
        '''
        img_root: 图像根目录
        gt_dmap_root: 真实扩张点地图的根目录（类别地图）
        gt_dots_root: 真实点地图的根目录
        gt_dmap_all_root: 真实扩张点全部地图的根目录（用于检测地图）
        class_indx: 返回来自真实扩张点地图的通道索引的逗号分隔列表
        split_filepath: 如果不为None，则仅使用文件中的图像
        phase: train or test
        fixed_size:  如果 > 0，在训练期间返回固定大小的裁剪
        max_side: 是否在训练期间有最大边长
        max_scale: 应用填充以使补丁边长可被max_scale整除
        return_padding: 返回由max_scale添加的x和y填充
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_dots_root=gt_dots_root
        self.gt_dmap_all_root = gt_dmap_all_root
        self.phase=phase
        self.return_padding = return_padding

        if(split_filepath is None):
            self.img_names=[filename for filename in os.listdir(img_root) \
                               if os.path.isfile(os.path.join(img_root,filename))]
        else:
            self.img_names=np.loadtxt(split_filepath, dtype=str).tolist()
            
        self.n_samples=len(self.img_names)

        self.fixed_size = fixed_size
        self.max_side = max_side
        self.max_scale = max_scale
        self.class_indx = class_indx
        self.class_indx_list = [int(x) for x in self.class_indx.split(',')]

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'

        # 读取图像，将其归一化，并确保其处于RGB格式
        img_name=self.img_names[index]
        # print('img_name',img_name)
        img=io.imread(os.path.join(self.img_root,img_name))/255  # 将像素值从[0,255]转换为[0,1]
        if len(img.shape) == 2:  # 将灰度图像扩展到三个通道
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), 2)

        # 读取真实扩张点地图
        gt_path = os.path.join(self.gt_dmap_root,img_name.replace('.png', '.npy'))
        if os.path.isfile(gt_path):
            gt_dmap = np.load(gt_path, allow_pickle=True)[:, :, self.class_indx_list].squeeze()
        else:
            gt_dmap = np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)))

        # 读取真实点地图（分类点注释映射）
        gt_dots_path = os.path.join(self.gt_dots_root,img_name.replace('.png', '_gt_dots.npy'))
        if os.path.isfile(gt_dots_path):
            gt_dots = np.load(gt_dots_path, allow_pickle=True)[:, :, self.class_indx_list].squeeze()
        else:
            gt_dots = np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)))

        # 读取真实扩张全部地图（用于细胞检测）
        gt_all_path = os.path.join(self.gt_dmap_all_root, img_name.replace('.png', '_all.npy'))
        if os.path.isfile(gt_all_path):
            gt_dmap_all = np.load(gt_all_path, allow_pickle=True).squeeze()
        else:
            gt_dmap_all = np.zeros((img.shape[0], img.shape[1]))

        # 如果是训练，应用随机翻转增强
        if random.randint(0, 1) == 1 and self.phase == 'train':
            img = img[:, ::-1].copy()  # 水平翻转
            gt_dmap = gt_dmap[:,::-1].copy()  # 水平翻转
            gt_dots = gt_dots[:,::-1].copy()  # 水平翻转
            gt_dmap_all = gt_dmap_all[:,::-1].copy()  # 水平翻转(根据维度自动处理，不必考虑具体维度差异)
        
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[::-1,:].copy()  # 垂直翻转
            gt_dmap=gt_dmap[::-1,:].copy()  # 垂直翻转
            gt_dots=gt_dots[::-1,:].copy()  # 垂直翻转
            gt_dmap_all = gt_dmap_all[::-1,:].copy()  # 垂直翻转

        # 如果是训练，确保宽度和高度 < max_side
        if(self.phase=='train' and self.max_side > 0):
            h = img.shape[0]
            w = img.shape[1]
            h2 = h
            w2 = w
            crop = False
            if(h > self.max_side):
                h2 = self.max_side
                crop = True
            if(w > self.max_side):
                w2 = self.max_side
                crop = True
            if(crop):
                y=0
                x=0
                if(not (h2 ==h)):
                    y = np.random.randint(0, high = h-h2)
                if(not (w2 ==w)):
                    x = np.random.randint(0, high = w-w2)
                img = img[y:y+h2, x:x+w2, :]
                gt_dmap = gt_dmap[y:y+h2, x:x+w2]
                gt_dots = gt_dots[y:y+h2, x:x+w2]
                gt_dmap_all = gt_dmap_all[y:y+h2, x:x+w2]

        # 如果是训练，进行大小为fixed_size的随机裁剪，如果fixed_size < 0，则使用图像维度的1/4
        if self.phase == 'train':
            i = -1
            img_pil = Image.fromarray(img.astype(np.uint8)*255)
            if(self.fixed_size < 0):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(img.shape[0]//4, img.shape[1]//4))
            elif(self.fixed_size < img.shape[0] or self.fixed_size < img.shape[1]):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(min(self.fixed_size,img.shape[0]), min(self.fixed_size,img.shape[1])))
            if(i >= 0):
                img = img[i:i+h, j:j+w, :]
                gt_dmap = gt_dmap[i:i+h, j:j+w]
                gt_dots = gt_dots[i:i+h, j:j+w]
                gt_dmap_all = gt_dmap_all[i:i+h, j:j+w]

        # 添加填充以确保图像维度可被max_scale整除
        pad_y1=0
        pad_y2=0
        pad_x1=0
        pad_x2=0
        if self.max_scale>1:  # 降采样图像和密度图以匹配深度模型
            ds_rows=int(img.shape[0]//self.max_scale)*self.max_scale
            ds_cols=int(img.shape[1]//self.max_scale)*self.max_scale
            pad_y1 = 0
            pad_y2 = 0
            pad_x1 = 0
            pad_x2 = 0
            if(ds_rows < img.shape[0]):
                pad_y1 = (self.max_scale - (img.shape[0] - ds_rows))//2
                pad_y2 = (self.max_scale - (img.shape[0] - ds_rows)) - pad_y1
            if(ds_cols < img.shape[1]):
                pad_x1 = (self.max_scale - (img.shape[1] - ds_cols))//2
                pad_x2 = (self.max_scale - (img.shape[1] - ds_cols)) - pad_x1
            img = np.pad(img, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(1,) )# padding constant differs by dataset based on bg color
            gt_dmap = np.pad(gt_dmap, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# padding constant differs by dataset based on bg color
            gt_dots = np.pad(gt_dots, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# padding constant differs by dataset based on bg color
            gt_dmap_all = np.pad(gt_dmap_all, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'constant', constant_values=(0,))

        # 将图像和真实值转换为Pytorch格式
        img=img.transpose((2,0,1))  # 转换为(channel, rows, cols)的顺序
        if(len(self.class_indx_list) > 1):
            gt_dmap=gt_dmap.transpose((2,0,1))  # 转换为(channel, rows, cols)的顺序
            gt_dots=gt_dots.transpose((2,0,1))  # 转换为(channel, rows, cols)的顺序
        else:
            gt_dmap=gt_dmap[np.newaxis,:,:]
            gt_dots=gt_dots[np.newaxis,:,:]
        img_tensor=torch.tensor(img,dtype=torch.float)
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        gt_dots_tensor=torch.tensor(gt_dots,dtype=torch.float)
        gt_dmap_all_tensor = torch.tensor(gt_dmap_all, dtype=torch.float)

        if(self.return_padding):
            return img_tensor,gt_dmap_tensor,gt_dots_tensor,gt_dmap_all_tensor,img_name, (pad_y1, pad_y2, pad_x1, pad_x2)
        else:
            return img_tensor,gt_dmap_tensor,gt_dots_tensor,gt_dmap_all_tensor,img_name
