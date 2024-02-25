# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:54:37 2023

@author: Runyu Jing
"""
'''
In this file, the code is refered from:
1) https://www.zhihu.com/question/545764550/answer/2670611518
2) https://github.com/CompVis/taming-transformers
3) https://github.com/CompVis/stable-diffusion
4) https://github.com/openai/improved-diffusion
Majorly from the first link, also compared with the other three repositories.

Since the code is to build a diffusion model for H&E img generation from RA image, 
the corss attention was added in the UNet model. 

Moreover, the code for token generation from RA image is in other file named 'repo_AE.py'.
'''
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from torch.nn import functional as F

# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
from abc import abstractmethod

import torch
import torch.nn as nn
# import torch.nn.functional as F
import copy

# import time

import numpy as np

import math
import pickle


RandHoriFlip = transforms.RandomHorizontalFlip(p=0.5)
RandVerFlip = transforms.RandomVerticalFlip(p=0.5)
RandRotate = transforms.RandomRotation(180)

class CustomDataset(Dataset):
    def __init__(self, root_dir, channelNum, splitHLen, splitWLen, splitStried, halfSize=False, randRotate = True):
        self.root_dir = root_dir
        
        self.halfSize = halfSize
        
        if channelNum > 1:
            transformations = [ transforms.ToTensor()]
        else:
            transformations = [transforms.Grayscale(num_output_channels=channelNum), transforms.ToTensor()]
        self.transform = transforms.Compose(transformations)

        self.randRotate = randRotate
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        
        self.splitHLen = splitHLen
        self.splitWLen = splitWLen
        self.splitStried = splitStried
        
        self.dataDict = {}
        self.fileIdx = 0
        self.idxLen = [0]
        
        for f in self.image_files:
            self.readOneFile(f)
        
        self.idxLen = np.array(self.idxLen)
    
    def readOneFile(self,f):
        img_name = os.path.join(self.root_dir, f)
        image = Image.open(img_name)
        if self.halfSize:
            new_size = (image.width // 2, image.height // 2)
            resize_transform = transforms.Resize(new_size)
            image = resize_transform(image)
        
        data = self.transform(image)
        
        
        h = data.size(-2)
        w = data.size(-1)
        
        # h_count = 0
        # local_length = self.splitStried
        # while local_length < h:
        #     h_count += 1
        #     local_length += self.splitStried
        
        # w_count = 0
        # local_length = self.splitStried
        # while local_length < w:
        #     w_count += 1
        #     local_length += self.splitStried
        h_count = (h - self.splitHLen ) // self.splitStried + 1
        w_count = (w - self.splitWLen ) // self.splitStried + 1
        
        self.dataDict[self.fileIdx] = (data,h_count,w_count)
        
        self.fileIdx += 1
        if len(self.idxLen) == 0:
            self.idxLen.append(h_count * w_count)
        else:
            self.idxLen.append(self.idxLen[-1] + h_count * w_count)
    
    def __len__(self):
        return self.idxLen[-1]

    
    def __getitem__(self, idx):
        tmpIdx = idx - self.idxLen
        tmpIdx[tmpIdx<0] = self.idxLen[-1] + 1
        fileIdx = np.argmin(tmpIdx)
        localLen = idx - self.idxLen[fileIdx]
        dataFull,hColNum,wColNum = self.dataDict[fileIdx]
        hCol = (localLen // wColNum) * self.splitStried
        wCol = (localLen % wColNum) * self.splitStried
        return dataFull[:,hCol:hCol+self.splitHLen,wCol:wCol+self.splitWLen]
    
    def exportRawData(self):
        tmpList = []
        for i in range(len(self.dataDict)):
            tmpList.append(self.dataDict[i][0])
        return torch.stack(tmpList,dim=0)

class CustomDatasetRA(Dataset):
    def __init__(self, root_dir, channelNum, splitHLen, splitWLen, splitStried, halfSize=False):
        self.root_dir = root_dir
        
        self.halfSize = halfSize
        
        

        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        
        self.splitHLen = splitHLen
        self.splitWLen = splitWLen
        self.splitStried = splitStried
        
        self.dataDict = {}
        self.fileIdx = 0
        self.idxLen = [0]
        
        for f in self.image_files:
            self.readOneFile(f)
        
        self.idxLen = np.array(self.idxLen)
    
    def readOneFile(self,f):
        filePath = os.path.join(self.root_dir, f)
        
        with open(filePath,'rb') as FID:
            data = pickle.load(FID)
        
        # image = Image.open(img_name)
        if self.halfSize:
            new_size = (data.size(-2) // 2, data.size(-1) // 2)
            resize_transform = transforms.Resize(new_size,antialias=False)
            data = resize_transform(data)
        
       
        h = data.size(-2)
        w = data.size(-1)
        
        h_count = (h - self.splitHLen ) // self.splitStried + 1
        w_count = (w - self.splitWLen ) // self.splitStried + 1
        
        self.dataDict[self.fileIdx] = (data,h_count,w_count)
        
        self.fileIdx += 1
        if len(self.idxLen) == 0:
            self.idxLen.append(h_count * w_count)
        else:
            self.idxLen.append(self.idxLen[-1] + h_count * w_count)
    
    def __len__(self):
        return self.idxLen[-1]

    
    def __getitem__(self, idx):
        tmpIdx = idx - self.idxLen
        tmpIdx[tmpIdx<0] = self.idxLen[-1] + 1
        fileIdx = np.argmin(tmpIdx)
        localLen = idx - self.idxLen[fileIdx]
        dataFull,hColNum,wColNum = self.dataDict[fileIdx]
        hCol = (localLen // wColNum) * self.splitStried
        wCol = (localLen % wColNum) * self.splitStried
        return dataFull[:,hCol:hCol+self.splitHLen,wCol:wCol+self.splitWLen]
    
    def exportRawData(self):
        tmpList = []
        for i in range(len(self.dataDict)):
            tmpList.append(self.dataDict[i][0])
        return torch.stack(tmpList,dim=0)

class CustomDatasetRANew(Dataset):
    def __init__(self, root_dir, channelNum, splitHLen, splitWLen, splitStriedH, splitStriedW, halfSize=False):
        self.root_dir = root_dir
        
        self.halfSize = halfSize
        
        

        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        
        self.splitHLen = splitHLen
        self.splitWLen = splitWLen
        self.splitStriedH = splitStriedH
        self.splitStriedW = splitStriedW
        
        self.dataDict = {}
        self.fileIdx = 0
        self.idxLen = [0]
        
        for f in self.image_files:
            self.readOneFile(f)
        
        self.idxLen = np.array(self.idxLen)
    
    def readOneFile(self,f):
        filePath = os.path.join(self.root_dir, f)
        
        with open(filePath,'rb') as FID:
            data = pickle.load(FID)
        
        # image = Image.open(img_name)
        if self.halfSize:
            new_size = (data.size(-2) // 2, data.size(-1) // 2)
            resize_transform = transforms.Resize(new_size,antialias=False)
            data = resize_transform(data)
        
       
        h = data.size(-2)
        w = data.size(-1)
        
        h_count = (h - self.splitHLen ) // self.splitStriedH + 1
        w_count = (w - self.splitWLen ) // self.splitStriedW + 1
        
        self.dataDict[self.fileIdx] = (data,h_count,w_count)
        
        self.fileIdx += 1
        if len(self.idxLen) == 0:
            self.idxLen.append(h_count * w_count)
        else:
            self.idxLen.append(self.idxLen[-1] + h_count * w_count)
    
    def __len__(self):
        return self.idxLen[-1]

    
    def __getitem__(self, idx):
        tmpIdx = idx - self.idxLen
        tmpIdx[tmpIdx<0] = self.idxLen[-1] + 1
        fileIdx = np.argmin(tmpIdx)
        localLen = idx - self.idxLen[fileIdx]
        dataFull,hColNum,wColNum = self.dataDict[fileIdx]
        hCol = (localLen // wColNum) * self.splitStriedH
        wCol = (localLen % wColNum) * self.splitStriedW
        return dataFull[:,hCol:hCol+self.splitHLen,wCol:wCol+self.splitWLen]
    
    def exportRawData(self):
        tmpList = []
        for i in range(len(self.dataDict)):
            tmpList.append(self.dataDict[i][0])
        return torch.stack(tmpList,dim=0)
    
class WrappedDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, dataset4):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4
        self.indices1 = np.arange(len(dataset1))
        self.indices2 = np.arange(len(dataset2))
        self.indices3 = np.arange(len(dataset3))
        self.indices4 = np.arange(len(dataset4))

    def __len__(self):
        return max(len(self.indices1), len(self.indices2), len(self.indices3), len(self.indices4))
    
    def shuffle(self):
        np.random.shuffle(self.indices1)
        np.random.shuffle(self.indices2)
        np.random.shuffle(self.indices3)
        np.random.shuffle(self.indices4)
    
    def __getitem__(self, idx):
        idx1 = idx % len(self.dataset1)
        idx2 = idx % len(self.dataset2)
        idx3 = idx % len(self.dataset3)
        idx4 = idx % len(self.dataset4)
        return self.dataset1[self.indices1[idx1]], self.dataset2[self.indices2[idx2]], self.dataset3[self.indices3[idx3]], self.dataset4[self.indices4[idx4]]
    
class DataGenerator:
    def __init__(self, dataset1, dataset2, batch_size=32):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size

    def __iter__(self):
        while True:  # 无限循环
            # 为每个数据集独立洗牌
            indices1 = np.arange(max(len(self.dataset1), len(self.dataset2)))
            indices2 = np.arange(max(len(self.dataset1), len(self.dataset2)))
            np.random.shuffle(indices1)
            np.random.shuffle(indices2)
            
            for start_idx in range(0, max(len(self.dataset1), len(self.dataset2)), self.batch_size):
                # 从每个数据集中挑选批次
                batch_indices1 = indices1[start_idx:start_idx + self.batch_size] % len(self.dataset1)
                batch_indices2 = indices2[start_idx:start_idx + self.batch_size] % len(self.dataset2)
                
                batch1 = [self.dataset1[idx] for idx in batch_indices1]
                batch2 = [self.dataset2[idx] for idx in batch_indices2]
                
                yield torch.stack(batch1), torch.stack(batch2)

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2)) // self.batch_size

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class ClsEbdBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, clsToken):
        """
        Apply the module to `x` given `clsToken` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock, ClsEbdBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ClsEbdBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x

def norm_layer(channels):
    return nn.GroupNorm(32, channels)

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

class CrossAttentionBlock(ClsEbdBlock):
    def __init__(self, channels, tokenDim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.norm_kv = nn.LayerNorm(tokenDim)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # self.kv = nn.Conv2d(tokenDim, channels * 2, kernel_size=1, bias=False)
        self.kv = nn.Linear(tokenDim, channels * 2, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, t):
        B, C, H, W = x.shape
        q = self.q(self.norm(x))
        L = t.size(1)
        kv = self.kv(self.norm_kv(t)).reshape(B,L,self.num_heads,-1)
        
        q = q.reshape(B*self.num_heads, -1, H*W)
        # k, v = kv.reshape(B*self.num_heads, -1, H*W).chunk(2, dim=1)
        k, v = kv.permute([0,2,3,1]).reshape(B*self.num_heads, -1, L).chunk(2, dim=1)
        
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


class UNetModel(nn.Module):

    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        cond_channels = 256,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.cond_channels = cond_channels
        
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        
        
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                    layers.append(CrossAttentionBlock(ch,cond_channels, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        
        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            CrossAttentionBlock(ch,cond_channels, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                    layers.append(CrossAttentionBlock(ch, cond_channels, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, cond):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond: an [N x S x D] Tensor of inputs.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x.dtype))
        
        # cond = self.condTrans(cond) #n * condLen * condDim
        
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb, cond)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb, cond)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb, cond)
        return self.out(h)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)



class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        # elif beta_schedule == 'cosine':
        #     betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, cond, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t.to(cond.dtype), t, cond)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, cond,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, cond, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), cond)
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, cond, channels=3):
        batch_size = cond.size(0)
        return self.p_sample_loop(model, cond,shape=(batch_size, channels, image_size, image_size))

    # compute train losses
    def train_losses(self, model, x_start, t, cond):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise).to(x_start.dtype)
        predicted_noise = model(x_noisy, t, cond)
        loss = F.mse_loss(noise, predicted_noise)
        return loss























