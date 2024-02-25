# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:03:46 2023

@author: tmp
"""

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from torch.nn import functional as F


import torch
import torch.nn as nn
# import torch.nn.functional as F
# import copy

# import time

import numpy as np


import pickle

# from torch.cuda.amp import autocast, GradScaler



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
            # data = (data + 1).float().log().half()
            # data = data.sign()*(data.abs() + 1).float().log().half()
        
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


class UnitDataHE(Dataset):
    def __init__(self,f, splitHLen, splitWLen, splitStriedH, splitStriedW, halfSize=False):
        
        self.fileName = f #just for recording
        self.halfSize = halfSize
        
        self.transform = transforms.Compose([ transforms.ToTensor()])

        self.splitHLen = splitHLen
        self.splitWLen = splitWLen
        self.splitStriedH = splitStriedH
        self.splitStriedW = splitStriedW
        
        image = Image.open(f)
        if halfSize:
            new_size = (image.width // 2, image.height // 2)
            resize_transform = transforms.Resize(new_size)
            image = resize_transform(image)
        
        data = self.transform(image)
        h = data.size(-2)
        w = data.size(-1)
        h_count = (h - splitHLen ) // splitStriedH + 1
        w_count = (w - splitWLen ) // splitStriedW + 1
        
        self.data = data
        self.h_count = h_count
        self.w_count = w_count
        
    
    def __len__(self):
        return self.h_count * self.w_count

    
    def __getitem__(self, idx):
        localLen = idx
        hCol = (localLen // self.w_count) * self.splitStriedH
        wCol = (localLen % self.w_count) * self.splitStriedW
        return self.data[:,hCol:hCol+self.splitHLen,wCol:wCol+self.splitWLen]

class UnitDataRA(Dataset):
    def __init__(self, f, splitHLen, splitWLen, splitStriedH, splitStriedW, halfSize=False):
        
        self.halfSize = halfSize
        self.fileName = f #just for recording
        

        
        self.splitHLen = splitHLen
        self.splitWLen = splitWLen
        self.splitStriedH = splitStriedH
        self.splitStriedW = splitStriedW
        
        with open(f,'rb') as FID:
            data = pickle.load(FID)
            # data = (data + 1).float().log().half()
            # data = data.sign()*(data.abs() + 1).float().log().half()
        
        if self.halfSize:
            new_size = (data.size(-2) // 2, data.size(-1) // 2)
            resize_transform = transforms.Resize(new_size,antialias=False)
            data = resize_transform(data)
        h = data.size(-2)
        w = data.size(-1)
        
        h_count = (h - self.splitHLen ) // self.splitStriedH + 1
        w_count = (w - self.splitWLen ) // self.splitStriedW + 1
        
        self.data = data
        self.h_count = h_count
        self.w_count = w_count
        
        
    def __len__(self):
        return self.h_count * self.w_count

    
    def __getitem__(self, idx):
        localLen = idx 
        hCol = (localLen // self.w_count) * self.splitStriedH
        wCol = (localLen % self.w_count) * self.splitStriedW
        return self.data[:,hCol:hCol+self.splitHLen,wCol:wCol+self.splitWLen]

class CustomDatasetHERA(Dataset):
    def __init__(self,HEFiles,RAFiles,HESplitPara,RASplitPara):
        self.sampleNum = None
        HEObjList = []
        HELens = [0]
        for f in HEFiles:
            sampleNum = int(os.path.split(f)[-1].split('-')[2].split(' ')[0])
            if self.sampleNum is None:
                self.sampleNum = sampleNum
            else:
                assert self.sampleNum == sampleNum
            splitHLen, splitWLen, splitStriedH, splitStriedW, halfSize = HESplitPara
            HEObj = UnitDataHE(f, splitHLen, splitWLen, splitStriedH, splitStriedW, halfSize)
            HEObjList.append(HEObj)
            HELens.append(len(HEObj))
            
        RAObjList = []
        RALens = [0]
        for f in RAFiles:
            sampleNum = int(os.path.split(f)[-1].split('-')[0])
            if self.sampleNum is None:
                self.sampleNum = sampleNum
            else:
                assert self.sampleNum == sampleNum
            splitHLen, splitWLen, splitStriedH, splitStriedW, halfSize = RASplitPara
            RAObj = UnitDataRA(f, splitHLen, splitWLen, splitStriedH, splitStriedW, halfSize)
            RAObjList.append(RAObj)
            RALens.append(len(RAObj))
        self.HEObjDict = dict(enumerate(HEObjList))
        self.RAObjDict = dict(enumerate(RAObjList))
        self.HELen = np.sum(HELens).astype(int)
        self.RALen = np.sum(RALens).astype(int)
        self.len = np.max([self.HELen, self.RALen])
        
        self.HERefList = np.cumsum(HELens)
        self.RARefList = np.cumsum(RALens)
        
        self.HEidx = np.arange(self.HELen)
        self.RAidx = np.arange(self.RALen)
        
    def __len__(self):
        return self.len
    
    def shuffle(self):
        np.random.shuffle(self.HEidx)
        np.random.shuffle(self.RAidx)
        
    def __getitem__(self, idx):
        HEidx = self.HEidx[idx % self.HELen]
        RAidx = self.RAidx[idx % self.RALen]
        
        tmpArr = HEidx - self.HERefList
        tmpArr[tmpArr < 0] = self.HELen
        HERefidx = np.argmin(np.abs(tmpArr))
        HEObj = self.HEObjDict[HERefidx]
        HEArr = HEObj[HEidx - self.HERefList[HERefidx]]
        
        tmpArr = RAidx - self.RARefList
        tmpArr[tmpArr < 0] = self.RALen
        RARefidx = np.argmin(np.abs(tmpArr))
        RAObj = self.RAObjDict[RARefidx]
        RAArr = RAObj[RAidx - self.RARefList[RARefidx]]
        
        return HEArr, RAArr
            
class WrapperedDatasetHERA(Dataset):
    def __init__(self,HERoot,RARoot,HESplitPara,RASplitPara):
        self.HERoot = HERoot
        self.HESplitPara = HESplitPara
        self.HEFiles = [f for f in os.listdir(HERoot) if os.path.isfile(os.path.join(HERoot, f))]
        
        self.RARoot = RARoot
        self.RASplitPara = RASplitPara
        self.RAFiles = [f for f in os.listdir(RARoot) if os.path.isfile(os.path.join(RARoot, f))]
        
        self.sampleFileDict = {}
        for f in self.HEFiles:
            sampleNum = int(f.split('-')[2].split(' ')[0])
            self.sampleFileDict.setdefault(sampleNum,{'HEFiles':[],'RAFiles':[]})
            self.sampleFileDict[sampleNum]['HEFiles'].append(HERoot+os.sep+f)
        
        for f in self.RAFiles:
            sampleNum = int(f.split('-')[0])
            self.sampleFileDict.setdefault(sampleNum,{'HEFiles':[],'RAFiles':[]})
            self.sampleFileDict[sampleNum]['RAFiles'].append(RARoot+os.sep+f)
        
        # self.sampleObjDict = {}        
        lenList = [0]
        self.objList = []
        for sampleNum in np.sort(list(self.sampleFileDict.keys())):
            HEFiles = self.sampleFileDict[sampleNum]['HEFiles']
            RAFiles = self.sampleFileDict[sampleNum]['RAFiles']
            if len(HEFiles) == 0 or len(RAFiles) == 0:
                print('skip sample %d because of %d HE - %d RA files' %(sampleNum, len(HEFiles), len(RAFiles)))
                continue
            HERAObj = CustomDatasetHERA(HEFiles,RAFiles,HESplitPara,RASplitPara)
            self.objList.append(HERAObj)
            lenList.append(len(HERAObj))
        
        self.refList = np.cumsum(lenList)
        self.len = np.sum(lenList).astype(int)
        
    def shuffle(self):
        for tmpobj in self.objList:
            tmpobj.shuffle()
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        tmpArr = idx - self.refList
        tmpArr[tmpArr < 0] = self.len
        refidx = np.argmin(np.abs(tmpArr))
        HERAObj = self.objList[refidx]
        HEArr,RAArr = HERAObj[idx - self.refList[refidx]]
        return HEArr,RAArr
        

def norm_layer(channels):
    return nn.GroupNorm(4, channels)


    

    
class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class AESimple(nn.Module):
    def __init__(self,dimIn,dropout_rate=0.15,fNum=None,activateLayer=None):
        super().__init__()  
        # self.fNum = [8,16,32,64]
        if fNum is None:
            self.fNum = [32,64,128]
        else:
            self.fNum = fNum
        
        self.initConv = nn.Conv2d(dimIn, self.fNum[0], kernel_size=1)
        self.encLayers = nn.ModuleList([ResblockRA(self.fNum[0], self.fNum[0], dropout_rate)])
        for i in range(len(self.fNum)-1):
            self.encLayers.append(ResblockRA(self.fNum[i], self.fNum[i],  dropout_rate))
            self.encLayers.append(Downsample(self.fNum[i]))
            self.encLayers.append(ResblockRA(self.fNum[i], self.fNum[i+1],  dropout_rate))
        
        self.enc= nn.Sequential(self.initConv,*self.encLayers)
        
        self.decLayers = nn.ModuleList([ResblockRA(self.fNum[-1], self.fNum[-1], dropout_rate)])
        for i in range(len(self.fNum)-1):
            self.decLayers.append(Upsample(self.fNum[-i-1], ))
            self.decLayers.append(ResblockRA(self.fNum[-i-1], self.fNum[-i-2], dropout_rate))
            self.decLayers.append(ResblockRA(self.fNum[-i-2], self.fNum[-i-2], dropout_rate))
        
        if activateLayer is None:
            activateLayer = nn.Identity()
        
        self.dec = nn.Sequential(
            *self.decLayers,
            norm_layer(self.fNum[0]),
            nn.SiLU(),
            nn.Conv2d(self.fNum[0], dimIn, kernel_size=1),
            activateLayer,
            )
        
        
        self.lossFunc = nn.MSELoss(reduction='mean')
        
    def forward(self,x):
        mid = self.enc(x)
        rec = self.dec(mid)
        loss = self.lossFunc(rec,x)
        return loss



class ResblockRA(nn.Module):
    def __init__(self,in_channels,out_channels,dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
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
            
    def forward(self, x):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        """
        h = self.conv1(x)
        h = self.conv2(h)
        return h + self.shortcut(x)



class ImgAlignNet(nn.Module):
    def __init__(self,imgDimIn,RADimIn,targetInitDim,
                   midDim,
                 # eDim, 
                 # RADimMid=4, 
                 clsNum=2, targetTransLayerNum=3,
                 targetNum=16, targetSize=(16,16),  imgStride=(8,8), raStride=(8,4), 
                  RAFNum=[512,256,128],
                 imgFNum=[32,64,128],
                 RAAECPPath = None, imgAECPPath = None,
                 dropout_rate=0.15):
        super().__init__()
        
        self.targetSize = targetSize
        self.imgStride = imgStride
        self.raStride = raStride
        self.clsNum = clsNum
        self.targetNum = targetNum
        
        self.temperatureImg = nn.Parameter(torch.ones([]) * np.log(1 / 0.5))
        self.temperatureRA = nn.Parameter(torch.ones([]) * np.log(1 / 0.5))
        
        tmpRAEnc = AESimple(dimIn=RADimIn,dropout_rate=dropout_rate,fNum=RAFNum,activateLayer=None)
        if RAAECPPath:
            tmpRAEnc.load_state_dict(parseCheckPoint(RAAECPPath))
        self.RAEnc = tmpRAEnc.enc
        
        
        tmpHEEnc = AESimple(dimIn=imgDimIn,dropout_rate=dropout_rate,fNum=imgFNum,activateLayer=nn.Sigmoid())
        if imgAECPPath:
            tmpHEEnc.load_state_dict(parseCheckPoint(imgAECPPath))
        self.imgEnc = tmpHEEnc.enc
            
        
        raOutDim = RAFNum[-1]
        
        imgMidDim = imgFNum[-1] * np.prod(targetSize)
        self.imgTrans = nn.Linear(imgMidDim, midDim)
        
        raMidDim = raOutDim * np.prod(targetSize)
        self.RATrans = nn.Linear(raMidDim, midDim)
        self.target = nn.Parameter(torch.randn([clsNum,targetNum,targetInitDim]),requires_grad=True)
        dimList = [targetInitDim] + [midDim] * targetTransLayerNum
        self.buildTargetProcessLayer(dimList)
        
        self.imgVotePara = nn.Parameter(torch.ones([1,clsNum,targetNum])/targetNum)
        self.raVotePara = nn.Parameter(torch.ones([1,clsNum,targetNum])/targetNum)
        
        self.lossFuncCls = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.15)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.eps = 1e-10
    
    def buildTransLayer(self,dimList):
        layers = []
        for i in range(len(dimList) - 1):
            dim1 = dimList[i]
            dim2 = dimList[i+1]
            layers.append(nn.Linear(dim1,dim2))
            layers.append(nn.LayerNorm(dim2))
            layers.append(nn.SiLU())
        return layers[:-2]
    
    def buildTargetProcessLayer(self,dimList):
        imgLayers = []
        raLayers = []
        for i in range(len(dimList) - 1):
            dim1 = dimList[i]
            dim2 = dimList[i+1]
            imgLayers.append(nn.Linear(dim1,dim2))
            imgLayers.append(nn.LayerNorm(dim2))
            imgLayers.append(nn.SiLU())
            
            raLayers.append(nn.Linear(dim1,dim2))
            raLayers.append(nn.LayerNorm(dim2))
            raLayers.append(nn.SiLU())
        self.imgTargetProcessLayer = nn.Sequential(
            *imgLayers[:-2]
            )
        self.raTargetProcessLayer = nn.Sequential(
            *raLayers[:-2]
            )
            
        
    
    def processImg(self, img):
        # img = F.normalize(img,p=2,dim=1)
        imgTK = self.imgEnc(img)
        n,c,_,_ = imgTK.shape
        s_h,s_w = self.targetSize
        imgTK = imgTK.unfold(2,s_h,self.imgStride[0]).unfold(3,s_w,self.imgStride[1] ).contiguous().view(n, c, -1, s_h * s_w)
        imgTK = imgTK.permute(0, 2, 1, 3).reshape( n, -1, c * s_h * s_w)
        imgTK = self.imgTrans(imgTK)
        return imgTK
    
    def processRA(self,ra):
        # ra = F.normalize(ra,p=2,dim=1)
        # raTK = self.RAEnc(ra.unsqueeze(1)).squeeze(1) #n * c * h * w        
        raTK = self.RAEnc(ra)
        n,c,_,_ = raTK.shape
        s_h,s_w = self.targetSize
        raTK = raTK.unfold(2,s_h,self.raStride[0]).unfold(3,s_w,self.raStride[1]).contiguous().view(n, c, -1, s_h * s_w)
        raTK = raTK.permute(0, 2, 1, 3).reshape( n, -1, c * s_h * s_w)
        raTK = self.RATrans(raTK)
        return raTK
    
    def generateTarget(self):
        imgLM = self.imgTargetProcessLayer(self.target) #cls * LMNum * d  
        raLM = self.raTargetProcessLayer(self.target) #cls * LMNum * d  
        return imgLM, raLM
    
    def computeDist(self,token,target,temperature,returnMidDist=False):
        # dist = (self.cos(token, target ) + 1).exp().flatten(3).sum(dim=-1) * temperature.exp()
        midDist = ((self.cos(token, target ) + 1)* temperature).exp() #n * cls * numLMK * l
        if returnMidDist:
            return midDist
        dist = midDist.flatten(3).sum(dim=-1) 
        
        return dist
    
    def forward(self,img,ra,clsArr):
        # loss = 0
        auxLoss = []
        imgTK = self.processImg(img).unsqueeze(1).unsqueeze(1)       #n * 1 * 1 * l1 * d
        raTK = self.processRA(ra).unsqueeze(1).unsqueeze(1)          #n * 1 * 1 * l2 * d
        
        imgTK = F.normalize(imgTK,p=2,dim=-1)
        raTK = F.normalize(raTK,p=2,dim=-1)
        
        imgLM, raLM = self.generateTarget()    # cls * numLMK * d
        
        
        imgLM = imgLM.unsqueeze(2).unsqueeze(0)  #1 * cls * numLMK * 1 * d
        raLM = raLM.unsqueeze(2).unsqueeze(0)    #1 * cls * numLMK * 1 * d
        
        distImg = self.computeDist(imgTK,imgLM,self.temperatureImg)
        distRA = self.computeDist(raTK, raLM,self.temperatureRA)
        
        predImg = (distImg * self.imgVotePara.abs()).mean(dim=-1)
        predRA = (distRA * self.raVotePara.abs()).mean(dim=-1)
        lossImg = self.lossFuncCls(predImg,clsArr)
        lossRA = self.lossFuncCls(predRA,clsArr)
        
        
        
        loss =  lossImg + lossRA
        # auxLoss = [lossImg.detach(), lossRA.detach()]
        auxLoss.append(lossImg.detach())
        auxLoss.append(lossRA.detach())
        
        return loss, auxLoss
        
    def predictRA(self,ra):
        raTK = self.processRA(ra).unsqueeze(1).unsqueeze(1)          #n * 1 * 1 * l2 * d
        raTK = F.normalize(raTK,p=2,dim=-1)
        imgLM, raLM = self.generateTarget()    # cls * numLMK * d
        raLM = raLM.unsqueeze(2).unsqueeze(0)    #1 * cls * numLMK * 1 * d
        distRA = self.computeDist(raTK, raLM,self.temperatureRA)
        predRA = (distRA * self.raVotePara.abs()).mean(dim=-1)
        return predRA
    
    def generateRACondOld(self,ra,p=2):
        raTKOri = self.processRA(ra) #n * l2 * d
        raTK = raTKOri.unsqueeze(2).unsqueeze(2) #n * l2 * 1 * 1 * d
        raTK = F.normalize(raTK,p=2,dim=-1)
        imgLM, raLM = self.generateTarget()    # cls * numLMK * d
        c,m,d = raLM.shape
        raLM = raLM.unsqueeze(0).unsqueeze(0)    #1 * 1 * cls * numLMK * d
        simi = ((self.cos(raTK, raLM ).flatten(2) + 1) ** p).softmax(-1) # n * l2 * cls * numLMK -> n * l2 * (cls * numLMK)
        cond = torch.einsum('nlm,md->nld',simi,raLM.reshape(c*m,d))
        return cond
        
    def generateRACond(self,ra,p=2,topK_token=None,topK_patch=None):
        raTK = self.processRA(ra).unsqueeze(1).unsqueeze(1)          #n * 1 * 1 * l2 * d
        raTK = F.normalize(raTK,p=2,dim=-1)
        imgLM, raLM = self.generateTarget()    # cls * numLMK * d
        raLM = raLM.unsqueeze(2).unsqueeze(0)    #1 * cls * numLMK * 1 * d
        
        
        if topK_token and topK_patch:
            distRA = self.computeDist(raTK, raLM,self.temperatureRA,returnMidDist=True)#n * cls * numLMK * l2
            n,n_cls,n_lmk,l = distRA.shape
            distRA = distRA.reshape([n,n_cls*n_lmk,l])
            topK_L = distRA.topk(topK_token,dim=-1).values #n * (cls * numLMK) * k
            topK_cls_lmk = topK_L.permute([0,2,1]).contiguous() #n * k * (cls * numLMK) 
            thres = topK_cls_lmk.topk(topK_patch,dim=-1).values[:,:,[-1]]
            topK_cls_lmk = topK_cls_lmk.masked_fill(topK_cls_lmk<thres,-torch.inf)
            topK_cls_lmk = topK_cls_lmk.softmax(dim=-1) #n * k * (cls * numLMK) 
            cond = torch.einsum('nkc,cd->nkd',topK_cls_lmk.float(),imgLM.reshape(n_cls*n_lmk,-1).float()).to(ra.dtype)
            return cond
        else:
            distRA = self.computeDist(raTK, raLM,self.temperatureRA) #n * cls * numLMK
            predRA = (distRA * self.raVotePara.abs()).mean(dim=-1).argmax(-1) #n 
            
            seleLM = raLM.squeeze(0)[predRA,:,:,:] #n * numLMK * 1 * d
            seleLM_img = imgLM[predRA,:,:] #n * numLMK * d
            # print(raTK.shape,seleLM.shape,predRA.shape)
            simi = ((self.cos(raTK.squeeze(1), seleLM )  + 1) ** p).permute([0,2,1]).softmax(-1) # n * l2 * numLMK
            # cond = torch.einsum('nlm,nmd->nld',simi,seleLM.squeeze(2))
            cond = torch.einsum('nlm,nmd->nld',simi,seleLM_img)
            return cond
    
    def predictImg(self,img):
        imgTK = self.processImg(img).unsqueeze(1).unsqueeze(1)       #n * 1 * 1 * l1 * d
        imgTK = F.normalize(imgTK,p=2,dim=-1)
        imgLM, raLM = self.generateTarget()    # cls * numLMK * d
        imgLM = imgLM.unsqueeze(2).unsqueeze(0)  #1 * cls * numLMK * 1 * d
        distImg = self.computeDist(imgTK,imgLM,self.temperatureImg)
        predImg = (distImg * self.imgVotePara.abs()).mean(dim=-1)
        return predImg
    

def calculate_positional_encoding(pos, ebdSize, ebdNum=1e-4, dtype=torch.float32, device='cpu'):
    #pos: n 
    # c,s = shape
    # ebdSize = c * s
    div_term = torch.exp(torch.arange(0, ebdSize, 2, dtype = dtype, device = device) * -(np.log(ebdNum) / ebdSize)).unsqueeze(0) # 1 *  d // 2
    position = pos.unsqueeze(-1) #n * 1 
    
    pos_enc = torch.zeros([position.shape[0], ebdSize], dtype=dtype, device=device) #n * d
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    
    return pos_enc
    # return pos_enc.reshape([pos.size(0),c,s])
                

                
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    


def parseCheckPoint(path):
    statDict = {}
    tmpDict = torch.load(path,map_location='cpu')
    for k in tmpDict:
        k1 = k
        if k.startswith('module.'):
            k1 = k.replace('module.','')
        statDict[k1] = tmpDict[k]
    return statDict

def lr_lambda(epoch):
    warmup_epochs = 10000
    target_lr = 1
    if epoch < warmup_epochs:
        return 1e-7 + (target_lr - 1e-7) * (epoch / warmup_epochs)
    else:
        return target_lr

        

