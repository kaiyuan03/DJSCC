
import torch
import torchsummary
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms
import warnings
from torchvision import datasets
from torch.utils.data import dataloader
from matplotlib import pyplot as plt
import time
import tqdm
import os
import pickle
from compressai.layers import GDN
import math
import numpy as np
ws_root = '/data1/kaiyuan/project/djscc_bcm/'

class Encoder(nn.Module):
    def __init__(self,c):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3,out_channels=256,kernel_size=9,stride=2,padding=4)),
            ('GDN1', GDN(256)),
            ('prelu1', nn.PReLU(num_parameters=256)),
            ('conv2', nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=2,padding=2)),
            ('GDN2', GDN(256)),
            ('prelu2', nn.PReLU(num_parameters=256)),
            ('conv3', nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2)),
            ('GDN3', GDN(256)),
            ('prelu3',nn.PReLU(num_parameters=256)),
            ('conv4', nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2)),
            ('GDN4', GDN(256)),
            ('prelu4',nn.PReLU(num_parameters=256)),
            ('conv5', nn.Conv2d(in_channels=256,out_channels=c,kernel_size=5,stride=1,padding=2)),
            ('GDN5', GDN(c)),
        ]))
    def forward(self, x):
        y = self.layers(x)
        return y
 

class Decoder(nn.Module):
    def __init__(self,c):
        
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('convtrans1', nn.ConvTranspose2d(in_channels=c,out_channels=256,kernel_size=5,stride=1,padding=2)),
            ('IGDN1', GDN(256, inverse=True)),
            ('prelu1', nn.PReLU(num_parameters=256)),
            ('convtrans2', nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2)),
            ('IGDN2', GDN(256, inverse=True)),
            ('prelu2', nn.PReLU(num_parameters=256)),
            ('convtrans3', nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2)),
            ('IGDN3', GDN(256, inverse=True)),
            ('prelu3', nn.PReLU(num_parameters=256)),
            ('convtrans4', nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=5,stride=2,padding=2,output_padding=1)),
            ('IGDN4', GDN(256, inverse=True)),
            ('prelu4', nn.PReLU(num_parameters=256)),
            ('convtrans5', nn.ConvTranspose2d(in_channels=256,out_channels=3,kernel_size=9,stride=2,padding=4,output_padding=1)),
            ('IGDN5', GDN(3, inverse=True)),
            ('sigmoid', nn.Sigmoid()),
        ]))
    def forward(self, x):
        y = self.layers(x)
        return y
    
class NormalizationNoise(nn.Module):
    def __init__(self, stddev):
        super(NormalizationNoise, self).__init__()
        self.stddev = stddev
    def forward(self, x):
        # noise power is the square of stddev
        noise = torch.normal(0, self.stddev, size=x.shape)
        noise = noise.to(x.device)
        return x+noise
    
class Autoencoder(nn.Module):
    def __init__(self, snr_db, bw_ratio, input_cwh):
        super(Autoencoder,self).__init__()
        self.snrdb_train=snr_db
        self.bwr = bw_ratio
        num_featuremap = self.calculate_filter_num(bw_ratio, input_cwh)
        print("Number of feature map: ", num_featuremap)
        self.encoder = Encoder(num_featuremap)
        self.decoder = Decoder(num_featuremap)
        stddev = 10 ** (-snr_db/20) # noise standard deviation of in-phase and quadrature compoments
        # precondition: symbol mean power 1
        self.channel = NormalizationNoise(stddev)
        self.trained_epoch = 0
    def forward(self, x):
        symbols = self.encoder(x)
        # concatenated symbol power sum: width*height*channel/2, i.e., number of elements/2,
        # i.e., number of elements in in-phase or quadrature compoment
        # average power of each element in in-phrase or quadrature component: 0.5
        symbols = F.normalize(symbols,dim=(1,2,3))*np.sqrt(np.prod(symbols.shape[1:]))
        symbols_corrupted = self.channel(symbols)
        symbols_corrupted = F.normalize(symbols_corrupted,dim=(1,2,3))*np.sqrt(np.prod(symbols.shape[1:]))
        x_hat = self.decoder(symbols_corrupted)
        return x_hat
    
    def calculate_filter_num(self, bw_ratio, input_cwh):
        def calculate_conv_outsize(x, kernel_size, stride, padding):
            return torch.floor((x-kernel_size+2*padding)/stride)+1
        dimensions0 = torch.tensor(input_cwh[1:])
        dimensions1 = calculate_conv_outsize(dimensions0,9,2,4)
        dimensions2 = calculate_conv_outsize(dimensions1,5,2,2)
        dimensions3 = calculate_conv_outsize(dimensions2,5,1,2)
        dimensions4 = calculate_conv_outsize(dimensions3,5,1,2)
        dimensions5 = calculate_conv_outsize(dimensions4,5,1,2)
        # total required real symbol number
        symbol_num = torch.prod(torch.tensor(input_cwh))*bw_ratio*2
        # number of real symbol in one feature map
        fmap_size=torch.prod(dimensions5)
        # number of required feature map
        filter_num = torch.div(symbol_num,fmap_size,rounding_mode='floor').item()+1
        return int(filter_num)
    # feature map的个数描述的是ratio，和输入维度无关


def timepref():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
def loss(x,y):
    return torch.mean((x-y)**2)
def train_one_epoch(model, loader, loss_func, optimizer, epochinfo):
    model.train()
    device = next(model.parameters()).device
    psnr_ls = []
    for batch_idx, x in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = loss_func(x, x_hat)
        loss.backward()
        optimizer.step()
        psnr = 10*torch.log10(1/loss).item()
        if batch_idx%1==0:
            msg = (f"[{device}][BWR {model.bwr}][SNR {model.snrdb_train}][****Train]"+
                   f"[Epoch {epochinfo[0]+1:2}|{epochinfo[1]}]"+
                   f"[Batch {batch_idx:4}|{len(loader)}]"+
                   f"[Loss {loss.item():.5f}]"+
                   f"[PSNR {psnr:.2f}]")
            print(msg)
        psnr_ls.append(psnr)
    return psnr_ls

@torch.no_grad()
def test_one_epoch(model, loader, loss_func, epochinfo):
    device = next(model.parameters()).device
    loss_ls = []
    for batch_idx,x in enumerate(loader):
        x = x.to(device)
        x_hat = model(x)
        lossv = loss_func(x, x_hat)
        psnr = 10*torch.log10(1/lossv).item()
        loss_ls.append(lossv)
        if batch_idx%1==0:
            msg = (f"[{device}][BWR {model.bwr}][SNR {model.snrdb_train}][-----Test]"+
                   f"[Epoch {epochinfo[0]+1:2}|{epochinfo[1]}]"+
                   f"[Batch {batch_idx:4}|{len(loader)}]"+
                   f"[Loss {lossv.item():.5f}]"+
                   f"[PSNR {psnr:.2f}]")
            print(msg)
    loss = sum(loss_ls)/len(loss_ls)
    psnr = 10*math.log10(1/loss)
    msg = (f"[{device}][BWR {model.bwr}][SNR {model.snrdb_train}][-----Test]"+ 
           f"[Epoch {epochinfo[0]+1:2}|{epochinfo[1]}]"+
           f"[Loss {loss:.5f}]"+
           f"[PSNR {psnr:.2f}]\n\n")
    print(msg)
    return loss.item(),psnr



if __name__ == "__main__":
    dev='cuda:5'
    load_pretrained = 1
    bw_ratio = 1/6
    snrdb_train = 13
    
    xtest = datasets.CIFAR10(root=ws_root+'datasets/cifar10',train=False,
                         download=True,transform=transforms.ToTensor())
    xtrain = datasets.CIFAR10(root=ws_root+'datasets/cifar10',train=True,
                            download=True,transform=transforms.ToTensor())
    trainloader = dataloader.DataLoader(dataset=xtrain,batch_size=128,shuffle=True)
    testloader = dataloader.DataLoader(dataset=xtest,batch_size=128,shuffle=True)
    if load_pretrained:
        with open(ws_root+
                  'ae_torch/savedmodels_torchprelu/snrdb_inf_bwratio_0.167_2024_01_16_16_06_03.pth.tar'
                  ,'rb') as f:
            autoencoder = pickle.load(f)
            autoencoder.channel.stddev = 10 ** (-snrdb_train/20)
            print("Loaded model trained epoch: ", autoencoder.trained_epoch)
    else:
        autoencoder = Autoencoder(snrdb_train,bw_ratio,(3,32,32))
    # torchsummary.summary(autoencoder,input_size=(3,32,32),batch_size=-1,device='cpu')
    optimizer = torch.optim.Adam(autoencoder.parameters(),lr=1e-4)
    autoencoder.to(dev)
    epoch_num = 500
    bestloss = torch.inf
    trainpsnr_ls = []
    testpsnr_ls = []
    for epoch in tqdm.trange(epoch_num):
        print(f"Training epoch: {epoch+1}/{epoch_num}")
        trainpsnr_ls += train_one_epoch(autoencoder,trainloader,loss,optimizer,(epoch,epoch_num))
        loss_test, psnr_test = test_one_epoch(autoencoder, testloader, loss, (epoch,epoch_num))
        autoencoder.trained_epoch += 1
        testpsnr_ls.append(psnr_test)
        if loss_test<bestloss:
            bestloss=loss_test
            _=os.remove(ws_root+modelfilename) if 'modelfilename' in dir() else 0
            modelfilename = f'ae_torch/savedmodels_torchprelu/snrdb_{snrdb_train}_bwratio_{bw_ratio:.3f}_{timepref()}.pth.tar'
            with open(ws_root+modelfilename,'wb') as f:
                pickle.dump(autoencoder,f)
        if 'logfilename' in dir():
            os.remove(ws_root+logfilename)
        logfilename = f'ae_torch/savedmodels_torchprelu/psnr_snrdb_{snrdb_train}_bwratio_{bw_ratio:.3f}_{timepref()}.txt'
        with open(ws_root+logfilename,'wb') as f:
            pickle.dump({'train':trainpsnr_ls,'test':testpsnr_ls},f)