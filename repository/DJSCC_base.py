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
ws_root = '/data1/kaiyuan/project/djscc_bcm/'

class Encoder(nn.Module):
    def __init__(self,c):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding=2)),
            ('prelu1', nn.PReLU(num_parameters=16)), # 16*14*14
            ('conv2', nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=2,padding=2)),
            ('prelu2', nn.PReLU(num_parameters=32)), # 80*5*5
            ('conv3', nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)),
            ('prelu3', nn.PReLU(num_parameters=32)), # 50*5*5
            ('conv4', nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)),
            ('prelu4', nn.PReLU(num_parameters=32)),
            ('conv5', nn.Conv2d(in_channels=32,out_channels=c,kernel_size=5,stride=1,padding=2)),
            ('prelu5', nn.PReLU(num_parameters=c)),
        ]))
    def forward(self, x):
        y = self.layers(x)
        return y
    
class Decoder(nn.Module):
    def __init__(self,c):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('convtrans1', nn.ConvTranspose2d(in_channels=c,out_channels=32,kernel_size=5,stride=1,padding=2)),
            ('prelu1', nn.PReLU(num_parameters=32)),
            ('convtrans2', nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)),
            ('prelu2', nn.PReLU(num_parameters=32)),
            ('convtrans3', nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)),
            ('prelu3', nn.PReLU(num_parameters=32)),
            ('convtrans4', nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5,stride=2,padding=2,output_padding=1)),
            ('prelu4', nn.PReLU(16)),
            ('convtrans5', nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=5,stride=2,padding=2,output_padding=1)),
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
        # x = x/torch.norm(x)
        noise = torch.normal(0, self.stddev, size=x.shape)
        noise = noise.to(x.device)
        return x+noise
    
class Autoencoder(nn.Module):
    def __init__(self, snr_db, bw_ratio, input_cwh):
        super(Autoencoder,self).__init__()
        num_featuremap = self.calculate_filter_num(bw_ratio, input_cwh)
        self.encoder = Encoder(num_featuremap)
        self.decoder = Decoder(num_featuremap)
        stddev = 10 ** (-snr_db/20)
        self.channel = NormalizationNoise(stddev)
        self.trained_epoch = 0
    def forward(self, x):
        symbols = self.encoder(x)
        symbols_corrupted = self.channel(symbols)
        x_hat = self.decoder(symbols_corrupted)
        return x_hat
    
    def calculate_filter_num(self, bw_ratio, input_cwh):
        def calculate_conv_outsize(x, kernel_size, stride, padding):
            return torch.floor((x-kernel_size+2*padding)/stride)+1
        dimensions0 = torch.tensor(input_cwh[1:])
        dimensions1 = calculate_conv_outsize(dimensions0,5,2,2)
        dimensions2 = calculate_conv_outsize(dimensions1,5,2,2)
        dimensions3 = calculate_conv_outsize(dimensions2,5,1,2)
        dimensions4 = calculate_conv_outsize(dimensions3,5,1,2)
        dimensions5 = calculate_conv_outsize(dimensions4,5,1,2)
        filter_num = torch.prod(torch.tensor(input_cwh))*bw_ratio/torch.prod(dimensions5)
        if round(filter_num.item())==0:
            msg = "\nBandwidth ratio too high. The minimul achievable bandwidth"+\
                f"ratio is much larger than {bw_ratio}. Set the filter number as 1."
            warnings.warn(message=msg)
        return max(round(filter_num.item()),1)

def timepref():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
def loss(x,y):
    return torch.mean((x-y)**2)
def train_one_epoch(model, loader, loss_func, optimizer, epochinfo):
    model.train()
    device = next(model.parameters()).device
    psnr_ls = []
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = loss_func(x, x_hat)
        loss.backward()
        optimizer.step()
        psnr = 10*torch.log10(1/loss).item()
        if batch_idx%100==0:
            msg = (f"[****Train] "+
                   f"[Epoch {epochinfo[0]+1:2}|{epochinfo[1]}] "+
                   f"[Batch {batch_idx:4}|{len(loader)}] "+
                   f"[Loss {loss.item():.5f}] "+
                   f"[PSNR {psnr:.2f}]")
            print(msg)
        psnr_ls.append(psnr)
    return psnr_ls

@torch.no_grad()
def test_one_epoch(model, loader, loss_func, epochinfo):
    device = next(model.parameters()).device
    loss_ls = []
    for x, _ in loader:
        x = x.to(device)
        x_hat = model(x)
        loss_ls.append(loss_func(x, x_hat))
    loss = sum(loss_ls)/len(loss_ls)
    psnr = 10*torch.log10(1/torch.tensor(loss)).item()
    msg = (f"[----Test]  "+ 
           f"[Epoch {epochinfo[0]+1:2}|{epochinfo[1]}] "+
           f"[Loss {loss:.5f}] "+
           f"[PSNR {psnr:.2f}]\n\n")
    print(msg)
    return loss.item(),psnr

if __name__ == "__main__":
    load_pretrained = False
    bw_ratio = 1/6
    snrdb_train = torch.inf
    
    xtest = datasets.CIFAR10(root=ws_root+'datasets/cifar10',train=False,
                         download=True,transform=transforms.ToTensor())
    xtrain = datasets.CIFAR10(root=ws_root+'datasets/cifar10',train=True,
                            download=True,transform=transforms.ToTensor())
    trainloader = dataloader.DataLoader(dataset=xtrain,batch_size=64,shuffle=True)
    testloader = dataloader.DataLoader(dataset=xtest,batch_size=64,shuffle=True)
    if load_pretrained:
        with open(ws_root+
                  'ae_torch/savedmodels_test/snrdb_inf_bwratio_0.167_2023_11_28_20_18_01.pth.tar'
                  ,'rb') as f:
            autoencoder = pickle.load(f)
    else:
        autoencoder = Autoencoder(snrdb_train,bw_ratio,(3,32,32))
    autoencoder.to('cuda:7')
    optim = torch.optim.Adam(autoencoder.parameters(),lr=1e-4)
    epoch_num = 1000
    bestloss = torch.inf
    trainpsnr_ls = []
    testpsnr_ls = []
    for epoch in tqdm.trange(epoch_num):
        print(f"Training epoch: {epoch+1}/{epoch_num}")
        trainpsnr_ls += train_one_epoch(autoencoder,trainloader,loss,optim,(epoch,epoch_num))
        loss_test, psnr_test = test_one_epoch(autoencoder, testloader, loss, (epoch,epoch_num))
        autoencoder.trained_epoch += 1
        testpsnr_ls.append(psnr_test)
        if loss_test<bestloss:
            bestloss=loss_test
            _=os.remove(ws_root+modelfilename) if 'modelfilename' in dir() else 0
            modelfilename = f'ae_torch/savedmodels_test/snrdb_{snrdb_train}_bwratio_{bw_ratio:.3f}_{timepref()}.pth.tar'
            with open(ws_root+modelfilename,'wb') as f:
                pickle.dump(autoencoder,f)
    logfilename = f'ae_torch/savedmodels_test/psnr_snrdb_{snrdb_train}_bwratio_{bw_ratio:.3f}_{timepref()}.txt'
    with open(ws_root+logfilename,'wb') as f:
        pickle.dump({'train':trainpsnr_ls,'test':testpsnr_ls},f)