import torch
import tqdm
import pickle
import os
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from functions import Imageset,timepref,setup_seed
from DJSCCv2 import Autoencoder,train_one_epoch,test_one_epoch#,loss
ws_root = '/data1/kaiyuan/project/djscc_bcm/'
# snrdb_train=20
# bwrs=[0.025431315104166668, 0.03814697265625, 0.050862630208333336, 0.0762939453125, 0.10172526041666667]
# snrdbs_train = [2,5,7,10,12]
snrdb_train = 26.859190196177387

dev='cuda:7'
bwr = 0.01

dev='cuda:6'
bwr = 0.02


epoch_num=100
istest=0
transform = transforms.Compose([transforms.RandomCrop((256,256))])
# transform = transforms.Compose([transforms.Resize((256,256))])
droot = '/data1/kaiyuan/project/compress_lip/Jianhao_datasets/Image/'
# droot = '/data1/kaiyuan/project/compress_lip/Tuning_Model/Data/bird_split_resized/'
xtrain = Imageset(droot+'train/data/',transform,istest)
xtest = Imageset(droot+'test/data/',transform,istest)
dltrain = DataLoader(xtrain,batch_size=64,num_workers=64)
dltest = DataLoader(xtest,batch_size=128,num_workers=64)
model = Autoencoder(snrdb_train,bwr,(3,256,256)).to(dev)
bestloss=torch.inf
trainpsnr_ls = []
testpsnr_ls = []
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
# loss = loss.to(dev)
loss = torch.nn.MSELoss()
for epoch in tqdm.trange(epoch_num):
    print(f"Training epoch: {epoch+1}/{epoch_num}")
    trainpsnr_ls += train_one_epoch(model,dltrain,loss,optimizer,(epoch,epoch_num))
    setup_seed(10,10,10,10)
    loss_test, psnr_test = test_one_epoch(model, dltest, loss, (epoch,epoch_num))
    seed = int(time.time()*10%1000)
    setup_seed(seed,seed,seed,seed)
    model.trained_epoch += 1
    testpsnr_ls.append(psnr_test)
    if loss_test<bestloss:
        bestloss=loss_test
        _=os.remove(ws_root+modelfilename) if 'modelfilename' in dir() else 0
        modelfilename = f'savedmodels/model_snrdb_{snrdb_train}_bwr_{bwr:.3f}_{timepref()}.pth.tar'
        with open(ws_root+modelfilename,'wb') as f:
            pickle.dump(model,f)
    if 'logfilename' in dir():
        os.remove(ws_root+logfilename)
    logfilename = f'savedmodels/history_{snrdb_train}_bwratio_{bwr:.3f}_{timepref()}.txt'
    with open(ws_root+logfilename,'wb') as f:
        pickle.dump({'train':trainpsnr_ls,'test':testpsnr_ls},f)