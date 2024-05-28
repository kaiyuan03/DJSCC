import torch
import tqdm
import pickle
import os
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from functions import Imageset,timepref,setup_seed
from DJSCC import Autoencoder,train_one_epoch,test_one_epoch,loss
ws_root = '/data1/kaiyuan/project/djscc_bcm/'
snrdb_train=10
bwrs=[0.02,0.045,0.0625,0.095,0.12]
dev='cuda:5'
bwr=bwrs[0]

dev='cuda:6'
bwr=bwrs[1]

dev='cuda:7'
bwr=bwrs[2]

dev='cuda:4'
bwr=bwrs[3]

dev='cuda:7'
bwr=bwrs[4]

epoch_num=30
istest=0
transform = transforms.Compose([transforms.RandomCrop((256,256))])
droot = '/data1/kaiyuan/project/compress_lip/Jianhao_datasets/Image/'
xtrain = Imageset(droot+'train/data/',transform,istest)
xtest = Imageset(droot+'test/data/',transform,istest)
dltrain = DataLoader(xtrain,batch_size=64)
dltest = DataLoader(xtest,batch_size=128)
model = Autoencoder(snrdb_train,bwr,(3,256,256)).to(dev)
bestloss=torch.inf
trainpsnr_ls = []
testpsnr_ls = []
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
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