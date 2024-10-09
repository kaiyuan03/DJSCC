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
"""
bwr (0.00000~0.01042), feature map num: 1
bwr (0.01042~0.02083), feature map num: 2
bwr (0.02083~0.03125), feature map num: 3
bwr (0.03125~0.04167), feature map num: 4
bwr (0.04167~0.05208), feature map num: 5
bwr (0.05208~0.06250), feature map num: 6
bwr (0.06250~0.07292), feature map num: 7
bwr (0.07292~0.08333), feature map num: 8
bwr (0.08333~0.09375), feature map num: 9
bwr (0.09375~0.10417), feature map num: 10
"""

# snrdb_train=20
bwrs=[0.010172526041666666, 0.020345052083333332, 0.030517578125, 0.040690104166666664, 0.050862630208333336]
# bwr = 0.051
# snrdbs_train = [-3.140809802515239,
#                 6.859190197484761,
#                 16.859190197484757,
#                 26.859190197484757,
#                 36.85919019748476]
snrdb_train = 26.859190197484757
dev='cuda:4'
bwr = bwrs[0]


epoch_num=20
istest=0
transform = transforms.Compose([transforms.RandomCrop((256,256))])
droot = '/data1/kaiyuan/project/compress_lip/Jianhao_datasets/Image/'
xtrain = Imageset(droot+'train/data/',transform,istest)
xtest = Imageset(droot+'test/data/',transform,istest)
dltrain = DataLoader(xtrain,batch_size=64,num_workers=32)
dltest = DataLoader(xtest,batch_size=128,num_workers=32)
modelname = 'model_snrdb_26.859190197484757_bwr_0.051_2024_09_05_13_24_16.pth.tar'
modelpath = './savedmodels/'+modelname
with open(modelpath,'rb') as f:
        model0=pickle.load(f)
model = Autoencoder(snrdb_train,model0.bwr,(3,256,256))
model.load_state_dict(model0.state_dict())
model = model.to(dev)
# model = Autoencoder(snrdb_train,bwr,(3,256,256)).to(dev)
bestloss=torch.inf
trainpsnr_ls = []
testpsnr_ls = []
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
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