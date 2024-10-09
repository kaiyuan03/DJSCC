from DJSCC import Autoencoder
import pickle
import torch
import tqdm
from PIL import Image
from compressai.ops import compute_padding
from torch.nn import functional as F
from torchvision import transforms
import os
import numpy as np
from collections import defaultdict
from pytorch_msssim import ms_ssim
from functions import timepref

@ torch.no_grad()
def eval_model(modelpath, test_snrdb,droot,fnames,dev,info):
    with open(modelpath,'rb') as f:
        model0=pickle.load(f)
    model = Autoencoder(test_snrdb,model0.bwr,(3,256,256))
    model.load_state_dict(model0.state_dict())
    model = model.to(dev)
    msels = []
    ssimls = []
    psnrls = []
    for fname in tqdm.tqdm(fnames,desc=f"[{dev}][{dataset}]"+
                           f"[Model {info['mid']+1}|{info['msum']}]"+
                           f"[SNR {info['sid']+1}|{info['sum']}]"):
        img = Image.open(droot+fname)
        x = transforms.ToTensor()(img)
        x=x.unsqueeze(0).to(dev)
        h,w=x.size(2),x.size(3)
        pad,unpad=compute_padding(h,w,min_div=2**6)
        x_padded = F.pad(x,pad,mode='constant',value=0)
        xhat = model(x_padded)
        x_unpad = F.pad(xhat,unpad)
        mse = torch.mean((x-x_unpad)**2)
        psnr = 10*np.log10(1/mse.item())
        ssim = ms_ssim(x,x_unpad,data_range=1.)
        msels.append(mse.item())
        ssimls.append(ssim.item())
        psnrls.append(psnr)
    return msels, ssimls, psnrls
channel = 1
dev='cuda:6'
print("device: ", dev)
modelnames = [
            #   'model_snrdb_10_bwr_0.020_2024_01_21_16_36_54.pth.tar',
            #   'model_snrdb_10_bwr_0.045_2024_01_21_16_43_30.pth.tar',
            #   'model_snrdb_10_bwr_0.062_2024_01_21_16_58_29.pth.tar',
            #   'model_snrdb_10_bwr_0.095_2024_01_21_14_41_27.pth.tar',
            #   'model_snrdb_10_bwr_0.120_2024_01_23_02_38_04.pth.tar',
            # 'model_snrdb_10_bwr_0.051_2024_06_28_06_33_22.pth.tar',
            # 'model_snrdb_20_bwr_0.051_2024_06_29_00_58_38.pth.tar'
            # D2JSCC SNR_train match SNR_test
            'model_snrdb_2_bwr_0.062_2024_08_21_07_06_29.pth.tar',
            'model_snrdb_5_bwr_0.062_2024_08_21_07_03_40.pth.tar',
            'model_snrdb_7_bwr_0.062_2024_08_21_03_57_23.pth.tar',
            'model_snrdb_10_bwr_0.062_2024_01_21_16_58_29.pth.tar',
            'model_snrdb_12_bwr_0.062_2024_08_21_06_59_49.pth.tar',
              ]
test_snrdbs = [2,5,7,10,12]
# test_snrdbs = [2]
# test_snrdbs = [5]


dataset = 'CLIC'
# dataset = 'Kodak'
droot = '../compress_lip/Jianhao_datasets/'+dataset+'/'
fnames = os.listdir(droot)*channel

# test
test = 0
if test:
    fnames = fnames[:3]
    modelnames = modelnames[:]
    test_snrdbs = test_snrdbs[:]
    channel = 1

results={}
results['test_snrdbs']=test_snrdbs
results['channel'] = channel
results['dataset'] = dataset

for mid, modelname in enumerate(modelnames):
    modelres = defaultdict(list)
    for sid,test_snrdb in enumerate(test_snrdbs):
        info={'sid':sid,'sum':len(test_snrdbs),
              'mid':mid,'msum':len(modelnames)}
        modelpath = './savedmodels/'+modelname
        msels, ssimls, psnrls = eval_model(modelpath,test_snrdb,droot,fnames,dev,info)
        mse = np.mean(msels)
        ssim = np.mean(ssimls)
        psnr1 = np.mean(psnrls)
        psnr = 10*np.log10(1/mse)
        modelres['mse'].append(mse)
        modelres['psnr'].append(psnr)
        modelres['ssim'].append(ssim)
        modelres['psnr1'].append(psnr1)
    results[modelname] = modelres
print(results)
resname = f"{dataset}_evalres_{timepref()}.pkl"
with open('evalresults/'+resname,'wb') as f:
    pickle.dump(results,f)

"""
model information
manually specify model name
passed information
compute msssim, test program passed.

results dictionary structrue:
snrdb_test: list
dataset: string
channel: integer
modelpath1: dictionary
    mse: [mse_under_snr1, mse_under_snr2, ...]
    psnr: [psnr_unser_snr1, psnr_under_snr2, ...]
    ssim: [ssim_under_snr1, ssim_under_snr2, ...]
modelpath2: dictionary
    mse: [mse_under_snr1, mse_under_snr2, ...]
    psnr: [psnr_unser_snr1, psnr_under_snr2, ...]
    ssim: [ssim_under_snr1, ssim_under_snr2, ...]
"""