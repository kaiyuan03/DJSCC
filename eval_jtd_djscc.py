from DJSCCv2 import Autoencoder
import pickle
import torch
import tqdm
from compressai.ops import compute_padding
from torch.nn import functional as F
from torchvision import transforms
import os
import numpy as np
from collections import defaultdict
from pytorch_msssim import ms_ssim
from functions import timepref
from torch.utils.data import DataLoader
from torchvision import datasets
from kaijsra import WSROOT
from collections import defaultdict
from kaijsra.utils.djscc import parse_snrdb_fmap

@ torch.no_grad()
def eval_model(compress_model_path, 
               classify_model_path,
               test_snrdb,
               dataloader,
               compress_dev,
               classify_dev,
               info):
    with open(compress_model_path,'rb') as f:
        model0=pickle.load(f)
    print("bwr: ", model0.bwr)
    compress_model = Autoencoder(test_snrdb,model0.bwr,(3,256,256))
    compress_model.load_state_dict(model0.state_dict())
    compress_model = compress_model.to(compress_dev)
    classify_model = torch.load(classify_model_path)
    classify_model = classify_model.to(classify_dev)
    lossf=torch.nn.CrossEntropyLoss(reduce=False)
    results = defaultdict(list)
    results['compress_model_path'] = compress_model_path
    results['classify_model_path'] = classify_model_path
    results['test_snrdb'] = test_snrdb
    results['compress_dev'] = compress_dev
    results['classify_dev'] = classify_dev
    results['info'] = info
    for x,y in tqdm.tqdm(dataloader,
                         desc=\
                         f"[{compress_dev}]" + 
                         f"[{classify_dev}]" + 
                         f"[Model {info['model_index']+1}|"+
                         f"{info['model_sum']}]" + 
                         f"[SNR {test_snrdb:6.2f}]"):
        x = x.to(compress_dev)
        y = y.to(classify_dev)
        h,w=x.size(2),x.size(3)
        pad,unpad=compute_padding(h,w,min_div=2**6)
        x_padded = F.pad(x,pad,mode='constant',value=0)
        xhat = compress_model(x_padded)
        x_unpad = F.pad(xhat,unpad)
        # process data related performance metrics
        mse_batch = torch.mean((x-x_unpad)**2,dim=(1,2,3))
        results['mses'] += mse_batch.tolist()
        psnr_batch = -10*np.log10(mse_batch.tolist())
        results['psnrs'] += psnr_batch.tolist()
        ssim_batch = ms_ssim(x,x_unpad,data_range=1.,size_average=False)
        results['ssims'] += ssim_batch.tolist()
        # process task related performance metrics
        x_unpad = x_unpad.to(classify_dev)
        llh = classify_model(x_unpad)
        _,yhat=torch.max(llh,1)
        num_top1_error = torch.count_nonzero(yhat-y).item()
        results['num_top1_error'].append(num_top1_error)
        _,predict_top5 = torch.topk(llh,5,1)
        diff_top5 = predict_top5-torch.reshape(y,(-1,1))
        num_accurate_top5 = torch.count_nonzero(diff_top5,1)-4
        num_error_top5 = torch.sum(num_accurate_top5).item()
        results['num_top5_error'].append(num_error_top5)
        cross_entropies = lossf(llh,y)
        results['cross_entropies'] += cross_entropies.tolist()
        results['top1_accuracy'] = 1-sum(results['num_top1_error'])\
            /len(results['cross_entropies'])
        results['top5_accuracy'] = 1-sum(results['num_top5_error'])\
            /len(results['cross_entropies'])
        results['running_mse'] = np.mean(results['mses'])
        results['runnin_psnr'] = np.mean(results['psnrs'])
        results['running_ssim'] = np.mean(results['ssims'])
        results['running_cross_entropy'] = np.mean(results['cross_entropies'])
    return results

channel = 1
compress_dev = 'cuda:6'
classify_dev = 'cuda:7'
modelnames = [
    'model_snrdb_26.859190196177387_fmap_1_2024_09_11_10_10_04.pth.tar',
    'model_snrdb_26.859190196177387_fmap_2_2024_09_11_10_57_52.pth.tar',
    'model_snrdb_26.859190196177387_fmap_3_2024_09_05_10_38_56.pth.tar',
    'model_snrdb_26.859190196177387_fmap_4_2024_09_05_11_47_30.pth.tar',
    'model_snrdb_26.859190197484757_fmap_5_2024_09_05_13_24_16.pth.tar'
]


# 在bird的测试集上做测试
transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()])
dataset=datasets.ImageFolder(
    WSROOT+'/classification/data/images/test/',
    transform)
dataloader=DataLoader(dataset,batch_size=512,num_workers=64)
# test
classify_model_path = WSROOT+\
    '/classification/models/classification/'+\
    'resnet152/caltech_birds_resnet152_full1.pth'
test = 0
if test:
    modelnames = modelnames[:]
    channel = 1

results={}
results['channel'] = channel

for mid, modelname in enumerate(modelnames):
    # 根据模型名字自动选择match的test SNR
    test_snrdb = parse_snrdb_fmap(modelname)
    info={'model_index':mid,'model_sum':len(modelnames)}
    compress_model_path = './savedmodels/'+modelname
    # 返回每张图片的评估结果
    model_results = eval_model(compress_model_path,
                                        classify_model_path,
                                        test_snrdb,
                                        dataloader,
                                        compress_dev,
                                        classify_dev,
                                           info)
    results[modelname] = model_results
# print(results)
resname = f"djscc_jtd_evalres_{timepref()}.pkl"
with open('evalresults/'+resname,'wb') as f:
    pickle.dump(results,f)
