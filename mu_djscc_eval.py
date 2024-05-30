import os,sys
sys.path.append('../')
import pickle
from DJSCC import *
import torch
from functions import *
from compressai.ops import compute_padding
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import warnings
from pytorch_msssim import ms_ssim
def save_variables(filename):
    with open(filename, 'wb') as f:
        # Filter out unpickleable objects
        variables = {k: v for k, v in globals().items() if not k.startswith('__') and not isinstance(v, types.ModuleType) and not isinstance(v, types.FunctionType)}
        pickle.dump(variables, f)
        
dev = 'cpu'

Nt = 2
U = 2
torch.manual_seed(10)
H = torch.Tensor([[1,0.1],[0.3,1]])
P = [0.6846,0.2837]
p_noise = [0.3**2,0.15**2]
mode = "ZF"
W_dict = {
    "ZF":F.normalize(H@torch.inverse(H.T@H),dim=0),
    "OPT":torch.Tensor([[0.9976,-0.2448],[-0.0693,0.9696]]),
    "Random":F.normalize(torch.randn(Nt,U),dim=0),
    "MRC":F.normalize(H,dim=0)
}
sinr_downlink(W_dict,mode,P,H,p_noise)
mu_miso = CHN_MU_MISO(W_dict[mode],H,P,p_noise)

# with open('../savedmodels/model_snrdb_10_bwr_0.120_2024_01_23_02_38_04.pth.tar','rb') as f:
#     model0=pickle.load(f)
model = Autoencoder(5,0.120,(3,256,256))
# model.load_state_dict(model0.state_dict())
# model = model.to(dev)
dset = Imageset('../compress_lip/Jianhao_datasets/CLIC/',transforms.Resize((256,256)),1,1,1)
dloaders = []
for u in range(U):
    dloaders.append(dataloader.DataLoader(dset,1,True))

torch.manual_seed(10)

perf_stats = [{"mse":RUN_STAT("mse"),"psnr":RUN_STAT("psnr"),"ssim":RUN_STAT("ssim")}
                 for _ in range(U)]
with torch.no_grad():
    for x in zip(*dloaders):
        # x is the list of all users' images
        # next: process the loaded images for the encoder
        x = [xu.to(dev) for xu in x]
        outenc = [model.encoder(xu) for xu in x]
        outchn = mu_miso(outenc)
        xhat = [torch.randn(1)*model.decoder(outchn_) for outchn_ in outchn]
        for xu,xhatu,perf_stat in zip(x,xhat,perf_stats):
            mse = torch.mean((xu-xhatu)**2)
            print("mse: ",u,mse)
            mse = torch.mean((xu-xhatu)**2)
            psnr = 10*np.log10(1/mse.item())
            ssim = ms_ssim(xu,xhatu,data_range=1.)
            perf_stat["mse"](mse)
            perf_stat["psnr"](psnr)
            perf_stat["ssim"](ssim)

# with open("test.txt",'wb') as f:
#     pickle.dump(perf_stats,f)

save_variables("test.txt")
