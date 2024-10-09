import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from kaijsra.utils.bpg_ldpc import bpg_ldpc_transmission
from kaijsra.utils.channel_coding.ldpc import LDPC_modulation_codec
import numpy as np
from kaijsra import dsetroot
import pickle
import sionna as sn

# fix SNR
dataset_path = dsetroot+"CLIC/"
image_paths = [dataset_path+ image for image in os.listdir(dataset_path)]
modulation_bps = 2
snr_db = 10
compression_values = list(range(35,50,2))
k = int(4096*3/4)
n = 4096
ldpc_transceiver = LDPC_modulation_codec(k,n,modulation_bps,snr_db)
results_over_compression_value = {
    "ldpc_k":k,
    "ldpc_n":n,
    "snr_db":snr_db,
    "modulation_bps":modulation_bps,
    "dataset_path":dataset_path,
    "compression_values":compression_values
}
sn.config.xla_compat=True
for compression_value in compression_values:
    eval_results = bpg_ldpc_transmission(image_paths,
                                        compression_value,
                                        ldpc_transceiver,
                                        run_mode="xla")
    results_over_compression_value[compression_value] = eval_results
sn.config.xla_compat=False

avg_psnrs = []
avg_bwrs = []
for compression_value in compression_values:
    avg_psnrs.append(np.mean(results_over_compression_value[compression_value]['psnrs']))
    avg_bwrs.append(np.mean(results_over_compression_value[compression_value]['bwrs']))
    print(np.mean(results_over_compression_value[compression_value]['psnrs']),
          np.mean(results_over_compression_value[compression_value]['bwrs']))
    

with open('../evalresults/CLIC_ldpc_fix_snr_3_4.pkl','wb') as f:
    pickle.dump(results_over_compression_value,f)
