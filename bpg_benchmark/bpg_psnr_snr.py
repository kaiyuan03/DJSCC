"""
Kodak：
    LDPC rate=2/3, 4QAM, compression value=41, BWR=0.0625
    completely distortion PSNR: 6.777101135874024
    perfect transmission PSNR: 30.01488457319007
CLIC:
    LDPC rate=3/4, 4QAM, compression value=37, BWR=0.0625
    completely distortion PSNR: 5.815777408856423
    perfect transmission PSNR: 34.21529261960708
"""
import os
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# dataset_name = "Kodak"
# ldpc_rate = 2/3
# snr_dbs = [3.375, 3.4375, 3.47, 3.5, 3.5625, 3.625, 4,8,12]
# compression_value = 41
# channel_num = 10

os.environ['CUDA_VISIBLE_DEVICES'] = "6"
dataset_name = "CLIC"
ldpc_rate = 3/4
snr_dbs = [4.5, 4.5625, 4.59375, 4.625, 4.6875, 4.75,6,8,12]
# 3 completely corrupted, 5 perfect transmission.
compression_value = 37
channel_num = 10

from kaijsra.simulation import bpg_ldpc_transmission
from kaijsra.utils.channel_coding.ldpc import LDPC_modulation_codec
from kaijsra import dsetroot
import pickle
# import sionna as sn


dataset_path = dsetroot+f"{dataset_name}/"
image_paths = [dataset_path+ image for image in os.listdir(dataset_path)]
image_paths = image_paths*channel_num
modulation_bps = 2
k = int(4096*ldpc_rate)
n = 4096
ldpc_transceiver = LDPC_modulation_codec(k,n,modulation_bps,0)
results_over_snr = {
    "ldpc_k":k,
    "ldpc_n":n,
    "snr_dbs":snr_dbs,
    "modulation_bps":modulation_bps,
    "dataset_path":dataset_path,
    "compression_value":compression_value
}
# sn.config.xla_compat=True
print("finished init")
for snr_db in snr_dbs:
    ldpc_transceiver.set_snr_db(snr_db)
    print(ldpc_transceiver.snr_db,ldpc_transceiver.noice_var)
    eval_results = bpg_ldpc_transmission(image_paths,
                                        compression_value,
                                        ldpc_transceiver,
                                        run_mode="eager",
                                        process_num=30)
    results_over_snr[snr_db] = eval_results
# sn.config.xla_compat=False

file_name = f"{dataset_name}_comp_{compression_value}"
file_name += f"_ldpc_fix_snr_{ldpc_rate:.2}"
file_name += f"_mod_{modulation_bps}_chn_{channel_num}.pkl"
with open(f'../evalresults/{file_name}','wb') as f:
    pickle.dump(results_over_snr,f)