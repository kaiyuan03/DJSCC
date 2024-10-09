import os
from kaijsra import timepref
from kaijsra import WSROOT
import pprint
import pickle
image_shape = (256,256)
image_folder = WSROOT+\
    'compress_lip/Tuning_Model/Data/bird_split_resized/test/'
image_paths = [image_folder+x for x in os.listdir(image_folder)]
ber = 1e-8
process_num = 70
classify_model_path = WSROOT+\
    'classification/models/classification/'+\
    'resnet152/caltech_birds_resnet152_full1.pth'
classify_dev = "cuda:7"
trans_dev = "cuda:6"
from kaijsra.simulation import BSC_BPG
bsc_bpg = BSC_BPG(
    image_paths,
    process_num,
    classify_model_path,
    classify_dev,
    image_shape,
    trans_dev)
compression_values = list(range(20,52))
# compression_values = [51]
result_all = {}
progress_info = {
    "exp_num":len(compression_values),
    "exp_idx":1,
}
for compression_value in compression_values:
    result = bsc_bpg.run(compression_value,ber,progress_info)
    progress_info["exp_idx"] += 1
    result['compression_value'] = compression_value
    result_all[compression_value] = result

result_all['compression_values'] = compression_values
result_all['image_folder'] = image_folder
result_all['classify_model_path'] = classify_model_path
result_all['ber'] = ber

file_path = f"/data1/kaiyuan/project/djscc_bcm/"+\
            f"bpg_benchmark/results/saved_{timepref()}.pkl"
with open(file_path, "wb") as f:
    pickle.dump(result_all, f)
