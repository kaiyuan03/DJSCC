from PIL import Image
from kaijsra import WSROOT
import os
image_folder = WSROOT+\
    'compress_lip/Tuning_Model/Data/bird_split_resized/train/'
image_names = os.listdir(image_folder)

def fix_image_srgb_profile(file_name):
    img = Image.open(image_folder+file_name)
    img.save("./tmp/"+file_name, icc_profile=None)

import tqdm
for image_name in tqdm.tqdm(image_names):
    fix_image_srgb_profile(image_name)
print("done")