
import os 
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path 
import argparse
import numpy as np
import torch
import shutil

from ade_config import images_folder, annotations_folder, prompts_folder, annotations_format, images_format, prompts_format, palette
from Utils import read_txt, index2color_annotation, device, resize_transform, totensor_transform
from Uncertainties import loss_brightness, entropy_loss, mcdropout_loss, smu_loss, lmu_loss, lcu_loss


if __name__ == "__main__":

    print("******************************")
    print("VISUALIZE SAMPLING POOL")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--data_path', type = str, default="./data/ade/ADEChallengeData2016/")
    parser.add_argument('--save_to', type = str, default="./data/ade/ADEChallengeData2016_10percent/")
    parser.add_argument('--percentage', type=int, default=0.1)





    args = parser.parse_args()
    print(f"Parameters: {args}")

    # TODO make folder structure
    os.makedirs(args.save_to, exist_ok=True)
    print(f"INFO:: Make directory {args.save_to + annotations_folder}")
    print(f"INFO:: Make directory {args.save_to + images_folder}")
    print(f"INFO:: Make directory {args.save_to + prompts_folder}")

    os.makedirs(args.save_to + annotations_folder, exist_ok=True)
    os.makedirs(args.save_to + images_folder, exist_ok=True)
    # os.makedirs(args.save_to + prompts_folder, exist_ok=True)

    # remove initial '/' so that its no absolute path 
    images_folder = images_folder[1:]
    annotations_folder = annotations_folder[1:]
    prompts_folder = prompts_folder[1:]
    

    # for first comparison
    real_image_paths = sorted(glob(os.path.join(args.data_path, images_folder)+"/*"+images_format))
    np.random.seed(7353)
    real_image_paths = np.random.choice(real_image_paths, size=(int(len(real_image_paths)*args.percentage),)) #[args.start_idx: args.end_idx]        

    for i,p in enumerate(real_image_paths): 
        if((i % 1000)==0):
            print(f"INFO::Image {i}/{len(real_image_paths)}. Source folder: {args.data_path}. Dest folder: {args.save_to}")
        dp_name = Path(p).stem

        real_path = f"{os.path.join(args.data_path, images_folder, dp_name)}{images_format}"
        
        # copy real image
        real_dst_img = f"{os.path.join(args.save_to, images_folder, dp_name)}{images_format}"
        shutil.copy(real_path, real_dst_img)

        # copy real annotation 
        real_dst_ann = f"{os.path.join(args.save_to, annotations_folder, dp_name)}{annotations_format}"
        real_src_ann = f"{os.path.join(args.data_path, annotations_folder, dp_name)}{annotations_format}"
        shutil.copy(real_src_ann, real_dst_ann)

    # copy prompts
    prompts_src = os.path.join(args.data_path, prompts_folder)
    prompts_dst = os.path.join(args.save_to, prompts_folder)
    print(f"INFO::Copy all prompts from {prompts_src} to {prompts_dst}")
    shutil.copytree(prompts_src, prompts_dst)
    print(f"INFO::Done. Saved data subset to {args.save_to}")


      

       









