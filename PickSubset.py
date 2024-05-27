
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

import random

# generates differnet subsets of input_list which are overlapping (draw random samples with replacement). 
# All subsets have the same size as input_list 
def bootstrap_aggregating(input_list, n):
    subsets = []
    for i in range(n):
        subset = np.random.choice(input_list, size = (len(input_list),), replace = True)
        subsets.append(subset)
    return subsets


def split_list_random(input_list, n):
    if n <= 0:
        raise ValueError("Number of subsets must be greater than 0")
    
    # Shuffle the input list to randomize the order
    random.shuffle(input_list)
    
    # Calculate the size of each subset
    k, m = divmod(len(input_list), n)
    
    # Split the list into n parts
    subsets = [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    return subsets

if __name__ == "__main__":

    print("******************************")
    print("PICK DATA AS TRAINING SUBSETS")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--data_path', type = str, default="./data/ade/ADEChallengeData2016/")
    parser.add_argument('--save_to', type = str, default="./data/ade/ADEChallengeData2016_10percent/")
    parser.add_argument('--percentage', type=float, default=0.1)
    parser.add_argument('--cross_val_splits', type=int, default=1)


    args = parser.parse_args()
    print(f"Parameters: {args}")

    # remove initial '/' so that its no absolute path 
    images_folder = images_folder[1:]
    annotations_folder = annotations_folder[1:]
    prompts_folder = prompts_folder[1:]

    real_image_paths = sorted(glob(os.path.join(args.data_path, images_folder)+"/*"+images_format))


    if((args.percentage > 0) and (args.cross_val_splits > 1)):
        print(f"ERROR::Can either generate cross validation splits or a subset of the training data, not both. Please set either args.percentage to -1 or args.corss_val_splits to 1.")
        import sys 
        sys.exit(-1)

    if(args.cross_val_splits > 1):
        save_paths = [args.save_to+"/"+str(i)+"/" for i in range(args.cross_val_splits)]
        np.random.seed(7353)
        subset_paths = bootstrap_aggregating(real_image_paths, args.cross_val_splits)
    else: 
        save_paths = [args.save_to+"/"]

        # get img paths of subset
        np.random.seed(7353)
        subset_paths = [np.random.choice(real_image_paths, size=(int(len(real_image_paths)*args.percentage),))] #[args.start_idx: args.end_idx]     

    for save_to, real_image_paths in zip(save_paths, subset_paths):
        # make folder structure
        os.makedirs(save_to, exist_ok=True)
        print(f"INFO:: Make directory {save_to + annotations_folder}")
        print(f"INFO:: Make directory {save_to + images_folder}")
        print(f"INFO:: Make directory {save_to + prompts_folder}")

        os.makedirs(save_to + annotations_folder, exist_ok=True)
        os.makedirs(save_to + images_folder, exist_ok=True)
        # os.makedirs(save_to + prompts_folder, exist_ok=True)

        
           
        for i,p in enumerate(real_image_paths): 
            if((i % 1000)==0):
                print(f"INFO::Image {i}/{len(real_image_paths)}. Source folder: {args.data_path}. Dest folder: {save_to}")
            dp_name = Path(p).stem

            real_path = f"{os.path.join(args.data_path, images_folder, dp_name)}{images_format}"
            
            # copy real image
            real_dst_img = f"{os.path.join(save_to, images_folder, dp_name)}{images_format}"
            shutil.copy(real_path, real_dst_img)

            # copy real annotation 
            real_dst_ann = f"{os.path.join(save_to, annotations_folder, dp_name)}{annotations_format}"
            real_src_ann = f"{os.path.join(args.data_path, annotations_folder, dp_name)}{annotations_format}"
            shutil.copy(real_src_ann, real_dst_ann)

        # copy prompts
        prompts_src = os.path.join(args.data_path, prompts_folder)
        prompts_dst = os.path.join(save_to, prompts_folder)
        print(f"INFO::Copy all prompts from {prompts_src} to {prompts_dst}")
        shutil.copytree(prompts_src, prompts_dst)
        print(f"INFO::Done. Saved data subset to {save_to}")


        

        









