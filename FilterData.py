from glob import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import time
import numpy as np


def filter_random(augmentations, path, num_augmentations):
    # add initial image (no augmentation)
    filtered_augmentations = [Path(path).stem+".jpg"]
    # remove real image from augmentations list
    augmentations.remove(Path(path).stem+".jpg")

    # get <num_augmentations> augmentations
    picked_augmentations = np.random.choice(augmentations, np.min([num_augmentations, len(augmentations)])).tolist()
    filtered_augmentations.extend(picked_augmentations)
    return filtered_augmentations

def filter_synthetic_only(augmentations, num_augmentations):
    # remove real image from pool
    augmentations.remove(Path(path).stem+".jpg")
    picked_augmentations = np.random.choice(augmentations, np.min([num_augmentations, len(augmentations)])).tolist()
    return picked_augmentations

def filter_real_only(path):
    # only add real image to pool
    filtered_augmentations = [Path(path).stem+".jpg"]
    return filtered_augmentations

if __name__ == "__main__":

    print("******************************")
    print("FILTER DATA")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    #parser.add_argument('--dataset', type = str, default="cocostuff10k", choices = ["ade", "cocostuff10k"])
    parser.add_argument('--data_path', type = str, default="./data/ade_augmented/canny_img2text/")
    parser.add_argument('--images_folder', type = str, default="/images/training/")


    parser.add_argument('--name', type = str, default="default")
    parser.add_argument('--filter_by', type = str, choices=["random", "synthetic_only", "real_only"])
    parser.add_argument('--num_augmentations', type = int, default=4)


    args = parser.parse_args()
    print(f"Parameters: {args}")

    start_time = time.time()

    save_path = args.data_path+"/"+args.name+".txt"
    print(f"Save as: {save_path}")

    # get all real images (no augmentations)
    image_paths = glob(args.data_path+"/"+args.images_folder+"*_000.jpg")
    lines = []

    for path in image_paths: 
        p = Path(path)
        n = p.stem
        n = ("_").join(n.split("_")[:-1])
        augmentations = glob(str(p.parent)+"/"+n+"_*.jpg")
        augmentations = [Path(a).stem+".jpg" for a in augmentations]
        if(args.filter_by == "random"):
            augmentations = filter_random(augmentations, path, args.num_augmentations)
        elif(args.filter_by == "synthetic_only"):
            augmentations = filter_synthetic_only(augmentations, args.num_augmentations)
        elif(args.filter_by == "real_only"):
            augmentations = filter_real_only(path)
        lines.extend(augmentations)

    with open(save_path, 'w') as f:
        for line in lines: 
            f.write(line+"\n")

    all_files = glob(args.data_path+"/"+args.images_folder+"*.jpg")
    print(f"INFO::Done writng to file {save_path} with {len(lines)}/{len(all_files)} images")
    print(f"INFO::Picked on average {(len(lines)/len(image_paths))-1} augmentations.")


    # TODO filter data 
