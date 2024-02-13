from glob import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import time
import numpy as np
import torch
import torchvision
from PIL import Image
from Utils import load_pkl

save_imgs = True
if(save_imgs):
    save_to_uncertainty = "./Debug_uncertainty/"
    os.makedirs(save_to_uncertainty, exist_ok=True)
    save_to_uncertainty_gt = "./Debug_uncertaintyGT/"
    os.makedirs(save_to_uncertainty_gt, exist_ok=True)



totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                                                      #torchvision.transforms.Resize((512, 2048))])
bce = torch.nn.BCEWithLogitsLoss()


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

def filter_al(dictonary, path, num_augmentations):
    real = Path(path).stem+".jpg"
    filtered_augmentations = [real]
    filtered_augmentations.extend(dictonary[real][:num_augmentations])
    return filtered_augmentations

if __name__ == "__main__":

    print("******************************")
    print("FILTER DATA")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--data_path', type = str, default="./data/ade_augmented/canny_img2text/")
    parser.add_argument('--images_folder', type = str, default="/images/training/")
    parser.add_argument('--filter_by', type = str, choices=["random", "synthetic_only", "real_only", "AL"])
    parser.add_argument('--path_to_aldict', type = str)
    parser.add_argument('--num_augmentations', type = int, default=1)


    args = parser.parse_args()
    print(f"Parameters: {args}")

    start_time = time.time()

    if(args.filter_by == "AL"):
        name = f"{Path(args.path_to_aldict).stem}_{args.num_augmentations}.txt"
    else: 
        name = f"{args.filter_by}_{args.num_augmentations}.txt"
    save_path = args.data_path+"/"+name
    print(f"Save as: {save_path}")
    # os.makedirs(save_path)

    # get all real images (no augmentations)
    image_paths = glob(args.data_path+"/"+args.images_folder+"*_0000.jpg")
    lines = []


    for idx, path in enumerate(image_paths): 
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
        elif(args.filter_by == "AL"):
            sorted_dict = load_pkl(args.path_to_aldict)# TODO load sorted dict
            augmentations = filter_al(sorted_dict, path, args.num_augmentations)
        

        lines.extend(augmentations)
        print(f"Image {idx}/{len(image_paths)}")

    with open(save_path, 'w') as f:
        for line in lines: 
            f.write(line+"\n")

    all_files = glob(args.data_path+"/"+args.images_folder+"*.jpg")
    print(f"INFO::Done writng to file {save_path} with {len(lines)}/{len(all_files)} images")
    print(f"INFO::Picked on average {(len(lines)/len(image_paths))-1} augmentations.")


    # TODO filter data 
