
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
    parser.add_argument('--sampling_pool_path', type = str, default="./data/ade_augmented/uncertainty/sampling_pool/")
    parser.add_argument('--save_to', type = str, default="")
    parser.add_argument('--model_path', type=str, default="./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/")
    parser.add_argument('--uncertainty', type=str, choices = ["mcdropout", "lcu", "lmu", "smu", "entropy"], default="mcdropout")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--top_n', type=int, default=1)





    args = parser.parse_args()
    print(f"Parameters: {args}")

    if(args.uncertainty == "entropy"):
        loss = entropy_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.uncertainty == "mcdropout"):
        loss = mcdropout_loss
        args.model_path = args.model_path + "train_model_scripted.pt"
    elif(args.uncertainty == "smu"):
        loss = smu_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.uncertainty == "lmu"):
        loss = lmu_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.uncertainty == "lcu"):
        loss = lcu_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"    
    
    if(args.uncertainty != ""):
        seg_model = torch.jit.load(args.model_path)
        seg_model = seg_model.to(device)
    else: 
        seg_model = None

    # TODO make folder structure
    os.makedirs(args.save_to, exist_ok=True)
    print(f"INFO:: Make directory {args.save_to + annotations_folder}")
    print(f"INFO:: Make directory {args.save_to + images_folder}")
    print(f"INFO:: Make directory {args.save_to + prompts_folder}")

    os.makedirs(args.save_to + annotations_folder, exist_ok=True)
    os.makedirs(args.save_to + images_folder, exist_ok=True)
    os.makedirs(args.save_to + prompts_folder, exist_ok=True)




    # remove initial / so that its no absolute path 
    images_folder = images_folder[1:]
    annotations_folder = annotations_folder[1:]
    prompts_folder = prompts_folder[1:]
    

    # for first comparison
    image_paths = sorted(glob(os.path.join(args.sampling_pool_path, images_folder)+"/*"+images_format))
    real_image_paths = [p for p in image_paths if "_0000"+images_format in p] # get only real images
    # real_image_paths = real_image_paths[:args.n_images]
    np.random.seed(7353)
    real_image_paths = real_image_paths[args.start_idx: args.end_idx]
    synthetic_image_paths_collection = [p for p in image_paths if not ("_0000"+images_format in p)]

    base_prompts_folder = prompts_folder

    """prompts_folder_tmp = glob(os.path.join(args.comparisons[0], prompts_folder)+"/*")[0]
    if(not os.path.isdir(prompts_folder_tmp)):
        prompts_folder = os.path.join(args.comparisons[0], prompts_folder)
    else: 
        prompts_folder = prompts_folder_tmp"""
        

    for i,p in enumerate(real_image_paths): 
        if(i % 1000):
            print(f"INFO::Image {i}/{len(real_image_paths)}")
        dp_name = Path(p).stem

       
        real_path = f"{os.path.join(args.sampling_pool_path, images_folder, dp_name)}{images_format}"
        synthetic_img_paths = sorted(glob(os.path.join(Path(real_path).parent,  "_".join(Path(real_path).stem.split("_")[:-1])+"_*"+images_format)))[1:]

        uncertainties = []

        for synthetic_img_path in synthetic_img_paths:
            # compute uncertainties
            compute_img,_ = resize_transform(Image.open(synthetic_img_path)) # resize shortest edge to 512
            compute_img = np.array(compute_img)
            if(len(compute_img.shape) != 3):
                compute_img = np.stack([compute_img,compute_img,compute_img], axis = 0).transpose(1,2,0)
            compute_img = totensor_transform(compute_img).unsqueeze(0)
            uncertainty, _ = loss(compute_img.to(device), None, None, seg_model, visualize = False)
            uncertainties.append(float(uncertainty.detach().cpu().numpy()[0]))

        # sort by uncertainties
        synthetic_img_paths = [x for _, x in sorted(zip(uncertainties, synthetic_img_paths))][0:args.top_n]
        uncertainties = sorted(uncertainties)

        # copy real image
        real_dst_img = f"{os.path.join(args.save_to, images_folder, dp_name)}{images_format}"
        print(f"INFO::Copy real image from {real_path} to {real_dst_img}")
        shutil.copy(real_path, real_dst_img)

        # copy real annotation 
        real_dst_ann = f"{os.path.join(args.save_to, annotations_folder, dp_name)}{annotations_format}"
        real_src_ann = f"{os.path.join(args.sampling_pool_path, annotations_folder, dp_name)}{annotations_format}"
        print(f"INFO::Copy real annotation from {real_src_ann} to {real_dst_ann}")
        shutil.copy(real_src_ann, real_dst_ann)
        
        # copy synthetic images
        for syn_src_img in synthetic_img_paths:
            syn_dst_img = f"{os.path.join(args.save_to, images_folder, Path(syn_src_img).stem)}{images_format}"
            print(f"INFO::Copy synthetic image from {syn_src_img} to {syn_dst_img}")
            shutil.copy(syn_src_img, syn_dst_img)

            syn_dst_ann = f"{os.path.join(args.save_to, annotations_folder, Path(syn_src_img).stem)}{annotations_format}"
            syn_src_ann = f"{os.path.join(args.sampling_pool_path, annotations_folder, Path(syn_src_img).stem)}{annotations_format}"
            print(f"INFO::Copy synthetic annotation from {syn_src_ann} to {syn_dst_ann}")
            shutil.copy(syn_src_ann, syn_dst_ann)



    # copy prompts
    prompts_src = os.path.join(args.sampling_pool_path, prompts_folder)
    prompts_dst = os.path.join(args.save_to, prompts_folder)
    print(f"INFO::Copy all prompts from {prompts_src} to {prompts_dst}")
    shutil.copytree(prompts_src, prompts_dst)


      

       









