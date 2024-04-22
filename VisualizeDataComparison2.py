
import os 
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path 
import argparse
import numpy as np
import torch

from ade_config import images_folder, annotations_folder, prompts_folder, annotations_format, images_format, prompts_format, palette
from Utils import read_txt, index2color_annotation, device, resize_transform, totensor_transform
from Uncertainties import loss_brightness, entropy_loss, mcdropout_loss, smu_loss, lmu_loss, lcu_loss


if __name__ == "__main__":

    print("******************************")
    print("VISUALIZE COMPARISON")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--comparisons', type = str, nargs="+", default=["./data/ade_augmented/controlnet1.1/visualization/"])
    parser.add_argument('--n_images', type = int, default=20)
    parser.add_argument('--save_to', type = str, default="")
    parser.add_argument('--model_path', type=str, default="./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/")
    parser.add_argument('--uncertainty', type=str, choices = ["", "mcdropout", "lcu", "lmu", "smu", "entropy"], default="mcdropout")


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

    os.makedirs(args.save_to, exist_ok=True)

    size = 7

    # remove initial / so that its no absolute path 
    images_folder = images_folder[1:]
    annotations_folder = annotations_folder[1:]
    prompts_folder = prompts_folder[1:]
    

    # for first comparison
    image_paths = sorted(glob(os.path.join(args.comparisons[0], images_folder)+"/*"+images_format))
    real_image_paths = [p for p in image_paths if "_0000"+images_format in p] # get only real images
    # real_image_paths = real_image_paths[:args.n_images]
    np.random.seed(7353)
    real_image_paths = np.random.choice(real_image_paths, size=(args.n_images,))
    synthetic_image_paths = [p for p in image_paths if not ("_0000"+images_format in p)]

    base_prompts_folder = prompts_folder

    prompts_folder_tmp = glob(os.path.join(args.comparisons[0], prompts_folder)+"/*")[0]
    if(not os.path.isdir(prompts_folder_tmp)):
        prompts_folder = os.path.join(args.comparisons[0], prompts_folder)
    else: 
        prompts_folder = prompts_folder_tmp
        


    for i,p in enumerate(real_image_paths): 
        dp_name = Path(p).stem

        real_img = np.array(Image.open(p))
        annotation = index2color_annotation(np.array(Image.open(f"{os.path.join(args.comparisons[0], annotations_folder, dp_name)}{annotations_format}")), palette)
        prompt = read_txt(f"{os.path.join(prompts_folder, dp_name)}{prompts_format}")[0]
        
        synthetic_img_path = [p for p in synthetic_image_paths if "_".join(dp_name.split("_")[:-1]) in p][0] 
        synthetic_img = np.array(Image.open(synthetic_img_path))
        
        # TODO 
        uncertainty_imgs = []
        uncertainties = []
        add_col = 1
        if(args.uncertainty == ""):
            add_col = 0
            uncertainty = ""

        fig, axis = plt.subplots(3+add_col, len(args.comparisons), figsize = (size*len(args.comparisons), (3+add_col)*size))

        """axis[0,0].imshow(real_img)
        axis[0,0].set_title(Path(args.comparisons[0]).stem + "\nReal", fontsize=24)

        axis[1,0].imshow(real_img)
        axis[1,0].imshow(annotation, alpha = 0.7)
        axis[1,0].set_title("Annotation", fontsize=24)

        axis[2,0].imshow(synthetic_img)
        axis[2,0].set_title("Synthetic", fontsize=24)

        axis[0,0].set_ylabel(Path(args.comparisons[0]).stem, fontsize=24)# + "\n" + prompt)"""

        # TODO add all comparisons
        for c in range(0,len(args.comparisons)):
            real_path = f"{os.path.join(args.comparisons[c], images_folder, dp_name)}{images_format}"
            real_img = np.array(Image.open(real_path))
            annotation = index2color_annotation(np.array(Image.open(f"{os.path.join(args.comparisons[c], annotations_folder, dp_name)}{annotations_format}")), palette)
            synthetic_path = os.path.join(Path(real_path).parent,  "_".join(Path(real_path).stem.split("_")[:-1])+"_0001"+images_format)
            synthetic_img = np.array(Image.open(synthetic_path))

            if(args.uncertainty != ""):
                compute_img,_ = resize_transform(Image.open(synthetic_path)) # resize shortest edge to 512
                compute_img = np.array(compute_img)
                if(len(compute_img.shape) != 3):
                    compute_img = np.stack([compute_img,compute_img,compute_img], axis = 0).transpose(1,2,0)
                compute_img = totensor_transform(compute_img).unsqueeze(0)
                uncertainty, uncertainty_img = loss(compute_img.to(device), None, None, seg_model, visualize = True)
                uncertainty_imgs.append(uncertainty_img.squeeze())
                uncertainties.append(float(uncertainty.detach().cpu().numpy()[0]))
            
            comp_prompts_folder_tmp = glob(os.path.join(args.comparisons[c], base_prompts_folder)+"/*")[0]
            if(not os.path.isdir(comp_prompts_folder_tmp)):
                comp_prompts_folder = os.path.join(args.comparisons[c], base_prompts_folder)
            else: 
                comp_prompts_folder = comp_prompts_folder_tmp

            prompt = read_txt(f"{os.path.join(comp_prompts_folder, dp_name)}{prompts_format}")[0]


            axis[0,c].imshow(real_img)
            axis[0,c].set_title(Path(args.comparisons[c]).stem + "\nReal", fontsize=24)
            # axis[0,c].set_ylabel(Path(args.comparisons[c]).stem+ "\n" + str(uncertainty.float()), fontsize=24) # + "\n" + prompt)

        
            axis[1,c].imshow(real_img)
            axis[1,c].imshow(annotation, alpha = 0.7)
            axis[1,c].set_title("Annotation", fontsize=24)

            axis[2,c].imshow(synthetic_img)
            axis[2,c].set_title("Synthetic", fontsize=24)


        # plot uncertainty heatmap
        if(args.uncertainty != ""):
            maximum = np.concatenate(uncertainty_imgs).max()
            minimum = np.concatenate(uncertainty_imgs).min()

            for c, (uncertainty_img, uncertainty) in enumerate(zip(uncertainty_imgs, uncertainties)): 
                axis[3,c].imshow((uncertainty_img-minimum)/(maximum-minimum), cmap="rainbow")
                axis[3,c].set_title(f"{args.uncertainty} = {uncertainty:.4f}",fontsize=24) 

        for axs in axis: 
            for a in axs: 
                a.set_axis_off()
        
        plt.tight_layout()
        plt.savefig(args.save_to + "/" + dp_name + ".jpg")
        plt.savefig(args.save_to + "/" + dp_name + ".pdf")
        plt.close()










