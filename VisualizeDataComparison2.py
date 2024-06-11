
import os 
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path 
import argparse
import numpy as np
import torch
import cv2

from ade_config import images_folder, annotations_folder, prompts_folder, annotations_format, images_format, prompts_format, palette
from Utils import read_txt, index2color_annotation, device, resize_transform, totensor_transform
from Loss import loss_fct, get_prediction


# Visualizes different generated datasets with their corresponding uncertainty. 

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
    parser.add_argument('--easy_model', type=str, default="./base_models/03-06-2024/BestEpoch/eval_model_scripted.pt")
    parser.add_argument('--hard_model', type=str, default="./base_models/03-06-2024/EarlyStopping25/eval_model_scripted.pt")
    parser.add_argument('--w_hard', type=float, default=1.0)
    parser.add_argument('--w_easy', type=float, default=1.0)


    args = parser.parse_args()
    print(f"Parameters: {args}")

    easy_model = torch.jit.load(args.easy_model)
    easy_model = easy_model.to(device)
    hard_model = torch.jit.load(args.hard_model)
    hard_model = hard_model.to(device)


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
        heatmaps = []
        losses = []
        losses_easy = []
        losses_hard = []


        
        with torch.no_grad():
            fig, axis = plt.subplots(8, len(args.comparisons), figsize = (size*len(args.comparisons), (3+add_col)*size))

            for c in range(0,len(args.comparisons)):
                real_path = f"{os.path.join(args.comparisons[c], images_folder, dp_name)}{images_format}"
                real_img = np.array(Image.open(real_path))
                annotation_indices = Image.open(f"{os.path.join(args.comparisons[c], annotations_folder, dp_name)}{annotations_format}")
                annotation = index2color_annotation(np.array(annotation_indices), palette)
                synthetic_path = os.path.join(Path(real_path).parent,  "_".join(Path(real_path).stem.split("_")[:-1])+"_0001"+images_format)
                synthetic_img = np.array(Image.open(synthetic_path))

                compute_img,_ = resize_transform(Image.open(synthetic_path)) # resize shortest edge to 512
                compute_img = np.array(compute_img)
                if(len(compute_img.shape) != 3):
                    compute_img = np.stack([compute_img,compute_img,compute_img], axis = 0).transpose(1,2,0)
                compute_img = totensor_transform(compute_img).unsqueeze(0)
                anotation_indices = resize_transform(annotation_indices)
                annotation_indices = np.array(annotation_indices)
                annotation_indices = totensor_transform(annotation_indices)
                print(annotation_indices.shape)
                print(compute_img.shape)
                
                easy_loss, hard_loss, loss, heatmap = loss_fct(compute_img.to(device), 
                        annotation, 
                        easy_model, 
                        args.w_easy, 
                        hard_model, 
                        args.w_hard, 
                        visualize = True, 
                        by_value = True) 
                
                # uncertainty,heatmap = uncertaintyloss_fct(compute_img.to(device), seg_model, loss, visualize = True)

                heatmaps.append(heatmap.squeeze())
                losses.append(float(loss))
                losses_easy.append(float(easy_loss))
                losses_hard.append(float(losses_hard))

                
                comp_prompts_folder_tmp = glob(os.path.join(args.comparisons[c], base_prompts_folder)+"/*")[0]
                if(not os.path.isdir(comp_prompts_folder_tmp)):
                    comp_prompts_folder = os.path.join(args.comparisons[c], base_prompts_folder)
                else: 
                    comp_prompts_folder = comp_prompts_folder_tmp

                prompt = read_txt(f"{os.path.join(comp_prompts_folder, dp_name)}{prompts_format}")[0]

                # TODO correctly scale images
                axis[0,c].imshow(real_img)
                axis[0,c].set_title(Path(args.comparisons[c]).stem + "\nReal", fontsize=24)

                axis[1,c].imshow(real_img)
                axis[1,c].imshow(annotation, alpha = 0.7)
                axis[1,c].set_title("Real - Annotation", fontsize=24)

                """real_img = totensor_transform(real_img)
                real_prediction = get_prediction(real_img.unsqueeze(0).cuda(),seg_model)
                axis[2,c].imshow(real_prediction) #, alpha = 0.7)
                axis[2,c].set_title("Real - Prediction", fontsize=24)"""

                axis[2,c].imshow(synthetic_img)
                axis[2,c].set_title("Synthetic", fontsize=24)

                axis[3,c].imshow(annotation) #, alpha = 0.7)
                axis[3,c].set_title("Synthetic - Annotation", fontsize=24)

                synthetic_img = totensor_transform(synthetic_img)               
                syn_prediction = get_prediction(synthetic_img.unsqueeze(0).cuda(),easy_model)
                axis[4,c].imshow(syn_prediction) #, alpha = 0.7)
                axis[4,c].set_title(f"Synthetic - Prediction Easy\nCE = {easy_loss:.4f}", fontsize=24)

                syn_prediction = get_prediction(synthetic_img.unsqueeze(0).cuda(),hard_model)
                axis[5,c].imshow(syn_prediction) #, alpha = 0.7)
                axis[5,c].set_title(f"Synthetic - Prediction Hard\nCE = {-1*hard_loss:.4f}", fontsize=24)


            # plot loss heatmap
            maximum = np.concatenate(heatmaps).max()
            minimum = np.concatenate(heatmaps).min()

            for c, (heatmap, loss) in enumerate(zip(heatmaps, losses)): 
                axis[6,c].imshow((heatmap-minimum)/(maximum-minimum), cmap="Reds") #, cmap="rainbow")
                axis[6,c].set_title(f"loss = {loss:.4f}",fontsize=24) 

            for axs in axis: 
                for a in axs: 
                    a.set_axis_off()
            
            plt.tight_layout()
            plt.savefig(args.save_to + "/" + dp_name + ".jpg")
            plt.savefig(args.save_to + "/" + dp_name + ".pdf")
            plt.close()










