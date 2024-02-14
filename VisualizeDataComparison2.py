
import os 
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path 
import argparse
import numpy as np

from ade_config import images_folder, annotations_folder, prompts_folder, annotations_format, images_format, prompts_format
from Utils import read_txt

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

    args = parser.parse_args()
    print(f"Parameters: {args}")

    os.makedirs(args.save_to, exist_ok=True)

    size = 7

    # remove initial / so that its no absolute path 
    images_folder = images_folder[1:]
    annotations_folder = annotations_folder[1:]
    prompts_folder = prompts_folder[1:]
    

    # for first comparison
    image_paths = sorted(glob(os.path.join(args.comparisons[0], images_folder)+"/*"+images_format))
    real_image_paths = [p for p in image_paths if "_0000"+images_format in p] # get only real images
    real_image_paths = real_image_paths[:args.n_images]
    # real_image_paths = np.random.choice(real_image_paths, size=(args.n_images,))
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
        annotation = np.array(Image.open(f"{os.path.join(args.comparisons[0], annotations_folder, dp_name)}{annotations_format}"))
        prompt = read_txt(f"{os.path.join(prompts_folder, dp_name)}{prompts_format}")[0]
        
        synthetic_img_path = [p for p in synthetic_image_paths if "_".join(dp_name.split("_")[:-1]) in p][0] 
        synthetic_img = np.array(Image.open(synthetic_img_path))

        fig, axis = plt.subplots(len(args.comparisons), 3, figsize = (size*3, size* len(args.comparisons)))

        axis[0,0].imshow(real_img)
        axis[0,0].set_title("Real")

        axis[0,1].imshow(real_img)
        axis[0,1].imshow(annotation, alpha = 0.7)
        axis[0,1].set_title("Annotation")

        axis[0,2].imshow(synthetic_img)
        axis[0,2].set_title("Synthetic")

        axis[0,0].set_ylabel(Path(args.comparisons[0]).stem + "\n" + prompt)

        # TODO add all comparisons
        for c in range(1,len(args.comparisons)):
            real_path = f"{os.path.join(args.comparisons[c], images_folder, dp_name)}{images_format}"
            real_img = np.array(Image.open(real_path))
            annotation = np.array(Image.open(f"{os.path.join(args.comparisons[c], annotations_folder, dp_name)}{annotations_format}"))
            synthetic_path = os.path.join(Path(real_path).parent,  "_".join(Path(real_path).stem.split("_")[:-1])+"_0001"+images_format)
            synthetic_img = np.array(Image.open(synthetic_path))
            
            comp_prompts_folder_tmp = glob(os.path.join(args.comparisons[c], base_prompts_folder)+"/*")[0]
            if(not os.path.isdir(comp_prompts_folder_tmp)):
                comp_prompts_folder = os.path.join(args.comparisons[c], base_prompts_folder)
            else: 
                comp_prompts_folder = comp_prompts_folder_tmp

            prompt = read_txt(f"{os.path.join(comp_prompts_folder, dp_name)}{prompts_format}")[0]


            axis[c,0].imshow(real_img)
            axis[c,0].set_title("Real")

            axis[c,1].imshow(real_img)
            axis[c,1].imshow(annotation, alpha = 0.7)
            axis[c,1].set_title("Annotation")

            axis[c,2].imshow(synthetic_img)
            axis[c,2].set_title("Synthetic")

            axis[c,0].set_ylabel(Path(args.comparisons[c]).stem + "\n" + prompt)
            
        
        plt.tight_layout()
        plt.savefig(args.save_to + "/" + dp_name + ".jpg")










