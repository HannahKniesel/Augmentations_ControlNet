
import os 
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path 
import argparse


if __name__ == "__main__":

    print("******************************")
    print("VISUALIZE COMPARISON")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--vis_folders', type = str, nargs="+", default=["./data/ade_augmented/controlnet1.1/visualization/"])
    parser.add_argument('--save_to', type = str, default="")



    args = parser.parse_args()
    print(f"Parameters: {args}")

    vis_folders = args.vis_folders #["./data/ade_augmented/baseline/visualization/", "./data/ade_augmented/baseline/visualization/", "./data/ade_augmented/baseline/visualization/"]
    save_to = args.save_to + "/Comparison/"
    os.makedirs(save_to, exist_ok = True)

    paths = []
    for vis_folder in vis_folders: 
        paths.append(glob(f"{vis_folder}/*.jpg"))


    # for each image
    for i in range(len(paths[0])):
        fig, axis = plt.subplots(1, len(paths), figsize = (10*len(paths), 10))
        plt.suptitle(Path(vis_folders[i]).stem)

        # for each comparison
        for n in range(len(paths)):
            img = Image.open(paths[n][i])
            axis[n].imshow(img)
            axis[n].set_axis_off()
            axis[n].set_title(vis_folders[n].split("/")[-3])
        
        plt.tight_layout()
        plt.savefig(f"{save_to}/{Path(paths[n][i]).stem}.jpg")
        plt.close()