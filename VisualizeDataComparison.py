
import os 
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path 

vis_folders = ["./data/ade_augmented/baseline/visualization/", "./data/ade_augmented/baseline/visualization/", "./data/ade_augmented/baseline/visualization/"]
save_to = "./Visualization/"
os.makedirs(save_to, exist_ok = True)

paths = []
for vis_folder in vis_folders: 
    paths.append(glob(f"{vis_folder}/*.jpg"))


# for each image
for i in range(len(paths[0])):
    fig, axis = plt.subplots(1, len(paths), figsize = (10*len(paths), 10))

    # for each comparison
    for n in range(len(paths)):
        img = Image.open(paths[n][i])
        axis[n].imshow(img)
        axis[n].set_axis_off()
        axis[n].set_title(vis_folders[n].split("/")[-3])
    
    plt.tight_layout()
    plt.savefig(f"{save_to}/{Path(paths[n][i]).stem}.jpg")
    plt.close()