import glob
import os
from pathlib import Path
import shutil

# Rename
img_dir = "data/ade_augmented/canny_img2text/images/training/"
mask_dir = "data/ade_augmented/canny_img2text/annotations/training/"

img_paths = sorted(glob.glob(img_dir +"*.jpg"))
mask_paths = sorted(glob.glob(mask_dir+"*.png"))


def rename(paths):
    for path in paths[::-1]: 
        new_name = str(Path(path).parent) +"/"+ "_".join(Path(path).stem.split("_")[:-1]) +"_"+str(int(Path(path).stem.split("_")[-1]) +1).zfill(4)+Path(path).suffix
        os.rename(path, new_name)

        # print(f"INFO::Rename {path} to {new_name}")

rename(img_paths)
print("Renamed all images")
rename(mask_paths)
print("Renamed all masks")


# Copy
path_to_real_imgs = glob.glob("data/ade/ADEChallengeData2016/images/training/*.jpg")
path_to_real_masks = glob.glob("data/ade/ADEChallengeData2016/annotations/training/*.png")

def copy(paths, dest_folder):
    for i,path in enumerate(paths): 
        new_name = dest_folder +"/"+ Path(path).stem+"_0000"+Path(path).suffix
        shutil.copyfile(path, new_name)
        """ if(i == 10):
            break"""



copy(path_to_real_imgs, img_dir)
print("Copied all images")
copy(path_to_real_masks, mask_dir)
print("Copied all masks")

