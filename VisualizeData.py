import torchvision
from glob import glob
import numpy as np
from PIL import Image
from pathlib import Path 
import matplotlib.pyplot as plt

from Utils import read_txt, index2color_annotation
from DataGeneration import get_name

from ade_config import palette

import argparse

from torch.utils.data import Dataset as TorchDataset


if __name__ == "__main__":

    print("******************************")
    print("VISUALIZE")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--data_path', type = str, default="./data/ade_augmented/controlnet1.1/")
    parser.add_argument('--num_images', type = int, default=20)



    args = parser.parse_args()
    print(f"Parameters: {args}")
    


    data_path = args.data_path
    base_path = "./data/ade/ADEChallengeData2016/"

    vis_folder = "/visualization/"
    images_folder = "/images/training/"
    annotations_folder = "/annotations/training/"
    prompts_folder = "/prompts/training/"


    annotations_format = ".png"
    images_format = ".jpg"
    prompts_format = ".txt"

    totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


    class Ade20kDataset(TorchDataset):
        def __init__(self, start_idx=-1, end_idx=-1, random_num = 20, seed = 42):
            super().__init__()
            data_paths = sorted(glob(data_path+images_folder+"*.jpg"))
            self.augmentation_paths = [p for p in data_paths if not ("_0000.jpg" in p)]
            real_paths = [p for p in data_paths if ("_0000.jpg" in p)]

            if((start_idx > 0) and (end_idx >= 0)):
                real_paths = real_paths[start_idx:end_idx]
                start_idx = start_idx
            elif(end_idx >= 0):
                real_paths = real_paths[:end_idx]
            elif(start_idx > 0):
                real_paths = real_paths[start_idx:]
                start_idx = start_idx

            if((random_num > 0) and (len(real_paths) > random_num)): 
                np.random.seed(seed)
                real_paths = np.random.choice(real_paths, random_num).tolist()

            self.annotations_dir = data_path+annotations_folder
            self.prompts_dir = glob(base_path+prompts_folder+"*")
            self.real_paths = real_paths
            self.seed = seed
            self.transform = totensor_transform

            # self.data_paths = ['./data/ade/ADEChallengeData2016//images/training/ADE_train_00000082.jpg']
            self.aspect_resize = torchvision.transforms.Resize(size=512)

        def __len__(self):
            return len(self.real_paths)

        def __getitem__(self, idx): 
            path = self.real_paths[idx]
            # open image
            init_image = self.aspect_resize(Image.open(path)) # resize shortest edge to 512
            init_image = np.array(init_image)
            if(len(init_image.shape) != 3):
                init_image = np.stack([init_image,init_image,init_image], axis = 0).transpose(1,2,0)

            augmentation_paths = [p for p in self.augmentation_paths if ("_".join(Path(p).stem.split("_")[:-1])) in path]
            augmentations = []
            for a in augmentation_paths:
                augmentations.append(np.array(self.aspect_resize(Image.open(a)))) # resize shortest edge to 512

            
            # open prompt
            prompt = read_txt(self.prompts_dir+"_".join(Path(path).stem.split("_")[:-1])+"_0000"+prompts_format)[0]
            
            # open mask
            annotation_path = self.annotations_dir+Path(path).stem+annotations_format
            annotation = Image.open(annotation_path)
            annotation = self.aspect_resize(annotation)
            annotation = np.array(annotation)

            condition = self.transform(index2color_annotation(annotation, palette))
            
            return init_image, augmentations, condition, annotation, prompt, path


    dataset = Ade20kDataset(random_num=args.num_images)

    for init_image, augmentations, condition, annotation, prompt, path in dataset: 
        fig, axis = plt.subplots(1+len(augmentations),2)

        axis[0,0].imshow(init_image)
        axis[0,0].set_ylabel("real")

        axis[0,1].imshow(init_image)
        axis[0,1].imshow(condition.permute(1,2,0), alpha = 0.7)

        for i, augmentation in enumerate(augmentations): 
            axis[i+1,0].imshow(augmentation)
            axis[i+1,0].set_ylabel("synthetic")

            axis[i+1,1].imshow(augmentation)
            axis[i+1,1].imshow(condition.permute(1,2,0), alpha = 0.7)


        plt.suptitle(prompt)

        """for ax in axis: 
            for a in ax:
                a.set_axis_off()"""


        plt.tight_layout()
        plt.savefig(data_path + vis_folder + Path(path).stem+".jpg")
        plt.close()