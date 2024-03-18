from torch.utils.data import Dataset as TorchDataset
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import torch

from Utils import totensor_transform, resize_transform
from Utils import read_txt, index2color_annotation

import ade_config


def get_name(path, idx):
    name = Path(path).stem.split(".")
    name[0] = name[0] + "_" + str(idx).zfill(4)
    name = (".").join(name)
    return name


# abstract dataset class to load image, annotation and prompt paths
class AbstractAde20k(TorchDataset):
    def __init__(self, start_idx, end_idx, prompt_type, seed = 42):
        data_paths = sorted(glob(ade_config.data_path+ade_config.images_folder+"*.jpg"))
        if((start_idx > 0) and (end_idx >= 0)):
            data_paths = data_paths[start_idx:end_idx]
            start_idx = start_idx
        elif(end_idx >= 0):
            data_paths = data_paths[:end_idx]
        elif(start_idx > 0):
            data_paths = data_paths[start_idx:]
            start_idx = start_idx
        self.annotations_dir = ade_config.data_path+ade_config.annotations_folder
        self.prompts_dir = ade_config.data_path+ade_config.prompts_folder
        self.prompts_dir = f"{self.prompts_dir}/{prompt_type}/" # set prompt dir based on prompt_type
        self.data_paths = data_paths
        self.seed = seed
        self.transform = totensor_transform

    def __len__(self):
        return len(self.data_paths)
    

# dataset for prompt generation
class Ade20kPromptDataset(AbstractAde20k):
    def __init__(self, start_idx, end_idx, num_captions_per_image, seed = 42):
        super().__init__(start_idx, end_idx, "", seed)
        res = [ele for ele in self.data_paths for i in range(num_captions_per_image)]
        self.aug_paths = [get_name(ele, i) for ele in self.data_paths for i in range(num_captions_per_image)]
        self.data_paths = res

    def __getitem__(self, idx):
        return self.data_paths[idx], self.aug_paths[idx]


class Ade20kDataset(AbstractAde20k):
    def __init__(self, start_idx, end_idx, prompt_type, copy_data = True, seed = 42):
        super().__init__(start_idx, end_idx, prompt_type, seed)
        self.aspect_resize = resize_transform
        self.copy_data = copy_data


    def __getitem__(self, idx): 
        path = self.data_paths[idx]
        # open image
        init_image = self.aspect_resize(Image.open(path)) # resize shortest edge to 512
        init_image = np.array(init_image)
        if(len(init_image.shape) != 3):
            init_image = np.stack([init_image,init_image,init_image], axis = 0).transpose(1,2,0)
        
        # open prompt
        try:
            prompt = read_txt(self.prompts_dir+Path(path).stem+"_0000"+ade_config.prompts_format)[0]
        except: 
            prompt = ""
        # open mask
        annotation_path = self.annotations_dir+Path(path).stem+ade_config.annotations_format
        annotation = Image.open(annotation_path)
        annotation = self.aspect_resize(annotation)
        annotation = np.array(annotation)

        condition = self.transform(index2color_annotation(annotation, ade_config.palette)) # also normalize condition image to [0,1]
        # condition = torch.from_numpy(index2color_annotation(annotation, ade_config.palette)).permute(2,0,1) # condition is in range [0,255]

        
        if(self.copy_data):
            # copy annotation and init image (used during data generation)
            name = get_name(path, 0)
            shutil.copy(annotation_path, ade_config.save_path+ade_config.annotations_folder+name+ade_config.annotations_format)
            shutil.copy(path, ade_config.save_path+ade_config.images_folder+name+ade_config.images_format)

        return init_image, condition, annotation, prompt, path
    


class SyntheticAde20kDataset(TorchDataset):
    def __init__(self, data_path, prompts="gt"):
        self.data_paths = sorted(glob(data_path+ade_config.images_folder+"*.jpg"))
        self.data_paths = [path for path in self.data_paths if not ("_0000.jpg" in path)] # only get synthetic data
        self.annotations_dir = data_path+ade_config.annotations_folder
        self.prompts_dir = f"{data_path}{ade_config.prompts_folder}/{prompts}/"
        self.transform = totensor_transform
        self.aspect_resize = resize_transform


    def __len__(self):
        return len(self.data_paths)
    

    def __getitem__(self, idx): 
        path = self.data_paths[idx]
        # open image
        init_image = self.aspect_resize(Image.open(path)) # resize shortest edge to 512
        init_image = np.array(init_image)
        if(len(init_image.shape) != 3):
            init_image = np.stack([init_image,init_image,init_image], axis = 0).transpose(1,2,0)
        
        init_image = self.transform(init_image)
        
        # open prompt
        prompt = read_txt(self.prompts_dir+"_".join(Path(path).stem.split("_")[:-1])+"_0000"+ade_config.prompts_format)[0]
        
        # open mask
        annotation_path = self.annotations_dir+Path(path).stem+ade_config.annotations_format
        annotation = Image.open(annotation_path)
        annotation = self.aspect_resize(annotation)
        annotation = np.array(annotation)

        condition = self.transform(index2color_annotation(annotation, ade_config.palette)) # also normalize condition image to [0,1]
        
       
        return init_image, condition, annotation, prompt, path
