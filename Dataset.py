import glob
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision
import cv2
from transformers import pipeline
import torch

# standard_transform = 
totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                                                    #  torchvision.transforms.Resize((512, 2048))])

def get_canny(init_image, canny_x = 100, canny_y = 250):
    canny_image = cv2.Canny(init_image, canny_x, canny_y)
    canny_image = canny_image[:,:,None]
    canny_image = np.concatenate([canny_image,canny_image,canny_image], axis = 2)
    canny_image = Image.fromarray(canny_image)
    return canny_image

def image2text_local(model, image, seed = 42):
    # image to text with vit-gpt2
    torch.manual_seed(seed)
    if(type(image) != Image.Image):
        image = Image.fromarray(image)
    input_text = model(image)
    input_text = input_text[0]['generated_text']
    return input_text


class Dataset(TorchDataset):
    def __init__(self, root_path, mode, transform = None, optimize = False):
        img_str = f"images/{mode}/*.jpg"
        mask_str = f"annotations/{mode}"

        self.image_data = glob.glob(root_path +"/"+ img_str)
        if(optimize):
            self.image_data = [np.random.choice(self.image_data)]
            self.image_data = np.repeat(self.image_data, 100)


        self.mask_root = root_path + "/" +mask_str
        self.mask_data = glob.glob(root_path +"/"+ mask_str)
        if(transform):
            self.transform = transform
        else: 
            self.transform = totensor_transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx): 
        img_path = self.image_data[idx]
        mask_path = self.mask_root+"/"+Path(img_path).stem+".png"

        img = np.array(Image.open(img_path))
        img = self.transform(img)
        mask = np.array(Image.open(mask_path))
        """if(self.transform):
            image = self.transform(img)"""

        return img, mask
        

class DatasetOptimize(TorchDataset):
    def __init__(self, root_path, mode, batch_size = 8, seed = 42):
        img_str = f"images/{mode}/*.jpg"
        mask_str = f"annotations/{mode}"
        self.batch_size = batch_size
        self.image_data = glob.glob(root_path +"/"+ img_str)
        np.random.seed(42)
        img_path = np.random.choice(self.image_data)
        mask_root = root_path + "/" +mask_str
        mask_path = mask_root+"/"+Path(img_path).stem+".png"
        img = np.array(Image.open(img_path))
        
        model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning") # prompt model TODO change when running on cluster

        self.condition = totensor_transform(get_canny(img, canny_x=100, canny_y=250))
        self.prompt = image2text_local(model, img, seed)
        self.mask = np.array(Image.open(mask_path))


    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx): 
        return self.prompt, self.condition, self.mask