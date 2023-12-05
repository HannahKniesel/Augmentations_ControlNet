import glob
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision

# standard_transform = 
totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                                                    #  torchvision.transforms.Resize((512, 2048))])

class Dataset(Dataset):
    def __init__(self, root_path, mode, transform = None):
        img_str = f"images/{mode}/*.jpg"
        mask_str = f"annotations/{mode}"

        self.image_data = glob.glob(root_path +"/"+ img_str)
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
        