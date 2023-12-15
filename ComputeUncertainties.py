import argparse
from glob import glob 
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import os
from Utils import save_pkl, load_pkl

trainmodel_file = "/train_model_scripted.pt"
evalmodel_file = "/eval_model_scripted.pt"

totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
resize_transform = torchvision.transforms.Resize((512,512)) # TODO look into what resize to use

sigmoid = torch.nn.Sigmoid()

class Ade20k(torch.utils.data.Dataset):
    def __init__(self, root_path, images_folder="/images/training/", annotations_folder="/annotations/training/", seed = 42):
        data_paths = sorted(glob(root_path+images_folder+"*.jpg"))
        self.annotations_dir = root_path+annotations_folder
        self.data_paths = [path for path in data_paths if int(Path(path).stem.split("_")[-1])>0] # only get augmentation images
        self.seed = seed

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx): 
        img_path = self.data_paths[idx]
        mask_path = self.annotations_dir+"/"+Path(img_path).stem+".png"

        img = np.array(Image.open(img_path))
        img = totensor_transform(img)
        img = resize_transform(img)

        mask = Image.open(mask_path)
        mask = resize_transform(mask)
        mask = torch.from_numpy(np.array(mask))

        return img, mask, Path(img_path).stem
    
    
def sort_dicts(paths, al_dict, best="minimum"):
    if(best=="minimum"):
        rev = False
    elif(best=="maximum"):
        rev = True
    sorted_aldict = {}
    for path in paths:
        aug_idx = 1
        aug_name = "_".join(Path(path).stem.split("_")[:-1])+"_"+str(aug_idx).zfill(4)+".jpg"
        collection_names = []
        while(os.path.isfile(str(Path(path).parent) + "/" + aug_name)):
            collection_names.append(aug_name)
            aug_idx += 1
            aug_name = "_".join(Path(path).stem.split("_")[:-1])+"_"+str(aug_idx).zfill(4)+".jpg"

        al_measures = [al_dict[name.split(".")[0]] for name in collection_names]
        sorted_collection = [x for _, x in sorted(zip(al_measures, collection_names),reverse=rev)]
        sorted_aldict[Path(path).stem+".jpg"] = sorted_collection
    return sorted_aldict

if __name__ == "__main__":

    print("******************************")
    print("COMPUTE UNCERTAINTIES")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    #parser.add_argument('--dataset', type = str, default="cocostuff10k", choices = ["ade", "cocostuff10k"])
    parser.add_argument('--data_path', type = str, default="./data/ade_augmented/test/")
    parser.add_argument('--num_uncertainty_samples', type = int, default=5)
    parser.add_argument('--batch_size', type = int, default=8)
    parser.add_argument('--model_path', type = str, default="./SegmentationModel/")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    log_path = args.data_path+"/"+Path(args.model_path).stem+"/"
    os.makedirs(log_path, exist_ok=True)

    dataset = Ade20k(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    mcdropout_dict = {} # --> select maximum
    smu_dict = {} # smallest margin --> select minimum
    lmu_dict = {} # largest margin --> select minimum
    lcu_dict = {} # --> select minimum
    entropy_dict = {} # --> select maximum

    train_segmentation = torch.jit.load(args.model_path+trainmodel_file).cuda()
    eval_segmentation = torch.jit.load(args.model_path+evalmodel_file).cuda()
    
    with torch.no_grad():
     for idx, (img,mask,path) in enumerate(dataloader):
         print(f"Image {idx*args.batch_size}/{len(dataset)}")
         img = img.cuda()
         mask = mask.cuda()

         # mc_dropout
         outputs = []
         for i in range(args.num_uncertainty_samples):
             out = train_segmentation(img).detach().cpu()
             outputs.append(out)
         outputs = torch.stack(outputs).numpy().transpose(1,0,2,3,4)
         outputs = np.mean(np.std(outputs, axis=(2,3,4)), axis = 1) # get mc dropout uncertainty

         for out, p in zip(outputs, path):
             mcdropout_dict[p] = out 

         # others
         out = eval_segmentation(img).detach().cpu()

         #smu
         best_two = torch.topk(out, 2, dim = 1).values
         smu = torch.mean(best_two[:,0,:,:]-best_two[:,1,:,:], dim=(1,2))
         for value, p in zip(smu, path):
             smu_dict[p] = float(value) 

         #lmu 
         maximum = torch.max(out, dim = 1).values
         minimum = torch.min(out, dim = 1).values
         lmu = torch.mean(maximum-minimum, dim=(1,2))
         for value, p in zip(lmu, path):
             lmu_dict[p] = float(value) 

         # lcu
         maximum = torch.max(out, dim = 1).values
         lcu = torch.mean(maximum, dim = (1,2))
         for value, p in zip(lcu, path):
             lcu_dict[p] = float(value)


         # entropy
         probs = sigmoid(out)
         logarithm = torch.log(probs)
         entropy = -1*torch.sum((probs*logarithm), dim = 1)
         mean_entropy = torch.mean(entropy, dim = (1,2))
         for value, p in zip(mean_entropy, path):
             entropy_dict[p] = float(value)


    # TODO sort dicts
    paths = glob(args.data_path+"/images/training/"+"*_0000.jpg") # get real images
    sorted_mcdropout = sort_dicts(paths, mcdropout_dict, best="maximum")
    sorted_smu = sort_dicts(paths, smu_dict, best="minimum")
    sorted_lmu = sort_dicts(paths, lmu_dict, best="minimum")
    sorted_lcu = sort_dicts(paths, lcu_dict, best="minimum")
    sorted_entropy = sort_dicts(paths, entropy_dict, best="maximum")


    # save raw data
    save_pkl(mcdropout_dict, log_path+"/mcdropout_raw")
    save_pkl(smu_dict, log_path+"/smu_raw")
    save_pkl(lmu_dict, log_path+"/lmu_raw")
    save_pkl(lcu_dict, log_path+"/lcu_raw")
    save_pkl(entropy_dict, log_path+"/entropy_raw")

    # save sorted data
    save_pkl(sorted_mcdropout, log_path+"/mcdropout")
    save_pkl(sorted_smu, log_path+"/smu")
    save_pkl(sorted_lmu, log_path+"/lmu")
    save_pkl(sorted_lcu, log_path+"/lcu")
    save_pkl(sorted_entropy, log_path+"/entropy")




    # TODO save dicts
    # TODO write filter.txt/adapt mmsegmentation
    # TODO cleanup with backup folder

        





