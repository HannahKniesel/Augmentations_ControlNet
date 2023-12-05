from glob import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import time
import numpy as np
import torch
import torchvision
from PIL import Image

save_imgs = True
if(save_imgs):
    save_to_uncertainty = "./Debug_uncertainty/"
    os.makedirs(save_to_uncertainty, exist_ok=True)
    save_to_uncertainty_gt = "./Debug_uncertaintyGT/"
    os.makedirs(save_to_uncertainty_gt, exist_ok=True)



totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                                                      #torchvision.transforms.Resize((512, 2048))])
bce = torch.nn.BCEWithLogitsLoss()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, mode, transform = None):
        img_str = f"images/{mode}/*.jpg"
        mask_str = f"annotations/{mode}"

        self.image_data = glob.glob(root_path +"/*.jpg"+ img_str)
        self.mask_root = root_path + "/" +mask_str
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
        


def filter_random(augmentations, path, num_augmentations):
    # add initial image (no augmentation)
    filtered_augmentations = [Path(path).stem+".jpg"]
    # remove real image from augmentations list
    augmentations.remove(Path(path).stem+".jpg")

    # get <num_augmentations> augmentations
    picked_augmentations = np.random.choice(augmentations, np.min([num_augmentations, len(augmentations)])).tolist()
    filtered_augmentations.extend(picked_augmentations)
    return filtered_augmentations

def filter_uncertainty(augmentations, path, num_augmentations, model, num_samples = 5):
    # add initial image (no augmentation)
    filtered_augmentations = [Path(path).stem+".jpg"]
    # remove real image from augmentations list
    augmentations.remove(Path(path).stem+".jpg")
    root = Path(path).parent
    uncertainties = []
    for augmentation in augmentations: 
        img = np.array(Image.open(root / augmentation))
        img = totensor_transform(img).cuda().float()[None]
        outs = []
        for i in range(num_samples):
            out = model(img).detach().cpu()
            outs.append(out)            
        mean_uncertainty = np.mean(np.std(outs, axis = (2,3)))
        uncertainties.append(mean_uncertainty)
    augmentations = [x for _, x in sorted(zip(uncertainties, augmentations), reverse=True)]

    if(save_imgs):
        # visualize
        fig,axis = plt.subplots(1,1+len(augmentations), figsize = ((1+len(augmentations))*5,5))
        img = np.array(Image.open(path))

        axis[0].imshow(img)
        for idx, (augmentation, uncertainty) in enumerate(zip(augmentations, sorted(uncertainties, reverse=True))): 
            img = np.array(Image.open(root / augmentation))
            axis[idx+1].imshow(img)
            axis[idx+1].set_title(uncertainty)
        for ax in axis: 
            ax.set_axis_off()
        plt.savefig(save_to_uncertainty+Path(path).stem+".jpg")
        plt.close()

    # pick augmentations 
    picked_augmentations = augmentations[:num_augmentations]
    filtered_augmentations.extend(picked_augmentations)
    return filtered_augmentations

def filter_uncertainty_gt(augmentations, path, num_augmentations, model, num_samples = 5):
    # add initial image (no augmentation)
    filtered_augmentations = [Path(path).stem+".jpg"]
    # remove real image from augmentations list
    augmentations.remove(Path(path).stem+".jpg")
    root = Path(path).parent
    minimize_mean = []
    maximize_std = []
    for augmentation in augmentations: 

        gt_path = str(root.parent.parent) + "/annotations/" + root.stem + "/" + augmentation.split(".")[0] + ".png"
        gt = torch.from_numpy(np.array(Image.open(gt_path)))
        gt_onehot = torch.zeros((1, 150, gt.shape[-2], gt.shape[-1]))
        for j in range(150):
            gt_onehot[0,j,:,:] = (gt == j)
        img = np.array(Image.open(root / augmentation))
        img = totensor_transform(img).cuda().float()[None]
        outs = []
        loss = []
        for i in range(num_samples):
            out = model(img).detach().cpu()
            resize = torchvision.transforms.Resize((out.shape[-2], out.shape[-1]), antialias=True)
            gt_resized = resize(gt_onehot)
            loss.append(bce(out,gt_resized))
            outs.append(out)       
        mean_prediction = torch.mean(torch.concatenate(outs), dim = 0) 
        minimize_mean.append(bce(mean_prediction[None], gt_resized)) # mean prediction should match gt
        maximize_std.append(torch.std(torch.Tensor(loss))) # std over all predictions should be high (similar as high std in losses)
        
    scores = []
    for minimize, maximize in zip(minimize_mean, maximize_std):
        scores.append(-1*minimize + maximize) # minimize loss and maximize uncertainty
    
    # TODO
    augmentations = [x for _, x in sorted(zip(scores, augmentations), reverse=True)]
    minimize_mean = [x for _, x in sorted(zip(scores, minimize_mean), reverse=True)]
    maximize_std = [x for _, x in sorted(zip(scores, maximize_std), reverse=True)]



    if(save_imgs):
        # visualize
        fig,axis = plt.subplots(1,1+len(augmentations), figsize = ((1+len(augmentations))*5,5))
        img = np.array(Image.open(path))

        axis[0].imshow(img)
        for idx, (augmentation, minimize, maximize, score) in enumerate(zip(augmentations, minimize_mean, maximize_std, sorted(scores, reverse=True))): 
            img = np.array(Image.open(root / augmentation))
            axis[idx+1].imshow(img)
            axis[idx+1].set_title(f"Min Mean: {minimize}\nMaximize std: {maximize}\nScore: {score}" )
        for ax in axis: 
            ax.set_axis_off()
        plt.savefig(save_to_uncertainty_gt+Path(path).stem+".jpg")
        plt.close()

    # pick augmentations 
    picked_augmentations = augmentations[:num_augmentations]
    filtered_augmentations.extend(picked_augmentations)
    return filtered_augmentations


def filter_synthetic_only(augmentations, num_augmentations):
    # remove real image from pool
    augmentations.remove(Path(path).stem+".jpg")
    picked_augmentations = np.random.choice(augmentations, np.min([num_augmentations, len(augmentations)])).tolist()
    return picked_augmentations

def filter_real_only(path):
    # only add real image to pool
    filtered_augmentations = [Path(path).stem+".jpg"]
    return filtered_augmentations

if __name__ == "__main__":

    print("******************************")
    print("FILTER DATA")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    #parser.add_argument('--dataset', type = str, default="cocostuff10k", choices = ["ade", "cocostuff10k"])
    parser.add_argument('--data_path', type = str, default="./data/ade_augmented/canny_img2text/")
    parser.add_argument('--images_folder', type = str, default="/images/training/")


    parser.add_argument('--name', type = str, default="default")
    parser.add_argument('--filter_by', type = str, choices=["random", "synthetic_only", "real_only", "uncertainty", "uncertainty_gt"])
    parser.add_argument('--num_augmentations', type = int, default=1)
    parser.add_argument('--num_uncertainty_samples', type = int, default=5)
    parser.add_argument('--model_path', type = str, default="./SegmentationModel/train_model_scripted.pt")




    args = parser.parse_args()
    print(f"Parameters: {args}")

    start_time = time.time()

    save_path = args.data_path+"/"+args.name+".txt"
    print(f"Save as: {save_path}")

    # get all real images (no augmentations)
    image_paths = glob(args.data_path+"/"+args.images_folder+"*_0000.jpg")
    lines = []

    if((args.filter_by == "uncertainty") or (args.filter_by == "uncertainty_gt")):
        segmentation_model = torch.jit.load(args.model_path).cuda()

    for path in image_paths: 
        p = Path(path)
        n = p.stem
        n = ("_").join(n.split("_")[:-1])
        augmentations = glob(str(p.parent)+"/"+n+"_*.jpg")
        augmentations = [Path(a).stem+".jpg" for a in augmentations]
        if(args.filter_by == "random"):
            augmentations = filter_random(augmentations, path, args.num_augmentations)
        elif(args.filter_by == "synthetic_only"):
            augmentations = filter_synthetic_only(augmentations, args.num_augmentations)
        elif(args.filter_by == "real_only"):
            augmentations = filter_real_only(path)
        elif(args.filter_by == "uncertainty"):
            augmentations = filter_uncertainty(augmentations, path, args.num_augmentations, segmentation_model, args.num_uncertainty_samples)
        elif(args.filter_by == "uncertainty_gt"):
            augmentations = filter_uncertainty_gt(augmentations, path, args.num_augmentations, segmentation_model, args.num_uncertainty_samples)

        lines.extend(augmentations)

    with open(save_path, 'w') as f:
        for line in lines: 
            f.write(line+"\n")

    all_files = glob(args.data_path+"/"+args.images_folder+"*.jpg")
    print(f"INFO::Done writng to file {save_path} with {len(lines)}/{len(all_files)} images")
    print(f"INFO::Picked on average {(len(lines)/len(image_paths))-1} augmentations.")


    # TODO filter data 
