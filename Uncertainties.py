from Dataset import Dataset
from torch.utils.data import DataLoader
import torch 
from ade_config import palette
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision 

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


save_to = "./TestUncertainties/"
os.makedirs(save_to, exist_ok = True)
num_samples = 5
num_imgs = 5


model_path = "./SegmentationModel/train_model_scripted.pt"
data_preprocessor_path = "./SegmentationModel/data_preprocessor.pth"



batch_size = 1
root_dir = "./data/ade/ADEChallengeData2016/"
mode = "training"
dataset = Dataset(root_dir, mode)
dataloader = DataLoader(dataset, batch_size, shuffle=False)


segmentation_model = torch.jit.load(model_path).cuda()

# torch.load(data_preprocessor_path)
# import pdb 
# pdb.set_trace()

for idx, batch in enumerate(dataloader): 
    img,mask = batch

    fig,axs = plt.subplots(1,1+num_samples, figsize = (5*num_samples, 5))
    axs[0].imshow(img.squeeze().permute(1,2,0))

    outs = []
    for i in range(num_samples):
        out = segmentation_model(img.cuda().float()).detach().cpu()
        outs.append(out)
        import pdb 
        pdb.set_trace()
        #visualization
        out_mask = np.zeros((out.shape[-2], out.shape[-1], 3), dtype=np.uint8)
        for label, color in enumerate(palette[1:]):
            out_mask[out[0,label, :,:]>0, :] = color
        axs[i+1].imshow(out_mask)
    mean_uncertainty = np.mean(np.std(outs, axis = (2,3)))
    
    
    plt.suptitle(mean_uncertainty)
    for ax in axs: 
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_to+"/"+str(idx).zfill(6)+".png")
    outs = np.stack(outs).squeeze()

    if(idx >= num_imgs):
        break



    # visualize
