import time
from torch.utils.data import DataLoader

# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import ControlNetModel, UniPCMultistepScheduler
from CNPipeline import StableDiffusionControlNetPipeline
import torch 
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import os
import wandb

# TODO include into DataGen + cleanup


from DataGeneration import Ade20kDataset


# TODO torch batchify loss for images
def loss_brightness(images): 
    if(type(images) is list): 
        m = [-1*torch.mean(i) for i in images] #[-1*np.mean(np.array(i)) for i in images]
        return np.mean(m)
    # blue_channel = images[:,:,:,2]  # N x 256 x 256
    # blue_channel = images[:,2,:,:]  # N x C x 256 x 256
    return -1*torch.mean(images)

additional_prompt = ", realistic looking, high-quality, extremely detailed, 4K, HQ, photorealistic"
negative_prompt = ", low quality, bad quality, sketches, flat, unrealistic"
controlnet_conditioning_scale = 1.0
guidance_scale = 7.5
inference_steps = 10
wandb_project = "Debug"


optimization_params = {"do_optimize": False, 
                        "visualize": f"./Visualizations/Optim/{wandb_project}/", 
                        "log_to_wandb": False, 
                        "lr": 1000., 
                        "iters": 1, 
                        "optim_every_n_steps": 1,
                        "loss": loss_brightness}


if(optimization_params["log_to_wandb"]):
    os.environ['WANDB_PROJECT']= wandb_project
    group = "Optimization" if optimization_params['do_optimize'] else "Base"
    wandb.init(config = optimization_params, reinit=True, group = group, mode="online")
    # wandb_name = self.wandb_name+"_"+str(wandb.run.id)
    # wandb.run.name = wandb_name


# load controlnet
checkpoint = "lllyasviel/control_v11p_sd15_seg" # Trained on COCO and Ade "lllyasviel/sd-controlnet-seg" # Only trained on Ade
controlnet = ControlNetModel.from_pretrained(checkpoint) #, torch_dtype="auto") #torch.float16)
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet) #, torch_dtype="auto") #torch.float16)
controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config) # TODO what happens if I remove this? 
controlnet_pipe.enable_model_cpu_offload()
controlnet_pipe.set_progress_bar_config(disable=False)


dataset = Ade20kDataset(-1, -1, prompt_type="", copy_data=False, seed=42)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
mean_time_img = []
for img_idx, (init_img, condition, annotation, prompt, path) in enumerate(dataloader):
    print(prompt)

    augmentations = []
    aug_annotations = []

    output, elapsed_time, loss = controlnet_pipe(prompt[0] + additional_prompt, #+"best quality, extremely detailed" # 
                            negative_prompt=negative_prompt, 
                            image=condition, 
                            controlnet_conditioning_scale=controlnet_conditioning_scale, 
                            guidance_scale = guidance_scale,
                            num_inference_steps=inference_steps, 
                            height = condition.shape[-2], 
                            width = condition.shape[-1],
                            num_images_per_prompt = 1, 
                            generator=None, 
                            img_name = Path(path[0]).stem,
                            optimization_arguments = optimization_params
                            )
        
    images = [elem for elem, nsfw in zip(output.images, output.nsfw_content_detected) if not nsfw]

    print(f"INFO:: Time elapsed = {elapsed_time} | Loss = {loss}")

    plt.figure()
    plt.imshow(images[0])
    plt.title(f"{prompt}\n{elapsed_time}s\nLoss={loss}")
    plt.savefig("./test.jpg")

    
    break
        
  


