from ade_config import *
import time
from torch.utils.data import DataLoader

# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import ControlNetModel, UniPCMultistepScheduler
from CNPipeline import StableDiffusionControlNetPipeline
import torch 

from DataGeneration import Ade20kDataset

import matplotlib.pyplot as plt 

def loss(images): 
    # blue_channel = images[:,:,:,2]  # N x 256 x 256
    blue_channel = images[:,2,:,:]  # N x C x 256 x 256
    return -1*torch.mean(blue_channel)


    # Calculate the mean of the blue channel for each image
    blueness = blue_channel.astype(float).mean(axis=(1,2))  # N
    blueness = (2 * blueness - 255)/255 # normalize to [0,1]

    return torch.from_numpy(blueness)
     

additional_prompt = ", realistic looking, high-quality, extremely detailed, 4K, HQ, photorealistic"
negative_prompt = ", low quality, bad quality, sketches, flat, unrealistic"
controlnet_conditioning_scale = 1.0
guidance_scale = 7.5
inference_steps = 20


# load controlnet
checkpoint = "lllyasviel/control_v11p_sd15_seg" # Trained on COCO and Ade "lllyasviel/sd-controlnet-seg" # Only trained on Ade
controlnet = ControlNetModel.from_pretrained(checkpoint) #, torch_dtype="auto") #torch.float16)
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet) #, torch_dtype="auto") #torch.float16)
controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config) # TODO what happens if I remove this? 
controlnet_pipe.enable_model_cpu_offload()
controlnet_pipe.set_progress_bar_config(disable=False)





dataset = Ade20kDataset(-1, -1, 42)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
mean_time_img = []
for img_idx, (init_img, condition, annotation, prompt, path) in enumerate(dataloader):
    starttime_img = time.time()

    print(prompt)

    
    # get augmentations
    augmentations = []
    aug_annotations = []

        
    output = controlnet_pipe(prompt[0] + additional_prompt, #+"best quality, extremely detailed" # 
                            negative_prompt=negative_prompt, 
                            image=condition, 
                            controlnet_conditioning_scale=controlnet_conditioning_scale, 
                            guidance_scale = guidance_scale,
                            num_inference_steps=inference_steps, 
                            height = condition.shape[-2], 
                            width = condition.shape[-1],
                            num_images_per_prompt = 1, # TODO are they computed in parallel or iteratively?
                            generator=None, 
                            latent_loss = loss)
        
    images = [elem for elem, nsfw in zip(output.images, output.nsfw_content_detected) if not nsfw]

    plt.figure()
    plt.imshow(images[0])
    plt.title(prompt)
    plt.savefig("./test.jpg")

    
    break
        
  


