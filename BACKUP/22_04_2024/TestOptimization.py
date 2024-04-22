from CNPipelineBasic import StableDiffusionControlNetPipeline as SDCNPipeline_Init
from diffusers import ControlNetModel, UniPCMultistepScheduler
from Datasets import Ade20kDataset
from torch.utils.data import DataLoader
import time
from pathlib import Path


checkpoint = "lllyasviel/control_v11p_sd15_seg" # Trained on COCO and Ade
sd_ckpt = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained(checkpoint) #, torch_dtype="auto") #torch.float16)
controlnet_pipe = SDCNPipeline_Init.from_pretrained(sd_ckpt, controlnet=controlnet) #, torch_dtype="auto") #torch.float16)


controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
controlnet_pipe.set_progress_bar_config(disable=True)

# get data
dataset = Ade20kDataset(0, 5, "gt", 7353)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


# iterate over dataset
for img_idx, (init_img, condition, annotation, prompt, path) in enumerate(dataloader):
    starttime_img = time.time()
    print(prompt)
    # TODO include new pipeline
    generator = torch.manual_seed(0)
    
    
    (image, has_nsfw_concept) = controlnet_pipe(prompt[0] + additional_prompt, #+"best quality, extremely detailed" # 
                                negative_prompt=negative_prompt, 
                                image=condition, 
                                controlnet_conditioning_scale=controlnet_conditioning_scale, 
                                guidance_scale = guidance_scale,
                                num_inference_steps=inference_steps, 
                                height = condition.shape[-2],
                                width = condition.shape[-1],
                                num_images_per_prompt = 1, # TODO are they computed in parallel or iteratively?
                                generator=generator,
                                return_dict = False
                                # img_name = Path(path[0]).stem,
                                # real_image = init_img
                                )
    
    print(f"INFO:: Time elapsed = {elapsed_time} | Loss = {loss}")


