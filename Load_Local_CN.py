
from diffusers import ControlNetModel, UniPCMultistepScheduler
from CNPipeline import StableDiffusionControlNetPipeline
import torch


# TODO safety checker

checkpoint = "./models/control_sd15_seg.pth"
checkpoint = "./models/control_sd15_ini.ckpt"


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg") #, torch_dtype="auto") #torch.float16)
controlnet_pipe = StableDiffusionControlNetPipeline.from_single_file(checkpoint, controlnet = controlnet)
controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
controlnet_pipe.enable_model_cpu_offload()
controlnet_pipe.set_progress_bar_config(disable=True)