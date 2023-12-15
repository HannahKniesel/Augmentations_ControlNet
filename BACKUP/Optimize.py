from Dataset import DatasetOptimize
from torch.utils.data import DataLoader
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, pipeline, Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import cv2

def image2text_local(model, image, seed = 42):
    # image to text with vit-gpt2
    torch.manual_seed(seed)
    if(type(image) != Image.Image):
        image = Image.fromarray(image)
    input_text = model(image)
    input_text = input_text[0]['generated_text']
    return input_text

def augment_image_controlnet(controlnet_pipe, canny_image, prompt, seed = 42):
    generator = torch.manual_seed(seed)
    images = controlnet_pipe(prompt, num_inference_steps=20, generator=generator, image=canny_image).images
    image = images[0]
    return image



def loss(segmentation_model, output, target):
    return

device = "cuda" if torch.cuda.is_available() else "cpu"


num_iterations = 1000
val_step = 100
batch_size = 8 
seed = 42
root_dir = "./data/ade/ADEChallengeData2016/"
model_path = "./SegmentationModel/train_model_scripted.pt"

# get segmentation model
segmentation_model = torch.jit.load(model_path).to(device)

# define controlnet
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16) # canny controlnet
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
controlnet_pipe.enable_model_cpu_offload()



# get image to optimize
mode = "training"
dataset = DatasetOptimize(root_dir, mode, batch_size = batch_size, seed=seed)
dataloader = DataLoader(dataset, batch_size, shuffle=False)
prompt, condition, mask = next(iter(dataloader))

# to optimize 
input = torch.nn.parameter.Parameter(torch.normal()

for i in range(num_iterations):
    noise = # TODO 
    images = controlnet_pipe(prompt, num_inference_steps=20, latents=generator, image=controlnet).images

    pass