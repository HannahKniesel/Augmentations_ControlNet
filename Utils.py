
import numpy as np
from PIL import Image
import torchvision
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
topil_transform = torchvision.transforms.ToPILImage()


def write_txt(file, content):
    with open(file, 'w') as f:
        f.write(content)

def read_txt(file):
    with open(file) as f:
        lines = f.readlines()
    return lines


# idx_annotation ... np.array with shape W,H indices as 
def index2color_annotation(idx_annotation, palette):
    color_annotation = np.zeros((idx_annotation.shape[0], idx_annotation.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(palette):
        color_annotation[idx_annotation == label, :] = color

    color_annotation = color_annotation.astype(np.uint8) 
    return color_annotation

# idx_annotation ... np.array with shape W,H indices as 
def color2index_annotation(color_annotation, palette):
    idx_annotation = np.zeros((color_annotation.shape[0], color_annotation.shape[1]), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(palette):
        color_annotation[idx_annotation == label, :] = color

    color_annotation = color_annotation.astype(np.uint8) 
    return color_annotation

# image to text generation
def image2text_gpt2(model, paths, seed = 42):
    # image to text with vit-gpt2
    # torch.manual_seed(seed)
    input_text = model(paths)
    prompts = [t[0]['generated_text'] for t in input_text]
    return prompts

def image2text_blip2(model, processor, paths, seed = 42):
    # image to text with vit-gpt2
    # torch.manual_seed(seed)
    prompts = []
    for path in paths: 
        pil_image = Image.open(path)
        prompt = "Question: What are shown in the photo? Answer:"#None
        inputs = processor(pil_image, prompt, return_tensors="pt").to(device, torch.float16)
        # inputs = processor(pil_image, prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs)

        input_text = (processor.decode(out[0], skip_special_tokens=True))
        prompts.append(input_text)
    return prompts


def augment_image_controlnet(controlnet_pipe, condition_image, prompt, height, width, batch_size, seed = None, controlnet_conditioning_scale = 1.0, guidance_scale = 0.5):
    if(seed):
        generator = torch.manual_seed(seed)
    else: 
        generator = None
    negative_prompt = 'low quality, bad quality, sketches'

    nsfw_content = batch_size
    curr_idx = 0
    augmentations = []
    while((nsfw_content > 0)):
        output = controlnet_pipe(prompt, #+", realistic looking, high-quality, extremely detailed, 4K, HQ", 
                                negative_prompt=negative_prompt, 
                                image=condition_image, 
                                controlnet_conditioning_scale=controlnet_conditioning_scale, 
                                guidance_scale = guidance_scale,
                                num_inference_steps=40, 
                                height = height, 
                                width = width,
                                num_images_per_prompt = nsfw_content,
                                generator=generator)
        
        images = [elem for elem, nsfw in zip(output.images, output.nsfw_content_detected) if not nsfw]
        augmentations.extend(images)
        nsfw_content = (len(augmentations)-batch_size)
        curr_idx += curr_idx+1
        if(curr_idx == 10):
            if(len(augmentations)<batch_size):
                augmentations.extend([output.images[0]]*(len(augmentations)-batch_size))
                print(f"WARNING:: augmentations contain {len(augmentations)-batch_size} nsfw")
            break
    # print(f"Expected: ({width},{height}) | Reality: {images[0].size}")
    print(f"Length augmentations {len(augmentations)}")
    return augmentations

# TODO
# Issue: cant backpropagate through controlnet pipe (torch.no_grad) --> look into DDPO implementation to backprob through forward pass of controlnet https://github.com/huggingface/trl/blob/main/trl/trainer/ddpo_trainer.py
def augmentandoptimize_image_controlnet(controlnet_pipe, condition_image, prompt, height, width, batch_size, seed = 42, controlnet_conditioning_scale = 1.0, guidance_scale = 0.5):
    if(seed):
        generator = torch.manual_seed(seed)
    else: 
        generator = None
    import pdb 
    pdb.set_trace()
    latents = controlnet_pipe.vae.encode(condition_image.type(torch.float16).cuda()).latent_dist.sample()
    images = controlnet_pipe(prompt, image = condition_image, latents = latents).images

    negative_prompt = 'low quality, bad quality, sketches'
    images = controlnet_pipe(prompt+", realistic looking, high-quality, extremely detailed, 4K, HQ", 
                             negative_prompt=negative_prompt, 
                             image=condition_image, 
                             controlnet_conditioning_scale=controlnet_conditioning_scale, 
                             guidance_scale = guidance_scale,
                             num_inference_steps=40, 
                             height = height, 
                             width = width,
                             num_images_per_prompt = batch_size,
                             generator=generator).images
    
    # print(f"Expected: ({width},{height}) | Reality: {images[0].size}")
    return images