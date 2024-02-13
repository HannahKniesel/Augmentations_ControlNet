
import numpy as np
from PIL import Image
import torchvision
import torch
import pickle
from pathlib import Path 

device = "cuda" if torch.cuda.is_available() else "cpu"
totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
resize_transform = torchvision.transforms.Resize(size=512)
topil_transform = torchvision.transforms.ToPILImage()


def get_name(path, idx):
    name = Path(path).stem.split(".")
    name[0] = name[0] + "_" + str(idx).zfill(4)
    name = (".").join(name)
    return name

def write_txt(file, content):
    with open(file, 'w') as f:
        f.write(content)

def read_txt(file):
    with open(file) as f:
        lines = f.readlines()
    return lines

def save_pkl(dictionary, path):
    with open(path +'.pkl', 'wb') as fp:
        pickle.dump(dictionary, fp)
        print(f'INFO::Saved content to file {path}.pkl')

def load_pkl(path):
    with open(path+'.pkl', 'rb') as fp:
        dictionary = pickle.load(fp)
        print(f'INFO::Loaded content from file {path}.pkl')
        return dictionary

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


# TODO move out of file
def augment_image_controlnet(controlnet_pipe, condition_image, prompt, 
                             height, width, batch_size, seed = None, 
                             controlnet_conditioning_scale = 1.0, guidance_scale = 7.5, 
                             negative_prompt = "low quality, bad quality, sketches", 
                             additional_prompt = ", realistic looking, high-quality, extremely detailed", 
                             inference_steps = 40):
    nsfw_content = batch_size
    curr_idx = 0
    augmentations = []
    while((nsfw_content > 0)):
        if(seed):
            generator = torch.manual_seed(seed + curr_idx)
        else: 
            generator = None
        output = controlnet_pipe(prompt + additional_prompt, #+"best quality, extremely detailed" # 
                                negative_prompt=negative_prompt, 
                                image=condition_image, 
                                controlnet_conditioning_scale=controlnet_conditioning_scale, 
                                guidance_scale = guidance_scale,
                                num_inference_steps=inference_steps, 
                                height = height, 
                                width = width,
                                num_images_per_prompt = nsfw_content, # TODO are they computed in parallel or iteratively?
                                generator=generator)
        
        images = [elem for elem, nsfw in zip(output.images, output.nsfw_content_detected) if not nsfw]
        augmentations.extend(images)
        nsfw_content = np.min(((len(augmentations)-batch_size), batch_size))
        curr_idx += 1
        if(curr_idx >= 5):
            break
    num_nsfw = 0
    if(len(augmentations)<batch_size):
        num_nsfw = batch_size- len(augmentations)
        print(f"WARNING:: augmentations contain {num_nsfw}/{batch_size} nsfw")
        augmentations.extend([output.images[0]]*(num_nsfw))
    
    assert len(augmentations) == batch_size, f"ERROR::Augmentations length ({len(augmentations)}) should equal the batch size ({batch_size})"
    # print(f"Expected: ({width},{height}) | Reality: {images[0].size}")
    return augmentations, num_nsfw




