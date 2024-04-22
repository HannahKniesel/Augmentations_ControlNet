
import numpy as np
from PIL import Image
import torchvision
import torch
import pickle
import ade_config
from pathlib import Path 

device = "cuda" if torch.cuda.is_available() else "cpu"


# used to crop image to a maximum side length of 1536 (small side length is 512). This is used to make the code able to run on 48GB GPU memory.
class CropToMaxSize(object):
    def __init__(self, max_size=1536):
        self.max_size = max_size

    # expects img to be PIL image
    def __call__(self, img):
        """
        Args:
            img (PIL Image or tensor): Image to be flipped.

        Returns:
            PIL Image or tensor: Randomly flipped image.
        """
        width = img.width
        height = img.height
        width, height = img.size
        resized = False

        if(width > self.max_size): 
            img = torchvision.transforms.functional.center_crop(img, (height, self.max_size))
            resized = True

        if(height > self.max_size): 
            img = torchvision.transforms.functional.center_crop(img, (self.max_size, width))
            resized = True

        return img, resized

totensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# resize_transform = torchvision.transforms.Resize(size=512)
resize_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=512), CropToMaxSize(max_size=1536)])
centercrop = torchvision.transforms.CenterCrop(512)
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
        prompts.extend(input_text)
    return prompts

def image2text_llava(processor, paths, seed = 42):
    # image to text with llava
    # torch.manual_seed(seed)
    prompts = []
    for path in paths: 
        pil_image = Image.open(path)
        prompt = "<image>\nUSER: What's the content of the photo?\nASSISTANT:"
        outputs = processor(pil_image, prompt=prompt, generate_kwargs={"max_new_tokens": 75})
        input_text = [o["generated_text"].split("\nASSISTANT: ")[1] for o in outputs]
        prompts.extend(input_text)
    return prompts



def image2text_small_llava_gt(processor, paths, seed = 42):
    # image to text with llava
    # torch.manual_seed(seed)
    prompts = []
    for path in paths: 
        pil_image = Image.open(path)
        
        # use gt_prompts
        mask = np.array(Image.open(f"{ade_config.data_path}{ade_config.annotations_folder}{Path(path).stem}{ade_config.annotations_format}"))
        available_classes = np.unique(mask)
        class_names = [ade_config.classes[i] for i in available_classes][1:]
        gt_classes = ", ".join(class_names)

        prompt1 = f"Can you describe the content of the photo using following words: '{gt_classes}'?"
        prompt2 = f"Can you make your answer shorter?"
        prompt3 = f"Can you make it even shorter?"

        prompt = f"USER: <image>\n{prompt1}\nASSISTANT:"
        outputs = processor(pil_image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})
        answer = [o["generated_text"].split("ASSISTANT: ")[1] for o in outputs]
        
        prompt = f"{prompt} {answer[0]} USER:{prompt2}\nASSISTANT:" 
        outputs = processor(pil_image, prompt=prompt, generate_kwargs={"max_new_tokens": 150})
        answer = [o["generated_text"].split("ASSISTANT: ")[2] for o in outputs]

        prompt = f"{prompt} {answer[0]} USER:{prompt3}\nASSISTANT:" 
        outputs = processor(pil_image, prompt=prompt, generate_kwargs={"max_new_tokens": 60})
        answer = [o["generated_text"].split("ASSISTANT: ")[3] for o in outputs]
        prompts.extend(answer)
    return prompts

def image2text_llava_gt(processor, paths, seed = 42):
    # image to text with llava
    # torch.manual_seed(seed)
    prompts = []
    for path in paths: 
        pil_image = Image.open(path)
        
        # use gt_prompts
        mask = np.array(Image.open(f"{ade_config.data_path}{ade_config.annotations_folder}{Path(path).stem}{ade_config.annotations_format}"))
        available_classes = np.unique(mask)
        class_names = [ade_config.classes[i] for i in available_classes][1:]
        gt_classes = ", ".join(class_names)
        prompt = f"<image>\nUSER: Can you describe the content of the photo using following words: '{gt_classes}'?\nASSISTANT:"

        outputs = processor(pil_image, prompt=prompt, generate_kwargs={"max_new_tokens": 70})
        input_text = [o["generated_text"].split("\nASSISTANT: ")[1] for o in outputs]
        prompts.extend(input_text)
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
        
        try:
            images = [elem for elem, nsfw in zip(output.images, output.nsfw_content_detected) if not nsfw]
        except: 
            images = output.images
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




