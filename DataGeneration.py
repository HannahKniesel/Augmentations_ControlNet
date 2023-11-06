from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, pipeline, Blip2Processor, Blip2ForConditionalGeneration
import os
from pathlib import Path
import argparse
import time
from datetime import timedelta
import shutil
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

def save_example(image, annotation, canny_image, augmentations, prompts, annotated_classes, folder, idx):
    fig,axs = plt.subplots(1,3+len(augmentations), figsize=(10+(5*len(augmentations)),15))
    axs[0].imshow(image)
    axs[1].imshow(annotation)
    axs[1].set_title(("\n").join(annotated_classes))
    axs[2].imshow(canny_image)
    for j, (augmented_image, prompt) in enumerate(zip(augmentations, prompts)):
        axs[j+3].imshow(augmented_image)
    axs[3].set_title(prompt)

    for ax in axs:
        ax.set_axis_off()

    str_idx = str(idx).zfill(4)
    save_dir = folder+"/"
    os.makedirs(save_dir, exist_ok = True)
    plt.tight_layout()
    plt.savefig(save_dir+str_idx+".jpg")
    plt.close()
    return 

def image2text_local(model, image, seed = 42):
    # image to text with vit-gpt2
    torch.manual_seed(seed)
    if(type(image) != Image.Image):
        image = Image.fromarray(image)
    input_text = model(image)
    input_text = input_text[0]['generated_text']
    return input_text

def image2text(model, processor, image, prompt, seed = 42):
    # image to text with vit-gpt2
    torch.manual_seed(seed)
    if(type(image) != Image.Image):
        image = Image.fromarray(image)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)
    generated_ids = model.generate(**inputs)
    input_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return input_text

def augment_image_controlnet(controlnet_pipe, canny_image, prompt, seed = 42):
    generator = torch.manual_seed(seed)
    images = controlnet_pipe(prompt, num_inference_steps=20, generator=generator, image=canny_image).images
    image = images[0]
    return image


def get_segmentation(annotation, palette):
    condition_image = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(palette):
        condition_image[annotation == label, :] = color

    condition_image = condition_image.astype(np.uint8) 
    condition_image = Image.fromarray(condition_image)
    return condition_image

def get_canny(init_image, canny_x = 100, canny_y = 250):
    canny_image = cv2.Canny(init_image, canny_x, canny_y)
    canny_image = canny_image[:,:,None]
    canny_image = np.concatenate([canny_image,canny_image,canny_image], axis = 2)
    canny_image = Image.fromarray(canny_image)
    return canny_image



if __name__ == "__main__":

    print("******************************")
    print("AUGMENTATIONS")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--prompt_definition', type = str, default="img2text", choices=["vqa", "img2text", "annotations"])
    parser.add_argument('--dataset', type = str, default="cocostuff10k", choices = ["ade", "cocostuff10k"])
    parser.add_argument('--condition', type = str, default="canny", choices = ["canny", "segmentation"])
    parser.add_argument('--num_augmentations', type = int, default=4)
    parser.add_argument('--local', action='store_true')


    parser.add_argument('--start_idx', type = int, default=-1)
    parser.add_argument('--end_idx', type = int, default=-1)


    args = parser.parse_args()
    print(f"Parameters: {args}")

    if(args.dataset == "cocotuff10k"):
        from coco_config import *
    elif(args.dataset == "ade"):
        from ade_config import *

    start_time = time.time()

    save_path = save_path+"/"+args.condition+"_"+args.prompt_definition
    print(f"Save to: {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+images_folder, exist_ok=True)
    os.makedirs(save_path+annotations_folder, exist_ok=True)

    data_paths = glob(data_path+images_folder+"*.jpg")

    if((args.start_idx >= 0) and (args.end_idx >= 0)):
        data_paths = data_paths[args.start_idx:args.end_idx]
        start_idx = args.start_idx
    elif(args.end_idx >= 0):
        data_paths = data_paths[:args.end_idx]
    elif(args.start_idx >= 0):
        data_paths = data_paths[args.start_idx:]
        start_idx = args.start_idx


    annotations_dir = data_path+annotations_folder

    # load models
    if(args.local):
        model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    else:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)  


    if(args.condition =="canny"):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    elif(args.condition == "segmentation"):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16)
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
    #controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
    controlnet_pipe.enable_model_cpu_offload()

    for img_idx, path in enumerate(data_paths): 
        print(f"Image {img_idx}/{len(data_paths)}")
        init_image = np.array(Image.open(path))
    
        annotation_path = annotations_dir+Path(path).stem+".png"
        annotation = np.array(Image.open(annotation_path))
        annotated_classes = [classes[x] for x in np.unique(annotation) if(classes[x] != 'bg')]

        if(args.condition == "segmentation"):
            condition_image = get_segmentation(annotation, palette)
        elif(args.condition == "canny"):
            condition_image = get_canny(init_image) 

        augmentations = []
        prompts = []
        image = init_image.copy()
        if(len(image.shape) != 3):
            image = np.stack([image,image,image], axis = 0)
        for i in range(args.num_augmentations):
            seed = torch.randint(high = 10000000, size = (1,))
            if(args.prompt_definition == "annotations"):
                anno_str = ", ".join(annotated_classes)
                prompt = "An image of "+anno_str
            else:
                if(args.prompt_definition == "vqa"):
                    prompt = "What is in the image?"
                elif(args.prompt_definition == "img2text"):
                    prompt = None
                elif(args.prompt_definition == "vqa_annotations"):
                    anno_str = ", ".join(annotated_classes)
                    prompt = "Describe the image using some of the following words: "+str(anno_str)
                    
                if(args.local): 
                    prompt = image2text_local(model, image, seed)
                else: 
                    prompt = image2text(model, processor, image, prompt, seed)

            if(prompt is not None):
                prompt = prompt+", realistic looking, high-quality, extremely detailed, 4K, HQ"
            

            image = augment_image_controlnet(controlnet_pipe, condition_image, prompt, seed)
            augmentations.append(image)
            prompts.append(prompt+"\n Seed = "+str(float(seed)))


            # save augmented image 
            name = Path(path).stem.split(".")
            name[0] = name[0] + "_" + str(i).zfill(3)
            name = (".").join(name)
            image.save(save_path+images_folder+name+".jpg")

            # copy annotation for this image
            shutil.copy(annotation_path, save_path+annotations_folder+name+annotations_format)

        save_example(init_image, annotation, condition_image, augmentations, prompts, annotated_classes, save_path, img_idx+start_idx)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed_time))
    print(f"Augmented {len(data_paths)} images with {args.num_augmentations} augmentations in {elapsed_time_str}")
    time_per_augmentation = elapsed_time/(len(data_paths)*args.num_augmentations)
    time_per_augmentation_str = time.strftime("%Hh%Mm%Ss", time.gmtime(time_per_augmentation))
    print(f"Time per augmentation = {time_per_augmentation_str}")


    