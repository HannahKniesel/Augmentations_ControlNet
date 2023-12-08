from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, pipeline, Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipProcessor, Blip2ForConditionalGeneration
import os
from pathlib import Path
import argparse
import time
from datetime import timedelta
import shutil
import torchvision
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_example(image, annotation, canny_image, augmentations, prompts, annotated_classes, folder, idx):
    fig,axs = plt.subplots(1,3+len(augmentations), figsize=(10+(5*len(augmentations)),15))
    axs[0].imshow(image)
    axs[0].set_xlabel(image.shape)
    axs[1].imshow(annotation)
    axs[1].set_xlabel(annotation.shape)
    axs[1].set_title(("\n").join(annotated_classes))
    axs[2].imshow(canny_image)
    axs[2].set_xlabel(canny_image.size)
    for j, (augmented_image, prompt) in enumerate(zip(augmentations, prompts)):
        axs[j+3].imshow(augmented_image)
        axs[j+3].set_xlabel(augmented_image.size)

    axs[3].set_title(prompt)

    """for ax in axs:
        ax.set_axis_off()"""

    str_idx = str(idx).zfill(6)
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
        print(image.shape)
        if(image.shape[0] <=3): 
            image = image.transpose(1,2,0)
        image = Image.fromarray(image)
    inputs = processor(image, prompt, return_tensors="pt").to(text_device)
    out = model.generate(**inputs)
    input_text = (processor.decode(out[0], skip_special_tokens=True))
    
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)
    # generated_ids = model.generate(**inputs)
    # input_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return input_text

def augment_image_controlnet(controlnet_pipe, canny_image, prompt, seed = 42, controlnet_conditioning_scale = 1.0):
    generator = torch.manual_seed(seed)
    negative_prompt = 'low quality, bad quality, sketches'
    images = controlnet_pipe(prompt, negative_prompt=negative_prompt, image=canny_image, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator).images
    # images = controlnet_pipe(prompt, num_inference_steps=20, generator=generator, image=canny_image, negative_prompt=negative_prompt).images
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
    start_idx = 0
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
        processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        text_device = "cpu"
        # model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    else:
        processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto")
        text_device = "cuda"

        # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)  



    if(args.condition =="canny"):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        controlnet_conditioning_scale = 0.5  # recommended for good generalization
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
        )

    elif(args.condition == "segmentation"):
        controlnet_conditioning_scale = 0.5  # recommended for good generalization
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16)
        controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
        #controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
        controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
    controlnet_pipe.enable_model_cpu_offload()

    for img_idx, path in enumerate(data_paths): 
        print(f"Image {img_idx+start_idx}/{len(data_paths)+start_idx}")
        init_image = np.array(Image.open(path))
    
        annotation_path = annotations_dir+Path(path).stem+".png"
        annotation = np.array(Image.open(annotation_path))
        annotated_classes = [classes[x] for x in np.unique(annotation) if(classes[x] != 'bg')]


        mask = Image.open(annotation_path)

        if(args.condition == "segmentation"):
            condition_image = get_segmentation(annotation, palette)
        elif(args.condition == "canny"):
            condition_image = get_canny(init_image) 
        

        """h,w = condition_image.size
        if((condition_image.size[0] % 2)!= 0):
            h = condition_image.size[0]-1
        if((condition_image.size[1] % 2)!= 0):
            w = condition_image.size[1] -1
        
        transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop((w,h))])
        condition_image = transform(condition_image)
        mask = transform(mask)"""

        # copy annotation for this image
        shutil.copy(annotation_path, save_path+annotations_folder+name+annotations_format)
        # mask.save(save_path+annotations_folder+name+annotations_format)

        augmentations = [] #init_image.copy()]
        prompts = []
        image = init_image.copy()

        # save init image 
        name = Path(path).stem.split(".")
        name[0] = name[0] + "_" + str(0).zfill(4)
        name = (".").join(name)
        image_pil = Image.fromarray(image)
        image_pil.save(save_path+images_folder+name+".jpg")

       

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
                    prompt = "Question: What are shown in the photo? Answer:"#None
                elif(args.prompt_definition == "vqa_annotations"):
                    anno_str = ", ".join(annotated_classes)
                    prompt = "Describe the image using some of the following words: "+str(anno_str)
                    
                # if(args.local): 
                #     prompt = image2text_local(model, image, seed)
                # else: 
                prompt = image2text(model, processor, image, prompt, seed)

            if(prompt is not None):
                prompt = prompt+", realistic looking, high-quality, extremely detailed, 4K, HQ"
            
            image = np.zeros((3,3))
            nsfw = 0
            while((np.array(image).sum() == 0) and (nsfw < 10)):
                image = augment_image_controlnet(controlnet_pipe, condition_image, prompt, seed+nsfw, controlnet_conditioning_scale)
                transform = torchvision.transforms.Compose([torchvision.transforms.Resize(mask.size[::-1])])
                image = transform(image)
                nsfw += 1

                # TODO resize
            augmentations.append(image)
            prompts.append(prompt+"\n Seed = "+str(float(seed)))


            # save augmented image 
            name = Path(path).stem.split(".")
            name[0] = name[0] + "_" + str(i+1).zfill(4)
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


    
