from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, pipeline
import os
from pathlib import Path
import argparse
import time
from datetime import timedelta
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO make full dataset with coco format


classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood']

def save_example(image, annotation, canny_image, augmentations, prompts, annotated_classes, folder, idx):
    fig,axs = plt.subplots(1,3+len(augmentations), figsize=(10+(5*len(augmentations)),15))
    axs[0].imshow(image)
    axs[1].imshow(annotation)
    axs[1].set_title(("\n").join(annotated_classes))
    axs[2].imshow(canny_image)
    for j, (augmented_image, prompt) in enumerate(zip(augmentations, prompts)):
        axs[j+3].imshow(augmented_image)
        axs[j+3].set_title(prompt)

    for ax in axs:
        ax.set_axis_off()

    str_idx = str(idx).zfill(4)
    save_dir = folder+"/"
    os.makedirs(save_dir, exist_ok = True)
    plt.tight_layout()
    plt.savefig(save_dir+str_idx+".jpg")
    plt.close()
    return 

image2text_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
controlnet_pipe.enable_model_cpu_offload()

def image2text(image, seed = 42):
    # image to text with vit-gpt2
    torch.manual_seed(seed)
    if(type(image) != Image.Image):
        image = Image.fromarray(image)
    input_text = image2text_model(image)
    input_text = input_text[0]['generated_text']
    return input_text

def vqa(image, seed = 42):
    torch.manual_seed(seed)
    question = "What is in the image?"
    inputs = vqa_processor(image, question, return_tensors="pt").to(device)
    out = vqa_model.generate(**inputs)
    generated_text = vqa_processor.decode(out[0], skip_special_tokens=True)
    input_text = "A photograph of " + generated_text
    return input_text

def augment_image_controlnet(image, canny_image, prompt, seed = 42):
    generator = torch.manual_seed(seed)
    images = controlnet_pipe(prompt, num_inference_steps=20, generator=generator, image=canny_image).images
    image = images[0]
    return image



if __name__ == "__main__":

    print("******************************")
    print("AUGMENTATIONS")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Binary')

    # General Parameters
    parser.add_argument('--prompt_definition', type = str, default="alternating", choices=["alternating", "vqa", "img2text", "annotations"])
    parser.add_argument('--coco_path', type = str, default="./coco_stuff10k/")
    parser.add_argument('--save_path', type = str, default="./coco_stuff10k_augmented/")


    parser.add_argument('--iterative_img', action='store_true')
    parser.add_argument('--save_number_images', type = int, default=50)
    parser.add_argument('--num_augmentations', type = int, default=4)
    args = parser.parse_args()
    print(f"Parameters: {args}")

    start_time = time.time()

    save_path = args.save_path+"/"+args.prompt_definition
    if(args.iterative_img): 
        save_path += "_iterative"

    data_paths = glob(args.coco_path+"/images/train2014/*.jpg")
    if(args.save_number_images >= 0):
        data_paths = data_paths[:args.save_number_images]
    annotations_dir = args.coco_path+"/annotations/train2014/"


    for img_idx, path in enumerate(data_paths): 
        print(f"Image {img_idx}/{len(data_paths)}")
        init_image = np.array(Image.open(path))
        canny_image = cv2.Canny(init_image, 100, 200)
        canny_image = canny_image[:,:,None]
        canny_image = np.concatenate([canny_image,canny_image,canny_image], axis = 2)
        canny_image = Image.fromarray(canny_image)

        annotation_path = annotations_dir+Path(path).stem+"_labelTrainIds.png"
        annotation = np.array(Image.open(annotation_path))
        annotated_classes = [classes[x-1] for x in np.unique(annotation)]

        augmentations = []
        prompts = []
        image = init_image.copy()
        if(len(image.shape) != 3):
            image = np.stack([image,image,image], axis = 0)
        for i in range(args.num_augmentations):
            if(not args.iterative_img):
                image = init_image
            seed = torch.randint(high = 10000000, size = (1,))
            if(args.prompt_definition == "alternating"):
                if((i%2) == 0):
                    prompt = image2text(image, seed)
                else: 
                    prompt = vqa(image, seed)
            elif(args.prompt_definition == "vqa"):
                prompt = vqa(image, seed)
            elif(args.prompt_definition == "img2text"):
                prompt = image2text(image, seed)
            elif(args.prompt_definition == "annotations"):
                anno_str = " ".join(annotated_classes)
                prompt = "A photograph of "+anno_str
            image = augment_image_controlnet(image, canny_image, prompt, seed)

            while(np.max(image) == np.min(image)):
                seed = torch.randint(high = 10000000, size = (1,))
                image = augment_image_controlnet(image, canny_image, prompt, seed)

            augmentations.append(image)
            prompts.append(prompt+"\n Seed = "+str(float(seed)))
        save_example(init_image, annotation, canny_image, augmentations, prompts, annotated_classes, save_path, img_idx)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed_time))
    print(f"Augmented {len(data_paths)} images with {args.num_augmentations} augmentations in {elapsed_time_str}")
    time_per_augmentation = elapsed_time/(len(data_paths)*args.num_augmentations)
    time_per_augmentation_str = time.strftime("%Hh%Mm%Ss", time.gmtime(time_per_augmentation))
    print(f"Time per augmentation = {time_per_augmentation_str}")


    