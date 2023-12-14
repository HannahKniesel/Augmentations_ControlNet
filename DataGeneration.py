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
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from Utils import *
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_name(path, idx):
    name = Path(path).stem.split(".")
    name[0] = name[0] + "_" + str(idx).zfill(4)
    name = (".").join(name)
    return name

def save_augmentations_with_gt(aug_annotations, augmentations, path):
    for idx, (annotation, augmentation) in enumerate(zip(aug_annotations, augmentations)):
        name = get_name(path, idx+1)
        annotation.save(save_path+annotations_folder+name+annotations_format)
        augmentation.save(save_path+images_folder+name+images_format)
    return

def visualize(aug_annotations, augmentations, init_image, init_annotation, prompt, name):
    fig, axis = plt.subplots(3, len(augmentations)+1, figsize = ((len(augmentations)+1)*5, 3*5))
    plt.suptitle(prompt)
    axis[0,0].imshow(init_image)
    axis[1,0].imshow(init_annotation)
    axis[2,0].imshow(init_image)
    axis[2,0].imshow(init_annotation, alpha = 0.5)
    axis[2,0].set_xlabel(f"Image res: {init_image.size()} | GT res: {init_annotation.size()}")

    for i, (annotation, augmentation) in enumerate(zip(aug_annotations, augmentations)):
        annotation = index2color_annotation(np.array(annotation), palette)
        axis[0,i+1].imshow(augmentation)
        axis[1,i+1].imshow(annotation)
        axis[2,i+1].imshow(augmentation)
        axis[2,i+1].imshow(annotation, alpha = 0.5)
        axis[2,i+1].set_xlabel(f"Image res: {augmentation.size} | GT res: {annotation.shape}")
    plt.savefig(save_path+vis_folder+name)
    plt.close()

class AbstractAde20k(TorchDataset):
    def __init__(self, start_idx, end_idx, seed = 42):
        data_paths = sorted(glob(data_path+images_folder+"*.jpg"))
        if((start_idx > 0) and (end_idx >= 0)):
            data_paths = data_paths[start_idx:end_idx]
            start_idx = start_idx
        elif(end_idx >= 0):
            data_paths = data_paths[:end_idx]
        elif(start_idx > 0):
            data_paths = data_paths[start_idx:]
            start_idx = start_idx
        self.annotations_dir = data_path+annotations_folder
        self.prompts_dir = data_path+prompts_folder
        self.data_paths = data_paths
        self.seed = seed
        self.transform = totensor_transform

    def __len__(self):
        return len(self.data_paths)

class Ade20kPromptDataset(AbstractAde20k):
    def __init__(self, start_idx, end_idx, num_captions_per_image, seed = 42):
        super().__init__(start_idx, end_idx, seed)
        res = [ele for ele in self.data_paths for i in range(num_captions_per_image)]
        self.aug_paths = [get_name(ele, i) for ele in self.data_paths for i in range(num_captions_per_image)]
        self.data_paths = res
        
        

    def __getitem__(self, idx):
        # p = self.data_paths[idx]
        # if(args.local):
        return self.data_paths[idx], self.aug_paths[idx]

        # image = Image.open(p)
        # image = totensor_transform(image)
        # return image, self.aug_paths[idx]


class Ade20kDataset(AbstractAde20k):
    def __init__(self, start_idx, end_idx, seed = 42):
        super().__init__(start_idx, end_idx, seed)

    def __getitem__(self, idx): 
        path = self.data_paths[idx]
        # open image
        init_image = (Image.open(path))
        init_image = np.array(init_image)
        if(len(init_image.shape) != 3):
            init_image = np.stack([init_image,init_image,init_image], axis = 0)
        
        # open prompt
        prompt = read_txt(self.prompts_dir+Path(path).stem+"_0000"+prompts_format)[0]
        
        # open mask
        annotation_path = self.annotations_dir+Path(path).stem+annotations_format
        annotation = np.array(Image.open(annotation_path))
        condition = self.transform(index2color_annotation(annotation, palette))
        
        # copy annotation and init image
        name = get_name(path, 0)
        shutil.copy(annotation_path, save_path+annotations_folder+name+annotations_format)
        shutil.copy(path, save_path+images_folder+name+images_format)



        # TODO return prompt

        return init_image, condition, annotation, prompt, path


if __name__ == "__main__":

    print("******************************")
    print("AUGMENTATIONS")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    # parser.add_argument('--prompt_definition', type = str, default="img2text", choices=["vqa", "img2text", "annotations"])
    # parser.add_argument('--dataset', type = str, default="ade", choices = ["ade", "cocostuff10k"])
    parser.add_argument('--experiment_name', type = str, default="")
    parser.add_argument('--num_augmentations', type = int, default=4)
    parser.add_argument('--seed', type = int, default=4)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--batch_size', type = int, default=4)
    parser.add_argument('--vis_every', type = int, default=1)
    parser.add_argument('--optimize', action='store_true')

    parser.add_argument('--start_idx', type = int, default=0)
    parser.add_argument('--end_idx', type = int, default=-1)


    args = parser.parse_args()
    print(f"Parameters: {args}")

    # import ade config
    from ade_config import *

    start_time = time.time()

    # save_path = save_path+"/"+args.condition+"_"+args.prompt_definition
    save_path = save_path+"/"+args.experiment_name+"/"
    print(f"Save to: {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+images_folder, exist_ok=True)
    os.makedirs(save_path+annotations_folder, exist_ok=True)
    os.makedirs(save_path+vis_folder, exist_ok=True)


   
    # check if prompts exist, if not generate prompts
    if(not Path(data_path+prompts_folder).is_dir()): 
        os.makedirs(data_path+prompts_folder, exist_ok=True)
        if(args.local):
            model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

        else:
            processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16, device_map="auto", load_in_8bit=True,)
            # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto")
            # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)  
        
        # dataset = Ade20kPromptDataset(args.start_idx, args.end_idx, args.num_augmentations, args.seed)
        dataset = Ade20kPromptDataset(args.start_idx, args.end_idx, 1, args.seed)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        for paths, aug_paths in tqdm(dataloader, desc="Generating prompts"): 
            if(args.local):
                prompts = image2text_gpt2(model, list(paths), args.seed)
            else: 
                prompts = image2text_blip2(model, processor, list(paths), args.seed)
            for p, prompt in zip(aug_paths, prompts):
                write_txt(data_path+prompts_folder+p+prompts_format, prompt)
                

    # copy prompts to new data folder 
    os.makedirs(save_path+prompts_folder, exist_ok=True)
    for filename in glob(os.path.join(data_path+prompts_folder, '*'+prompts_format)):
        shutil.copy(filename, save_path+prompts_folder)

    # load controlnet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16)
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
    controlnet_pipe.enable_model_cpu_offload()
    controlnet_pipe.set_progress_bar_config(disable=True)

    # get data
    dataset = Ade20kDataset(args.start_idx, args.end_idx, args.seed)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean_time_img = []
    for img_idx, (init_img, condition, annotation, prompt, path) in enumerate(dataloader):
        starttime_img = time.time()

        print(prompt)

        # get augmentations
        augmentations = []
        aug_annotations = []
        while(len(augmentations)<args.num_augmentations):
            curr_batch_size = np.min((args.batch_size, (args.num_augmentations - len(augmentations))))
            # image = np.zeros((3,3))
            # nsfw = 0
            if(args.optimize):
                augmented = augmentandoptimize_image_controlnet(controlnet_pipe, condition, prompt[0], condition.shape[-2], condition.shape[-1], curr_batch_size, controlnet_conditioning_scale = 1.0, guidance_scale = 0.5)
            else: 
                augmented = augment_image_controlnet(controlnet_pipe, condition, prompt[0], condition.shape[-2], condition.shape[-1], curr_batch_size, controlnet_conditioning_scale = 1.0, guidance_scale = 0.5)
            augmentations.extend(augmented)

            transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(augmented[0].size[::-1]), torchvision.transforms.ToPILImage()])
            aug_annotation = transform(annotation[0])
            aug_annotations.extend([aug_annotation]*len(augmented))
            # image = transform(image)
            # nsfw += 1

        # save augmentations
        save_augmentations_with_gt(aug_annotations, augmentations, path[0])

        if((args.vis_every >0 ) and ((img_idx %args.vis_every) == 0)): 
            visualize(aug_annotations, augmentations, init_img[0], condition[0].permute(1,2,0), prompt, Path(path[0]).stem)
        
        endtime_img = time.time()
        elapsedtime_img = endtime_img - starttime_img
        mean_time_img.append(elapsedtime_img)
        remaining_time = np.mean(mean_time_img)*(len(dataset)-img_idx)
        elapsedtime_img_str = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsedtime_img))
        remainingtime_img_str = str(timedelta(seconds=remaining_time))
        # remainingtime_img_str = time.strftime("%Hh%Mm%Ss", time.gmtime(remaining_time))
        print(f"Image {img_idx+args.start_idx}/{len(dataset)+args.start_idx} | Time for image = {elapsedtime_img_str} | Remaining time = {remainingtime_img_str}")

    end_time = time.time()
    elapsedtime = end_time - start_time
    elapsedtime_str = str(timedelta(seconds=elapsedtime))
    print(f"Time to generate {args.num_augmentations} augmentations for {len(dataset)} images was {elapsedtime_str}")


    
