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
import json


def get_name(path, idx):
    name = Path(path).stem.split(".")
    name[0] = name[0] + "_" + str(idx).zfill(4)
    name = (".").join(name)
    return name

def save_augmentations_with_gt(aug_annotations, augmentations, path, start_aug_idx):
    for idx, (annotation, augmentation) in enumerate(zip(aug_annotations, augmentations)):
        name = get_name(path, idx+1+start_aug_idx)
        annotation.save(ade_config.save_path+ade_config.annotations_folder+name+ade_config.annotations_format)
        augmentation.save(ade_config.save_path+ade_config.images_folder+name+ade_config.images_format)
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
        annotation = index2color_annotation(np.array(annotation), ade_config.palette)
        axis[0,i+1].imshow(augmentation)
        axis[1,i+1].imshow(annotation)
        axis[2,i+1].imshow(augmentation)
        axis[2,i+1].imshow(annotation, alpha = 0.5)
        axis[2,i+1].set_xlabel(f"Image res: {augmentation.size} | GT res: {annotation.shape}")
    plt.savefig(ade_config.save_path+ade_config.vis_folder+name)
    plt.close()

import ade_config
class AbstractAde20k(TorchDataset):
    def __init__(self, start_idx, end_idx, prompt_type, seed = 42):
        data_paths = sorted(glob(ade_config.data_path+ade_config.images_folder+"*.jpg"))
        if((start_idx > 0) and (end_idx >= 0)):
            data_paths = data_paths[start_idx:end_idx]
            start_idx = start_idx
        elif(end_idx >= 0):
            data_paths = data_paths[:end_idx]
        elif(start_idx > 0):
            data_paths = data_paths[start_idx:]
            start_idx = start_idx
        self.annotations_dir = ade_config.data_path+ade_config.annotations_folder
        self.prompts_dir = ade_config.data_path+ade_config.prompts_folder
        self.prompts_dir = f"{self.prompts_dir}/{prompt_type}/"
        self.data_paths = data_paths
        self.seed = seed
        self.transform = totensor_transform

    def __len__(self):
        return len(self.data_paths)

class Ade20kPromptDataset(AbstractAde20k):
    def __init__(self, start_idx, end_idx, num_captions_per_image, seed = 42):
        super().__init__(start_idx, end_idx, "", seed)
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
    def __init__(self, start_idx, end_idx, prompt_type, copy_data = True, seed = 42, crop = False):
        super().__init__(start_idx, end_idx, prompt_type, seed)
        # self.data_paths = ['./data/ade/ADEChallengeData2016//images/training/ADE_train_00000082.jpg']
        if(crop):
            self.aspect_resize = torchvision.transforms.Compose([torchvision.transforms.Resize(size=512), torchvision.transforms.CenterCrop((512,512))])
        else:
            self.aspect_resize = torchvision.transforms.Resize(size=512)
        self.copy_data = copy_data


    def __getitem__(self, idx): 
        path = self.data_paths[idx]
        # open image
        init_image = self.aspect_resize(Image.open(path)) # resize shortest edge to 512
        init_image = np.array(init_image)
        if(len(init_image.shape) != 3):
            init_image = np.stack([init_image,init_image,init_image], axis = 0).transpose(1,2,0)
        
        # open prompt
        try:
            prompt = read_txt(self.prompts_dir+Path(path).stem+"_0000"+ade_config.prompts_format)[0]
        except: 
            prompt = ""
        # open mask
        annotation_path = self.annotations_dir+Path(path).stem+ade_config.annotations_format
        annotation = Image.open(annotation_path)
        annotation = self.aspect_resize(annotation)
        annotation = np.array(annotation)

        condition = self.transform(index2color_annotation(annotation, ade_config.palette))


        
        if(self.copy_data):
            # copy annotation and init image
            name = get_name(path, 0)
            shutil.copy(annotation_path, ade_config.save_path+ade_config.annotations_folder+name+ade_config.annotations_format)
            shutil.copy(path, ade_config.save_path+ade_config.images_folder+name+ade_config.images_format)

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
    parser.add_argument('--num_augmentations', type = int, default=1)
    parser.add_argument('--seed', type = int, default=4)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--batch_size', type = int, default=4)
    parser.add_argument('--vis_every', type = int, default=1)
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--controlnet', type=str, choices=["1.1", "1.0", "2.1"], default="1.1")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--crop', action='store_true')


    parser.add_argument('--prompts', type=str, choices=["gt", "blip2", "llava", "llava_gt"], default="gt")

    
    parser.add_argument('--negative_prompt', type=str, default="low quality, bad quality, sketches")
    parser.add_argument('--additional_prompt', type=str, default=", realistic looking, high-quality, extremely detailed") # , high-quality, extremely detailed, 4K, HQ
    parser.add_argument('--controlnet_conditioning_scale', type = float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--inference_steps', type=int, default=80)


    parser.add_argument('--start_idx', type = int, default=0)
    parser.add_argument('--end_idx', type = int, default=-1)

    parser.add_argument('--start_idx_aug', type = int, default=0)



    args = parser.parse_args()
    print(f"Parameters: {args}")


    start_time = time.time()

    ade_config.save_path = ade_config.save_path+"/"+args.experiment_name+"/"
    print(f"Save to: {ade_config.save_path}")

    
    os.makedirs(ade_config.save_path, exist_ok=True)
    os.makedirs(ade_config.save_path+ade_config.images_folder, exist_ok=True)
    os.makedirs(ade_config.save_path+ade_config.annotations_folder, exist_ok=True)
    os.makedirs(ade_config.save_path+ade_config.vis_folder, exist_ok=True)

    with open(ade_config.save_path +'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # with open('commandline_args.txt', 'r') as f:
        # args.__dict__ = json.load(f)


    # PROMPT GENERATION
    # check if prompts exist, if not generate prompts
    if(args.prompts == "blip2"):
        if(not Path(ade_config.data_path+ade_config.prompts_folder+"/blip2/").is_dir()): 
            os.makedirs(ade_config.data_path+ade_config.prompts_folder+"/blip2/", exist_ok=True)
            if(args.local):
                model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

            else:
                processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
                model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16, device_map="auto", load_in_8bit=True,)
            
            # dataset = Ade20kPromptDataset(args.start_idx, args.end_idx, args.num_augmentations, args.seed)
            dataset = Ade20kPromptDataset(args.start_idx, args.end_idx, 1, args.seed)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            for paths, aug_paths in tqdm(dataloader, desc="Generating prompts"): 
                if(args.local):
                    prompts = image2text_gpt2(model, list(paths), args.seed)
                else: 
                    prompts = image2text_blip2(model, processor, list(paths), args.seed)
                for p, prompt in zip(aug_paths, prompts):
                    write_txt(ade_config.data_path+ade_config.prompts_folder+"/blip2/"+p+ade_config.prompts_format, prompt)
                    

        # copy prompts to new data folder 
        os.makedirs(ade_config.save_path+ade_config.prompts_folder+"/blip2/", exist_ok=True)
        for filename in glob(os.path.join(ade_config.data_path+ade_config.prompts_folder+"/blip2/", '*'+ade_config.prompts_format)):
            shutil.copy(filename, ade_config.save_path+ade_config.prompts_folder+"/blip2/")
    elif(args.prompts == "gt"):
        if(not Path(ade_config.data_path+ade_config.prompts_folder+"/gt/").is_dir()): 
            os.makedirs(ade_config.data_path+ade_config.prompts_folder+"/gt/", exist_ok=True)
            dataset = Ade20kPromptDataset(args.start_idx, args.end_idx, 1, args.seed)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            for paths, aug_paths in tqdm(dataloader, desc="Generating prompts"):                 
                for p in aug_paths:
                    mask = np.array(Image.open(ade_config.data_path + ade_config.annotations_folder + "_".join(p.split("_")[:-1]) + ade_config.annotations_format))
                    available_classes = np.unique(mask)
                    class_names = [ade_config.classes[i] for i in available_classes][1:]
                    prompt = ", ".join(class_names)
                    write_txt(ade_config.data_path+ade_config.prompts_folder+"/gt/"+p+ade_config.prompts_format, prompt)
                    

        # copy prompts to new data folder 
        os.makedirs(ade_config.save_path+ade_config.prompts_folder+"/gt/", exist_ok=True)
        for filename in glob(os.path.join(ade_config.data_path+ade_config.prompts_folder+"/gt/", '*'+ade_config.prompts_format)):
            shutil.copy(filename, ade_config.save_path+ade_config.prompts_folder+"/gt/")



    # load controlnet
    if(args.controlnet == "2.1"):
        checkpoint = "thibaud/controlnet-sd21-ade20k-diffusers" # ""
        sd_ckpt = "stabilityai/stable-diffusion-2-1-base"
    elif(args.controlnet =="1.1"):
        checkpoint = "lllyasviel/control_v11p_sd15_seg" # Trained on COCO and Ade
        sd_ckpt = "runwayml/stable-diffusion-v1-5"
    elif(args.controlnet =="1.0"):
        checkpoint = "lllyasviel/sd-controlnet-seg" # Only trained on Ade
        sd_ckpt = "runwayml/stable-diffusion-v1-5"
    controlnet = ControlNetModel.from_pretrained(checkpoint) #, torch_dtype="auto") #torch.float16)    
    # load controlnet from pretrained checkpoint
    if(args.checkpoint != ""):
        controlnet_pipe = StableDiffusionControlNetPipeline.from_single_file(args.checkpoint, controlnet = controlnet)
    else:
        controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet) #, torch_dtype="auto") #torch.float16)



    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
    controlnet_pipe.enable_model_cpu_offload()
    controlnet_pipe.set_progress_bar_config(disable=True)

    # get data
    dataset = Ade20kDataset(args.start_idx, args.end_idx, args.prompts, args.seed, crop = args.crop)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean_time_img = []
    total_nsfw = 0
    batch_size = args.batch_size
    for img_idx, (init_img, condition, annotation, prompt, path) in enumerate(dataloader):
        starttime_img = time.time()

        print(prompt)

        # TODO reduce batch size based on resolution
        max_res = 500000
        current_res = init_img.shape[1]*init_img.shape[2] # TODO change to condition
        batch_size = int(np.floor(max_res / current_res * args.batch_size))
        if(batch_size == 0):
            resize= torchvision.transforms.Resize((init_img.shape[1]//2, init_img.shape[2]//2))
            condition = resize(condition)
            annotation = resize(annotation)
            # current_res = init_img.shape[1]*init_img.shape[2]
            batch_size = 1 #int(np.floor(max_res / current_res * args.batch_size))

            print(f"INFO::Resize image to {(init_img.shape[1]//2, init_img.shape[2]//2)}. Initial shape was {(init_img.shape[1], init_img.shape[2])}")


        # get augmentations
        augmentations = []
        aug_annotations = []
        while(len(augmentations)<args.num_augmentations):            
            curr_batch_size = np.min((batch_size, (args.num_augmentations - len(augmentations))))
            augmented, num_nsfw = augment_image_controlnet(controlnet_pipe, condition, prompt[0], 
                                                           condition.shape[-2], condition.shape[-1], curr_batch_size, 
                                                           negative_prompt=args.negative_prompt, 
                                                           additional_prompt=args.additional_prompt, 
                                                           controlnet_conditioning_scale=args.controlnet_conditioning_scale, 
                                                           guidance_scale=args.guidance_scale, 
                                                           inference_steps=args.inference_steps)
            total_nsfw += num_nsfw
            augmentations.extend(augmented)

            transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(augmented[0].size[::-1]), torchvision.transforms.ToPILImage()])
            aug_annotation = transform(annotation[0])
            aug_annotations.extend([aug_annotation]*len(augmented))
            # image = transform(image)
            # nsfw += 1

        # save augmentations
        save_augmentations_with_gt(aug_annotations, augmentations, path[0], args.start_idx_aug)

        if((args.vis_every >0 ) and ((img_idx %args.vis_every) == 0)): 
            visualize(aug_annotations, augmentations, init_img[0], condition[0].permute(1,2,0), prompt, Path(path[0]).stem)
        
        endtime_img = time.time()
        elapsedtime_img = endtime_img - starttime_img
        mean_time_img.append(elapsedtime_img)
        remaining_time = np.mean(mean_time_img)*(len(dataset)-img_idx)
        elapsedtime_img_str = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsedtime_img))
        remainingtime_img_str = str(timedelta(seconds=remaining_time))
        # remainingtime_img_str = time.strftime("%Hh%Mm%Ss", time.gmtime(remaining_time))
        print(f"Image {img_idx+args.start_idx}/{len(dataset)+args.start_idx} | Resolution = {init_img.shape} | Batch size = {batch_size} | Number of augmentations = {len(augmentations)} | Time for image = {elapsedtime_img_str} | Average time for image = {str(timedelta(seconds=np.mean(mean_time_img)))} | Remaining time = {remainingtime_img_str} | {total_nsfw}/{len(augmentations)*(img_idx+1)} = {int((total_nsfw*100)/(len(augmentations)*(img_idx+1)))}% contain NSFW")

    end_time = time.time()
    elapsedtime = end_time - start_time
    elapsedtime_str = str(timedelta(seconds=elapsedtime))
    print(f"Time to generate {args.num_augmentations} augmentations for {len(dataset)} images was {elapsedtime_str}")


    
