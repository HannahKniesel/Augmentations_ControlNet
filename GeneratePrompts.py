import argparse
from pathlib import Path
import os
from torch.utils.data import DataLoader
from transformers import BlipProcessor, Blip2ForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, pipeline

from tqdm import tqdm
import torch
import numpy as np 
from PIL import Image

import ade_config
from Datasets import Ade20kPromptDataset
from Utils import image2text_blip2, image2text_llava, image2text_llava_gt, write_txt


if __name__ == "__main__":

    print("******************************")
    print("GENERATE PROMPTS")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--batch_size', type = int, default=4)
    parser.add_argument('--seed', type = int, default=7353)
    parser.add_argument('--prompt_type', type=str, choices=["gt", "blip2", "llava", "llava_gt"], default="blip2")

    args = parser.parse_args()
    print(f"Parameters: {args}")


    # PROMPT GENERATION
    # prompt_path = os.path.join(ade_config.data_path, ade_config.prompts_folder, args.prompt_type)
    prompt_path = f"{ade_config.data_path}/{ade_config.prompts_folder}/{args.prompt_type}/"
    assert not os.path.isdir(prompt_path), f"Prompt path {prompt_path} already exists."
    os.makedirs(prompt_path, exist_ok=True)

    dataset = Ade20kPromptDataset(-1, -1, 1, args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if(args.prompt_type == "blip2"): 
        processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16, device_map="auto", load_in_8bit=True,)

        for paths, aug_paths in tqdm(dataloader, desc="Generating prompts"): 
            prompts = image2text_blip2(model, processor, list(paths), args.seed)
            for p, prompt in zip(aug_paths, prompts):
                write_txt(os.path.join(prompt_path,p)+ade_config.prompts_format, prompt)
                    

    elif(args.prompt_type == "gt"):
        for paths, aug_paths in tqdm(dataloader, desc="Generating prompts"):                 
            for p in aug_paths:
                mask = np.array(Image.open(ade_config.data_path + ade_config.annotations_folder + "_".join(p.split("_")[:-1]) + ade_config.annotations_format))
                available_classes = np.unique(mask)
                class_names = [ade_config.classes[i] for i in available_classes][1:]
                prompt = ", ".join(class_names)
                write_txt(os.path.join(prompt_path,p)+ade_config.prompts_format, prompt)

    elif(args.prompt_type == "llava"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
        for paths, aug_paths in tqdm(dataloader, desc="Generating prompts"): 
            prompts = image2text_llava(processor, list(paths), args.seed)
            for p, prompt in zip(aug_paths, prompts):
                write_txt(os.path.join(prompt_path,p)+ade_config.prompts_format, prompt)

    elif(args.prompt_type == "llava_gt"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
        for paths, aug_paths in tqdm(dataloader, desc="Generating prompts"): 
            prompts = image2text_llava_gt(processor, list(paths), args.seed)
            for p, prompt in zip(aug_paths, prompts):
                write_txt(os.path.join(prompt_path,p)+ade_config.prompts_format, prompt)


    print(f"INFO:: Saved image prompts to {os.path.join(prompt_path,p,ade_config.prompts_format)}.")
                    
