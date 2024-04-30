import argparse
import os
import json
from glob import glob
import shutil
from torch.utils.data import DataLoader
from diffusers import ControlNetModel, UniPCMultistepScheduler
import time
import torchvision
import numpy as np
from datetime import timedelta
import torch
import wandb 
from pathlib import Path

import ade_config
from Datasets import Ade20kDataset
from Utils import get_name, device
from CNPipeline import StableDiffusionControlNetPipeline as SDCNPipeline_Latents
from Uncertainties import loss_brightness, entropy_loss, mcdropout_loss, mse_loss

from Uncertainties import loss_brightness, entropy_loss, mcdropout_loss, smu_loss, lmu_loss, lcu_loss, segment_entropy_loss, min_max_segment_entropy_loss


# TODO load dotenv
# TODO batchify
# TODO make loss computation faster? Decoding to image space is costy.


def save_augmentations_with_gt(aug_annotations, augmentations, path, start_aug_idx):
    for idx, (annotation, augmentation) in enumerate(zip(aug_annotations, augmentations)):
        name = get_name(path, idx+1+start_aug_idx)
        annotation.save(ade_config.save_path+ade_config.annotations_folder+name+ade_config.annotations_format)
        augmentation.save(ade_config.save_path+ade_config.images_folder+name+ade_config.images_format)
    return




if __name__ == "__main__":

    print("******************************")
    print("GENERATE IMAGES")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--experiment_name', type = str, default="")
    parser.add_argument('--data_to_augment', type = str, default="./data/ade/ADEChallengeData2016/")
    parser.add_argument('--num_augmentations', type = int, default=1)
    parser.add_argument('--seed', type = int, default=7353)

    parser.add_argument('--controlnet', type=str, choices=["1.1", "1.0", "2.1"], default="1.1")
    parser.add_argument('--finetuned_checkpoint', type=str, default="")
    parser.add_argument('--prompt_type', type=str, choices=["gt", "blip2", "llava", "llava_gt", "short_llava_gt", "no_prompts"], default="gt")
    parser.add_argument('--negative_prompt', type=str, default="low quality, bad quality, sketches") # "low quality, bad quality, sketches, flat, unrealistic" 
    parser.add_argument('--additional_prompt', type=str, default=", realistic looking, high-quality, extremely detailed") # , realistic looking, high-quality, extremely detailed, 4K, HQ, photorealistic" # , high-quality, extremely detailed, 4K, HQ
    parser.add_argument('--controlnet_conditioning_scale', type = float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--inference_steps', type=int, default=80)

    parser.add_argument('--start_idx', type = int, default=0)
    parser.add_argument('--end_idx', type = int, default=-1)
    parser.add_argument('--start_idx_aug', type = int, default=0)

    # optimization parameters
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--wandb_mode', type=str, choices = ["off", "standard", "detailed"], default = "standard")
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--lr', type=float, default=1000.)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--optimizer', type=str, choices=["adam", "sgd"], default="sgd")
    parser.add_argument('--optim_every_n_steps', type=int, default=1)
    parser.add_argument('--start_t', type=int, default=0)
    parser.add_argument('--end_t', type=int, default=80)
    parser.add_argument('--loss', type=str, choices=["brightness", "entropy", "mcdropout", "smu", "lmu", "lcu", "mse", "segment_entropy", "min_max_segment_entropy"], default="min_max_segment_entropy")
    parser.add_argument('--w_pixel', type=float, default=1.0)
    parser.add_argument('--w_class', type=float, default=1.0)
    parser.add_argument('--model_path', type=str, default="./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/")

    parser.add_argument('--mixed_precision', type=str, choices=["bf16", "fp16"], default="bf16")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    if(args.loss == "brightness"):
        loss = loss_brightness
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.loss == "entropy"):
        loss = entropy_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.loss == "mcdropout"):
        loss = mcdropout_loss
        args.model_path = args.model_path + "train_model_scripted.pt"
    elif(args.loss == "smu"):
        loss = smu_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.loss == "lmu"):
        loss = lmu_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.loss == "lcu"):
        loss = lcu_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"    
    elif(args.loss == "mse"):
        loss = mse_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.loss == "segment_entropy"):
        loss = segment_entropy_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    elif(args.loss == "min_max_segment_entropy"):
        loss = min_max_segment_entropy_loss
        args.model_path = args.model_path + "eval_model_scripted.pt"
    else: 
        print(f"ERROR:: Could not match the defined loss {args.loss}")



    if(args.prompt_type == "no_prompts"):
        args.additional_prompt = ""
        args.negative_prompt = ""
        guess_mode = True
    else: 
        guess_mode = False

    if(not args.optimize):
        print("WARNING::Standard data generation. No optimization.")
        args.iters = 0

    optimization_params = {"do_optimize": args.optimize, 
                            "wandb_mode": args.wandb_mode, 
                            "lr": args.lr, 
                            "iters": args.iters,
                            "optimizer": args.optimizer, 
                            "optim_every_n_steps": args.optim_every_n_steps,
                            "start_t": args.start_t, 
                            "end_t": args.end_t,
                            "loss": loss, 
                            "w_pixel": args.w_pixel,
                            "w_class": args.w_class,
                            "mixed_precision": args.mixed_precision}
    

    if(bool(args.wandb_mode in ["standard", "detailed"])):
        os.environ['WANDB_PROJECT']= args.wandb_project
        """group = "Optimization" if optimization_params['do_optimize'] else "Base"
        if  optimization_params['do_optimize']: """
        group = args.loss
        wandb.init(config = optimization_params, reinit=True, group = group, mode="online")
        # wandb_name = self.wandb_name+"_"+str(wandb.run.id)
        name = f"{args.experiment_name}_{args.loss}_{wandb.run.id}"
        if(args.optimize):
            name = f"{name}_optim"
        
        wandb.run.name = name
    start_time = time.time()

    ade_config.save_path = ade_config.save_path+"/"+args.experiment_name+"/"
    print(f"Save to: {ade_config.save_path}")

    
    os.makedirs(ade_config.save_path, exist_ok=True)
    os.makedirs(ade_config.save_path+ade_config.images_folder, exist_ok=True)
    os.makedirs(ade_config.save_path+ade_config.annotations_folder, exist_ok=True)
    os.makedirs(ade_config.save_path+ade_config.vis_folder, exist_ok=True)
    

    # save args parameter to json 
    # Could be loaded with: 
    # with open('commandline_args.txt', 'r') as f:
        # args.__dict__ = json.load(f)
    with open(ade_config.save_path +'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    # copy prompts to new data folder for logging purposes
    save_prompt_path = f"{ade_config.save_path}/{ade_config.prompts_folder}/{args.prompt_type}/"
    load_prompt_path = f"{ade_config.data_path}/{ade_config.prompts_folder}/{args.prompt_type}/"
    os.makedirs(save_prompt_path, exist_ok=True)
    for filename in glob(f"{load_prompt_path}/*{ade_config.prompts_format}"):
        shutil.copy(filename, save_prompt_path)

    
    if(args.model_path != ""):
        seg_model = torch.jit.load(args.model_path)
        seg_model = seg_model.to(device)
    else: 
        seg_model = None


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
    
    if(args.finetuned_checkpoint != ""):
        controlnet = ControlNetModel.from_pretrained(args.finetuned_checkpoint) 
        print(f"INFO::load controlnet from finetuned checkpoint {args.finetuned_checkpoint}")
    else: 
        controlnet = ControlNetModel.from_pretrained(checkpoint) #, torch_dtype="auto") #torch.float16)
        print(f"INFO::load default controlnet {checkpoint}")

    controlnet_pipe = SDCNPipeline_Latents.from_pretrained(sd_ckpt, controlnet=controlnet) #, torch_dtype="auto") #torch.float16)
    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
    controlnet_pipe.enable_model_cpu_offload()
    controlnet_pipe.set_progress_bar_config(disable=True)

    # get data
    dataset = Ade20kDataset(args.start_idx, args.end_idx, args.prompt_type, root_path = args.data_to_augment, copy_data = True, seed = args.seed)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean_time_img = []
    mean_time_augmentation = []
    total_nsfw = 0
    avg_loss = []
    avg_loss_classentropy = []
    avg_loss_pixelentropy = []

    # iterate over dataset
    for img_idx, (init_img, condition, annotation, prompt, path) in enumerate(dataloader):
        starttime_img = time.time()
        print(prompt)

        
        # generate augmentations
        augmentations = []
        aug_annotations = []
        aug_index = 0
        while(len(augmentations)<args.num_augmentations):            
            # TODO include new pipeline
            generator = torch.manual_seed(0 + aug_index)
            aug_index += 1
            output, elapsed_time, loss, loss_classentropy, loss_pixelentropy = controlnet_pipe(prompt[0] + args.additional_prompt, #+"best quality, extremely detailed" # 
                                    negative_prompt=args.negative_prompt, 
                                    image=condition, 
                                    controlnet_conditioning_scale=args.controlnet_conditioning_scale, 
                                    guidance_scale = args.guidance_scale,
                                    num_inference_steps=args.inference_steps, 
                                    guess_mode = guess_mode,
                                    height = condition.shape[-2], 
                                    width = condition.shape[-1],
                                    num_images_per_prompt = 1, 
                                    generator=generator, 
                                    img_name = Path(path[0]).stem,
                                    optimization_arguments = optimization_params, 
                                    seg_model = seg_model, 
                                    real_image = init_img, 
                                    annotation = annotation,
                                    )

            #try:
            #    augmented = [elem for elem, nsfw in zip(output.images, output.nsfw_content_detected) if not nsfw]
            #    num_nsfw = np.sum(output.nsfw_content_detected)
            #except: 
            augmented = output.images
            num_nsfw = 0

            print(f"INFO:: Time elapsed = {elapsed_time} | Loss = {loss} | Loss Class Entropy = {loss_classentropy} | Loss Pixel Entropy = {loss_pixelentropy}")

            total_nsfw += num_nsfw
            augmentations.extend(augmented)
            avg_loss.append(loss)
            avg_loss_classentropy.append(loss_classentropy)
            avg_loss_pixelentropy.append(loss_pixelentropy)


            mean_time_augmentation.append(elapsed_time)

            # make sure annotation and images have similar resolution
            transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(augmented[0].size[::-1]), torchvision.transforms.ToPILImage()])
            aug_annotation = transform(annotation[0])
            aug_annotations.extend([aug_annotation]*len(augmented))
            

        # save augmentations
        save_augmentations_with_gt(aug_annotations, augmentations, path[0], args.start_idx_aug)

        
        endtime_img = time.time()
        elapsedtime_img = endtime_img - starttime_img
        mean_time_img.append(elapsedtime_img)
        remaining_time = np.mean(mean_time_img)*(len(dataset)-img_idx)
        elapsedtime_img_str = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsedtime_img))
        remainingtime_img_str = str(timedelta(seconds=remaining_time))
        print(f"Image {img_idx+args.start_idx}/{len(dataset)+args.start_idx} | \
              Avg Loss = {np.mean(avg_loss)} | \
              Avg Loss Classentropy = {np.mean(avg_loss_classentropy)} | \
              Avg Loss Pixelentropy = {np.mean(avg_loss_pixelentropy)} | \
              Number of augmentations = {len(augmentations)} | \
              Time for image = {elapsedtime_img_str} | \
              Avg time for image = {str(timedelta(seconds=np.mean(mean_time_img)))} | \
              Avg time per augmentation = {str(timedelta(seconds=np.mean(mean_time_augmentation)))} | \
              Remaining time = {remainingtime_img_str} | \
              {total_nsfw}/{len(augmentations)*(img_idx+1)} = {int((total_nsfw*100)/(len(augmentations)*(img_idx+1)))}% contain NSFW |\
              Crop Images = {dataset.resized_counter}")
        
        if(optimization_params["wandb_mode"] in ["standard", "detailed"]):
            wandb.log({"AvgLoss": np.mean(avg_loss), 
                    "AvgLoss Class Entropy": np.mean(avg_loss_classentropy), 
                    "AvgLoss Pixel Entropy": np.mean(avg_loss_pixelentropy), 
                    "AvgTime Augmentation": np.mean(mean_time_augmentation)})

    end_time = time.time()
    elapsedtime = end_time - start_time
    elapsedtime_str = str(timedelta(seconds=elapsedtime))
    print(f"Time to generate {args.num_augmentations} augmentations for {len(dataset)} images was {elapsedtime_str}")
    print(f"Average loss over dataset is {np.mean(avg_loss)}.")
    if(optimization_params["wandb_mode"] in ["standard", "detailed"]):
        wandb.log({"Dataset Loss": np.mean(avg_loss), 
                    "Final Time Avg" : np.mean(mean_time_augmentation)})





    

