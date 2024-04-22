import argparse
import os
import wandb
import time 
import json
import torch 
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

from torch.utils.data import DataLoader

from Datasets import SyntheticAde20kDataset
from Utils import device
from Uncertainties import mcdropout_loss, entropy_loss, lcu_loss, lmu_loss, smu_loss
import ade_config

# Log number of black images 


if __name__ == "__main__":

    print("******************************")
    print("COMPUTE AVERAGE UNCERTAINTY OVER SYNTHETIC DATASET")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')

    # General Parameters
    parser.add_argument('--experiment_name', type = str, default="")
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--uncertainty', type=str, nargs="+", choices=["entropy", "mc_dropout", "lcu", "lmu", "smu"], default=["mc_dropout"])
    parser.add_argument('--data_path', type=str, default="./data/ade_augmented/finetuned_cn10/") 
    parser.add_argument('--model_path', type=str, default="./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/")
    parser.add_argument('--remove_black_images', action='store_true')



    args = parser.parse_args()
    print(f"Parameters: {args}")

    if("mc_dropout" in args.uncertainty):
        args.model_path = args.model_path + "/train_model_scripted.pt"
        print("WARNING:: MCDropout uncertainty detected. Open model in train mode.")
    else: 
        args.model_path = args.model_path + "/eval_model_scripted.pt"
        print("INFO::Open model in eval mode.")


    seg_model_name = Path(args.model_path).parent.parent.stem

    if(args.wandb_project != ""):
        os.environ['WANDB_PROJECT']= args.wandb_project
        wandb.init(config = args, reinit=True, mode="online")
        # wandb_name = self.wandb_name+"_"+str(wandb.run.id)
        wandb.run.name = f"{seg_model_name}_{args.experiment_name}_{wandb.run.id}"

    start_time = time.time()

    dataset = SyntheticAde20kDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    seg_model = torch.jit.load(args.model_path)
    seg_model = seg_model.to(device)


    try: 
        with open(args.data_path +'/uncertainties.txt', 'r') as f:
            results = json.load(f)
            print(f"INFO::Found logged uncertainties at {args.data_path}/uncertainties.txt.")
            print(results)
            print()

            if(not(seg_model_name in results.keys())):
                results[seg_model_name] = {}
                print(f"INFO::Model not found in uncertainties. Make new entry.")
            else: 
                to_remove = []
                for k in args.uncertainty: 
                    if k in results[seg_model_name].keys():
                        print(f"WARNING:: There are already entries for {k} in the existing dict ({results[seg_model_name][k]}). Do not overwrite these entries.")
                        to_remove.append(k)
                    
                args.uncertainty = [k for k in args.uncertainty if not(k in to_remove)]

    except:
        print(f"INFO::Could not find logged uncertainties at {args.data_path}/uncertainties.txt. Hence make new file.")
        results = {seg_model_name : {}}

    if(args.remove_black_images):
        black_images_path = f"{args.data_path}/{ade_config.images_folder.split("/")[:-1]}/black_images/"
        black_annotations_path = f"{args.data_path}/{ade_config.annotations_folder.split("/")[:-1]}/black_images/"
        os.makedirs(black_images_path, exist_ok = True)
        os.makedirs(black_annotations_path, exist_ok = True)
        print(f"INFO::Generate folder for black images at {black_images_path} and at {black_annotations_path}")

    
    PRINT_STEP = 1000
    black_img_counter = 0
    with torch.no_grad():
        for idx, (init_image, condition, annotation, prompt, path) in enumerate(dataloader):
            if((idx % PRINT_STEP) == 0): 
                print(f"INFO:: Image {idx}/{len(dataloader)}")
            if(init_image.max() == init_image.min()):
                black_img_counter += 1
                if(args.remove_black_images):
                    curr_img_path =  str(Path(path[0]).stem) + ade_config.images_format
                    curr_ann_path =  str(Path(path[0]).stem) + ade_config.annotations_format
                    print(f"INFO::Move black image from {args.data_path + ade_config.images_folder + curr_img_path} to {black_images_path + curr_img_path}")
                    print(f"INFO::Move annotation from black image from {args.data_path + ade_config.annotations_folder + curr_ann_path} to {black_annotations_path + curr_ann_path}\n")
                    shutil.move(args.data_path + ade_config.images_folder + curr_img_path, black_images_path + curr_img_path)
                    shutil.move(args.data_path + ade_config.annotations_folder + curr_ann_path, black_annotations_path + curr_ann_path)
                    

            init_image = init_image.to(device)
            if("mc_dropout" in args.uncertainty):
                mc_dropout = float(mcdropout_loss(init_image, None, None, seg_model, mc_samples = 5)[0].cpu())  #generated_image, real_images, gt_mask, model, mc_samples = 5, visualize = False
                try: 
                    results[seg_model_name]["mc_dropout"].append(mc_dropout)
                except: 
                    results[seg_model_name]["mc_dropout"] = [mc_dropout]

            if("entropy" in args.uncertainty):
                entropy = float(entropy_loss(init_image, None, seg_model)[0].cpu())    # generated_image, real_images, model,
                try: 
                    results[seg_model_name]["entropy"].append(entropy)
                except: 
                    results[seg_model_name]["entropy"] = [entropy]

            if("lcu" in args.uncertainty):
                lcu = float(lcu_loss(init_image, None, seg_model)[0].cpu())  
                try: 
                    results[seg_model_name]["lcu"].append(lcu)
                except: 
                    results[seg_model_name]["lcu"] = [lcu]

            if("lmu" in args.uncertainty):
                lmu = float(lmu_loss(init_image, None, seg_model)[0].cpu())
                try: 
                    results[seg_model_name]["lmu"].append(lmu)
                except: 
                    results[seg_model_name]["lmu"] = [lmu]
            
            if("smu" in args.uncertainty):
                smu = float(smu_loss(init_image, None, seg_model)[0].cpu())
                try: 
                    results[seg_model_name]["smu"].append(smu)
                except: 
                    results[seg_model_name]["smu"] = [smu]

    print("INFO::Compute mean and std...")
    for key in args.uncertainty: #results[seg_model_name]: 
        values = results[seg_model_name][key]
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[seg_model_name][key] = {"mean": mean_val, "std": std_val}

        if(args.wandb_project != ""):
            wandb.log({f"{key}_Mean" : mean_val})
            wandb.log({f"{key}_Std" : std_val})
    
    results["black_images"] = black_img_counter
    results["black_images_percent"] = int(100*black_img_counter/len(dataloader))

    # save args parameter to json 
    # Could be loaded with: 
    # with open('commandline_args.txt', 'r') as f:
        # args.__dict__ = json.load(f)
    with open(args.data_path +'/uncertainties.txt', 'w') as f:
        json.dump(results, f, indent=2)

    print(results)
    print(f"INFO::Save results to {args.data_path}/uncertainties.txt")

    

    

            

