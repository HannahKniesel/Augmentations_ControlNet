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
from Loss import loss_fct


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
    parser.add_argument('--data_path', type=str, default="./data/ade_augmented/finetuned_cn10/") 
    parser.add_argument('--easy_model', type=str, default="./base_models/03-06-2024/BestEpoch/eval_model_scripted.pt")
    parser.add_argument('--hard_model', type=str, default="./base_models/03-06-2024/EarlyStopping25/eval_model_scripted.pt")
    parser.add_argument('--w_hard', type=float, default=1.0)
    parser.add_argument('--w_easy', type=float, default=1.0)
    parser.add_argument('--remove_black_images', action='store_true')



    args = parser.parse_args()
    print(f"Parameters: {args}")

    


    easy_model_name = Path(args.easy_model).parent.stem
    hard_model_name = Path(args.hard_model).parent.stem
    exp_name = f"Easy-{easy_model_name}-Hard-{hard_model_name}"


    if(args.wandb_project != ""):
        os.environ['WANDB_PROJECT']= args.wandb_project
        wandb.init(config = args, reinit=True, mode="online")
        # wandb_name = self.wandb_name+"_"+str(wandb.run.id)
        wandb.run.name = f"{exp_name}_{args.experiment_name}_{wandb.run.id}"

    start_time = time.time()

    dataset = SyntheticAde20kDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    easy_model = torch.jit.load(args.easy_model)
    easy_model = easy_model.to(device)
    hard_model = torch.jit.load(args.hard_model)
    hard_model = hard_model.to(device)


    try: 
        with open(args.data_path +'/uncertainties.txt', 'r') as f:
            results = json.load(f)
            print(f"INFO::Found logged uncertainties at {args.data_path}/uncertainties.txt.")
            print(results)
            print()

            if(not(exp_name in results.keys())):
                results[exp_name] = {}
                print(f"INFO::Experiment not found in uncertainties. Make new entry '{exp_name}'.")
            else: 
                to_remove = []
                for k in args.uncertainty_loss: 
                    if k in results[exp_name].keys():
                        print(f"WARNING:: There are already entries for {k} in the existing dict ({results[exp_name][k]}). Do not overwrite these entries.")
                        to_remove.append(k)
                    
                args.uncertainty_loss = [k for k in args.uncertainty_loss if not(k in to_remove)]

    except:
        print(f"INFO::Could not find logged uncertainties at {args.data_path}/uncertainties.txt. Hence make new file.")
        results = {exp_name : {}}

    if(args.remove_black_images):
        black_images_path = f"{args.data_path}/black_images/"
        black_annotations_path = f"{args.data_path}/black_annotations/"
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
            easy_loss, hard_loss, loss, heatmap = loss_fct(init_image, 
                            annotation, 
                            easy_model, 
                            args.w_easy, 
                            hard_model, 
                            args.w_hard, 
                            visualize = True, 
                            by_value = True) 
            try: 
                results[exp_name]["CE-easy"].append(easy_loss)
            except: 
                results[exp_name]["CE-easy"] = [easy_loss]

            try: 
                results[exp_name]["CE-hard"].append(-1*hard_loss)
            except: 
                results[exp_name]["CE-hard"] = [-1*hard_loss]

            try: 
                results[exp_name]["Loss"].append(loss)
            except: 
                results[exp_name]["Loss"] = [loss]

            
    print("INFO::Compute mean and std...")
    for key in args.uncertainty_loss: #results[seg_model_name]: 
        values = results[exp_name][key]
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[exp_name][key] = {"mean": mean_val, "std": std_val}

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

    

    

            

