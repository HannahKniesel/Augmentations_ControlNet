# Train ControlNet
First, we require a well trained ControlNet on the collected prompts.

## Installation 
https://huggingface.co/docs/diffusers/training/controlnet

cd diffusers
pip install . 

cd examples/controlnet
pip install -r requirements_sdxl.txt

## Run docker image
```bash
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /media/hansel/SSD/Code/SyntheticData/ControlNet_HF/:/ControlNet_HF/ -v /media/hansel/SSD/Data/StandardCV/:/ControlNet_HF/data/ -w /ControlNet_HF/ --name cn_c hannahkniesel/cn_hf bash
```

## Prepare dataset with prompts
Prepare the datasets as huggingface dataset with the corresponding prompts.
```bash
python prepare_ade20k.py
```
Adapt lines 13 - 19:
```bash
additional_prompt = ", realistic looking, high-quality, extremely detailed"
dataset_name = "hannahkniesel/ade20k_gt_ap"
private = False

imgs_paths = sorted(glob.glob("./data/ade/ADEChallengeData2016/images/training/*.jpg"))
annotations_paths = sorted(glob.glob("./data/ade/ADEChallengeData2016/annotations/training/*.png"))
text_path = "./data/ade/ADEChallengeData2016/prompts/training/gt/"
```

## Train ControlNet

```bash
# 3. Train ControlNet
cd /ControlNet_HF/diffusers/examples/controlnet/

export WANDB_API_KEY="f9ac711f43521f970835a198be72917607413691"
wandb init

export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export RESUME_DIR="lllyasviel/sd-controlnet-seg" 
export OUTPUT_DIR="trained_model/CN10_ade20k_gt_ap"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$RESUME_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=hannahkniesel/ade20k_gt_ap \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=120000 \
 --validation_image 'validation/ADE_train_00007970.png' 'validation/ADE_train_00011548.png' 'validation/ADE_train_00012753.png' 'validation/ADE_train_00015154.png' 'validation/ADE_train_00017975.png'  \
 --validation_prompt 'sky, tree, person, earth, mountain, railing, stairs' 'wall, floor, ceiling, cabinet, door, column, chandelier' 'building, sky, tree, road, grass, sidewalk, door, car, streetlight' 'wall, floor, ceiling, painting, desk, fireplace, book, stool, vase' 'wall, building, sky, tree, road, sidewalk, plant, car, signboard, streetlight'  \
 --train_batch_size=4 \
 --gradient_accumulation_steps=64 \

```

## Train SDXL 
VAE for SDXL is unstable. Using mixed precision fp16 can lead to black images. 

https://huggingface.co/madebyollin/sdxl-vae-fp16-fix

--> Use different VAE when using fp16 or try not to use mixed precision training

Use with different VAE during inference:

```bash
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
```
https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9
Also use SDXL0.9 ("stabilityai/stable-diffusion-xl-base-0.9") based on compatibility reasons to sdxl-vae-fp16-fix
https://huggingface.co/stabilityai/sdxl-vae/discussions/6
Licence to SDXL 0.9: https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/blob/main/LICENSE.md

Default config that seemed to work:
https://github.com/huggingface/diffusers/issues/4185#issuecomment-1645544125



# Segmentation Training 
uses mmsegmentation + additional hooks for logging

```bash
# clone git repo
git clone git@github.com:open-mmlab/mmsegmentation.git

# run docker
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /media/hansel/SSD/Code/SyntheticData/mmsegmentation_basic/:/mmsegmentation/ -v /media/hansel/SSD/Data/StandardCV/:/mmsegmentation/data/ -w /mmsegmentation/ --name mmsegmentation_c hannahkniesel/mmsegmentation:v03 bash

# convert trained model to jit model -> saves model in ckpt_file directory 
python convert_to_jit.py --config_file ./configs/sem_fpn/fpn_r50_4xb4-160k_ade20k-512x512_noaug.py --ckpt_file "./work_dirs/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/best_mIoU_epoch_136.pth" 
```

# Generate Data

## Run docker 
```bash 
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /media/hansel/SSD/Code/SyntheticData/Augmentations_ControlNet/:/Augmentations_ControlNet/ -v /media/hansel/SSD/Data/StandardCV/:/Augmentations_ControlNet/data/ -v /media/hansel/SSD/Code/SyntheticData/CN_Training/ControlNet/models/:/Augmentations_ControlNet/models/ -v /media/hansel/SSD/Code/SyntheticData/segmentationAL/work_dirs/:/Augmentations_ControlNet/seg_models/ -v /media/hansel/SSD/Code/SyntheticData/ControlNet_HF/diffusers/examples/controlnet/trained_model/:/Augmentations_ControlNet/controlnet/ -w /Augmentations_ControlNet/ --name augmentation_c hannahkniesel/augmentation_controlnet bash
```

## Generate and Validate data

```bash

# 1. Generate Data
python -u GenerateData.py --experiment_name "debug" --controlnet "1.0" --finetuned_checkpoint "./controlnet/CN10_ade20k_gt_ap/checkpoint-20000/controlnet" --prompt_type "gt" --wandb_project "Debug" --inference_steps 5 --optimize --lr 10 --loss lcu --iters 10 --start_t 0 --end_t 1 --mixed_precision "bf16" --end_idx 10 --model_path "./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/" --num_augmentations 1
# use --wandb_mode "detailed" for debugging purposes (defaults to "standard")

# 2. Evaluate based on uncertainty and remove black images
python -u AvgUncertainty.py --uncertainty mc_dropout --data_path ./data/ade_augmented/uncertainty/baseline/ --model_path "./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/" --remove_black_images
# Saves the result to {args.data_path}/uncertainties.txt

```

## Pick data from sampling pool for comparison
```bash
# 1. Generate data
python -u GenerateData.py --experiment_name "debug" --controlnet "1.0" --finetuned_checkpoint "./controlnet/CN10_ade20k_gt_ap/checkpoint-20000/controlnet" --prompt_type "gt" --wandb_project "Debug" --inference_steps 5 --loss lcu --mixed_precision "bf16" --end_idx 10 --model_path "./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/" --num_augmentations 100

# 2. Pick from pool
python -u PickImagedFromPool.py --sampling_pool_path "./data/ade_augmented/uncertainty/sampling_pool/" --save_to "./data/ade_augmented/uncertainty/sampled_lcu_10/" --model_path "./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/" --uncertainty lcu --top_n 10

# 3. Evaluate based on uncertainty and remove black images
python -u AvgUncertainty.py --uncertainty lcu --data_path ./data/ade_augmented/uncertainty/sampled_lcu_10/ --model_path "./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/" --remove_black_images
# Saves the result to {args.data_path}/uncertainties.txt

```

## Visualize
```bash
python VisualizeDataComparison2.py --comparisons ./data/ade_augmented/uncertainty/baseline/ ./data/ade_augmented/uncertainty/mc0/ ./data/ade_augmented/uncertainty/mc1/ ./data/ade_augmented/uncertainty/mc2/ ./data/ade_augmented/uncertainty/mc3/ ./data/ade_augmented/uncertainty/mc4/  --save_to "./Visualizations/MCDropout/" --n_images 50 --uncertainty "mcdropout"
# visualizes generated images within list side by side. Also shows uncertainty image when --uncertainty is defined.
```
## Generate subset of Ade20k
This is helpful for further ablations. 
```bash
python -u PickSubset.py
```

# How To
1. Train segmentation model with no augmentations + convert to jit model
2. Generate data based on pretrained segmentation model 
3. Remove black images and evaluate uncertainty of dataset
4. Retrain segmentation model with new augmentations

