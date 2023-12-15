# Start docker locally
```bash
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /media/hansel/SSD/Code/SyntheticData/Augmentations_ControlNet/:/Augmentations_ControlNet/ -v /media/hansel/SSD/Data/StandardCV/:/Augmentations_ControlNet/data/ -w /Augmentations_ControlNet/ --name augmentation_c hannahkniesel/augmentation_controlnet bash

# install mmsegmentation
pip install . 

# install diffusers library from source
apt-get update 
apt install git-all -Y
pip install git+https://github.com/huggingface/diffusers.git transformers accelerate xformers==0.0.16 wandb
pip install torchvision==0.14.1
accelerate config default
pip install datasets
```

# Run Code Sampling Experiment
```bash

# 1. Train segmentation model on real data 

# 2. Generate Augmentations using ControlNet
python -u DataGeneration.py --experiment_name baseline --num_augmentations 100 --batch_size 4 --vis_every -1 --end_idx -1

# 3. Compute uncertainties for the generated data
python ComputeUncertainties.py --data_path ./data/ade_augmented/baseline/ --num_uncertainty_samples 5 --batch_size 8 --model_path ./SegmentationModel/

# 4. Write txt files to filter data 
python FilterData.py --data_path ./data/ade_augmented/baseline/ --filter_by AL --path_to_aldict ./data/ade_augmented/test/SegmentationModel/entropy --num_augmentations 5 

# 5. Train segmentation model with additional synthetic data

```

# Run Code Finetuning Experiment