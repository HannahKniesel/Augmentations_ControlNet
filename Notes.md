# Start docker locally
```bash
    docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v ~/Documents/Code/RL_SyntheticData/Augmentations_ControlNet/:/workspace/ -v /media/hansel/Volume/Datasets/CV/coco_stuff10k/:/workspace/coco_stuff10k/ -w /workspace/ --name augmentation_c hannahkniesel/augmentation_controlnet bash
```

# Run Code for augmentations 
```bash
    python Augmentations.py --coco_path "./coco_stuff10k/" --save_path "./coco_stuff10k_augmented/" --prompt_definition vqa --iterative_img --save_number_images 50 --num_augmentations 4

```