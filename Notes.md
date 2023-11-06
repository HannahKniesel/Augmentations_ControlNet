# Start docker locally
```bash
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v ~/Documents/Code/RL_SyntheticData/Augmentations_ControlNet/:/workspace/ -v ~/Documents/Data/StandardCV/coco_stuff10k/:/workspace/coco_stuff10k/ -w /workspace/ --name augmentation_c hannahkniesel/augmentation_controlnet bash

    docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /media/hansel/SSD/Code/SyntheticData/Augmentations_ControlNet/:/Augmentations_ControlNet/ -v /media/hansel/SSD/Data/StandardCV/:/Augmentations_ControlNet/data/ -w /Augmentations_ControlNet/ --name augmentation_c hannahkniesel/augmentation_controlnet bash
```

# Run Code for augmentations 
```bash
    python Augmentations.py --coco_path "./coco_stuff10k/" --save_path "./coco_stuff10k_augmented/" --prompt_definition vqa --iterative_img --save_number_images 50 --num_augmentations 4
    python DataGeneration_CocoStuff.py --coco_path "./coco_stuff10k/" --save_path "./coco_stuff10k_augmented/" --prompt_definition vqa --iterative_img --num_augmentations 4
    python DataGeneration.py --prompt_definition img2text --num_augmentations 4 --dataset ade --condition canny --start_idx 1797
```