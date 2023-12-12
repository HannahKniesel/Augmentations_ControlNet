export WANDB_API_KEY="f9ac711f43521f970835a198be72917607413691"
wandb init
wandb login

export SD_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/sd-controlnet-seg"
export OUTPUT_DIR="controlnet_model_out"


accelerate launch FinetuneControlNet.py \
 --pretrained_model_name_or_path=$SD_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./data/ade/ADEChallengeData2016/annotations/training/ADE_train_00000001.png" "./data/ade/ADEChallengeData2016/annotations/training/ADE_train_00020183.png" \
 --validation_prompt "a fountain" "a bunk bed, a window, and a woman" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision="fp16" \
 --report_to wandb 


# accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --controlnet_model_name_or_path="lllyasviel/sd-controlnet-seg" \
#  --output_dir="controlnet_model_out" \
#  --dataset_name=multimodalart/facesyntheticsspigacaptioned \
#  --conditioning_image_column=spiga_seg \
#  --image_column=image \
#  --caption_column=image_caption \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "./face_landmarks1.jpeg" "./face_landmarks2.jpeg" "./face_landmarks3.jpeg" \
#  --validation_prompt "High-quality close-up dslr photo of man wearing a hat with trees in the background" "Girl smiling, professional dslr photograph, dark background, studio lights, high quality" "Portrait of a clown face, oil on canvas, bittersweet expression" \
#  --train_batch_size=4 \
#  --num_train_epochs=3 \
#  --tracker_project_name="controlnet" \
#  --enable_xformers_memory_efficient_attention \
#  --checkpointing_steps=5000 \
#  --validation_steps=5000 \
#  --report_to wandb \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --use_8bit_adam \
#  --set_grads_to_none


#   --push_to_hub \ 


