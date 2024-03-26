# import torch 
# from Utils import device
# from Uncertainties import entropy_loss

# seg_model_path = "./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/train_model_scripted.pt"
# seg_model = torch.jit.load(seg_model_path)
# seg_model = seg_model.to(device)

# save_image = torch.rand(1,3,512,512).to(device)
# uncertainty = entropy_loss(save_image, seg_model)

from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline
import torch 

checkpoint = "lllyasviel/control_v11p_sd15_seg" # Trained on COCO and Ade
sd_ckpt = "runwayml/stable-diffusion-v1-5"

controlnet = ControlNetModel.from_pretrained(checkpoint) #, torch_dtype="auto") #torch.float16)
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_ckpt, controlnet=controlnet) #, torch_dtype="auto") #torch.float16)
controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
# controlnet_pipe.enable_model_cpu_offload()
controlnet_pipe.set_progress_bar_config(disable=True)

latents = torch.randn((1, 4, 64, 97), requires_grad=True)
optimizer = torch.optim.SGD([latents], lr=1)


controlnet_pipe.vae.requires_grad_(False)
controlnet_pipe.unet.requires_grad_(False)
controlnet_pipe.text_encoder.requires_grad_(False)
controlnet_pipe.controlnet.requires_grad_(False)

decoded_image = controlnet_pipe.vae.decode(latents / controlnet_pipe.vae.config.scaling_factor, return_dict=False)[0]
decoded_image = controlnet_pipe.image_processor.denormalize(decoded_image)
loss = torch.mean(decoded_image)

loss.backward()
print(f"Gradients Final: {latents.grad}")
optimizer.step()
optimizer.zero_grad(set_to_none=True)
