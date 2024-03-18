import torch 
from Utils import device
from Uncertainties import entropy_loss

seg_model_path = "./seg_models/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/train_model_scripted.pt"
seg_model = torch.jit.load(seg_model_path)
seg_model = seg_model.to(device)

save_image = torch.rand(1,3,512,512).to(device)
uncertainty = entropy_loss(save_image, seg_model)