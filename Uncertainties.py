import torch 
import torchvision
import numpy as np

from Utils import device, totensor_transform

softmax = torch.nn.Softmax(dim = 1)
crossentropy = torch.nn.CrossEntropyLoss(reduction="none")



def forward_model(input,model):
    # input = input.cuda()
    out = model(input) #, mode="tensor") 
    # return features of mask2former to compute uncertainties
    if(type(out) == tuple):
        cls_pred_list, mask_pred_list = out
        mask_cls_results = cls_pred_list[-1]
        mask_pred_results = mask_pred_list[-1]
        cls_score = torch.nn.functional.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        out = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
    return out

#     best = "maximum"
def mcdropout_loss(input, real_images, gt_mask, model, mc_samples = 5, visualize = False):
    # TODO fix torch random seed? 
    torch.manual_seed(0)
    # mc_dropout
    outputs = []
    for i in range(mc_samples):
        out = forward_model(input,model)
        outputs.append(out)
    outputs = torch.stack(outputs).permute(1,0,2,3,4)
    std = torch.std(outputs, axis=(1)) # compute std over samples

    if(visualize):
        uncertainty_img = torch.mean(std, axis = (1)).detach().cpu()
        # print(f"Shape uncertainty img: {uncertainty_img.shape} | Shape std: {std.shape}")
        uncertainty = -1*torch.mean(std, axis = (1,2,3)) # get mc dropout uncertainty: mean std over all classes and pixles
        return uncertainty, uncertainty_img
    uncertainty = -1*torch.mean(std, axis = (1,2,3)) # get mc dropout uncertainty: mean std over all classes and pixles

    return uncertainty,None

#     best = "minimum"
def smu_loss(input, real_images, gt_mask, model, visualize = False):
    out = forward_model(input,model)
    best_two = torch.topk(out, 2, dim = 1).values
    difference = best_two[:,0,:,:]-best_two[:,1,:,:]

    if(visualize):
        uncertainty_img = (1-difference).detach().cpu()
        uncertainty = torch.mean(difference, dim=(1,2)) 
        return uncertainty, -1*uncertainty_img

    uncertainty = torch.mean(difference, dim=(1,2)) 
    return uncertainty, None

#     best = "minimum"
def lmu_loss(input,real_images, gt_mask, model, visualize = False):
    out = forward_model(input,model)
    maximum = torch.max(out, dim = 1).values
    minimum = torch.min(out, dim = 1).values
    difference = maximum - minimum 

    if(visualize):
        uncertainty_img = (1-difference).detach().cpu()
        uncertainty = torch.mean(difference, dim=(1,2)) 
        return uncertainty, -1*uncertainty_img

    uncertainty = torch.mean(difference, dim=(1,2))
    return uncertainty, None

#     best = "minimum"
def lcu_loss(input, real_images, gt_mask, model, visualize = False):
    out = forward_model(input,model)
    maximum = torch.max(out, dim = 1).values
    
    if(visualize):
        uncertainty_img = (1-maximum).detach().cpu()
        uncertainty = torch.mean(maximum, dim=(1,2)) 
        return uncertainty, -1*uncertainty_img
    
    uncertainty = torch.mean(maximum, dim = (1,2))
    return uncertainty, None


def entropy(logits):
    probability = softmax(logits) # compute softmax over the class dimension to get probability of class --> shape: bs, classes, w, h
    entropy = -1*torch.sum(probability*torch.log(probability), dim = 1) # compute entropy over all classes (for each pixel value) --> shape: bs, w, h
    return entropy

def segment_entropy_loss(input, real_images, gt_mask, model, visualize = False):
    import pdb 
    pdb.set_trace()
    return


#     best = "maximum"
def entropy_loss(input, real_images, gt_mask, model, visualize = False):
    out = forward_model(input,model) # bs, classes, w, h
    probability = softmax(out) # compute softmax over the class dimension to get probability of class --> shape: bs, classes, w, h
    entropy = -1*torch.sum(probability*torch.log(probability), dim = 1) # compute entropy over all classes (for each pixel value) --> shape: bs, w, h

    if(visualize):
        uncertainty_img = entropy.detach().cpu()
        uncertainty = -1*torch.mean(entropy) 
        return uncertainty, uncertainty_img

    uncertainty = -1*torch.mean(entropy) # compute mean of the resulting entropy
    return uncertainty, None


# TODO torch batchify loss for images
def loss_brightness(images, real_images, gt_mask, model, visualize = False): 
    if(type(images) is list): 
        m = [-1*torch.mean(i) for i in images] #[-1*np.mean(np.array(i)) for i in images]
        return np.mean(m)

    if(visualize):
        return -1*torch.mean(images), (images[:,:,:,0]).detach().cpu()
    # blue_channel = images[:,:,:,2]  # N x 256 x 256
    # blue_channel = images[:,2,:,:]  # N x C x 256 x 256
    return -1*torch.mean(images), None

# idea: maximize difference to training data point to create dataset with the highest variance. 
def mse_loss(generated_images, real_images, gt_mask, model, visualize = False):
    real_images = (real_images).permute(0,3,1,2)/255
    centercrop = torchvision.transforms.CenterCrop((generated_images.shape[2], generated_images.shape[3]))
    real_images = centercrop(real_images).to(device)
    print(f"Generated images: {generated_images.shape} | Real images: {real_images.shape}")
    print(f"Generated images: {generated_images.dtype} | Real images: {real_images.dtype}")
    print(f"Generated images: {type(generated_images)} | Real images: {type(real_images)}")
    print(f"Generated images: {(generated_images).max()} | Real images: {(real_images).max()}")

    if(visualize):
        return -1*torch.nn.functional.mse_loss(generated_images, real_images), (generated_images[:,:,:,0]).detach().cpu()

    return -1*torch.nn.functional.mse_loss(generated_images, real_images), None
    # torch.Size([1, 3, 512, 680]) | Real images: torch.Size([1, 512, 683, 3])

