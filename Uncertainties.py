import torch 
import torchvision
import numpy as np

from Utils import device, totensor_transform, index2color_annotation
import ade_config



softmax = torch.nn.Softmax(dim = 1)
crossentropy = torch.nn.CrossEntropyLoss(reduction="none")


"""#     best = "maximum"
def mcdropout_loss(input, real_images, gt_mask, model, mc_samples = 5, visualize = False):
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
def lcu_loss(input, real_images, gt_mask, model, w_pixel = 0, w_class = 1, visualize = False):
    out = forward_model(input,model)

    maximum = torch.max(out, dim = 1).values
    
    if(visualize):
        uncertainty_img = (1-maximum).detach().cpu()
        uncertainty = torch.mean(maximum, dim=(1,2)) 
        return uncertainty, -1*uncertainty_img
    
    uncertainty = torch.mean(maximum, dim = (1,2))
    return uncertainty, None"""


def entropy_fct(logits, dim = 1):
    probability = softmax(logits) # compute softmax over the class dimension to get probability of class --> shape: bs, classes, w, h
    entropy = -1*torch.sum(probability*torch.log(probability), dim = dim) # compute entropy over all classes (for each pixel value) --> shape: bs, w, h
    return entropy

#     best = "maximum"
def entropy_loss(logits, gt, visualize = False):
    r"""
        Compute entropy over classes (for each pixel value) --> shape: BS x W x H

        Args:
            logits (`torch.cuda.FloatTensor`): 
                logits based on the prediction of the segmentation model for the generated model. Shape = 1 x C x W x H
            visualize (`bool`): 
                weather to visualize the entropy
    """
    # compute entropy over all classes (for each pixel value) --> shape: bs, w, h
    entropy = entropy_fct(logits) 
    uncertainty_loss = -1*torch.mean(entropy)
    if(visualize):
        uncertainty_img = entropy.detach().cpu()
        print(f"Uncertainty Loss: {uncertainty_loss}")
        return uncertainty_loss, uncertainty_img
    return uncertainty_loss, None


def easy_fct(logits, gt, visualize = False):
    loss = crossentropy(logits, gt.unsqueeze(0).type(torch.LongTensor).to(device))
    uncertainty_loss = -1*torch.mean(loss)
    if(visualize):
        uncertainty_img = loss.detach().cpu().squeeze()
        print(f"Uncertainty Loss: {uncertainty_loss}")
        return uncertainty_loss, uncertainty_img
    return uncertainty_loss, None


def hard_fct(logits, gt, visualize = False):
    loss = -1*crossentropy(logits, gt.unsqueeze(0).type(torch.LongTensor).to(device))
    uncertainty_loss = -1*torch.mean(loss)
    if(visualize):
        uncertainty_img = loss.detach().cpu().squeeze()
        print(f"Uncertainty Loss: {uncertainty_loss}")
        return uncertainty_loss, uncertainty_img
    return uncertainty_loss, None


