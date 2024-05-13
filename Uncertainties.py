import torch 
import torchvision
import numpy as np

from Utils import device, totensor_transform, index2color_annotation
import ade_config



softmax = torch.nn.Softmax(dim = 1)
crossentropy = torch.nn.CrossEntropyLoss(reduction="none")

def get_size(input, model):
    with torch.no_grad():
        return forward_model(input,model).shape

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
    return out # bchw


    """if(gt_mask != None):
        class_indices = np.unique(gt_mask)
        out = out.permute(1,0,2,3) #cbhw

        for class_idx in class_incides:
            mask = (gt_mask == class_idx) #bhw
            logits = out[:,mask]"""

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
def lcu_loss(input, real_images, gt_mask, model, w_pixel = 0, w_class = 1, visualize = False):
    out = forward_model(input,model)

    maximum = torch.max(out, dim = 1).values
    
    if(visualize):
        uncertainty_img = (1-maximum).detach().cpu()
        uncertainty = torch.mean(maximum, dim=(1,2)) 
        return uncertainty, -1*uncertainty_img
    
    uncertainty = torch.mean(maximum, dim = (1,2))
    return uncertainty, None


def entropy(logits, dim = 1):
    probability = softmax(logits) # compute softmax over the class dimension to get probability of class --> shape: bs, classes, w, h
    entropy = -1*torch.sum(probability*torch.log(probability), dim = dim) # compute entropy over all classes (for each pixel value) --> shape: bs, w, h
    return entropy

# only works for single image
def segment_entropy_loss(input, real_images, gt_mask, model, visualize = False):
    torch.manual_seed(0)

    logits = forward_model(input,model) # bs, classes, w, h

    gt_mask = torchvision.transforms.functional.center_crop(gt_mask, input.shape[-2:])
    gt_mask = torchvision.transforms.functional.resize(gt_mask, logits.shape[-2:], antialias = False, interpolation = torchvision.transforms.functional.InterpolationMode.NEAREST).squeeze()
    class_indices = np.unique(gt_mask)
    if(visualize):
        uncertainty_img = torch.zeros(gt_mask.shape)
    uncertainty = 0
    for class_idx in class_indices: 
        mask = (gt_mask == class_idx)
        class_logits = logits[:,:,mask]
        class_logits = torch.mean(class_logits, dim = -1)
        class_uncertainty = entropy(class_logits)
        if(visualize):
            uncertainty_img[mask] = class_uncertainty
        uncertainty += class_uncertainty
    
    if(visualize):
        return -1*uncertainty, uncertainty_img
        
    return -1*uncertainty, None

def get_prediction(input,model):
    torch.manual_seed(0)
    logits = forward_model(input,model) # bs, classes, w, h

    # prediction for visualization purposes
    prediction = softmax(logits).argmax(1)
    prediction = index2color_annotation(prediction.cpu().squeeze(), ade_config.palette)
    return prediction

# idea: minimize std over pixel values, maximize it over classes
def min_max_segment_entropy_loss(input, real_images, gt_mask, model, w_pixel = 0, w_class = 1, visualize = False):
    torch.manual_seed(0)

    # import matplotlib.pyplot as plt
    # import wandb

    """import pdb 
    pdb.set_trace()

    fig,axs = plt.subplots(1,2)
    axs[0].imshow(input.squeeze().permute(1,2,0).cpu().to(torch.float32).numpy())
    axs[1].imshow(real_images.squeeze().numpy())
    plt.savefig("./debug.jpg")

    import pdb 
    pdb.set_trace()"""


    logits = forward_model(input,model) # bs, classes, w, h

    # prediction for visualization purposes
    # prediction = softmax(logits).argmax(1)
    # prediction = index2color_annotation(prediction.cpu().squeeze(), ade_config.palette)

    gt_mask = torchvision.transforms.functional.center_crop(gt_mask, input.shape[-2:])
    gt_mask = torchvision.transforms.functional.resize(gt_mask, logits.shape[-2:], antialias = False, interpolation = torchvision.transforms.functional.InterpolationMode.NEAREST).squeeze() #softmax(logits).argmax(1).squeeze().detach().cpu().numpy() #
    class_indices = np.unique(gt_mask)
    if(visualize):
        uncertainty_img = torch.zeros(gt_mask.shape)
        uncertainty_img_pixles = torch.zeros(gt_mask.shape)
        uncertainty_img_classes = torch.zeros(gt_mask.shape)

    uncertainty = 0
    for class_idx in class_indices: 
        mask = (gt_mask == class_idx)
        class_logits = logits[:,:,mask]
        # for each class prediction compute the entropy over the segment. This is supposed to be low (all class predictions within the same segment should show the same object)
        pixel_entropy = entropy(torch.mean(class_logits, dim = -2))  #torch.mean(entropy(class_logits.permute(0,2,1)))
        pixel_entropy
        # for each pixel compute the entropy over all classes. This entropy is supposed to be high (class probabilities should be uniformly distributed.)
        class_entropy = -1*entropy(torch.mean(class_logits, dim = -1)) #-1*torch.mean(entropy(class_logits))
        # wandb.log({f"class_entropy img": class_entropy})
        # w_pixel = 1
        # w_class = 1 #0.5 #10
        segment_uncertainty = (w_pixel*pixel_entropy) + (w_class*class_entropy)
        # print(f"pixel_entropy: {pixel_entropy}")
        # print(f"class_entropy: {class_entropy}")
        # print(f"segment_uncertainty: {segment_uncertainty}")

        
        if(visualize):
            uncertainty_img[mask] = segment_uncertainty
            # uncertainty_img_pixles[mask] = pixel_entropy
            # uncertainty_img_classes[mask] = -class_entropy
        uncertainty += torch.sum(mask) * segment_uncertainty

    """with torch.no_grad():
        s = 7
        fig,axs = plt.subplots(1,5, figsize = (5*s, s))
        plt.suptitle(f"Uncertainty = {uncertainty}")
        axs[0].imshow(input.squeeze().permute(1,2,0).cpu().to(torch.float32).numpy())
        axs[0].set_title("Generated image")
        axs[1].imshow(prediction)
        axs[1].set_title("Prediction")
        axs[2].imshow(uncertainty_img_classes)
        axs[2].set_title("Class Entropy")
        axs[3].imshow(uncertainty_img_pixles)
        axs[3].set_title("Pixel Entropy")
        axs[4].imshow(uncertainty_img)
        axs[4].set_title("Uncertainty loss")

        for ax in axs: 
            ax.set_axis_off()

        plt.savefig("./MinMaxEntropy_Pred.jpg")
        plt.close()
        import sys
        sys.exit()"""


    uncertainty = uncertainty / (gt_mask.shape[0]*gt_mask.shape[1])
    if(visualize):
        return uncertainty[0], -1*uncertainty_img
        
    return uncertainty, None
# idea: minimize entropy over pixel values within one segment, maximize it over classes


#     best = "maximum"
def entropy_loss(input, real_images, gt_mask, model, w_pixel = 0, w_class = 0, visualize = False):
    out = forward_model(input,model) # bs, classes, w, h
    probability = softmax(out) # compute softmax over the class dimension to get probability of class --> shape: bs, classes, w, h
    entropy = -1*torch.sum(probability*torch.log(probability), dim = 1) # compute entropy over all classes (for each pixel value) --> shape: bs, w, h

    if(visualize):
        uncertainty_img = entropy.detach().cpu()
        uncertainty = -1*torch.mean(entropy) 
        print(f"Uncertainty: {uncertainty}")
        return uncertainty, uncertainty_img

    uncertainty = -1*torch.mean(entropy) # compute mean of the resulting entropy
    return uncertainty, None

# p should have shape of BS, dim (with BS being the distributions to compare and dim the axis which sums to 1 for every dist )
def pairwise_kld(p):
    import torch.nn.functional as F
    dist = F.softmax(p, dim = 1)

    reps = dist.shape[0]
    p_dist = torch.Tensor.repeat(dist, repeats = (reps,1))
    q_dist = torch.repeat_interleave(dist, repeats = torch.tensor([reps]), dim = 0)

    return torch.sum(p_dist * torch.log(p_dist/q_dist), dim = 1)

# make distributions of the prediction on the real image fit to the distribution of the generated image.
# nonsense as this will work against entropy
# rather make distributions within one segment fit 
def kl_loss(input, real_images, gt_mask, model, w_pixel, w_class, visualize = False):
    torch.manual_seed(0)
    import torch.nn.functional as F
    kl_loss = torch.nn.KLDivLoss(reduction="sum", log_target = True)

    logits = forward_model(input,model) # bs, classes, w, h
    log_dist = F.log_softmax(logits, dim=1)

    print(f"Shape log_dist: {log_dist.shape}")
    print(f"Shape logits: {logits.shape}")
    print(f"Shape gt_mask: {gt_mask.shape}")




    gt_mask = torchvision.transforms.functional.center_crop(gt_mask, input.shape[-2:])
    gt_mask = torchvision.transforms.functional.resize(gt_mask, logits.shape[-2:], antialias = False, interpolation = torchvision.transforms.functional.InterpolationMode.NEAREST).squeeze() #softmax(logits).argmax(1).squeeze().detach().cpu().numpy() #
    class_indices = np.unique(gt_mask)
    if(visualize):
        uncertainty_img = torch.zeros(gt_mask.shape)
        uncertainty_img_pixles = torch.zeros(gt_mask.shape)
        uncertainty_img_classes = torch.zeros(gt_mask.shape)

    uncertainty = 0
    for class_idx in class_indices: 
        mask = (gt_mask == class_idx)
        class_logits = logits[:,:,mask]
        class_log_dist = log_dist[:,:,mask]

        # minimize the KLD between all predictions within one segment.
        pixel_entropy = kl_loss(class_log_dist.squeeze().permute(1,0), class_log_dist.squeeze().permute(1,0))
        print(f"Shape KLdiv: {pixel_entropy.shape}")
        # for each pixel compute the entropy over all classes. This entropy is supposed to be high (class probabilities should be uniformly distributed.)
        class_entropy = -1*torch.mean(entropy(class_logits, dim = 1))#-1*torch.mean(entropy(class_logits))
        
        # class_entropy = -1*entropy(torch.mean(class_logits, dim = -1)) #-1*torch.mean(entropy(class_logits))
        # wandb.log({f"class_entropy img": class_entropy})
        # w_pixel = 1
        # w_class = 1 #0.5 #10
        segment_uncertainty = (w_pixel*pixel_entropy) + (w_class*class_entropy)
        # print(f"pixel_entropy: {pixel_entropy}")
        # print(f"class_entropy: {class_entropy}")
        # print(f"segment_uncertainty: {segment_uncertainty}")

        
        if(visualize):
            uncertainty_img[mask] = segment_uncertainty
            # uncertainty_img_pixles[mask] = pixel_entropy
            # uncertainty_img_classes[mask] = -class_entropy
        uncertainty += torch.sum(mask) * segment_uncertainty

    """with torch.no_grad():
        s = 7
        fig,axs = plt.subplots(1,5, figsize = (5*s, s))
        plt.suptitle(f"Uncertainty = {uncertainty}")
        axs[0].imshow(input.squeeze().permute(1,2,0).cpu().to(torch.float32).numpy())
        axs[0].set_title("Generated image")
        axs[1].imshow(prediction)
        axs[1].set_title("Prediction")
        axs[2].imshow(uncertainty_img_classes)
        axs[2].set_title("Class Entropy")
        axs[3].imshow(uncertainty_img_pixles)
        axs[3].set_title("Pixel Entropy")
        axs[4].imshow(uncertainty_img)
        axs[4].set_title("Uncertainty loss")

        for ax in axs: 
            ax.set_axis_off()

        plt.savefig("./MinMaxEntropy_Pred.jpg")
        plt.close()
        import sys
        sys.exit()"""


    uncertainty = uncertainty / (gt_mask.shape[0]*gt_mask.shape[1])
    if(visualize):
        return uncertainty, -1*uncertainty_img
        
    return uncertainty, None


def mse_loss_ours(input, real_images, gt_mask, model, w_pixel, w_class, visualize = False):
    torch.manual_seed(0)
    logits = forward_model(input,model) # bs, classes, w, h

    gt_mask = torchvision.transforms.functional.center_crop(gt_mask, input.shape[-2:])
    gt_mask = torchvision.transforms.functional.resize(gt_mask, logits.shape[-2:], antialias = False, interpolation = torchvision.transforms.functional.InterpolationMode.NEAREST).squeeze() #softmax(logits).argmax(1).squeeze().detach().cpu().numpy() #
    class_indices = np.unique(gt_mask)
    if(visualize):
        uncertainty_img = torch.zeros(gt_mask.shape)
        uncertainty_img_pixles = torch.zeros(gt_mask.shape)
        uncertainty_img_classes = torch.zeros(gt_mask.shape)

    uncertainty = 0
    for class_idx in class_indices: 
        mask = (gt_mask == class_idx)
        class_logits = logits[:,:,mask]

        print(f"class logits: {class_logits.shape}") # bs, class, pixels
        print(f"distances: {torch.cdist(class_logits.permute(0,2,1), class_logits.permute(0,2,1)).shape}")

        # the pairwise error between the pixel prediction should be minimized such that all predictions predict the same class within one segment.
        pixel_entropy = torch.mean(torch.cdist(class_logits.permute(0,2,1), class_logits.permute(0,2,1)))
        # for each pixel compute the entropy over all classes. This entropy is supposed to be high (class probabilities should be uniformly distributed.)
        class_entropy = -1*torch.mean(entropy(class_logits, dim = 1))#-1*torch.mean(entropy(class_logits))
        
        # class_entropy = -1*entropy(torch.mean(class_logits, dim = -1)) #-1*torch.mean(entropy(class_logits))
        # wandb.log({f"class_entropy img": class_entropy})
        # w_pixel = 1
        # w_class = 1 #0.5 #10
        segment_uncertainty = (w_pixel*pixel_entropy) + (w_class*class_entropy)
        # print(f"pixel_entropy: {pixel_entropy}")
        # print(f"class_entropy: {class_entropy}")
        # print(f"segment_uncertainty: {segment_uncertainty}")

        
        if(visualize):
            uncertainty_img[mask] = segment_uncertainty
            # uncertainty_img_pixles[mask] = pixel_entropy
            # uncertainty_img_classes[mask] = -class_entropy
        uncertainty += torch.sum(mask) * segment_uncertainty

    """with torch.no_grad():
        s = 7
        fig,axs = plt.subplots(1,5, figsize = (5*s, s))
        plt.suptitle(f"Uncertainty = {uncertainty}")
        axs[0].imshow(input.squeeze().permute(1,2,0).cpu().to(torch.float32).numpy())
        axs[0].set_title("Generated image")
        axs[1].imshow(prediction)
        axs[1].set_title("Prediction")
        axs[2].imshow(uncertainty_img_classes)
        axs[2].set_title("Class Entropy")
        axs[3].imshow(uncertainty_img_pixles)
        axs[3].set_title("Pixel Entropy")
        axs[4].imshow(uncertainty_img)
        axs[4].set_title("Uncertainty loss")

        for ax in axs: 
            ax.set_axis_off()

        plt.savefig("./MinMaxEntropy_Pred.jpg")
        plt.close()
        import sys
        sys.exit()"""


    uncertainty = uncertainty / (gt_mask.shape[0]*gt_mask.shape[1])
    if(visualize):
        return uncertainty, -1*uncertainty_img
        
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

