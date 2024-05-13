import torch
from Utils import index2color_annotation
import ade_config
import torchvision

softmax = torch.nn.Softmax(dim = 1)

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

def get_prediction(input,model):
    torch.manual_seed(0)
    logits = forward_model(input,model) # bs, classes, w, h

    # prediction for visualization purposes
    prediction = softmax(logits).argmax(1)
    prediction = index2color_annotation(prediction.cpu().squeeze(), ade_config.palette)
    return prediction

def loss_fct(generated_image, real_image, gt_segments, model, uncertainty_loss_fct, reg_fct, w_loss, w_reg, base_segments = "gt", normalize = True, visualize = False, by_value = False):
    r"""
        Compute loss by combining regularization and uncertainty computation

        Args:
            generated_image (`torch.cuda.FloatTensor`): 
                generated image from guided inference process. Shape = 1 x 3 x W x H
            real_image (`torch.cuda.FloatTensor`): 
                real image on which the generated image is based on. Shape = 1 x 3 x W x H
            gt_segments (`torch.IntegerTensor`): 
                GT segments for regularization. Shape = 1 x W x H
            gt_segments (`torch.cuda.JitModule`): 
                segmentation model for uncertainty computation
            uncertainty_loss_fct (`Callable`): 
                Uncertainty loss for loss computation
            reg_fct (`Callable`): 
                Regularization loss for loss computation
            w_loss (`float`): 
                Weight for uncertainty loss
            w_reg (`float`): 
                Weight for regularization loss
            base_segments (`str`, optional):
                One of ['gt', 'real'] to retrieve the segments either from the prediction on the real image or the GT mask.
                Default = 'gt'
            normalize (`bool`, optional): 
                weather to normalize the loss by the sizes of the segments     
                Default = True   
            visualize (`bool`, optional): 
                weather to visualize the entropy
                Default = False
    """
    logits = forward_model(generated_image, model) # bs x c x h x w

    if(base_segments == "real"):
        real_image = torchvision.transforms.functional.center_crop(real_image, generated_image.shape[-2:])
        segments = softmax(forward_model(real_image, model)) # bs x c x h x w
        segments = torch.argmax(segments, dim = 1).squeeze().detach().cpu() # W x H #softmax(logits).argmax(1).squeeze().detach().cpu().numpy() #
    elif(base_segments == "gt"):
        # crop and resize to logits shape
        segments = torchvision.transforms.functional.center_crop(gt_segments, generated_image.shape[-2:])
        segments = torchvision.transforms.functional.resize(segments, logits.shape[-2:], antialias = False, interpolation = torchvision.transforms.functional.InterpolationMode.NEAREST).squeeze() 
    else: 
        print(f"ERROR::Can not retrieve segments from {base_segments}. Define as 'gt' or 'real'.")
    loss_value, uncertainty_img = uncertainty_loss_fct(logits, visualize = visualize)
    reg_value = reg_fct(logits, segments, normalize = normalize)
    loss_value = (w_loss * loss_value) + (w_reg * reg_value)

    if(by_value):
        return loss_value.item(), reg_value.item(), loss_value.item(), uncertainty_img
    
    return loss_value, uncertainty_img
    
