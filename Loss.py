import torch
from Utils import index2color_annotation
import ade_config
import torchvision
from Utils import device

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

def get_prediction(input,model):
    torch.manual_seed(0)
    logits = forward_model(input,model) # bs, classes, w, h

    # prediction for visualization purposes
    prediction = softmax(logits).argmax(1)
    prediction = index2color_annotation(prediction.cpu().squeeze(), ade_config.palette)
    return prediction



def loss_fct(generated_image, gt, easy_model, w_easy, hard_model, w_hard, visualize = False, by_value = False):
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
    # get predictions of base model
    if(easy_model is not None):
        logits_easy = easy_model(generated_image) #, mode="tensor") 
        gt_shape = logits_easy.shape[-2:] 
   

    if(hard_model is not None):
        logits_hard = hard_model(generated_image) #, mode="tensor") 
        gt_shape = logits_hard.shape[-2:] 
    else: 
        logits_hard = torch.Tensor([0.], device = device)

    # prepare GT masks
    gt = torchvision.transforms.functional.center_crop(gt, generated_image.shape[-2:])
    gt = torchvision.transforms.functional.resize(gt, gt_shape, antialias = False, interpolation = torchvision.transforms.functional.InterpolationMode.NEAREST).type(torch.LongTensor).to(device)
    
    # compute loss
    if(easy_model is not None):
        # minimize the crossentropy to get an "easy" example for the trained model
        easy_loss = crossentropy(logits_easy, gt)
    else: 
        easy_loss = torch.Tensor([0.], device = device)

    if(hard_model is not None):
        # maximize the crossentropy (minimize the negative crossentropy) to get an "hard" example for the trained model
        hard_loss = -1*crossentropy(logits_hard, gt)
    else: 
        hard_loss = torch.Tensor([0.], device = device)

    loss = (w_easy*easy_loss) + (w_hard*hard_loss)

    #TODO check shape of loss and implement visualize == True
    if(visualize):
        return torch.mean(loss), loss.detach().cpu().squeeze()

    if(by_value):
        return torch.mean(easy_loss).item(), torch.mean(hard_loss).item(), torch.mean(loss).item(), loss.detach().cpu().squeeze()
        
    
    return torch.mean(loss), None


