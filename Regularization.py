import torch 
import numpy as np
import torch.nn.functional as F



def mse_reg(logits, segments, normalize = True):
    r"""
        MSE regularization to supress noise within predictions leading to adversarial examples.

        Args:
            logits (`torch.cuda.FloatTensor`): 
                logits based on the prediction of the segmentation model for the generated model. Shape = 1 x C x W x H
            segments (`torch.IntegerTensor`):
                segmentation mask based on indices. Can be derived by the prediction of the model based on the real image or the GT mask. Shape = W x H
            normalize (`bool`): 
                weather to normalize the loss by the sizes of the segments
    """
    class_indices = np.unique(segments)
    mse_value = 0
    for class_idx in class_indices:
        mask = (segments == class_idx)
        class_logits = logits[:,:,mask]
        # the pairwise error between the pixel prediction should be minimized such that all predictions predict the same class within one segment.
        mse_segment_value = torch.mean(torch.cdist(class_logits.permute(0,2,1), class_logits.permute(0,2,1)))
        if(normalize):
            mse_value += torch.sum(mask) * mse_segment_value
        else: 
            mse_value += mse_segment_value
    
    if(normalize):
        mse_value = mse_value / (segments.shape[0]*segments.shape[1])
    
    return mse_value

def kld_reg(logits, segments, normalize = True):
    class_indices = np.unique(segments)
    kld_value = 0
    for class_idx in class_indices:
        mask = (segments == class_idx)
        # class_logits should have shape of BS, dim (with BS being the distributions to compare and dim the axis which sums to 1 for every dist )
        class_logits = logits[:,:,mask]
        class_logits = class_logits.squeeze().permute(1,0)
        # the pairwise KLD between the pixel prediction should be minimized such that all predictions predict the same class within one segment.
        dist = F.softmax(class_logits, dim = 1)
        reps = dist.shape[0]
        p_dist = torch.Tensor.repeat(dist, repeats = (reps,1))
        q_dist = torch.repeat_interleave(dist, repeats = torch.tensor([reps]), dim = 0)
        kld_segment_value = torch.sum(p_dist * torch.log(p_dist/q_dist), dim = 1)

        if(normalize):
            kld_value += torch.sum(mask) * kld_segment_value
        else: 
            kld_value += kld_segment_value
    
    if(normalize):
        kld_value = kld_value / (segments.shape[0]*segments.shape[1])
    
    return kld_value




# p should have shape of BS, dim (with BS being the distributions to compare and dim the axis which sums to 1 for every dist )
def pairwise_kld(p):
    dist = F.softmax(p, dim = 1)

    reps = dist.shape[0]
    p_dist = torch.Tensor.repeat(dist, repeats = (reps,1))
    q_dist = torch.repeat_interleave(dist, repeats = torch.tensor([reps]), dim = 0)

    return torch.sum(p_dist * torch.log(p_dist/q_dist), dim = 1)




