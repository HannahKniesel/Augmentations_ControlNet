import torch 
import numpy as np
import torch.nn.functional as F

def no_reg(logits, segments, normalize = True):
    return 0


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



def kl_divergence(p, q, dim = None):
    return torch.sum(p * torch.log(p / q), dim=dim)

# Takes VERY long
def pairwise_kld_greedy(tensor):
    bs, n = tensor.shape
    pairwise_kld_value = 0 #torch.zeros(bs, bs)
    for i in range(bs):
        # print(f"{i}/{bs}")
        for j in range(i):
            pairwise_kld_value += torch.sum(kl_divergence(torch.stack((tensor[i], tensor[j])), torch.stack((tensor[j], tensor[i])), dim = 1)) 
            # kl_divergence(tensor[i], tensor[j]) + kl_divergence(tensor[j], tensor[i])
    return pairwise_kld_value

# leads to CUDA OOM
def pairwise_kld_vectorized(tensor, n_samples = 512):
    """import pdb 
    pdb.set_trace()"""
    # sampling? 
    # pytorch KLD? 
    bs, n = tensor.shape
    # sample randomly from segment to use less memory
    tensor = tensor[torch.randint(bs, (n_samples,)), :]  
    # n_samples = bs
    expanded_tensor = tensor.unsqueeze(1)  # Add an extra dimension for broadcasting
    expanded_tensor = expanded_tensor.expand(n_samples, n_samples, n)  # Broadcast to create pairs
    # print(expanded_tensor.shape)
    # print(kl_divergence(expanded_tensor, expanded_tensor.transpose(0, 1), dim = 2).shape)

    pairwise_kld_matrix = torch.sum(expanded_tensor * torch.log(expanded_tensor / expanded_tensor.transpose(0,1))) #torch.sum(kl_divergence(expanded_tensor, expanded_tensor.transpose(0, 1), dim = 2), dim=1)
    return pairwise_kld_matrix

def kld_reg(logits, segments, normalize = True):
    class_indices = np.unique(segments)
    kld_value = 0
    for class_idx in class_indices:
        mask = (segments == class_idx)
        # class_logits should have shape of BS, dim (with BS being the distributions to compare and dim the axis which sums to 1 for every dist )
        class_logits = logits[:,:,mask]
        class_logits = class_logits.squeeze().permute(1,0)
        # the pairwise KLD between the pixel prediction should be minimized such that all predictions predict the same class within one segment.
        kld_segment_value = pairwise_kld_vectorized(F.softmax(class_logits, dim = 1))

        if(normalize):
            kld_value += torch.sum(mask) * kld_segment_value
        else: 
            kld_value += kld_segment_value
    
    if(normalize):
        kld_value = kld_value / (segments.shape[0]*segments.shape[1])
    
    return kld_value






# p should have shape of BS, dim (with BS being the distributions to compare and dim the axis which sums to 1 for every dist )
"""def pairwise_kld_vectorized(p):
    dist = F.softmax(p, dim = 1)

    reps = dist.shape[0]
    p_dist = torch.Tensor.repeat(dist, repeats = (reps,1))
    q_dist = torch.repeat_interleave(dist, repeats = torch.tensor([reps]), dim = 0)

    return torch.sum(p_dist * torch.log(p_dist/q_dist), dim = 1)"""



