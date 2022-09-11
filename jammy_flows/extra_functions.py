from torch import nn
import torch
import numpy
import time

def log_one_plus_exp_x_to_a_minus_1(x, a):

    """
    Calculates the logarithm of ((1+exp(x))**a - 1) / ((1+exp(x))**a)

    Parameters:
        x (Tensor): Size (B,D)
        a (Tensor): Size (B,1)

    Returns:
        Tensor
            Result
    """

    assert(x.dtype==torch.double), "This function requires double precision to work properly"
    assert(x.shape[1:]==a.shape[1:])

    ## softplus result must be positive by definition
    softplus_result=a*nn.functional.softplus(x)

    ## small x values
    x_small_mask=x<=-20

    res=torch.where(x_small_mask, torch.log(a)+x, 0.0)

    ### check the value of softplus
    soft_plus_large=softplus_result > 20

    ## we can neglect the final -1 since the soft plus term is large
    mask_2=(~x_small_mask) & soft_plus_large
    
    res=torch.where(mask_2, softplus_result, res)

    ### check the value of softplus
    soft_plus_small=softplus_result < 1e-8

    mask_3=(~x_small_mask) & soft_plus_small

    res=torch.masked_scatter(input=res, mask=mask_3, source=torch.log(softplus_result[mask_3]))

    mask_4=(~x_small_mask) & (~soft_plus_large) & (~soft_plus_small)
    
    res=torch.masked_scatter(input=res, mask=mask_4, source=torch.log(torch.exp(softplus_result[mask_4])-1.0))

    if( (torch.isfinite(res)==False).sum()>0):
        print("LOGPLUS1 RES", res)
        raise Exception("Non-finite values")
    return res-softplus_result

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}

def list_from_str(spec):
    if(spec==""):
        return []
        
    return list(tuple(map(int, spec.split("-"))))
