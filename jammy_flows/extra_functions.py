from torch import nn
import torch
import numpy
import time

def log_one_plus_exp_x_to_a_minus_1(x, a):

    """
    Calculates the logarithm of ((1+exp(x))**a - 1) / ((1+exp(x))**a)
    """

    assert(x.dtype==torch.double), "This function requires double precision to work properly"
    assert(x.shape[1:]==a.shape[1:])
    #res=torch.zeros_like(x)

    #print("RES", res.shape)
    #print(x.shape)
    #print(a.shape)
    #print("-----------------")



    ## softplus result must be positive by definition
    softplus_result=a*nn.functional.softplus(x)

    ## small x values
    x_small_mask=x<=-20

    res=torch.where(x_small_mask, torch.log(a)+x, 0.0)

    #print(res.shape)
    
    #res=torch.masked_scatter(input=res, mask=x_small_mask, source=(torch.log(a)+x)[x_small_mask])
    
    ### check the value of softplus
    soft_plus_large=softplus_result > 20

    ## we can neglect the final -1 since the soft plus term is large
    mask_2=(~x_small_mask) & soft_plus_large

    #res=torch.masked_scatter(input=res, mask=mask_2, source=softplus_result)

    
    res=torch.where(mask_2, softplus_result, res)
    #res=res.masked_scatter(mask_2, softplus_result[mask_2])
   
    ### check the value of softplus
    soft_plus_small=softplus_result < 1e-8

    mask_3=(~x_small_mask) & soft_plus_small

    #print(mask_3)
    ##print("log sofplus")
    #print(torch.log(softplus_result))
    #res.masked_scatter_(mask_3, torch.log(softplus_result[mask_3]))
    res=torch.masked_scatter(input=res, mask=mask_3, source=torch.log(softplus_result[mask_3]))
    #res=torch.where(mask_3, torch.log(softplus_result), res)


    #print("AFTER 3")
    #print(res)

   
    mask_4=(~x_small_mask) & (~soft_plus_large) & (~soft_plus_small)
    #res.masked_scatter_(mask_4, torch.log(torch.exp(softplus_result[mask_4])-1.0))
    res=torch.masked_scatter(input=res, mask=mask_4, source=torch.log(torch.exp(softplus_result[mask_4])-1.0))

    #res=torch.where(mask_4,torch.log(torch.exp(softplus_result)-1.0), res)

    #print("after 4")
    #print(res)
    """
    tbef=time.time()
    for i in range(1000):
        res=torch.where(mask_4, torch.log(softplus_result), torch.log(torch.exp(softplus_result)-1.0))

    print(time.time()-tbef)

    tbef=time.time()
    for i in range(1000):
        res.masked_scatter_(mask_4, torch.log(torch.exp(softplus_result[mask_4])-1.0))
    print(time.time()-tbef)    

    tbef=time.time()
    for i in range(1000):
        res=torch.masked_scatter(input=res, mask=mask_4, source=torch.log(torch.exp(softplus_result[mask_4])-1.0))
    print(time.time()-tbef)    

    tbef=time.time()
    for i in range(1000):
        res[mask_4]=torch.log(torch.exp(softplus_result[mask_4])-1.0)
        #res.masked_scatter_(mask_4, torch.log(torch.exp(softplus_result[mask_4])-1.0))
    print(time.time()-tbef)    

    sys.exit(-1)
    """
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
