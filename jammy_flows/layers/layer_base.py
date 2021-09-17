import torch
from torch import nn

class layer_base(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()

        self.total_param_num=0
        
        self.dimension = dimension

    def get_total_param_num(self):
        return self.total_param_num

    ## return the potentially desired initalization params of this layer
    def get_desired_init_parameters(self):
        return torch.randn(self.total_param_num)

    ####################################################################
    
    ## initialize params of the layer with vector of params
    def init_params(self, params): 
        raise NotImplementedError
    ## every layer must implement this to define how other sub-pdfs (defined by other layers) take this layer dimension as an input. If this is a 1-sphere, for example, the input is potentially better given as x,y pair instead of 0-2pi angle due to the 0/2pi discontinuity of the mapping.
    def _embedding_conditional_return(self, x):
        raise NotImplementedError

    def _embedding_conditional_return_num(self): 
        raise NotImplementedError

    def obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""):
        """
        Returns the parameter names along with actual values in a dict to get an idea about the layer structure.
        Some layers (e.g. "n") might require previous x inputs to determine layer param structure.
        """
        raise NotImplementedError


    