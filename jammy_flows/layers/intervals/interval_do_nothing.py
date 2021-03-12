import torch
from torch import nn
import numpy

from . import interval_base

class interval_do_nothing(interval_base.interval_base):
    def __init__(self, dimension, euclidean_to_interval_as_first=0, use_permanent_parameters=False, low_boundary=0.0, high_boundary=1.0):
    
        super().__init__(dimension=dimension, euclidean_to_interval_as_first=euclidean_to_interval_as_first, use_permanent_parameters=use_permanent_parameters, low_boundary=low_boundary, high_boundary=high_boundary)

        self.use_permanent_parameters=use_permanent_parameters

        self.low_boundary=low_boundary
        self.high_boundary=high_boundary
        self.interval_width=high_boundary-low_boundary

        self.euclidean_to_interval_as_first=euclidean_to_interval_as_first

        assert(self.high_boundary > self.low_boundary)
    
    #############################################################################

    ## implement the following by specific interval child layers

    def _init_params(self, params):

        assert(len(params)==0)

    def _get_desired_init_parameters(self):
        
        return torch.Tensor([])

    def _inv_flow_mapping(self, inputs, extra_inputs=None):
        
        [x,log_det]=inputs

        return x, log_det

    def _flow_mapping(self, inputs, extra_inputs=None):
    
        [x,log_det]=inputs

        return x, log_det

    




    