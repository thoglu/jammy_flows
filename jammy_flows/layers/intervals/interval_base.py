import torch
from torch import nn
import numpy
import collections

from .. import layer_base

class interval_base(layer_base.layer_base):
    def __init__(self, 
                 dimension=1, 
                 euclidean_to_interval_as_first=0, 
                 use_permanent_parameters=False, 
                 low_boundary=0.0, 
                 high_boundary=1.0):
        """
        
        """
        super().__init__(dimension=dimension)

        ## we only allow 1-dimensional interval flows
        assert(self.dimension==1)

        self.use_permanent_parameters=use_permanent_parameters

        self.low_boundary=low_boundary
        self.high_boundary=high_boundary
        self.interval_width=high_boundary-low_boundary

        self.euclidean_to_interval_as_first=euclidean_to_interval_as_first
        
        assert(self.high_boundary > self.low_boundary)
    
    def real_line_to_interval(self, inputs):

        [x, log_det] = inputs

       
        res=0.5+0.5*torch.erf(x/numpy.sqrt(2.0))

   
        res=res*self.interval_width+self.low_boundary
     
        log_det=log_det-(x[:,0]**2)/2.0-0.5*numpy.log(2*numpy.pi)+numpy.log(self.interval_width)

        return res, log_det

    def interval_to_real_line(self, inputs):

        [x, log_det] = inputs

        res=(x-self.low_boundary)/self.interval_width

        res=torch.erfinv(2.0*res-1.0)*numpy.sqrt(2.0)

     
        ## similar to other log_det, just different sign
        log_det=log_det-(-(res[:,0]**2)/2.0-0.5*numpy.log(2*numpy.pi)+numpy.log(self.interval_width))

        return res, log_det

    def inv_flow_mapping(self, inputs, extra_inputs=None, force_embedding_coordinates=False, force_intrinsic_coordinates=False):
        
        res, log_det=self._inv_flow_mapping(inputs, extra_inputs=extra_inputs)

        if(self.euclidean_to_interval_as_first):

            res, log_det=self.interval_to_real_line([res, log_det])

        return res, log_det

    def flow_mapping(self, inputs, extra_inputs=None, force_embedding_coordinates=False, force_intrinsic_coordinates=False):

        new_inputs=inputs

        if(self.euclidean_to_interval_as_first):

            new_inputs=self.real_line_to_interval(inputs)

        return self._flow_mapping(new_inputs, extra_inputs=extra_inputs)

    def get_desired_init_parameters(self):

        ## offset + specific layer params

        return self._get_desired_init_parameters()

    def init_params(self, params):

        assert(len(params)==self.total_param_num)

        self._init_params(params)

    def _embedding_conditional_return(self, x):
        return x

    def _embedding_conditional_return_num(self): 
        return self.dimension

    def _get_layer_base_dimension(self):
       
        
        return self.dimension

    def transform_target_space(self, x, log_det=0.0, transform_from="default", transform_to="embedding"):
        
        return x, log_det

    #############################################################################

    ## implement the following by specific interval child layers

    def _init_params(self, params):
        raise NotImplementedError
    def _get_desired_init_parameters(self):
        raise NotImplementedError

    def _inv_flow_mapping(self, inputs, extra_inputs=None):
        raise NotImplementedError

    def _flow_mapping(self, inputs, extra_inputs=None):
        raise NotImplementedError


    def obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 

        self._obtain_layer_param_structure(param_dict, extra_inputs=extra_inputs, previous_x=None, extra_prefix="")

        #return param_dict

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Implemented by Euclidean sublayers.
        """
     
        raise NotImplementedError

    




    