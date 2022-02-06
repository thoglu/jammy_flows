import torch
from torch import nn
import collections
import numpy

from .. import layer_base

class euclidean_base(layer_base.layer_base):
    def __init__(self, dimension=1, use_permanent_parameters=False, model_offset=0):
    
        super().__init__(dimension=dimension)

        self.use_permanent_parameters=use_permanent_parameters
        self.model_offset=model_offset

        self.offsets=None

        if(self.model_offset):
            self.offsets = torch.zeros(dimension).type(torch.double).unsqueeze(0)

            if(self.use_permanent_parameters):
                self.offsets = nn.Parameter(torch.randn(dimension).type(torch.double).unsqueeze(0))
            
            self.total_param_num+=dimension
            

    def inv_flow_mapping(self, inputs, extra_inputs=None, force_embedding_coordinates=False, force_intrinsic_coordinates=False):
        
        if(self.model_offset):

            [x,logdet]=inputs

            this_offset=self.offsets.to(x)

            if(extra_inputs is not None):
                this_offset=this_offset+extra_inputs[:,:self.dimension]

                return self._inv_flow_mapping([x-this_offset, logdet], extra_inputs=extra_inputs[:,self.dimension:])

            else:
                return self._inv_flow_mapping([x-this_offset, logdet], extra_inputs=None)
        else:

            return self._inv_flow_mapping(inputs, extra_inputs=extra_inputs)

    def flow_mapping(self, inputs, extra_inputs=None, force_embedding_coordinates=False, force_intrinsic_coordinates=False):

        if(self.model_offset):

            [x,logdet]=inputs

            this_offset=self.offsets.to(x)

            if(extra_inputs is not None):

                this_offset=this_offset+extra_inputs[:,:self.dimension]

               
                x, logdet=self._flow_mapping([x, logdet], extra_inputs=extra_inputs[:,self.dimension:])

                return [x+this_offset, logdet]

            else:
                x, logdet=self._flow_mapping([x, logdet], extra_inputs=None)

                return [x+this_offset, logdet]
        else:

            return self._flow_mapping(inputs, extra_inputs=extra_inputs)

    def get_desired_init_parameters(self):

        ## offset + specific layer params

        par_list=[]
        if(self.model_offset):
            par_list.append(torch.ones(self.dimension)*0.001)

        par_list.append(self._get_desired_init_parameters())

        return torch.cat(par_list)

    def init_params(self, params):

        assert(len(params)==self.total_param_num)

        ## this function should only be called when the pdf has permanent parameters
        assert(self.use_permanent_parameters==1)

        if(self.model_offset):

            self.offsets.data=params[:self.dimension]

            ## initialize child layers
            self._init_params(params[self.dimension:])
        else:
            self._init_params(params)

    def _embedding_conditional_return(self, x):
        return x

    def _embedding_conditional_return_num(self): 
        return self.dimension

    def _get_layer_base_dimension(self):
        
        return self.dimension   

    #############################################################################

    ## implement the following by specific euclidean child layers

    def _init_params(self, params):
        raise NotImplementedError
    def _get_desired_init_parameters(self):
        raise NotImplementedError

    def _inv_flow_mapping(self, inputs, extra_inputs=None):
        raise NotImplementedError

    def _flow_mapping(self, inputs, extra_inputs=None):
        raise NotImplementedError

    def obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """

        """
        if(self.model_offset):
            if(extra_inputs is not None):
                param_dict["offset"]=extra_inputs[:,:self.dimension]
                self._obtain_layer_param_structure(param_dict, extra_inputs=extra_inputs[:,self.dimension:], previous_x=previous_x, extra_prefix=extra_prefix)
            else:
                param_dict["offset"]=self.offsets.data
                self._obtain_layer_param_structure(param_dict, extra_inputs=None, previous_x=previous_x, extra_prefix=extra_prefix)
        else:
            self._obtain_layer_param_structure(param_dict, extra_inputs=extra_inputs, previous_x=previous_x, extra_prefix=extra_prefix)


    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Implemented by Euclidean sublayers.
        """
     
        raise NotImplementedError




    