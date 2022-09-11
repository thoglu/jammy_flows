import torch
from torch import nn

class layer_base(nn.Module):
    def __init__(self, dimension=1, always_parametrize_in_embedding_space=0):
        """
        Base class for all flow layers.

        Parameters:
            dimension (int): The intrinsic dimension of the flow.
            always_parametrize_in_embedding_space (int): Required for subclasses to make decisions for embedding transformations.
        """
        super().__init__()

        self.total_param_num=0
        
        self.dimension = dimension

        ## only relevant for manifold-based distributions... do we work in embedding space, or intrinsic coordinates?
        ## If True: Everything is done in the higher-dimensional embedding space (evaluating and sampling new coordinates)
        ## If False: Everything is evaluated in intrinsic space.

        self.always_parametrize_in_embedding_space=always_parametrize_in_embedding_space

    def get_total_param_num(self):
        return self.total_param_num

    ## return the potentially desired initalization params of this layer
    def get_desired_init_parameters(self):
        """
        Returns desired init parameters for given flow.
        Should be overwritten by subclass.
        """
        return torch.randn(self.total_param_num)

    def get_layer_embedded_target_dimension(self):
        return self._embedding_conditional_return_num()

    def get_layer_intrinsic_target_dimension(self):
        return self.dimension

    def get_layer_base_dimension(self):
        
        return self._get_layer_base_dimension()

   

    ####################################################################
    
    ## initialize params of the layer with vector of params
    def init_params(self, params): 
        """
        Initializes permanent params. Can only be invoked when PDF is non-conditional.
        Must be overwritten by subclass.
        """
        raise NotImplementedError

    def flow_mapping(self, input, extra_inputs=None): 
        """
        Implements the flow forward mapping, which is required for sampling.
        Must be overwritten by subclass.
        """
        raise NotImplementedError

    def inv_flow_mapping(self, input, extra_inputs=None): 
        """
        Implements the flow inverse mapping, which is required for evaluation.
        Must be overwritten by subclass.
        """
        raise NotImplementedError
    ## every layer must implement this to define how other sub-pdfs (defined by other layers) take this layer dimension as an input. If this is a 1-sphere, for example, the input is potentially better given as x,y pair instead of 0-2pi angle due to the 0/2pi discontinuity of the mapping.
    
    def _embedding_conditional_return(self, x):
        """
        Transforms a vector into embedded space. Is flow specific and must be implemented by subclass.
        """
        raise NotImplementedError

    def _embedding_conditional_return_num(self): 
        """
        Returns the number of embedding space dimensions for this flow. Must be implemented by subclass.
        """
        raise NotImplementedError

    def transform_target_space(self, x, log_det=0.0, trafo_from="default", trafo_to="embedding"):
        """
        Transforms the target vector from a given parametrization to another parametrization.
        Must be implemented by subclass.
        """
        raise NotImplementedError()

    def obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""):
        """
        Useful for debugging purposes. Must be overwritten by subclass.

        Parameters:
            param_dict (dict): Is handed over from *jammy_flows.pdf* class and filled in this function with current values via param_dict[*par_name*]=*par_value*.

        """
        raise NotImplementedError


    