import torch
from torch import nn
import numpy

from . import sphere_base
from ..bisection_n_newton import inverse_bisection_n_newton

class spherical_do_nothing(sphere_base.sphere_base):
    def __init__(self, dimension, euclidean_to_sphere_as_first=False, use_permanent_parameters=True, add_rotation=0):
        """ Spherical identity mapping: Symbol "y" """
        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_permanent_parameters=use_permanent_parameters, add_rotation=add_rotation)

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x,log_det]=inputs

        sf_extra=None


        if(self.always_parametrize_in_embedding_space==True):
            # s2 flow is defined in embedding space,
            #x, log_det=self.eucl_to_spherical_embedding(x, log_det)

            x, log_det=self.eucl_to_spherical_embedding(x, log_det)
       
        if(self.higher_order_cylinder_parametrization):

            cos_coords=0.5-0.5*torch.cos(x[:, :self.dimension-1])

            zero_mask=cos_coords==0
            one_mask=cos_coords==1

            cos_coords[zero_mask]=cos_coords[zero_mask]+0.000001
            cos_coords[one_mask]=cos_coords[one_mask]-0.000001
            
            ln_cyl=torch.log(cos_coords)
            sf_extra=torch.log(1.0-cos_coords)

            ## ddx = sin(x)/2
            log_det+=0.5*(ln_cyl+sf_extra).sum(axis=-1)

            x[:,:self.dimension-1]=ln_cyl

        if(self.always_parametrize_in_embedding_space==True):
            # s2 flow is defined in embedding space,
            #x, log_det=self.eucl_to_spherical_embedding(x, log_det)

            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        return x, log_det, sf_extra

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
        
       
        [x,log_det]=inputs
        
        if(self.always_parametrize_in_embedding_space==True):
            # s2 flow is defined in embedding space,
            #x, log_det=self.eucl_to_spherical_embedding(x, log_det)

            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        if(self.higher_order_cylinder_parametrization):

            ln_cyl=x[:, :self.dimension-1]

            ## ddx = sin(x)/2
            log_det+=-0.5*(ln_cyl+sf_extra).sum(axis=-1)

            x[:,:self.dimension-1]=torch.acos(1.0-2.0*ln_cyl.exp())

        if(self.always_parametrize_in_embedding_space==True):
            # s2 flow is defined in embedding space,
            #x, log_det=self.eucl_to_spherical_embedding(x, log_det)

            x, log_det=self.spherical_to_eucl_embedding(x, log_det)
    
        return x, log_det

    def _init_params(self, params):

        assert(len(params)==0)

    def _get_desired_init_parameters(self):
        
        return torch.Tensor([])

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        # do nothing
        return