import torch
from torch import nn
import numpy

from . import sphere_base
from ..bisection_n_newton import inverse_bisection_n_newton
from ..spline_fns import rational_quadratic_spline



class spline_1d(sphere_base.sphere_base):
    def __init__(self, dimension=1, 
                       euclidean_to_sphere_as_first=True, 
                       add_rotation=1,
                       natural_direction=0, 
                       use_permanent_parameters=False,  
                       num_basis_functions=10):
        """
        A circular spline variant. Symbol: "o"

        Follows https://arxiv.org/abs/2002.02428. Still experimental and has not really been tested.
    
        """
        super().__init__(dimension=1, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, add_rotation=add_rotation, use_permanent_parameters=use_permanent_parameters)
        
        if(dimension!=1):
            raise Exception("The moebius flow is defined for dimension 1, but dimension %d is handed over" % (dimension))
           
        self.num_basis_functions=num_basis_functions

        ## add parameters from this layer (on top of householder params defined in the class parent)
        self.total_param_num+=self.num_basis_functions*3

        if(use_permanent_parameters):
            self.spline_pars=nn.Parameter(torch.randn(self.num_basis_functions,3).unsqueeze(0))
            
        ## natural direction means no bisection in the forward pass, but in the backward pass
        self.natural_direction=natural_direction

    def _inv_flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):

        [x,log_det]=inputs

        if(extra_inputs is not None):
            spline_pars=torch.reshape(extra_inputs, [-1, self.num_basis_functions, 3])
        else:
            spline_pars=self.spline_pars.to(x)

        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        # mirror derivatives at the endpoints
        derivs=torch.cat([spline_pars[:,:,2], spline_pars[:,0:1,2]], dim=1)

        use_inverse=True
        if(self.natural_direction==0):
            use_inverse=False

        x, log_deriv=rational_quadratic_spline(x,
                      spline_pars[:,:,0],
                      spline_pars[:,:,1],
                      derivs,
                      inverse=use_inverse,
                      left=0, right=2*numpy.pi, bottom=0, top=2*numpy.pi,
                      rel_min_bin_width=1e-3,
                      rel_min_bin_height=1e-3,
                          min_derivative=1e-3)

      
        log_deriv=log_deriv.sum(axis=-1)
        
        log_det=log_det+log_deriv

        
        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        
        return x, log_det

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
        
       
        [x,log_det]=inputs

        
        if(extra_inputs is not None):
            spline_pars=torch.reshape(extra_inputs, [-1, self.num_basis_functions, 3])
        else:
            spline_pars=self.spline_pars.to(x)
        
        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)
        
        # mirror derivatives at the endpoints
        derivs=torch.cat([spline_pars[:,:,2], spline_pars[:,0:1,2]], dim=1)

        use_inverse=False
        if(self.natural_direction==0):
            use_inverse=True

        x, log_deriv=rational_quadratic_spline(x,
                      spline_pars[:,:,0],
                      spline_pars[:,:,1],
                      derivs,
                      inverse=use_inverse,
                      left=0, right=2*numpy.pi, bottom=0, top=2*numpy.pi,
                      rel_min_bin_width=1e-3,
                      rel_min_bin_height=1e-3,
                          min_derivative=1e-3)

        log_deriv=log_deriv.sum(axis=-1)

        log_det=log_det+log_deriv

        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        return x, log_det

    def _init_params(self, params):

        self.spline_pars.data=params.reshape(1, self.num_basis_functions, 3)

    def _get_desired_init_parameters(self):

        ## fixed params for the spline
        return torch.ones((self.num_basis_functions*3))*0.54

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
     
        if(extra_inputs is not None):
            spline_pars=extra_inputs.reshape(-1, self.num_basis_functions, 3)
        else:
            spline_pars=self.spline_pars

        param_dict[extra_prefix+"moebius"]=spline_pars.data
        

