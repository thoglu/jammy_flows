import torch
from torch import nn
import numpy

from . import sphere_base
from . import moebius_1d
from .. import spline_fns
from ..bisection_n_newton import inverse_bisection_n_newton_sphere, inverse_bisection_n_newton_sphere_fast
from ...amortizable_mlp import AmortizableMLP
from ...extra_functions import list_from_str

from ..euclidean.polynomial_stretch_flow import psf_block
from ..euclidean.euclidean_do_nothing import euclidean_do_nothing

from ...extra_functions import NONLINEARITIES

from .cnf_specific import utils

import sys
import os
import time
import copy

import torch.autograd


"""
Implementation of the Fisher-von-Mises distribution as a normalizing flow. 
"""




class fisher_von_mises_2d(sphere_base.sphere_base):

    def __init__(self, 
                 dimension, 
                 euclidean_to_sphere_as_first=False, 
                 use_permanent_parameters=False):
        """
        Uses the spherical exponential map. Symbol: "v"

        Uses linear and quadratic potential as described in https://arxiv.org/abs/0906.0874, and exponential potential as described in https://arxiv.org/abs/2002.02428.
        Additionally added a spline-based potential.

        Parameters:
        
            exp_map_type (str): Defines the potential of the exponential map. One of ["linear", "quadratic", "exponential", "splines"].
            natural_direction (int). If 0, log-probability evaluation is faster. If 1, sampling is faster.
            num_components (int): How many components to sum over in the exponential map.
            add_rotation (int): Add a rotation after the main flow?.
            num_newton_iter (int): Number of newton iterations.
        """
        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_permanent_parameters=use_permanent_parameters, higher_order_cylinder_parametrization=False, add_rotation=1)
        
        if(dimension!=2):
            raise Exception("The moebius flow should be used for dimension 2!")
        
        self.min_kappa=1e-10

        if(self.use_permanent_parameters):
            self.log_kappa=nn.Parameter(torch.randn(1).unsqueeze(0))

        self.total_param_num+=1

    def _inv_flow_mapping(self, inputs, extra_inputs=None):
        


        [x,log_det]=inputs
        
        sf_extra=None
        
        if(self.always_parametrize_in_embedding_space):
           
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        if(extra_inputs is not None):

            kappa=extra_inputs.exp()+self.min_kappa
        else:
            kappa=self.log_kappa.to(x).exp()+self.min_kappa

        ## go to cylinder from angle
        prev_ret=torch.cos(x[:,:1])
        fw_upd=torch.log(torch.sin(x[:,0]))
        log_det=log_det+fw_upd

        ## intermediate [-1,1]->[-1,1] transformation
        safe_part=2*kappa
        smaller_mask=kappa[:,0]<100
        
        safe_part=torch.masked_scatter(input=safe_part, mask=smaller_mask[:,None], source=torch.log(torch.exp(2*kappa[smaller_mask])-1.0))

        safe_ld_update=(torch.log(2*kappa)+kappa*(prev_ret+1)-safe_part)[:,0]
        
        ## 1 + 1 - 2*k -2*(1+k(x-1)) / (-2k) = 2 - 2k -2 -2k(x-1) / -2k = 1 + 1(x-1)
        ret= (1.0+torch.exp(-2*kappa)-2*torch.exp(kappa*(prev_ret-1)))/(-1+torch.exp(-2*kappa))

        #approx_result=ret # nothing happens for k->0
        if(x.dtype==torch.float32):
            kappa_mask=kappa<1e-4
        elif(x.dtype==torch.float64):
            kappa_mask=kappa<1e-8
        else:
            raise Exception("Require 32 or 64 bit float")

           
        ret=torch.where(kappa_mask, prev_ret, ret)  

        log_det=log_det+safe_ld_update

        ### we have to make the angles safe here...TODO: change to external transformation
        ret=torch.where(ret<=-1.0, -1.0+1e-7, ret)
        ret=torch.where(ret>=1.0, 1.0-1e-7, ret)
        ## go back to angle
        ret=torch.acos(ret)
        rev_upd=torch.log(torch.sin(ret))[:,0]
        log_det=log_det-rev_upd

        ret=torch.cat([ret, x[:,1:]], dim=1)
        
        if(self.always_parametrize_in_embedding_space):

            ret, log_det=self.spherical_to_eucl_embedding(ret, log_det)

        return ret, log_det, sf_extra

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
       
        [x,log_det]=inputs

        
        if(self.always_parametrize_in_embedding_space):
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)
        
        if(extra_inputs is not None):
            kappa=extra_inputs.exp()+self.min_kappa
        else:
            kappa=self.log_kappa.to(x).exp()+self.min_kappa
        
        ## go to cylinder from angle
        prev_ret=torch.cos(x[:,:1])
        fw_upd=torch.log(torch.sin(x[:,0]))
        log_det=log_det+fw_upd

        ## kappa->0 

        ## 0.5+0.5x + (0.5-0.5x)*(1-2k) = 1 -k+kx = (1+k(x-1))^(1/k)
        ## intermediate [-1,1]->[-1,1] transformation
        
        log_det=log_det-torch.log(kappa*prev_ret+kappa/torch.tanh(kappa))[:,0]
        
        ret=1.0+(1.0/kappa)*torch.log( 0.5*(1.0+prev_ret) + (0.5-0.5*prev_ret)*torch.exp(-2.0*kappa) )
        if(x.dtype==torch.float32):
            kappa_mask=kappa<1e-4
        elif(x.dtype==torch.float64):
            kappa_mask=kappa<1e-8
        else:
            raise Exception("Require 32 or 64 bit float")

        ret=torch.where(kappa_mask, prev_ret, ret)  
        
        ## go back to angle
        ret=torch.acos(ret)
        rev_upd=torch.log(torch.sin(ret))[:,0]
        log_det=log_det-rev_upd

        ret=torch.cat([ret, x[:,1:]], dim=1)

        if(self.always_parametrize_in_embedding_space):
            ret, log_det=self.spherical_to_eucl_embedding(ret, log_det)
     
        return ret, log_det

    def _init_params(self, params):

        assert(len(params)== 1)

        self.log_kappa.data=params.reshape(1, 1)

    def _get_desired_init_parameters(self):

       
        gaussian_init=torch.randn((1))

        return gaussian_init


    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Implemented by Euclidean sublayers.
        """

        if(extra_inputs is not None):
            log_kappa=extra_inputs
        else:
            log_kappa=self.log_kappa

        param_dict[extra_prefix+"log_kappa"]=log_kappa.data