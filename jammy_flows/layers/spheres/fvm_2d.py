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
from ...main import default

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
                 use_permanent_parameters=False,
                 fisher_parametrization="split",
                 add_vertical_rq_spline_flow=0,
                 add_circular_rq_spline_flow=0,
                 vertical_flow_defs="r",
                 circular_flow_defs="o",
                 add_correlated_rq_spline_flow=0,
                 correlated_max_rank=3): # split / joint
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

        add_rotation=0
        if(fisher_parametrization=="split"):
            add_rotation=1

        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_permanent_parameters=use_permanent_parameters, higher_order_cylinder_parametrization=False, add_rotation=add_rotation)
        
        if(dimension!=2):
            raise Exception("2-D Flow")
        
        ####

        self.fisher_parametrization=fisher_parametrization

        if(fisher_parametrization=="split"):
            self.min_kappa=1e-10

            if(self.use_permanent_parameters):
                self.log_kappa=nn.Parameter(torch.randn(1).unsqueeze(0))

            self.total_param_num+=1
        else:
            self.unnormalized_mu=nn.Parameter(torch.randn(3).unsqueeze(0))

            self.total_param_num+=3

        self.add_vertical_rq_spline_flow=add_vertical_rq_spline_flow
        self.add_circular_rq_spline_flow=add_circular_rq_spline_flow
        self.add_correlated_rq_spline_flow=add_correlated_rq_spline_flow

        self.total_num_vertical_params=0
        if(add_vertical_rq_spline_flow):
           
            flow_dict=dict()

            self.vertical_rqspline_flow=default.pdf("i1_-1.0_1.0", vertical_flow_defs, 
                                  options_overwrite=flow_dict,
                                  amortize_everything=True,
                                  amortization_mlp_use_custom_mode=True,
                                  use_as_passthrough_instead_of_pdf=True)

            self.total_num_vertical_params=self.vertical_rqspline_flow.total_number_amortizable_params
            self.total_param_num+=self.total_num_vertical_params

            if(use_permanent_parameters):
                self.vertical_flow_params = nn.Parameter(torch.randn(1, self.total_num_vertical_params))
        
        self.total_num_circular_params=0
        if(add_circular_rq_spline_flow):
           
            flow_dict=dict()

            self.circular_rqspline_flow=default.pdf("s1", circular_flow_defs, 
                                  options_overwrite=flow_dict,
                                  amortize_everything=True,
                                  amortization_mlp_use_custom_mode=True,
                                  use_as_passthrough_instead_of_pdf=True)

            self.total_num_circular_params=self.circular_rqspline_flow.total_number_amortizable_params
            self.total_param_num+=self.total_num_circular_params

            if(use_permanent_parameters):
                self.circular_flow_params = nn.Parameter(torch.randn(1, self.total_num_circular_params))

        self.total_num_correlated_params=0
        if(add_correlated_rq_spline_flow):
            assert(add_circular_rq_spline_flow==0)
            assert(add_vertical_rq_spline_flow==0)
           
            flow_dict=dict()

            self.correlated_rqspline_flow=default.pdf("i1_-1.0_1.0+s1", vertical_flow_defs+"+"+circular_flow_defs, 
                                  options_overwrite=flow_dict,
                                  amortize_everything=True,
                                  amortization_mlp_use_custom_mode=True,
                                  amortization_mlp_dims="64",
                                  amortization_mlp_ranks=correlated_max_rank,
                                  use_as_passthrough_instead_of_pdf=True)

            self.total_num_correlated_params=self.correlated_rqspline_flow.total_number_amortizable_params
            self.total_param_num+=self.total_num_correlated_params

            if(use_permanent_parameters):
                self.correlated_flow_params = nn.Parameter(torch.randn(1, self.total_num_correlated_params))

        
           
    def _inv_flow_mapping(self, inputs, extra_inputs=None):
        
        [x,log_det]=inputs
        
        sf_extra=None
        
        if(self.always_parametrize_in_embedding_space):
           
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        vertical_params=None
        circular_params=None
        correlated_params=None

        if(extra_inputs is not None):

            kappa=extra_inputs[:,0:1].exp()+self.min_kappa

            if(self.add_correlated_rq_spline_flow):
                correlated_params=extra_inputs[:,1:self.total_num_correlated_params+1]
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=extra_inputs[:,1:self.total_num_vertical_params+1]

                if(self.add_circular_rq_spline_flow):
                    circular_params=extra_inputs[:,1+self.total_num_vertical_params:self.total_num_circular_params+self.total_num_vertical_params+1]


        else:
            kappa=self.log_kappa.to(x).exp()+self.min_kappa

            if(self.add_correlated_rq_spline_flow):
                correlated_params=self.correlated_flow_params.to(x)
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=self.vertical_flow_params.to(x)
                if(self.add_circular_rq_spline_flow):
                    circular_params=self.circular_flow_params.to(x)

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

        angle=x[:,1:]

        if(correlated_params is not None):
            
            comb=torch.cat([ret, angle],dim=1)
            comb, log_det=self.correlated_rqspline_flow.all_layer_inverse(comb, log_det, None, amortization_parameters=correlated_params)
            ret=comb[:,:1]
            angle=comb[:,1:]
        else:
            ########## do vertical transformation here if desired
            if(vertical_params is not None):
                ret,log_det=self.vertical_rqspline_flow.all_layer_inverse(ret, log_det, None, amortization_parameters=vertical_params)
            
            
            if(circular_params is not None):
                angle, log_det=self.circular_rqspline_flow.all_layer_inverse(angle, log_det, None, amortization_parameters=circular_params)
            
        ## go back to angle
        ret=torch.acos(ret)
        rev_upd=torch.log(torch.sin(ret))[:,0]
        log_det=log_det-rev_upd

        ret=torch.cat([ret, angle], dim=1)
        
        if(self.always_parametrize_in_embedding_space):

            ret, log_det=self.spherical_to_eucl_embedding(ret, log_det)

        return ret, log_det, sf_extra

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
       
        [x,log_det]=inputs

        
        if(self.always_parametrize_in_embedding_space):
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)
        
        vertical_params=None
        circular_params=None
        correlated_params=None

        if(extra_inputs is not None):
            kappa=extra_inputs[:,:1].exp()+self.min_kappa

            if(self.add_correlated_rq_spline_flow):
                correlated_params=extra_inputs[:,1:self.total_num_correlated_params+1]
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=extra_inputs[:,1:self.total_num_vertical_params+1]

                if(self.add_circular_rq_spline_flow):
                    circular_params=extra_inputs[:,1+self.total_num_vertical_params:self.total_num_circular_params+self.total_num_vertical_params+1]

        else:
            kappa=self.log_kappa.to(x).exp()+self.min_kappa
            
            if(self.add_correlated_rq_spline_flow):
                correlated_params=self.correlated_flow_params.to(x)
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=self.vertical_flow_params.to(x)
                if(self.add_circular_rq_spline_flow):
                    circular_params=self.circular_flow_params.to(x)

        ## go to cylinder from angle
        prev_ret=torch.cos(x[:,:1])
        fw_upd=torch.log(torch.sin(x[:,0]))
        log_det=log_det+fw_upd

        angle=x[:,1:]

        if(correlated_params is not None):

            comb=torch.cat([prev_ret, angle],dim=1)
            comb, log_det=self.correlated_rqspline_flow.all_layer_forward(comb, log_det, None, amortization_parameters=correlated_params)
            prev_ret=comb[:,:1]
            angle=comb[:,1:]

        else:

            if(circular_params is not None):

                angle, log_det=self.circular_rqspline_flow.all_layer_forward(angle, log_det, None, amortization_parameters=circular_params)
            
            ########## do vertical transformation here if desired
            if(vertical_params is not None):
                prev_ret,log_det=self.vertical_rqspline_flow.all_layer_forward(prev_ret, log_det, None, amortization_parameters=vertical_params)
            

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

        ret=torch.cat([ret, angle], dim=1)

        if(self.always_parametrize_in_embedding_space):
            ret, log_det=self.spherical_to_eucl_embedding(ret, log_det)
     
        return ret, log_det

    def _init_params(self, params):

        self.log_kappa.data=params[:1].reshape(1, 1)

        if(self.add_correlated_rq_spline_flow):
            assert(len(params)== (1+self.total_num_correlated_params))

            self.correlated_flow_params.data=params[1:1+self.total_num_correlated_params].reshape(1,self.total_num_correlated_params)
        else:
            assert(len(params)== (1+self.total_num_vertical_params+self.total_num_circular_params))

            if(self.add_vertical_rq_spline_flow):
                self.vertical_flow_params.data=params[1:1+self.total_num_vertical_params].reshape(1,self.total_num_vertical_params)

            if(self.add_circular_rq_spline_flow):
                self.circular_flow_params.data=params[1+self.total_num_vertical_params:1+self.total_num_vertical_params+self.total_num_circular_params].reshape(1,self.total_num_circular_params)


    def _get_desired_init_parameters(self):

        gaussian_init=torch.randn((1))

        overall_init=gaussian_init

        if(self.add_correlated_rq_spline_flow):

            desired_params=self.correlated_rqspline_flow.init_params()

            overall_init=torch.cat([overall_init, desired_params])

        else:
            if(self.add_vertical_rq_spline_flow):
               
                extra_params=[l.get_desired_init_parameters() for l in self.vertical_rqspline_flow.layer_list[0]]
                overall_init=torch.cat([overall_init]+extra_params)
            
            if(self.add_circular_rq_spline_flow):
              
                extra_params=[l.get_desired_init_parameters() for l in self.circular_rqspline_flow.layer_list[0]]
                overall_init=torch.cat([overall_init]+extra_params)

        return overall_init


    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Implemented by Euclidean sublayers.
        """

        vertical_params=None
        circular_params=None
        correlated_params=None

        if(extra_inputs is not None):
            log_kappa=extra_inputs[:,:1]

            if(self.add_correlated_rq_spline_flow):
                correlated_params=extra_inputs[:,1:self.total_num_correlated_params+1]
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=extra_inputs[:,1:self.total_num_vertical_params+1]
                if(self.add_circular_rq_spline_flow):
                    circular_params=extra_inputs[:,1+self.total_num_vertical_params:self.total_num_circular_params+self.total_num_vertical_params+1]

        else:
            log_kappa=self.log_kappa
            
            if(self.add_correlated_rq_spline_flow):
                correlated_params=self.correlated_flow_params
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=self.vertical_flow_params
                if(self.add_circular_rq_spline_flow):
                    circular_params=self.circular_flow_params

        param_dict[extra_prefix+"log_kappa"]=log_kappa.data

        if(self.add_correlated_rq_spline_flow):
            param_dict[extra_prefix+"correlated_params"]=correlated_params.data
        else:
            if(self.add_vertical_rq_spline_flow):

                param_dict[extra_prefix+"vertical_params"]=vertical_params.data

            if(self.add_circular_rq_spline_flow):

                param_dict[extra_prefix+"circular_params"]=circular_params.data