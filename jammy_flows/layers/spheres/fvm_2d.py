import torch
from torch import nn
import numpy

from . import sphere_base
from . import moebius_1d
from .. import spline_fns
from ..bisection_n_newton import inverse_bisection_n_newton_sphere, inverse_bisection_n_newton_sphere_fast
from ...amortizable_mlp import AmortizableMLP
from ...extra_functions import list_from_str

from ..euclidean.euclidean_do_nothing import euclidean_do_nothing

from ...extra_functions import NONLINEARITIES
from ...main import default
from torch.nn import functional as F

from .cnf_specific import utils

import sys
import os
import time
import copy

import torch.autograd


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
                 correlated_max_rank=3,
                 inverse_z_scaling=1,
                 spline_num_basis_functions=5,
                 boundary_cos_theta_identity_region=0.0,
                 vertical_smooth=0,
                 vertical_restrict_max_min_width_height_ratio=-1.0,
                 vertical_fix_boundary_derivative=1,
                 vertical_fix_first_width_n_height_to_zero=0,
                 vertical_also_fix_second_width_to_zero=0,
                 vertical_independent_width_height_parametrization=0,
                 circular_add_rotation=1,
                 min_kappa=1e-10,
                 kappa_prediction="direct_log_real_bounded",
                 add_extra_rotation_inbetween=0,
                 kappa_clamping=0,
                 add_rotation=1,
                 rotation_mode="householder",
                 num_householder_iter=-1): 
        """
        Symbol: "f"

        Based off of https://arxiv.org/abs/2002.02428 (conditional rational splines) and 
        https://arxiv.org/abs/2604.19846, (smooth rational splines and von-mises Fisher scaling functions).

        Parameters:
        
            add_vertical_rq_spline_flow (int): Add vertical rq flow?
            add_circular_rq_spline_flow (int): Add 1-d rq flow for phi?
            vertical_flow_defs (str): Which vertical flow layers to use.
            circular_flow_defs (str): Which circular flow layers to use.
            add_correlated_rq_spline_flow (int): Add a correlation between vertical and circular flow via a NN ?
            correlated_max_rank (int): Max rank of matrices in NN to connect vertical and circular flow.
            inverse_z_scaling (int): Define flow -z instead of z. Should be set to 1 to work well with standard stereographic projection.
            spline_num_basis_functions (int): Number of basis functions for the vertical/circular spline flows.
            boundary_cos_theta_identity_region (float): Defines the distance in cos(theta) from -1/1 respectively in which only an identity mapping is used.
            vertical_smooth (int): Smooth 2nd derivatives for vertical rq flow.
            vertical_restrict_max_min_width_height_ratio (float): Restrict the ratio between widths/heights to a maximum value.
            vertical_fix_boundary_derivative (int): Fix boundary derivative to 1.0 or not?
            min_kappa (float): Minimum allowed kappa value.
            kappa_prediction (str): Either "direct_log_real_bounded", or "log_bounded", which uses softplus in log space to lower-bound kappa.
        """

        
        super().__init__(dimension=dimension, 
                         euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, 
                         use_permanent_parameters=use_permanent_parameters, 
                         higher_order_cylinder_parametrization=False, 
                         add_rotation=add_rotation, 
                         num_householder_iter=num_householder_iter,
                         rotation_mode=rotation_mode)
        
        if(dimension!=2):
            raise Exception("2-D Flow")
        
        ####

        if(inverse_z_scaling):
            self.z_scaling_factor=-1.0
        else:
            self.z_scaling_factor=1.0

        self.fisher_parametrization=fisher_parametrization
        assert(fisher_parametrization == "split")

        self.min_kappa=min_kappa

        

        self.kappa_clamping=kappa_clamping
        self.kappa_prediction=kappa_prediction
        if(kappa_clamping):
            if(kappa_prediction=="direct_log_real_bounded"):
                self.kappa_fn=lambda x: torch.clamp(x, min=-5.0).exp()+self.min_kappa
            elif(kappa_prediction=="softplus_real_bounded"):
                self.kappa_fn=lambda x: F.softplus(torch.clamp(x, min=-5.0))+self.min_kappa
            elif(kappa_prediction=="log_bounded"):
                log_min_kappa=numpy.log(self.min_kappa)

                self.kappa_fn=lambda x: (torch.clamp(F.softplus(x), min=-5.0)+log_min_kappa).exp()
            else:
                self.kappa_fn=None
        else:
            if(kappa_prediction=="direct_log_real_bounded"):
                self.kappa_fn=lambda x: x.exp()+self.min_kappa
            elif(kappa_prediction=="softplus_real_bounded"):
                self.kappa_fn=lambda x: F.softplus(x)+self.min_kappa
            elif(kappa_prediction=="log_bounded"):
                log_min_kappa=numpy.log(self.min_kappa)

                self.kappa_fn=lambda x: (F.softplus(x)+log_min_kappa).exp()
            else:
                self.kappa_fn=None

        if(self.kappa_prediction=="mu" or self.kappa_prediction=="mu_squared"):
            assert(self.add_rotation)
            assert(self.rotation_mode=="xyz")
        if(self.kappa_prediction=="quatvec" or self.kappa_prediction=="quatvec_squared"):
            assert(self.add_rotation)
            assert(self.rotation_mode=="quaternion"), ("ROTATION MODE?!", self.rotation_mode)

        self.num_loglike_kappa_params=0
        if(self.kappa_fn is not None):
            if(self.use_permanent_parameters):
                self.loglike_kappa=nn.Parameter(torch.randn(1).unsqueeze(0))

            self.total_param_num+=1
            self.num_loglike_kappa_params=1

        self.add_vertical_rq_spline_flow=add_vertical_rq_spline_flow
        self.add_circular_rq_spline_flow=add_circular_rq_spline_flow
        self.add_correlated_rq_spline_flow=add_correlated_rq_spline_flow
        self.boundary_cos_theta_identity_region=boundary_cos_theta_identity_region
        self.spline_num_basis_functions=spline_num_basis_functions
        if(self.spline_num_basis_functions==-1):
            assert(vertical_smooth==1), "num_basis_functions=-1 means alternating 2/3 as basis functions and requires smooth splines."
        self.total_num_vertical_params=0

        if(add_vertical_rq_spline_flow):
            
            num_vertical_flows=len(vertical_flow_defs)

            flow_dict=dict()
            flow_dict["r"]=dict()
            flow_dict["r"]["fix_boundary_derivatives"]=-1.0 if vertical_fix_boundary_derivative==0 else 1.0
            
            flow_dict["r"]["smooth_second_derivative"]=vertical_smooth
            flow_dict["r"]["restrict_max_min_width_height_ratio"]=vertical_restrict_max_min_width_height_ratio
            flow_dict["r"]["fix_first_width_n_height_to_zero"]=vertical_fix_first_width_n_height_to_zero
            flow_dict["r"]["also_fix_second_width_to_zero"]=vertical_also_fix_second_width_to_zero
            flow_dict["r"]["independent_width_height_parametrization"]=vertical_independent_width_height_parametrization

            # fluctuate 2/3 basis functions
            if(spline_num_basis_functions==-1):
                for cur_r in range(num_vertical_flows):
                    flow_dict[(0,cur_r)]=dict()
                    flow_dict[(0,cur_r)]["r"]=dict()

                    cur_num_basis=2
                    if(cur_r%2==1):
                        cur_num_basis=3
                    flow_dict[(0,cur_r)]["r"]["num_basis_functions"]=cur_num_basis

                    flow_dict[(0,cur_r)]["r"]["fix_boundary_derivatives"]=-1.0 if vertical_fix_boundary_derivative==0 else 1.0
            
                    flow_dict[(0,cur_r)]["r"]["smooth_second_derivative"]=vertical_smooth
                    flow_dict[(0,cur_r)]["r"]["restrict_max_min_width_height_ratio"]=vertical_restrict_max_min_width_height_ratio
                    flow_dict[(0,cur_r)]["r"]["fix_first_width_n_height_to_zero"]=vertical_fix_first_width_n_height_to_zero
                    flow_dict[(0,cur_r)]["r"]["also_fix_second_width_to_zero"]=vertical_also_fix_second_width_to_zero
                    flow_dict[(0,cur_r)]["r"]["independent_width_height_parametrization"]=vertical_independent_width_height_parametrization

            else:
                flow_dict["r"]["num_basis_functions"]=spline_num_basis_functions

            self.vertical_rqspline_flow=default.pdf("i1_-%.2f_%.2f"%(1.0-boundary_cos_theta_identity_region,1.0-boundary_cos_theta_identity_region), vertical_flow_defs, 
                                  options_overwrite=flow_dict,
                                  amortize_everything=True,
                                  amortization_mlp_use_custom_mode=True,
                                  use_as_passthrough_instead_of_pdf=True)

            self.total_num_vertical_params=self.vertical_rqspline_flow.total_number_amortizable_params
            self.total_param_num+=self.total_num_vertical_params

            if(use_permanent_parameters):
                self.vertical_flow_params = nn.Parameter(torch.randn(1, self.total_num_vertical_params))
        
        self.total_num_circular_params=0
        self.circular_add_rotation=circular_add_rotation
        if(add_circular_rq_spline_flow):
            
            ## TODO: dont take vertical params here
            flow_dict=dict()
            flow_dict["o"]=dict()
            flow_dict["o"]["num_basis_functions"]=2
            flow_dict["o"]["smooth_second_derivative"]=1
            flow_dict["o"]["fix_first_width_n_height_to_zero"]=vertical_fix_first_width_n_height_to_zero
            flow_dict["o"]["also_fix_second_width_to_zero"]=vertical_also_fix_second_width_to_zero
            flow_dict["o"]["independent_width_height_parametrization"]=vertical_independent_width_height_parametrization
            flow_dict["o"]["add_rotation"]=circular_add_rotation

            assert(circular_add_rotation==0), "Currently not allowing additional S-1 rotations due to potential complications at the poles. Please set *circular_add_rotation* of f-flow to 0. Possible fix: make householder matrix dependent on zenith and switch of rotation smoothly at poles!"

            self.circular_rqspline_flow=default.pdf("s1", circular_flow_defs, 
                                  options_overwrite=flow_dict,
                                  amortize_everything=True,
                                  amortization_mlp_use_custom_mode=True,
                                  use_as_passthrough_instead_of_pdf=True)

            self.total_num_circular_params=self.circular_rqspline_flow.total_number_amortizable_params
            self.circular_rqspline_flow_param_masks=[]

            tot_count=0
            for flow_ind,flow_obj in enumerate(self.circular_rqspline_flow.layer_list[0]):

                self.circular_rqspline_flow_param_masks.append(torch.zeros(flow_obj.num_householder_params, dtype=torch.bool))
                self.circular_rqspline_flow_param_masks.append(torch.ones(flow_obj.total_param_num-flow_obj.num_householder_params, dtype=torch.bool))

            self.circular_rqspline_flow_param_masks=torch.cat(self.circular_rqspline_flow_param_masks)

            self.total_param_num+=self.total_num_circular_params

            if(use_permanent_parameters):
                self.circular_flow_params = nn.Parameter(torch.randn(1, self.total_num_circular_params))

        self.total_num_correlated_params=0
        if(add_correlated_rq_spline_flow):
            assert(add_circular_rq_spline_flow==0)
            assert(add_vertical_rq_spline_flow==0)
           
            flow_dict=dict()

            self.correlated_rqspline_flow=default.pdf("i1_-%.2f_%.2f+s1" % (1.0-boundary_cos_theta_identity_region,1.0-boundary_cos_theta_identity_region), vertical_flow_defs+"+"+circular_flow_defs, 
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

        
        self.add_extra_rotation_inbetween=add_extra_rotation_inbetween
    
    def _obtain_azimuthal_flow_scaling(self, cos_theta):
        
        scaling=torch.where(cos_theta<=0, 6*cos_theta**5+15*cos_theta**4+10*cos_theta**3+1.0, -6*cos_theta**5+15*cos_theta**4-10*cos_theta**3+1.0)
        
        return scaling
 
    def _inv_flow_mapping(self, inputs, extra_inputs=None, extra_inputs_base=None):

        [x,log_det]=inputs
            
        sf_extra=None
        
        if(self.always_parametrize_in_embedding_space):
           
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        vertical_params=None
        circular_params=None
        correlated_params=None

        if(extra_inputs is not None):

            if(self.kappa_fn is not None):
                kappa=self.kappa_fn(extra_inputs[:,0:1])
            else:
                assert(extra_inputs_base is not None)

                if(self.kappa_prediction=="mu"):
                    assert(extra_inputs_base.shape[-1]==3)
                    kappa=(extra_inputs_base**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="mu_squared"):
                    assert(extra_inputs_base.shape[-1]==3)
                    kappa=(extra_inputs_base**2).sum(dim=-1,keepdims=True)
                elif(self.kappa_prediction=="quatvec"):
                    assert(extra_inputs_base.shape[-1]==4)
                    kappa=(extra_inputs_base[...,1:]**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="quatvec_squared"):
                    assert(extra_inputs_base.shape[-1]==4)
                    kappa=(extra_inputs_base[...,1:]**2).sum(dim=-1,keepdims=True)

            if(self.add_correlated_rq_spline_flow):
                correlated_params=extra_inputs[:,self.num_loglike_kappa_params:self.total_num_correlated_params+self.num_loglike_kappa_params]
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=extra_inputs[:,self.num_loglike_kappa_params:self.total_num_vertical_params+self.num_loglike_kappa_params]

                if(self.add_circular_rq_spline_flow):
                    circular_params=extra_inputs[:,self.num_loglike_kappa_params+self.total_num_vertical_params:self.total_num_circular_params+self.total_num_vertical_params+self.num_loglike_kappa_params]


        else:

            if(self.kappa_fn is not None):
                kappa=self.kappa_fn(self.loglike_kappa.to(x))
            else:

                if(self.kappa_prediction=="mu"):
                    kappa=(self.householder_params.to(x)**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="mu_squared"):
                    kappa=(self.householder_params.to(x)**2).sum(dim=-1,keepdims=True)
                elif(self.kappa_prediction=="quatvec"):
                    kappa=(self.householder_params.to(x)[...,1:]**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="quatvec_squared"):
                    kappa=(self.householder_params.to(x)[...,1:]**2).sum(dim=-1,keepdims=True)

            if(self.add_correlated_rq_spline_flow):
                correlated_params=self.correlated_flow_params.to(x)
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=self.vertical_flow_params.to(x)
                if(self.add_circular_rq_spline_flow):
                    circular_params=self.circular_flow_params.to(x)



        ## go to cylinder from angle
        prev_ret=torch.cos(x[:,:1])

        

        fw_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(x[:,0])))

        log_det=log_det+fw_upd

        ## intermediate [-1,1]->[-1,1] transformation
        safe_part=2*kappa
        smaller_mask=kappa[:,0]<100

        ## safe_part only involves kappa
        safe_part=torch.masked_scatter(input=safe_part, mask=smaller_mask[:,None], source=torch.log(torch.exp(2*kappa[smaller_mask])-1.0))
        safe_ld_update=(torch.log(2*kappa)+kappa*(self.z_scaling_factor*prev_ret+1)-safe_part)[:,0]

        switched=self.z_scaling_factor*prev_ret
       
        ret= self.z_scaling_factor*((1.0+torch.exp(-2*kappa)-2*torch.exp(kappa*(self.z_scaling_factor*prev_ret-1)))/(-1+torch.exp(-2*kappa)))
       
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
        ret=sphere_base.return_safe_costheta(ret)
        angle=x[:,1:]
        

        if(self.add_extra_rotation_inbetween):

            ret=torch.acos(ret)
            
            rev_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(ret[:,0])))
            log_det=log_det-rev_upd
            
            comb=torch.cat([ret, angle],dim=1)

            comb, log_det=self.spherical_to_eucl_embedding(comb, log_det)
            
            inbetween_matrix=torch.Tensor([[0.0,0.0,1.0],[0.0,1.0,0.0],[-1.0,0.0,0.0]]).to(ret).type_as(ret).unsqueeze(0)

            comb=torch.einsum("...ij,...j->...i", inbetween_matrix.permute(0,2,1), comb)

            comb, log_det=self.eucl_to_spherical_embedding(comb, log_det)

            ret=torch.cos(comb[:,:1])
            fw_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(comb[:,0])))
            log_det=log_det+fw_upd

            angle=comb[:,1:]

        if(self.boundary_cos_theta_identity_region==0.0):
           
            if(correlated_params is not None):
            
                comb=torch.cat([ret, angle],dim=1)
                comb, log_det=self.correlated_rqspline_flow.all_layer_inverse(comb, log_det, None, amortization_parameters=correlated_params)
                ret=comb[:,:1]
                angle=comb[:,1:]
            else:
                ########## do vertical transformation here if desired

                ## fix up circular params if necessary
                if(self.add_circular_rq_spline_flow):
                
                    scalings=self._obtain_azimuthal_flow_scaling(ret)
                   
                    if(circular_params.shape[0]==1):
                        circular_params=circular_params.repeat(scalings.shape[0], 1)
                    
                    used_mask=self.circular_rqspline_flow_param_masks.to(circular_params.device)

                    circular_params=torch.masked_scatter(input=circular_params, mask=used_mask[None,:], source=circular_params[:,used_mask]*scalings)
                  
                    angle, log_det=self.circular_rqspline_flow.all_layer_inverse(angle, log_det, None, amortization_parameters=circular_params)
                    

                if(vertical_params is not None):

                    ret,log_det=self.vertical_rqspline_flow.all_layer_inverse(ret, log_det, None, amortization_parameters=vertical_params)
                

        else:
            contained_mask=((ret>(-1.0+self.boundary_cos_theta_identity_region)) & (ret<(1.0-self.boundary_cos_theta_identity_region)))[:,0]
            assert(contained_mask.dim()==1)
            if(contained_mask.sum()>0):
                if(correlated_params is not None):
                    
                    comb=torch.cat([ret, angle],dim=1)
                    comb_contained, log_det_contained=self.correlated_rqspline_flow.all_layer_inverse(comb[contained_mask], log_det[contained_mask], None, amortization_parameters=correlated_params if self.use_permanent_parameters else correlated_params[contained_mask])
                    
                    assert(contained_mask.dim()==comb[:,0].dim())
                    comb=torch.masked_scatter(input=comb, mask=contained_mask[:,None], source=comb_contained)
                    log_det=torch.masked_scatter(input=log_det, mask=contained_mask, source=log_det_contained)
                      
                    ret=comb[:,:1]
                    angle=comb[:,1:]
                else:
                    ########## do vertical transformation here if desired

                    if(self.add_circular_rq_spline_flow):

                       
                        scalings=self._obtain_azimuthal_flow_scaling(ret)
                       

                        if(circular_params.shape[0]==1):
                            circular_params=circular_params.repeat(scalings.shape[0], 1)
                        
                        used_mask=self.circular_rqspline_flow_param_masks.to(circular_params.device)

                        circular_params=torch.masked_scatter(input=circular_params, mask=used_mask[None,:], source=circular_params[:,used_mask]*scalings)
                      

                        angle_contained, log_det_contained=self.circular_rqspline_flow.all_layer_inverse(angle[contained_mask], log_det[contained_mask], None, amortization_parameters=circular_params[contained_mask])
                       
                        assert(contained_mask.dim()==angle_contained[:,0].dim())
                        angle=torch.masked_scatter(input=angle, mask=contained_mask[:,None], source=angle_contained)
                        
                        log_det=torch.masked_scatter(input=log_det, mask=contained_mask, source=log_det_contained)
        

                    if(vertical_params is not None):
                        
                        ret_contained,log_det_contained=self.vertical_rqspline_flow.all_layer_inverse(ret[contained_mask], log_det[contained_mask], None, amortization_parameters=vertical_params if self.use_permanent_parameters else vertical_params[contained_mask])
                        
                        assert(contained_mask.dim()==ret_contained[:,0].dim())
                        ret=torch.masked_scatter(input=ret, mask=contained_mask[:,None], source=ret_contained)
                        
                        log_det=torch.masked_scatter(input=log_det, mask=contained_mask, source=log_det_contained)
                      
                    
        ret=sphere_base.return_safe_costheta(ret)

        ## go back to angle in a safe way
        ret=torch.acos(ret)
        rev_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(ret[:,0])))

        log_det=log_det-rev_upd

        ret=torch.cat([ret, angle], dim=1)
        
        if(self.always_parametrize_in_embedding_space):

            ret, log_det=self.spherical_to_eucl_embedding(ret, log_det)


        return ret, log_det, sf_extra

    def _flow_mapping(self, inputs, extra_inputs=None, extra_inputs_base=None, sf_extra=None):
        
        [x,log_det]=inputs
       
        if(self.always_parametrize_in_embedding_space):
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)
        
        vertical_params=None
        circular_params=None
        correlated_params=None

        if(extra_inputs is not None):

            if(self.kappa_fn is not None):
                kappa=self.kappa_fn(extra_inputs[:,0:1])
            else:
                assert(extra_inputs_base is not None)

                if(self.kappa_prediction=="mu"):
                    assert(extra_inputs_base.shape[-1]==3)
                    kappa=(extra_inputs_base**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="mu_squared"):
                    assert(extra_inputs_base.shape[-1]==3)
                    kappa=(extra_inputs_base**2).sum(dim=-1,keepdims=True)
                elif(self.kappa_prediction=="quatvec"):
                    assert(extra_inputs_base.shape[-1]==4)
                    kappa=(extra_inputs_base[...,1:]**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="quatvec_squared"):
                    assert(extra_inputs_base.shape[-1]==4)
                    kappa=(extra_inputs_base[...,1:]**2).sum(dim=-1,keepdims=True)

            if(self.add_correlated_rq_spline_flow):
                correlated_params=extra_inputs[:,self.num_loglike_kappa_params:self.total_num_correlated_params+self.num_loglike_kappa_params]
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=extra_inputs[:,self.num_loglike_kappa_params:self.total_num_vertical_params+self.num_loglike_kappa_params]

                if(self.add_circular_rq_spline_flow):
                    circular_params=extra_inputs[:,self.num_loglike_kappa_params+self.total_num_vertical_params:self.total_num_circular_params+self.total_num_vertical_params+self.num_loglike_kappa_params]

        else:
            if(self.kappa_fn is not None):
                kappa=self.kappa_fn(self.loglike_kappa.to(x))
            else:

                if(self.kappa_prediction=="mu"):
                    kappa=(self.householder_params.to(x)**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="mu_squared"):
                    kappa=(self.householder_params.to(x)**2).sum(dim=-1,keepdims=True)
                elif(self.kappa_prediction=="quatvec"):
                    kappa=(self.householder_params.to(x)[...,1:]**2).sum(dim=-1,keepdims=True).sqrt()
                elif(self.kappa_prediction=="quatvec_squared"):
                    kappa=(self.householder_params.to(x)[...,1:]**2).sum(dim=-1,keepdims=True)
            
            if(self.add_correlated_rq_spline_flow):
                correlated_params=self.correlated_flow_params.to(x)
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=self.vertical_flow_params.to(x)
                if(self.add_circular_rq_spline_flow):
                    circular_params=self.circular_flow_params.to(x)

        ## go to cylinder from angle
    
        prev_ret=torch.cos(x[:,:1])

        fw_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(x[:,0])))
        log_det=log_det+fw_upd

        angle=x[:,1:]

        if(self.boundary_cos_theta_identity_region==0.0):

            if(correlated_params is not None):

                comb=torch.cat([prev_ret, angle],dim=1)
                comb, log_det=self.correlated_rqspline_flow.all_layer_forward(comb, log_det, None, amortization_parameters=correlated_params)
                prev_ret=comb[:,:1]
                angle=comb[:,1:]

            else:
                """
                if(circular_params is not None):
                    print("angle bef (FW) ", angle)
                    print("CIRCX PARAMS", circular_params)
                    angle, log_det=self.circular_rqspline_flow.all_layer_forward(angle, log_det, None, amortization_parameters=circular_params)
                    print("naglge af (FW) ", angle)
                """
                ########## do vertical transformation here if desired
                if(vertical_params is not None):
                    prev_ret,log_det=self.vertical_rqspline_flow.all_layer_forward(prev_ret, log_det, None, amortization_parameters=vertical_params)
                

                ## order of circular/vertical flow matters now
                if(self.add_circular_rq_spline_flow):
                    
                    scalings=self._obtain_azimuthal_flow_scaling(prev_ret)
                    
                    if(circular_params.shape[0]==1):
                        circular_params=circular_params.repeat(scalings.shape[0], 1)
                    
                    used_mask=self.circular_rqspline_flow_param_masks.to(circular_params.device)

                    circular_params=torch.masked_scatter(input=circular_params, mask=used_mask[None,:], source=circular_params[:,used_mask]*scalings)
                    
                    angle, log_det=self.circular_rqspline_flow.all_layer_forward(angle, log_det, None, amortization_parameters=circular_params)
            

        else:
            ## should be a 1-d mask in batch dim only
            contained_mask=((prev_ret>(-1.0+self.boundary_cos_theta_identity_region)) & (prev_ret<(1.0-self.boundary_cos_theta_identity_region)))[:,0]
            assert(contained_mask.dim()==1)
            if(contained_mask.sum()>0):
                if(correlated_params is not None):

                    
                    comb=torch.cat([prev_ret, angle],dim=1)
                    comb_contained, log_det_contained=self.correlated_rqspline_flow.all_layer_forward(comb[contained_mask], log_det[contained_mask], None, amortization_parameters=correlated_params if self.use_permanent_parameters else correlated_params[contained_mask])

                    assert(contained_mask.dim()==comb[:,0].dim())
                    comb=torch.masked_scatter(input=comb, mask=contained_mask[:,None], source=comb_contained)
                        
                    log_det=torch.masked_scatter(input=log_det, mask=contained_mask, source=log_det_contained)
                      
                    prev_ret=comb[:,:1]
                    angle=comb[:,1:]

                else:
                  
                    
                    ########## do vertical transformation here if desired
                    if(vertical_params is not None):
                        
                        ret_contained,log_det_contained=self.vertical_rqspline_flow.all_layer_forward(prev_ret[contained_mask], log_det[contained_mask], None, amortization_parameters=vertical_params if self.use_permanent_parameters else vertical_params[contained_mask])
                        
                        assert(contained_mask.dim()==ret_contained[:,0].dim())
                        prev_ret=torch.masked_scatter(input=prev_ret, mask=contained_mask[:,None], source=ret_contained)
                       
                        log_det=torch.masked_scatter(input=log_det, mask=contained_mask, source=log_det_contained)
                    
                    ## order of circular/vertical flow matters now
                    if(self.add_circular_rq_spline_flow):
                       
                        scalings=self._obtain_azimuthal_flow_scaling(prev_ret)
                       
                        if(circular_params.shape[0]==1):
                            circular_params=circular_params.repeat(scalings.shape[0], 1)
                        
                        used_mask=self.circular_rqspline_flow_param_masks.to(circular_params.device)

                        circular_params=torch.masked_scatter(input=circular_params, mask=used_mask[None,:], source=circular_params[:,used_mask]*scalings)
                        

                        angle_contained, log_det_contained=self.circular_rqspline_flow.all_layer_forward(angle[contained_mask], log_det[contained_mask], None, amortization_parameters=circular_params[contained_mask])
                        
               
                        assert(contained_mask.dim()==angle_contained[:,0].dim())
                        angle=torch.masked_scatter(input=angle, mask=contained_mask[:,None], source=angle_contained)
                        
                        log_det=torch.masked_scatter(input=log_det, mask=contained_mask, source=log_det_contained)
                      

        if(self.add_extra_rotation_inbetween):

            ## go back to angle

            prev_ret=torch.acos(prev_ret)
            
            rev_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(prev_ret[:,0])))
            log_det=log_det-rev_upd
            
            comb=torch.cat([prev_ret, angle],dim=1)

            comb, log_det=self.spherical_to_eucl_embedding(comb, log_det)
            
            inbetween_matrix=torch.Tensor([[0.0,0.0,1.0],[0.0,1.0,0.0],[-1.0,0.0,0.0]]).to(prev_ret).type_as(prev_ret).unsqueeze(0)
                        
            comb=torch.einsum("...ij,...j->...i", inbetween_matrix, comb)

            comb, log_det=self.eucl_to_spherical_embedding(comb, log_det)

            prev_ret=torch.cos(comb[:,:1])
            fw_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(comb[:,0])))
            log_det=log_det+fw_upd

            angle=comb[:,1:]


        ## kappa->0 

        ## 0.5+0.5x + (0.5-0.5x)*(1-2k) = 1 -k+kx = (1+k(x-1))^(1/k)
        ## intermediate [-1,1]->[-1,1] transformation

        ## perform the flow with z axis swapped
        #prev_ret_inverse=self.z_scaling_factor*prev_ret
        
        log_det=log_det-torch.log(kappa*self.z_scaling_factor*prev_ret+kappa/torch.tanh(kappa))[:,0]
        
        
        ret=self.z_scaling_factor*(1.0+(1.0/kappa)*torch.log( 0.5*(1.0+self.z_scaling_factor*prev_ret) + (0.5-0.5*self.z_scaling_factor*prev_ret)*torch.exp(-2.0*kappa) ))
        
        if(x.dtype==torch.float32):
            kappa_mask=kappa<1e-4
        elif(x.dtype==torch.float64):
            kappa_mask=kappa<1e-8
        else:
            raise Exception("Require 32 or 64 bit float")

        ret=torch.where(kappa_mask, prev_ret, ret)  

        ret=sphere_base.return_safe_costheta(ret)
       
        ## go back to angle
        ret=torch.acos(ret)
        
        rev_upd=torch.log(torch.sin(sphere_base.return_safe_angle_within_pi(ret[:,0])))
        log_det=log_det-rev_upd

        # join angles again
        ret=torch.cat([ret, angle], dim=1)

        if(self.always_parametrize_in_embedding_space):
            ret, log_det=self.spherical_to_eucl_embedding(ret, log_det)
        
        return ret, log_det

    def _init_params(self, params):

        if(self.kappa_fn is not None):
            self.loglike_kappa.data=params[:1].reshape(1, 1)

        if(self.add_correlated_rq_spline_flow):
            assert(len(params)== (self.num_loglike_kappa_params+self.total_num_correlated_params))

            self.correlated_flow_params.data=params[self.num_loglike_kappa_params:self.num_loglike_kappa_params+self.total_num_correlated_params].reshape(1,self.total_num_correlated_params)
        else:
            assert(len(params)== (self.num_loglike_kappa_params+self.total_num_vertical_params+self.total_num_circular_params))

            if(self.add_vertical_rq_spline_flow):
                self.vertical_flow_params.data=params[self.num_loglike_kappa_params:self.num_loglike_kappa_params+self.total_num_vertical_params].reshape(1,self.total_num_vertical_params)

            if(self.add_circular_rq_spline_flow):
                self.circular_flow_params.data=params[self.num_loglike_kappa_params+self.total_num_vertical_params:self.num_loglike_kappa_params+self.total_num_vertical_params+self.total_num_circular_params].reshape(1,self.total_num_circular_params)


    def _get_desired_init_parameters(self):

        if(self.kappa_fn is not None):
            gaussian_init=torch.randn((1))-3.0#0.01-5.0

            overall_init=gaussian_init
        else:
            overall_init=torch.Tensor([])

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

        loglike_kappa=None

        if(extra_inputs is not None):

            if(self.num_loglike_kappa_params==1):
                loglike_kappa=extra_inputs[:,:1]

            if(self.add_correlated_rq_spline_flow):
                correlated_params=extra_inputs[:,self.num_loglike_kappa_params:self.total_num_correlated_params+self.num_loglike_kappa_params]
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=extra_inputs[:,self.num_loglike_kappa_params:self.total_num_vertical_params+self.num_loglike_kappa_params]
                if(self.add_circular_rq_spline_flow):
                    circular_params=extra_inputs[:,self.num_loglike_kappa_params+self.total_num_vertical_params:self.total_num_circular_params+self.total_num_vertical_params+self.num_loglike_kappa_params]

        else:
            if(self.num_loglike_kappa_params==1):
                loglike_kappa=self.loglike_kappa
            
            if(self.add_correlated_rq_spline_flow):
                correlated_params=self.correlated_flow_params
            else:
                if(self.add_vertical_rq_spline_flow):
                    vertical_params=self.vertical_flow_params
                if(self.add_circular_rq_spline_flow):
                    circular_params=self.circular_flow_params

        if(loglike_kappa is not None):
            param_dict[extra_prefix+"loglike_kappa"]=loglike_kappa.data

        if(self.add_correlated_rq_spline_flow):
            param_dict[extra_prefix+"correlated_params"]=correlated_params.data
        else:
            if(self.add_vertical_rq_spline_flow):

                param_dict[extra_prefix+"vertical_params"]=vertical_params.data

            if(self.add_circular_rq_spline_flow):

                param_dict[extra_prefix+"circular_params"]=circular_params.data