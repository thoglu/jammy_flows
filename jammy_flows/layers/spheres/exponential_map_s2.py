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

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

"""
An implementation of exponential map flows as suggested in https://arxiv.org/abs/2002.02428 ("Normalizing Flows on Tori and Spheres"),
which intself is based upon earlier work https://arxiv.org/abs/0906.0874 ("A Jacobian inequality for gradient maps on the sphere and its application to directional statistics").
"""

def generate_normalization_function(max_value=1.0,stretch_factor=10.0):
    """ 
    Generates a function that bounds the input to be smaller than *max_value*. The input is assumed to be positive.
    """
   
    def f(x):
      
        res=-torch.log(1.0+(numpy.exp(1.0)-1.0)*torch.exp(-x/stretch_factor))+max_value

        return res

    return f

def generate_log_function_bounded_in_logspace(min_val_normal_space=1, max_val_normal_space=10, center=False):
    
    ## min and max values are in normal space -> must be positive
    assert(min_val_normal_space > 0)

    ln_max=numpy.log(max_val_normal_space)
    ln_min=numpy.log(min_val_normal_space)

    ## this shift makes the function equivalent to a normal exponential for small values
    center_val=ln_max

    ## can also center around zero (it will be centered in exp space, not in log space)
    if(center==False):
        center_val=0.0


    def f(x):

        res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-x+center_val).unsqueeze(-1)], dim=-1)

        first_term=ln_max-torch.logsumexp(res, dim=-1, keepdim=True)

        return torch.logsumexp( torch.cat([first_term, torch.ones_like(first_term)*ln_min], dim=-1), dim=-1)

    return f



class exponential_map_s2(sphere_base.sphere_base):

    def __init__(self, 
                 dimension, 
                 euclidean_to_sphere_as_first=False, 
                 use_permanent_parameters=False, 
                 exp_map_type="linear", 
                 natural_direction=0, 
                 num_components=10,
                 add_rotation=0,
                 max_num_newton_iter=1000,
                 mean_parametrization="old"):
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
            mean_parametrization (str): "old" (x,y,z, directly), or "householder", which desribes each mean by a unit vec and rotates them to new positions.
        """
        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_permanent_parameters=use_permanent_parameters, higher_order_cylinder_parametrization=False, add_rotation=add_rotation)
        
        if(dimension!=2):
            raise Exception("The moebius flow should be used for dimension 2!")
        
        self.num_components=num_components
        self.exp_map_type=exp_map_type
        self.natural_direction=natural_direction
        self.max_num_newton_iter=max_num_newton_iter
        self.num_potential_pars=0
        
        self.num_spline_basis_functions=10

        ## mean parmetrization has an impact on the number of parameters
        self.mean_parametrization=mean_parametrization
        if(mean_parametrization=="old"):
            self.num_mu_params=3
            ## we go from positive to [0,1], used custom function here .. maybe not the best
            self.mu_norm_function=generate_normalization_function(stretch_factor=10.0, max_value=1.0)
        else:
            self.num_mu_params=3*3+1 ## 3*3 householder params + one normalization param
            # we go from reals to [0,1], just use sigmoid
            self.mu_norm_function=torch.nn.Sigmoid()

        if(self.exp_map_type=="linear" or self.exp_map_type=="quadratic"):
            self.num_potential_pars=self.num_mu_params+1 # mu vector + weights
        elif(self.exp_map_type=="exponential" or self.exp_map_type=="polynomial"):
            self.num_potential_pars=self.num_mu_params+2 # mu vector + weights + exponential
        elif(self.exp_map_type=="splines"):

            self.num_potential_pars=self.num_mu_params+1+self.num_spline_basis_functions*3+1 # mu vector + weights + spline parameters (3*basis functions +1)
        elif(self.exp_map_type=="nn"):
            raise Exception("Only used for testing, nn exponential flow does not work currently.")
            self.amortized_mlp=AmortizableMLP(3, "64-64", 3, low_rank_approximations=-1, use_permanent_parameters=use_permanent_parameters,svd_mode="smart")
            self.num_mlp_params=self.amortized_mlp.num_amortization_params
            self.total_param_num+=self.num_mlp_params

            self.num_potential_pars=0 # mu vector + weights + 3*spline parameters (3*basis functions +1)
            self.num_components=1

        else:
            raise Exception("Unknown exponential map parameterization %s. Allowed: linear/quadratic/exponential" % self.exp_map_type)

        ## potential parameters
        if(self.use_permanent_parameters):
            self.potential_pars=nn.Parameter(torch.randn(self.num_potential_pars, self.num_components).unsqueeze(0))
        

        self.exponent_log_norm_function=generate_log_function_bounded_in_logspace(min_val_normal_space=1.0, max_val_normal_space=30.0, center=False)

        ## param num = potential_pars*num_potentials 
        self.total_param_num+=self.num_potential_pars*self.num_components
  
    def basic_exponential_map(self, start, v_unit, v_norm):
        """
        Expoential map at base point X with tangent vec v_unit with normalization v_norm.
        """

       
        #return start*torch.cos(v_norm)+v_new*utils.sindiv(v_norm)#*torch.sin(v_norm)

        return start*torch.cos(v_norm)+v_unit*torch.sin(v_norm)

    def unnormalized_logarithmic_map(self, base, unnormalized_target, return_jacobian=False,jacobian_on_unnormalized_target=None):

        assert(len(base.shape)==len(unnormalized_target.shape)), (base.shape, unnormalized_target.shape)
        
        #b=base/(base**2).sum(axis=1,keepdims=True).sqrt()
        b=base
        target_norm=(unnormalized_target**2).sum(axis=1,keepdims=True).sqrt()
        normalized_target=unnormalized_target/target_norm

        cos_alpha=(normalized_target*b).sum(axis=1,keepdims=True)
        alpha=torch.arccos(cos_alpha)
       
        normalized_tangent_vec=(normalized_target-b*cos_alpha)/torch.sin(alpha)

        # project unnormalized target on tangent vector

        projection=(unnormalized_target*normalized_tangent_vec).sum(axis=1,keepdims=True)

        if(return_jacobian):

            ## calculate jacobian w.r.t to *base*
            d_tangent_d_base=torch.diag_embed((-cos_alpha/torch.sin(alpha)).repeat(1,3))
            d_tangent_d_theta=((b-normalized_target*cos_alpha)/(torch.sin(alpha)**2)).unsqueeze(-1)
            d_theta_d_base=((-1.0/torch.sqrt(1.0-((normalized_target*b).sum(axis=1,keepdims=True))**2))*normalized_target).unsqueeze(1)

            sec_fac=d_tangent_d_theta @ d_theta_d_base

            # total jacobian on the normalized tangent vector
            total_jac_tangent_vec=d_tangent_d_base+sec_fac

            # total jacobian on the projection onto the tangent vector
            total_jac_projection=((total_jac_tangent_vec*unnormalized_target.unsqueeze(-1)).sum(axis=1, keepdims=True))

            if(jacobian_on_unnormalized_target is not None):

                assert(jacobian_on_unnormalized_target.shape[1:]==total_jac_tangent_vec.shape[1:]), (jacobian_on_unnormalized_target.shape, total_jac_tangent_vec.shape)

                # we also have a jacobian of unnormalized_target w.r.t to base given
                # calculate jaobian w.r.t. unnormlized target and propagate via chain rule

                # d/d_unnormalized_target (normalized_tangent_vec) = 
                d_theta_d_norm=((-1.0/torch.sqrt(1.0-((normalized_target*b).sum(axis=1,keepdims=True))**2))*b).unsqueeze(1)
                d_norm_d_unnorm=(-unnormalized_target/target_norm**2).unsqueeze(-1) @ normalized_target.unsqueeze(1)+torch.diag_embed(1.0/target_norm.repeat(1,3))

                d_tangent_d_norm=torch.diag_embed(1.0/torch.sin(alpha).repeat(1,3))
            
                # add jacobian parts on tangent vector for unnormalized vector
                total_jac_tangent_vec=total_jac_tangent_vec+d_tangent_d_theta @ d_theta_d_norm @ d_norm_d_unnorm @ jacobian_on_unnormalized_target
                total_jac_tangent_vec=total_jac_tangent_vec+d_tangent_d_norm @ d_norm_d_unnorm @ jacobian_on_unnormalized_target

                # add jacobian parts for the projection aswell
                total_jac_projection=total_jac_projection + (normalized_tangent_vec.unsqueeze(-1)*jacobian_on_unnormalized_target).sum(axis=1, keepdims=True)
                
            return normalized_tangent_vec, projection ,total_jac_tangent_vec, total_jac_projection
        else:

            return normalized_tangent_vec, projection

    def basic_logarithmic_map(self, base, target):
        

        assert(len(base.shape)==len(target.shape)), (base.shape, target.shape)
      
        alternative_base=torch.zeros_like(base)
        alternative_base[:,0]=1.0

        cos_alpha=(target*base).sum(axis=1,keepdims=True)
        alternative_cos_alpha=(target*alternative_base).sum(axis=1,keepdims=True)

        converged_mask=cos_alpha>=1

        cos_alpha=torch.masked_scatter(input=cos_alpha, mask=converged_mask, source=alternative_cos_alpha[converged_mask])
        alpha=torch.arccos(cos_alpha)
        
        used_base=torch.masked_scatter(input=base, mask=converged_mask, source=alternative_base[converged_mask[:,0]])

        normalized_tangent_vec=(target-used_base*cos_alpha)/torch.sin(alpha)

        ## set alphas to 0 where we are close
        alpha=torch.masked_scatter(input=alpha, mask=converged_mask, source=torch.zeros_like(alternative_cos_alpha)[converged_mask])

        return normalized_tangent_vec, alpha
   
   

    def get_exp_map_and_jacobian(self, x, potential_pars):
        """
        Calculates the exponential map and its jacobian in one go, since Jacobian always requires to calculate the exponential map anyway.
        """

        if(self.exp_map_type!="nn"):

            if(self.mean_parametrization=="old"):

                norm=((potential_pars[:,:3,:]**2).sum(axis=1, keepdim=True)).sqrt()

                zero_mask=norm==0.0
                assert(zero_mask.sum()==0)

                normalized_mu=potential_pars[:,:3,:]/norm

                fake_norm=self.mu_norm_function(norm)

            else:

                reshaped_pars=potential_pars[:,:9,:].permute(0,2,1).reshape(-1, 3,3)
               
                hh_matrices=self.compute_householder_matrix(reshaped_pars, 3, device=x.device)

                
                hh_matrices=hh_matrices.reshape(-1,self.num_components,9).reshape(-1,self.num_components, 3,3).permute(0,2,3,1)
                
                pre_mu=torch.zeros_like(potential_pars[:,:3,:])
                pre_mu[:,2,:]=1.0

                normalized_mu=torch.einsum("bijm,bjm->bim", hh_matrices, pre_mu)

                pre_norm=potential_pars[:,9:10,:]
                fake_norm=self.mu_norm_function(pre_norm)

        
        #print("exm map type ", self.exp_map_type)
        if(self.exp_map_type=="exponential"):
           
    
            log_weights=potential_pars[:,self.num_mu_params:self.num_mu_params+1,:]-torch.logsumexp(potential_pars[:,self.num_mu_params:self.num_mu_params+1,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            #xl=x/(x**2).sum(axis=1,keepdims=True).sqrt()

            #xl=xl/(xl**2).sum(axis=1,keepdims=True).sqrt()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            scaling_beta=potential_pars[:,self.num_mu_params+1:self.num_mu_params+2,:].exp()

           

            pure_grad_vec=(weights*normalized_mu*torch.exp(scaling_beta*(x_times_mu-1.0))).sum(axis=-1)

        

            # B X 3 X 1 X num_compoents
            pure_grad_vec_jacobian=(scaling_beta*weights*normalized_mu*torch.exp(scaling_beta*(x_times_mu-1.0))).unsqueeze(2)

            # B X 1 X 3 X num_components
            d_x_time_u_dx=normalized_mu.unsqueeze(1)
           
            pure_grad_vec_jacobian= torch.einsum("...iju, ...jlu -> ...ilu", pure_grad_vec_jacobian, d_x_time_u_dx)

            ## sum over components
            pure_grad_vec_jacobian=pure_grad_vec_jacobian.sum(axis=-1)
        elif(self.exp_map_type=="linear"):

            log_weights=potential_pars[:,self.num_mu_params:self.num_mu_params+1,:]-torch.logsumexp(potential_pars[:,self.num_mu_params:self.num_mu_params+1,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)
            
            pure_grad_vec=(weights*normalized_mu).sum(axis=-1)

            pure_grad_vec_jacobian=None
        elif(self.exp_map_type=="quadratic"):

            log_weights=potential_pars[:,self.num_mu_params:self.num_mu_params+1,:]-torch.logsumexp(potential_pars[:,self.num_mu_params:self.num_mu_params+1,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)
            
            pure_grad_vec=(weights*normalized_mu*x_times_mu).sum(axis=-1)

            # B X 3 X 1 X num_compoents
            pure_grad_vec_jacobian=(weights*normalized_mu).unsqueeze(2)

            # B X 1 X 3 X num_components
            d_x_time_u_dx=normalized_mu.unsqueeze(1)
           
            pure_grad_vec_jacobian= torch.einsum("...iju, ...jlu -> ...ilu", pure_grad_vec_jacobian, d_x_time_u_dx)

            ## sum over components
            pure_grad_vec_jacobian=pure_grad_vec_jacobian.sum(axis=-1)

       
        elif(self.exp_map_type=="splines"):

            log_weights=potential_pars[:,self.num_mu_params:self.num_mu_params+1,:]-torch.logsumexp(potential_pars[:,self.num_mu_params:self.num_mu_params+1,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            unnormalized_widths=potential_pars[:,self.num_mu_params+1:self.num_spline_basis_functions+self.num_mu_params+1,:].permute(0,2,1)
            unnormalized_heights=potential_pars[:,self.num_spline_basis_functions+self.num_mu_params+1:2*self.num_spline_basis_functions+self.num_mu_params+1,:].permute(0,2,1)
            unnormalized_derivatives=potential_pars[:,2*self.num_spline_basis_functions+self.num_mu_params+1:3*self.num_spline_basis_functions+self.num_mu_params+2,:].permute(0,2,1)

            res, log_deriv=spline_fns.rational_quadratic_spline(x_times_mu.permute(0,2,1),
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=-1.0, right=1.0, bottom=-1.0, top=1.0,
                              rel_min_bin_width=1e-3,
                              rel_min_bin_height=1e-3,
                              min_derivative=1e-3)

            ## back to normal order
            deriv=log_deriv.exp().permute(0,2,1)
            res=res.permute(0,2,1)


            # actual potential is integral of spline function
            # grad is the actual spline function * inner part (normalized mu)
            pure_grad_vec=(weights*normalized_mu*res).sum(axis=-1)



            # B X 3 X 1 X num_compoents
            pure_grad_vec_jacobian=(weights*normalized_mu*deriv).unsqueeze(2)

            # B X 1 X 3 X num_components
            d_x_time_u_dx=normalized_mu.unsqueeze(1)
           
            pure_grad_vec_jacobian= torch.einsum("...iju, ...jlu -> ...ilu", pure_grad_vec_jacobian, d_x_time_u_dx)

            ## sum over components
            pure_grad_vec_jacobian=pure_grad_vec_jacobian.sum(axis=-1)

        elif(self.exp_map_type=="nn"):
            # not really sure if this is truly bijective, but it seems to work
            raise Exception("doees not work - only used for testing - bijection doesnt work ")
            
            ## grad vec between -1 and 1
            apply_mlp=lambda xx: NONLINEARITIES["tanh"](self.amortized_mlp(xx))*1.0-0.5

            pure_grad_vec=NONLINEARITIES["tanh"](self.amortized_mlp(x))*1.0-0.5

            pure_grad_vec_jacobian=torch.autograd.functional.jacobian(apply_mlp, x, create_graph=False).sum(axis=2)

        else:
            raise Exception()

        
        # calculate quantities related to unnormalized logarithmic map
        tangent_vec, tangent_vec_norm, tangent_vec_jac, tangent_vec_norm_jac=self.unnormalized_logarithmic_map(x, pure_grad_vec, jacobian_on_unnormalized_target=pure_grad_vec_jacobian, return_jacobian=True)

        # calculate exponential map

        #print("tangent vec ", tangent_vec[19], tangent_vec_norm[19])
        total_exp_map_result=self.basic_exponential_map(x, tangent_vec, tangent_vec_norm)
        #print("RESULT ", total_exp_map_result[19])
        ## unnormed_tangent_vec
        #unnormed_tangent_vec=tangent_vec*tangent_vec_norm
        
        # jacobian of total tangent_vec
        #jac_unnormed_tangent_vec=tangent_vec.unsqueeze(-1) @ tangent_vec_norm_jac+torch.diag_embed(tangent_vec_norm.repeat(1,3)) @ tangent_vec_jac

        
        outer=torch.einsum("...ij, ...jl -> ...il", (-x*torch.sin(tangent_vec_norm)).unsqueeze(-1), tangent_vec_norm_jac)

        first=torch.diag_embed(torch.cos(tangent_vec_norm).repeat(1, 3))+outer

        # d/dx (v/|v|)* sin(phi) 
        second=tangent_vec_jac*torch.sin(tangent_vec_norm.unsqueeze(-1))+torch.einsum("...ij, ...jl -> ...il", (tangent_vec*torch.cos(tangent_vec_norm)).unsqueeze(-1), tangent_vec_norm_jac)

        ## total jacobian of exponential map in embedding (3-dim) space
        total_exp_map_jac=first+second
        ####################################################################
            
        # project jacobian onto tangential plane
        second_tangent_vec=torch.cross(x,tangent_vec, dim=1)

        tangent_basis=torch.cat([tangent_vec.unsqueeze(2), second_tangent_vec.unsqueeze(2)], dim=2)

        # project jacobian
        projected_jacobian=torch.bmm(total_exp_map_jac, tangent_basis)
            
        # calculate p^T*p
        projected_jacobian_squared=torch.bmm(projected_jacobian.permute(0,2,1), projected_jacobian)

        ## return exponential map, the Jacobian^2 projected onto the tangent plane (3X3 Jacobian -> 2X3 Jacobian, -> squareding 2X3 X3X2=2X2), and the full Jacobian^2 (3x3 Jacobian)
        return total_exp_map_result, projected_jacobian_squared, total_exp_map_jac, tangent_vec

   

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x,log_det]=inputs
        
        assert(x.dtype==torch.float64), "V flow requires float64, otherwise it often will not converge correctly!"
            
      
        if(extra_inputs is not None):
           
            potential_pars=extra_inputs.reshape(x.shape[0], self.num_potential_pars, self.num_components)
        else:
            potential_pars=self.potential_pars.to(x)

        if(self.always_parametrize_in_embedding_space==False):
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)
           
        if(self.natural_direction):


            result=inverse_bisection_n_newton_sphere(self.get_exp_map_and_jacobian, self.basic_logarithmic_map, self.basic_exponential_map, x, potential_pars, num_newton_iter=self.max_num_newton_iter )

            _, jac_squared, _,_=self.get_exp_map_and_jacobian(result, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)
           
            log_det=log_det-0.5*slog_det
        else:


            result, new_projected, new_standard,_=self.get_exp_map_and_jacobian(x, potential_pars)
            
            _,ld_res=torch.slogdet(new_projected)
                
            log_det_update=0.5*ld_res

            log_det=log_det+log_det_update


        if(self.always_parametrize_in_embedding_space==False):
            # embedding to intrinsic
            result, log_det=self.eucl_to_spherical_embedding(result, log_det)

        return result, log_det, None

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
        #print("calculating forward flow ?!")
        [x,log_det]=inputs

        assert(x.dtype==torch.float64), "V flow requires float64, otherwise it often will not converge correctly!"
        
        if(self.always_parametrize_in_embedding_space==False):
            # s2 flow is defined in embedding space,
            #x, log_det=self.eucl_to_spherical_embedding(x, log_det)

            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        if(extra_inputs is not None):
            potential_pars=extra_inputs.reshape(x.shape[0], self.num_potential_pars, self.num_components)
        else:
            potential_pars=self.potential_pars.to(x)
        

        if(self.natural_direction):
            result, jac_squared, _,_=self.get_exp_map_and_jacobian(x, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)

            log_det=log_det+0.5*slog_det

        else:

           

            result=inverse_bisection_n_newton_sphere_fast(self.get_exp_map_and_jacobian,self.basic_logarithmic_map,  self.basic_exponential_map, x, potential_pars, num_newton_iter=self.max_num_newton_iter )

            _, jac_squared, _,_=self.get_exp_map_and_jacobian(result, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)
           
            log_det=log_det-0.5*slog_det


        if(self.always_parametrize_in_embedding_space==False):
            result, log_det=self.eucl_to_spherical_embedding(result, log_det)

        return result, log_det

    def _init_params(self, params):

        if(self.exp_map_type!="nn"):

            assert(len(params)== (self.num_potential_pars*self.num_components))

            self.potential_pars.data=params.reshape(1, self.num_potential_pars, self.num_components)

        else:
            mlp_params=params[:self.num_mlp_params]

            bias_mlp_pars=mlp_params[-3:]

            self.amortized_mlp.initialize_uvbs(init_b=bias_mlp_pars)

    def _get_desired_init_parameters(self):

        if(self.exp_map_type!="nn"):
            gaussian_init=torch.randn((self.num_potential_pars*self.num_components))

            
        else:
            gaussian_init=torch.randn((self.num_mlp_params))
        return gaussian_init


    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Implemented by Euclidean sublayers.
        """

        
        if(extra_inputs is not None):
            potential_pars=extra_inputs
        else:
            potential_pars=self.potential_pars

        param_dict[extra_prefix+"potential_pars"]=potential_pars.data