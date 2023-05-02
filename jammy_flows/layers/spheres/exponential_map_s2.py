import torch
from torch import nn
import numpy

from . import sphere_base
from . import moebius_1d
from .. import spline_fns
from ..bisection_n_newton import inverse_bisection_n_newton_sphere
from ...amortizable_mlp import AmortizableMLP
from ...extra_functions import list_from_str

from ..euclidean.polynomial_stretch_flow import psf_block
from ..euclidean.euclidean_do_nothing import euclidean_do_nothing

from ...extra_functions import NONLINEARITIES


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
                 add_rotation=0):
        """
        Uses the spherical exponential map. Symbol: "v"

        Uses linear and quadratic potential as described in https://arxiv.org/abs/0906.0874, and exponential potential as described in https://arxiv.org/abs/2002.02428.
        Additionally added a spline-based potential.

        Parameters:
        
            exp_map_type (str): Defines the potential of the exponential map. One of ["linear", "quadratic", "exponential", "splines"].
            natural_direction (int). If 0, log-probability evaluation is faster. If 1, sampling is faster.
            num_components (int): How many components to sum over in the exponential map.
        """
        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_permanent_parameters=use_permanent_parameters, higher_order_cylinder_parametrization=False, add_rotation=add_rotation)
        
        if(dimension!=2):
            raise Exception("The moebius flow should be used for dimension 2!")
        
        self.num_components=num_components
        self.exp_map_type=exp_map_type
        self.natural_direction=natural_direction
        self.num_potential_pars=0
        
        self.num_spline_basis_functions=10

        if(self.exp_map_type=="linear" or self.exp_map_type=="quadratic"):
            self.num_potential_pars=3+1 # mu vector + weights
        elif(self.exp_map_type=="exponential" or self.exp_map_type=="polynomial"):
            self.num_potential_pars=3+2 # mu vector + weights + exponential
        elif(self.exp_map_type=="splines"):

            self.num_potential_pars=3+1+self.num_spline_basis_functions*3+1 # mu vector + weights + spline parameters (3*basis functions +1)
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
            self.potential_pars=nn.Parameter(torch.randn(self.num_potential_pars, self.num_components).type(torch.double).unsqueeze(0))
        else:
            self.potential_pars=torch.zeros(self.num_potential_pars, self.num_components).type(torch.double).unsqueeze(0)

        self.mu_norm_function=generate_normalization_function(stretch_factor=10.0, max_value=1.0)

        self.exponent_log_norm_function=generate_log_function_bounded_in_logspace(min_val_normal_space=1.0, max_val_normal_space=30.0, center=False)

        ## param num = potential_pars*num_potentials 
        self.total_param_num+=self.num_potential_pars*self.num_components
    """
    def all_vs(self, x, normalized_mu, in_newton=False):
        
        mu_1=normalized_mu[:,0:1,:]
        mu_2=normalized_mu[:,1:2,:]
        mu_3=normalized_mu[:,2:3,:]


        x_1=x[:,0:1,None]
        x_2=x[:,1:2,None]
        x_3=x[:,2:3,None]

        v_1_top=-mu_1*x_2**2+x_1*x_2*mu_2-mu_1*x_3**2+x_1*x_3*mu_3
        v_2_top=-mu_2*x_1**2+x_2*x_1*mu_1-mu_2*x_3**2+x_2*x_3*mu_3
        v_3_top=-mu_3*x_1**2+x_1*x_3*mu_1-mu_3*x_2**2+x_2*x_3*mu_2


        x_norm=(x_1**2+x_2**2+x_3**2)**(1.0/2.0)

        bbef=((mu_2**2+mu_3**2)*x_1**2 + (mu_1**2+mu_2**2)*x_3**2 + (mu_1**2+mu_3**2)*x_2**2 - 2*x_1*x_3*mu_1*mu_3 - 2*x_1*x_2*mu_1*mu_2 - 2*x_2*x_3*mu_2*mu_3)

        ### this mask is >0 if due to floating point issues the value is negative .. this haas to be handled later for correct non-NAN gradfients
        issue_mask=bbef<=0

        bbef=bbef.masked_fill(bbef<=0, 1e-10)
        b=(bbef).sqrt()
        v_bottom=b*x_norm

        

        ## if this function is used outside of newton iterations, we should be fine.. No need to check
        if(in_newton==False):

            v_1=v_1_top/v_bottom
            v_2=v_2_top/v_bottom
            v_3=v_3_top/v_bottom

            all_vs=torch.cat([v_1,v_2,v_3], dim=1)

            return all_vs



        v_1=v_1_top/v_bottom
        v_2=v_2_top/v_bottom
        v_3=v_3_top/v_bottom

        ### Generate a vector that is orthogonal (but in principle arbitrary orientation)  
        arb_z=(-x[:,0:1]*0.5-x[:,1:2]*0.5)/x[:,2:3]

        arb=torch.cat([ torch.ones_like(x[:,0:1])*0.5, torch.ones_like(x[:,0:1])*0.5, arb_z], dim=1).unsqueeze(2).repeat(1,1,normalized_mu.shape[2])

        arb=arb/((arb**2).sum(axis=1,keepdims=True).sqrt())

      

        ### those values that have an issue with potential division by zero are replaced with arbitrary vecs .. doesnt make a difference really since target is basically reached already
        v_1 = torch.masked_scatter(input=v_1, mask=issue_mask, source=arb[:,0:1,:][issue_mask])
        v_2 = torch.masked_scatter(input=v_2, mask=issue_mask, source=arb[:,1:2,:][issue_mask])
        v_3 = torch.masked_scatter(input=v_3, mask=issue_mask, source=arb[:,2:3,:][issue_mask])

   

        all_vs=torch.cat([v_1,v_2,v_3], dim=1)
            
        return all_vs

    """

    """
    ## jacobian of vs x,y,z axes
    def all_vs_jacobians(self, x, normalized_mu):
        
        mu_1=normalized_mu[:,0:1,:]
        mu_2=normalized_mu[:,1:2,:]
        mu_3=normalized_mu[:,2:3,:]


        x_1=x[:,0:1,None]
        x_2=x[:,1:2,None]
        x_3=x[:,2:3,None]

        v_1_top=-mu_1*x_2**2+x_1*x_2*mu_2-mu_1*x_3**2+x_1*x_3*mu_3
        v_2_top=-mu_2*x_1**2+x_2*x_1*mu_1-mu_2*x_3**2+x_2*x_3*mu_3
        v_3_top=-mu_3*x_1**2+x_1*x_3*mu_1-mu_3*x_2**2+x_2*x_3*mu_2

        v_top_vec=torch.cat([v_1_top, v_2_top, v_3_top], dim=1)

       
        #### jacobian d_v_top / d_x
        v_1_top_x=x_2*mu_2+x_3*mu_3
        v_1_top_y=-2*mu_1*x_2+x_1*mu_2
        v_1_top_z=-2*x_3*mu_1+mu_3*x_1

        v_2_top_x=-2*mu_2*x_1+x_2*mu_1
        v_2_top_y=x_1*mu_1+x_3*mu_3
        v_2_top_z=-2*mu_2*x_3+x_2*mu_3

        v_3_top_x=-2*mu_3*x_1+x_3*mu_1
        v_3_top_y=-2*mu_3*x_2+x_3*mu_2
        v_3_top_z=x_1*mu_1+x_2*mu_2

        v_top_jac_row1=torch.cat([v_1_top_x, v_1_top_y, v_1_top_z], dim=1)
        v_top_jac_row2=torch.cat([v_2_top_x, v_2_top_y, v_2_top_z], dim=1)
        v_top_jac_row3=torch.cat([v_3_top_x, v_3_top_y, v_3_top_z], dim=1)

       
        v_top_jacobian=torch.cat([v_top_jac_row1.unsqueeze(1),v_top_jac_row2.unsqueeze(1),v_top_jac_row3.unsqueeze(1) ], dim=1)
        
       
        ###############

        x_norm=(x_1**2+x_2**2+x_3**2)**(1.0/2.0)

        x_norm_dx=x_1/x_norm
        x_norm_dy=x_2/x_norm
        x_norm_dz=x_3/x_norm

        b=((mu_2**2+mu_3**2)*x_1**2 + (mu_1**2+mu_2**2)*x_3**2 + (mu_1**2+mu_3**2)*x_2**2 - 2*x_1*x_3*mu_1*mu_3 - 2*x_1*x_2*mu_1*mu_2 - 2*x_2*x_3*mu_2*mu_3).sqrt()

        b_dx=((mu_2**2+mu_3**2)*x_1  - x_3*mu_1*mu_3 - x_2*mu_1*mu_2)/b
        b_dy=( (mu_1**2+mu_3**2)*x_2  - x_1*mu_1*mu_2 - x_3*mu_2*mu_3)/b
        b_dz=((mu_1**2+mu_2**2)*x_3  - x_1*mu_1*mu_3  - x_2*mu_2*mu_3)/b

        v_bottom=(b*x_norm).unsqueeze(1)

        ## d v_bottom / d X

        v_bottom_dx=b_dx*x_norm+b*x_norm_dx
        v_bottom_dy=b_dy*x_norm+b*x_norm_dy
        v_bottom_dz=b_dz*x_norm+b*x_norm_dz

        v_bottom_grad=torch.cat([v_bottom_dx, v_bottom_dy, v_bottom_dz], dim=1)

        
        ### D/DX ( V/v_bottom) = (JAC(V)*v_bottom- V X grad(v_bottom) )/v_bottom**2

        outer_product=torch.einsum("aib, ajb -> aijb", v_top_vec , v_bottom_grad)

      
        final=v_top_jacobian/v_bottom - outer_product/(v_bottom**2)

        return final

    """
    """
    def exp_map(self, x, potential_pars):

        ## calculate normalized tangent vector v of exponential map
        norm=((potential_pars[:,:3,:]**2).sum(axis=1, keepdim=True)).sqrt()

        normalized_mu=potential_pars[:,:3,:]/norm
       
        all_vs=self.all_vs(x, normalized_mu)


        ## specific choices have different "grad_phi" s, i.e. normalizations of v
        if(self.exp_map_type=="linear"):
            
            log_weights=potential_pars[:,3:,:]
        
            weights=torch.softmax(log_weights, dim=2)

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            #alphas=torch.arccos( x_times_mu )

            ##
            projections=(all_vs*weights*normalized_mu).sum(axis=1, keepdims=True)

     
            tangent_vecs=projections*all_vs

            aggregated_vs=tangent_vecs.sum(axis=2)

            grad_phi_norm=(aggregated_vs**2).sqrt()
            

            new_vs=aggregated_vs/grad_phi_norm

            res=x*torch.cos(grad_phi_norm)+new_vs*torch.sin(grad_phi_norm)

        elif(self.exp_map_type=="exponential"):


            log_weights=potential_pars[:,3:4,:]
            weights=torch.softmax(log_weights, dim=2)
                
            scaling_beta=potential_pars[:,4:5,:].exp()


            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            pure_grad=weights*normalized_mu*torch.exp(scaling_beta*(x_times_mu-1.0))


            projections=(all_vs*pure_grad).sum(axis=1, keepdims=True)

            tangent_vecs=projections*all_vs


            aggregated_vs=tangent_vecs.sum(axis=2)


            grad_phi_norm=(aggregated_vs**2).sum(axis=1,keepdims=True).sqrt()
            

            new_vs=aggregated_vs/grad_phi_norm

            res=x*torch.cos(grad_phi_norm)+new_vs*torch.sin(grad_phi_norm)


        return res, new_vs
    """
    """
    def get_explicit_formula_result(self, x, potential_pars):

        ## works only for linear atm
       
        
        norm=((potential_pars[:,:3,:]**2).sum(axis=1, keepdim=True)).sqrt()

        zero_mask=norm==0.0
        assert(zero_mask.sum()==0)

        normalized_mu=potential_pars[:,:3,:]/norm

       
        fake_norm=self.mu_norm_function(norm)

     
        log_weights=potential_pars[:,3:4,:]-torch.logsumexp(potential_pars[:,3:4,:],dim=2,keepdims=True)+fake_norm.log()

     
        ############################

        unsqueezed_x=x.unsqueeze(-1)

        # angle between each kernel mu_i and x
        alpha_i=torch.arccos((unsqueezed_x*normalized_mu).sum(axis=1, keepdims=True))

        # tangent normal vector from x pointint towards mu_i
        e_i=(normalized_mu-unsqueezed_x*torch.cos(alpha_i))/torch.sin(alpha_i)
      
        # xxT

        x_x_t=torch.einsum("bia,bja->bija", unsqueezed_x, unsqueezed_x)

        e_e_t=torch.einsum("bia,bja->bija", e_i, e_i)

        if(self.exp_map_type=="linear"):
            f_prime=-torch.sin(alpha_i)
            f_double_prime=-torch.cos(alpha_i)

        elif(self.exp_map_type=="quadratic"):
            

            #f_prime=-0.5*torch.sin(2*alpha_i)
            #f_double_prime=-torch.cos(2*alpha_i)
            # different parametrization more amenable to generalization
            f_prime=-torch.cos(alpha_i)*torch.sin(alpha_i)
            f_double_prime=torch.sin(alpha_i)**2-torch.cos(alpha_i)**2

        elif(self.exp_map_type=="polynomial"):
            
            exponents=self.exponent_log_norm_function(potential_pars[:,4:5,:]).exp()

        
            f_prime=-(1.0/exponents)*torch.sin(exponents*alpha_i)
            f_double_prime=-torch.cos(exponents*alpha_i)

            #f_prime=-torch.sin(alpha_i)*torch.cos(alpha_i)**(exponents-1.0)
            #f_double_prime=((exponents-1.0))*(torch.sin(alpha_i)**2)*torch.cos(alpha_i)**(exponents-2.0)-(1.0)*torch.cos(alpha_i)**(exponents)

        elif(self.exp_map_type=="exponential"):
            betas=potential_pars[:,4:5,:].exp()
            
            exp_factor=torch.exp( betas*(torch.cos(alpha_i)-1.0))
            f_prime=-torch.sin(alpha_i)*exp_factor

            f_double_prime=-torch.cos(alpha_i)*exp_factor+torch.sin(alpha_i)*betas*torch.sin(alpha_i)*exp_factor
        else:
            raise Exception("unsupported exp map type: %s", self.exp_map_type)

      
        v_total=-(log_weights.exp()*f_prime*e_i).sum(axis=-1)

        norm_v_total=((v_total**2).sum(axis=-1, keepdims=True).sqrt())

      
        e_v_total=v_total/norm_v_total

        e_v_e_v_t=torch.einsum("bi,bj->bij", e_v_total, e_v_total)

       
        H_v=e_v_e_v_t+norm_v_total.unsqueeze(-1)*torch.cos(norm_v_total.unsqueeze(-1))/torch.sin(norm_v_total.unsqueeze(-1))*(torch.eye(3).unsqueeze(0).to(norm_v_total)-x_x_t.squeeze(-1)-e_v_e_v_t)

        
        K_i=f_double_prime.unsqueeze(1)*e_e_t+f_prime.unsqueeze(1)*(torch.cos(alpha_i).unsqueeze(1)/torch.sin(alpha_i).unsqueeze(1))*(torch.eye(3).unsqueeze(0).unsqueeze(-1).to(norm_v_total)-x_x_t-e_e_t)
        
        K_tot=((log_weights.unsqueeze(1).exp())*K_i).sum(axis=-1)

        
        signs,log_det_fac=torch.linalg.slogdet(x_x_t.squeeze(-1)+H_v+K_tot)

      
        log_det_fac=log_det_fac+torch.log(torch.sin(norm_v_total).squeeze(-1))-torch.log(norm_v_total).squeeze(-1)
        
       
        new_res=self.basic_exponential_map(x, e_v_total, norm_v_total)


        full_jacobian=x_x_t.squeeze(-1)+H_v+K_tot
        full_jacobian=full_jacobian*((torch.sin(norm_v_total)/norm_v_total)**(1/3)).unsqueeze(-1)
            
        fin_mask=(torch.isfinite(log_det_fac)==0)

        if(fin_mask.sum()>0):
            print("NON finite")
            print(fin_mask.shape)

            print("xxt")
            print(x_x_t[fin_mask])

            print("HV")
            print(H_v[fin_mask])

            print("ktot")
            print(K_tot[fin_mask])

            print(log_det_fac[fin_mask])

            sign_res, bad_res=torch.linalg.slogdet(  (x_x_t.squeeze(-1)+H_v+K_tot)[fin_mask] )

            print(sign_res)
            print(bad_res)
            sys.exit(-1)
        
        return new_res, log_det_fac, full_jacobian
    """
    def basic_exponential_map(self, start, v_unit, v_norm):
        """
        Expoential map at base point X with tangent vec v_unit with normalization v_norm.
        """
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
        """
        Logarithmic map at base point "base" towards "target".
        """

        assert(len(base.shape)==len(target.shape)), (base.shape, target.shape)
       
        cos_alpha=(target*base).sum(axis=1,keepdims=True)
     
        alpha=torch.arccos(cos_alpha)

        # make sure we get sane division by sin(alpha) for small (zero) alphas
        alpha=torch.where(alpha<1e-6, 1e-6, alpha)

        normalized_tangent_vec=(target-base*cos_alpha)/torch.sin(alpha)

        return normalized_tangent_vec, alpha

   

    def get_exp_map_and_jacobian(self, x, potential_pars):
        """
        Calculates the exponential map and its jacobian in one go, since Jacobian always requires to calculate the exponential map anyway.
        """

        if(self.exp_map_type!="nn"):
            norm=((potential_pars[:,:3,:]**2).sum(axis=1, keepdim=True)).sqrt()

            zero_mask=norm==0.0
            assert(zero_mask.sum()==0)

            normalized_mu=potential_pars[:,:3,:]/norm

            fake_norm=self.mu_norm_function(norm)
       
        #print("exm map type ", self.exp_map_type)
        if(self.exp_map_type=="exponential"):
           
    
            log_weights=potential_pars[:,3:4,:]-torch.logsumexp(potential_pars[:,3:4,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            #xl=x/(x**2).sum(axis=1,keepdims=True).sqrt()

            #xl=xl/(xl**2).sum(axis=1,keepdims=True).sqrt()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            scaling_beta=potential_pars[:,4:5,:].exp()

            

            pure_grad_vec=(weights*normalized_mu*torch.exp(scaling_beta*(x_times_mu-1.0))).sum(axis=-1)

        

            # B X 3 X 1 X num_compoents
            pure_grad_vec_jacobian=(scaling_beta*weights*normalized_mu*torch.exp(scaling_beta*(x_times_mu-1.0))).unsqueeze(2)

            # B X 1 X 3 X num_components
            d_x_time_u_dx=normalized_mu.unsqueeze(1)
           
            pure_grad_vec_jacobian= torch.einsum("...iju, ...jlu -> ...ilu", pure_grad_vec_jacobian, d_x_time_u_dx)

            ## sum over components
            pure_grad_vec_jacobian=pure_grad_vec_jacobian.sum(axis=-1)
        elif(self.exp_map_type=="linear"):

            log_weights=potential_pars[:,3:4,:]-torch.logsumexp(potential_pars[:,3:4,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)
            
            pure_grad_vec=(weights*normalized_mu).sum(axis=-1)

            pure_grad_vec_jacobian=None
        elif(self.exp_map_type=="quadratic"):

            log_weights=potential_pars[:,3:4,:]-torch.logsumexp(potential_pars[:,3:4,:],dim=2,keepdims=True)+fake_norm.log()
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

            log_weights=potential_pars[:,3:4,:]-torch.logsumexp(potential_pars[:,3:4,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            unnormalized_widths=potential_pars[:,4:self.num_spline_basis_functions+4,:].permute(0,2,1)
            unnormalized_heights=potential_pars[:,self.num_spline_basis_functions+4:2*self.num_spline_basis_functions+4,:].permute(0,2,1)
            unnormalized_derivatives=potential_pars[:,2*self.num_spline_basis_functions+4:3*self.num_spline_basis_functions+5,:].permute(0,2,1)

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
        total_exp_map_result=self.basic_exponential_map(x, tangent_vec, tangent_vec_norm)

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

    """
    def get_exp_map_and_jacobian_old(self, x, potential_pars):
        

        norm=((potential_pars[:,:3,:]**2).sum(axis=1, keepdim=True)).sqrt()


        zero_mask=norm==0.0
        assert(zero_mask.sum()==0)

        
        normalized_mu=potential_pars[:,:3,:]/norm

        fake_norm=self.mu_norm_function(norm)
        
        if(normalized_mu.shape[0]==1):
            normalized_mu=normalized_mu.repeat(x.shape[0], 1,1)

        vs_jac=self.all_vs_jacobians(x, normalized_mu)#.detach()
        vs=self.all_vs(x,normalized_mu)#.detach()

       
        #print("exm map type ", self.exp_map_type)
        if(self.exp_map_type=="exponential"):
           
    
            log_weights=potential_pars[:,3:4,:]-torch.logsumexp(potential_pars[:,3:4,:],dim=2,keepdims=True)+fake_norm.log()
            weights=log_weights.exp()

            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            scaling_beta=potential_pars[:,4:5,:].exp()

            ## grad PHI
            pure_grad_vec=weights*normalized_mu*torch.exp(scaling_beta*(x_times_mu-1.0))

            
            outer_jac=torch.einsum("aib, ajb -> aijb", pure_grad_vec, normalized_mu*scaling_beta)

            res1=torch.einsum("aijb, ajb -> aib", outer_jac, vs)
            res2=torch.einsum("ajib, ajb -> aib", vs_jac, pure_grad_vec)

            ## projections

            jacobian_projections=res1+res2
            projections=(vs*pure_grad_vec).sum(axis=1, keepdims=True)

            ####### tangent and sebsequent sum of all tangent vecs in joint vec

            tangent_vec_jacs=vs_jac*projections.unsqueeze(1)+torch.einsum("aib, ajb -> aijb", vs, jacobian_projections)

            joint_vec_jac=tangent_vec_jacs.sum(axis=3)

            ## v vec tangential to sphere (the velocity vec!)
            joint_vec=(projections*vs).sum(axis=2)
          
            ####### len of the the velocity vec (grad phi norm) -> the velocity to flow around sphere
            grad_phi_norm=(joint_vec**2).sum(axis=1, keepdims=True).sqrt()


            jac_grad_phi=(1.0/grad_phi_norm)*torch.einsum("ai , aij -> aj", joint_vec, joint_vec_jac)

            ################ final normalized vec

            final_vec=joint_vec/grad_phi_norm

            outer_product=torch.einsum("ai, aj -> aij", joint_vec , jac_grad_phi)

            final_vec_jac=joint_vec_jac/grad_phi_norm.unsqueeze(2) - outer_product/((grad_phi_norm.unsqueeze(2))**2)

            ### exp map result jacobian

            # d/dx (x * cos(phi) = diag(cos(phi))+ x X grad_phi *(-sin(phi))
            outer=torch.einsum("ai, aj -> aij", x, (-jac_grad_phi*torch.sin(grad_phi_norm)))

            first=torch.diag_embed(torch.cos(grad_phi_norm).repeat(1, 3))+outer

            # d/dx (x * cos(phi) = diag(cos(phi))+ x X grad_phi *(-sin(phi))
            second=final_vec_jac*torch.sin(grad_phi_norm.unsqueeze(2))+torch.einsum("ai, aj -> aij", final_vec, (jac_grad_phi*torch.cos(grad_phi_norm)))

            ## total jacobian in embedding (3-dim) space
            total_exp_map_jac=first+second

           
            total_exp_map=x*torch.cos(grad_phi_norm)+final_vec*torch.sin(grad_phi_norm)

            ## get a basis within the tangent plane to project onto

         
            second_tangent_vec=torch.cross(x,final_vec, dim=1)

            tangent_basis=torch.cat([final_vec.unsqueeze(2), second_tangent_vec.unsqueeze(2)], dim=2)

            #####

            half_projected_jacobian=torch.bmm(total_exp_map_jac, tangent_basis)
            
            projected_jacobian=torch.bmm(half_projected_jacobian.permute(0,2,1), half_projected_jacobian)

        elif(self.exp_map_type=="linear" or self.exp_map_type=="quadratic" or self.exp_map_type=="polynomial"):

            # sum(weights)=fake_norm [eps,1.0]
            #log_weights=potential_pars[:,3:4,:]-torch.logsumexp(potential_pars[:,3:4,:],dim=2,keepdims=True)+fake_norm.log()

            
            #pure_grad_vec=(log_weights.exp()*normalized_mu).sum(axis=-1)
            #len_of_grad_vec=(pure_grad_vec**2).sum(axis=-1, keepdims=True).sqrt()

            #total_jacobian=torch.diag_embed(len_of_grad_vec.repeat(1,3))


            total_exp_map, _, full_jacobian=self.get_explicit_formula_result(x, potential_pars)
            projected_jacobian=full_jacobian
            total_exp_map_jac=full_jacobian
            

        ## return exponential map, the Jacobian^2 projected onto the tangent plane (3X3 Jacobian -> 2X3 Jacobian, -> squareding 2X3 X3X2=2X2), and the full Jacobian^2 (3x3 Jacobian)
        return total_exp_map, projected_jacobian, total_exp_map_jac
    """
  

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x,log_det]=inputs
        
       
        potential_pars=self.potential_pars.to(x)
        if(extra_inputs is not None):
            potential_pars=potential_pars+extra_inputs.reshape(x.shape[0], self.potential_pars.shape[1], self.potential_pars.shape[2])
      

        if(self.always_parametrize_in_embedding_space==False):
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)
           
        if(self.natural_direction):


            result=inverse_bisection_n_newton_sphere(self.get_exp_map_and_jacobian, self.basic_logarithmic_map, self.basic_exponential_map, x, potential_pars )

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
        
        if(self.always_parametrize_in_embedding_space==False):
            # s2 flow is defined in embedding space,
            #x, log_det=self.eucl_to_spherical_embedding(x, log_det)

            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        potential_pars=self.potential_pars.to(x)

        if(extra_inputs is not None):
            potential_pars=potential_pars+extra_inputs.reshape(x.shape[0], self.potential_pars.shape[1], self.potential_pars.shape[2])

        

        if(self.natural_direction):
            result, jac_squared, _,_=self.get_exp_map_and_jacobian(x, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)

            log_det=log_det+0.5*slog_det

        else:
            result=inverse_bisection_n_newton_sphere(self.get_exp_map_and_jacobian,self.basic_logarithmic_map,  self.basic_exponential_map, x, potential_pars )

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

        potential_pars=self.potential_pars
        if(extra_inputs is not None):
            potential_pars=potential_pars+extra_inputs

        param_dict[extra_prefix+"potential_pars"]=potential_pars.data