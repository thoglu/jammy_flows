import torch
from torch import nn
import numpy

from . import sphere_base
from . import moebius_1d
from ..bisection_n_newton import inverse_bisection_n_newton_sphere
from ...amortizable_mlp import AmortizableMLP
from ...extra_functions import list_from_str
from ..euclidean.gaussianization_flow import gf_block, find_init_pars_of_chained_gf_blocks
from ..euclidean.polynomial_stretch_flow import psf_block
from ..euclidean.euclidean_do_nothing import euclidean_do_nothing


import sys
import os
import copy

import torch.autograd

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

"""
An implementation of exponential map flows as suggested in https://arxiv.org/abs/2002.02428 ("Normalizing Flows on Tori and Spheres"),
which intself is based upon earlier work https://arxiv.org/abs/0906.0874 ("A Jacobian inequality for gradient maps on the sphere and its application to directional statistics").
"""


class exponential_map_s2(sphere_base.sphere_base):

    def __init__(self, dimension, euclidean_to_sphere_as_first=False, use_extra_householder=False, use_permanent_parameters=False, exp_map_type="linear", natural_direction=0, num_components=10, higher_order_cylinder_parametrization=False):

        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_extra_householder=use_extra_householder, use_permanent_parameters=use_permanent_parameters, higher_order_cylinder_parametrization=higher_order_cylinder_parametrization)
        
        if(dimension!=2):
            raise Exception("The moebius flow should be used for dimension 2!")
        
        self.num_components=num_components
        self.exp_map_type=exp_map_type
        self.natural_direction=natural_direction
        self.num_potential_pars=0

        if(self.exp_map_type=="linear"):
            self.num_potential_pars=3+1 # mu vector + weights
        elif(self.exp_map_type=="exponential"):
            self.num_potential_pars=3+2 # mu vector + weights
        else:
            raise Exception("Unknown exponential map parameterization %s. Allowed: linear/quadratic/exponential" % self.exp_map_type)

        ## potential parameters
        if(self.use_permanent_parameters):
            self.potential_pars=nn.Parameter(torch.randn(self.num_potential_pars, self.num_components).type(torch.double).unsqueeze(0))
        else:
            self.potential_pars=torch.zeros(self.num_potential_pars, self.num_components).type(torch.double).unsqueeze(0)

        ## param num = potential_pars*num_potentials 
        self.total_param_num+=self.num_potential_pars*self.num_components
    
    def all_vs(self, x, normalized_mu, in_newton=False):
        """ 
        Calculates the normalized tangent vector of the projection of "normalized mu" onto x on the surface of the sphere.
        """
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

        """ 
        make sure the tangent vector can be calculated. If x and normalized_mu are almost parallel, the calculaten can return nans. In these cases overwrite tangent
        vector with an arbitrary one.
        """
       

        #non_fin_mask=((torch.isfinite(b)==False) )

        ### Generate a vector that is orthogonal (but in principle arbitrary orientation)  
        arb_z=(-x[:,0:1]*0.5-x[:,1:2]*0.5)/x[:,2:3]

        arb=torch.cat([ torch.ones_like(x[:,0:1])*0.5, torch.ones_like(x[:,0:1])*0.5, arb_z], dim=1).unsqueeze(2).repeat(1,1,normalized_mu.shape[2])

        arb=arb/((arb**2).sum(axis=1,keepdims=True).sqrt())

        #if(non_fin_mask.sum()>0):
        #    return arb
        """
        v_1=arb[:,0:1,:]*non_fin_mask+v_1*(1.0-non_fin_mask)
        v_2=arb[:,1:2,:]*non_fin_mask+v_2*(1.0-non_fin_mask)
        v_3=arb[:,2:3,:]*non_fin_mask+v_3*(1.0-non_fin_mask)
        """
        #v_1=arb[:,0:1,:].detach()
        #v_2=arb[:,1:2,:].detach()
        #v_3=arb[:,2:3,:].detach()

        ### those values that have an issue with potential division by zero are replaced with arbitrary vecs .. doesnt make a difference really since target is basically reached already
        v_1 = torch.masked_scatter(input=v_1, mask=issue_mask, source=arb[:,0:1,:][issue_mask])
        v_2 = torch.masked_scatter(input=v_2, mask=issue_mask, source=arb[:,1:2,:][issue_mask])
        v_3 = torch.masked_scatter(input=v_3, mask=issue_mask, source=arb[:,2:3,:][issue_mask])

        """
        print("V1 / v2 / v3")
        print(v_1)
        print(v_2)
        print(v_3)
        print("---- arb")
        print(arb)
        
        arb[:,0:1,:][non_fin_mask==False]=(v_1_top[non_fin_mask==False]/v_bottom[non_fin_mask==False])
        arb[:,1:2,:][non_fin_mask==False]=(v_2_top[non_fin_mask==False]/v_bottom[non_fin_mask==False])
        arb[:,2:3,:][non_fin_mask==False]=(v_3_top[non_fin_mask==False]/v_bottom[non_fin_mask==False])
        """
       

        all_vs=torch.cat([v_1,v_2,v_3], dim=1)
            
        return all_vs

        """

        v_1=v_1_top/v_bottom
        v_2=v_2_top/v_bottom
        v_3=v_3_top/v_bottom

        if(non_fin_mask.sum()>0):
                
            v_1[non_fin_mask]=arb[:,0:1,:][non_fin_mask]
            v_2[non_fin_mask]=arb[:,1:2,:][non_fin_mask]
            v_3[non_fin_mask]=arb[:,2:3,:][non_fin_mask]

            all_vs=torch.cat([v_1,v_2,v_3], dim=1)
            
            return all_vs

        """
        

        #v_1=torch.cat([torch.masked_select(v_1, non_fin_mask), torch.masked_select(v_1_top/v_bottom, ~non_fin_mask)],dim=1)
        #v_2=torch.cat([v_2, torch.masked_select(v_2_top/v_bottom, ~non_fin_mask).unsqueeze(0).unsqueeze(0)],dim=1)
        #v_3=torch.cat([v_3, torch.masked_select(v_3_top/v_bottom, ~non_fin_mask).unsqueeze(0).unsqueeze(0)],dim=1)
        #print("after ", v_1.shape)
   
        #v_1[~non_fin_mask]=(v_1_top[~non_fin_mask]/v_bottom[~non_fin_mask])
        #v_2[~non_fin_mask]=(v_2_top[~non_fin_mask]/v_bottom[~non_fin_mask])
        #v_3[~non_fin_mask]=(v_3_top[~non_fin_mask]/v_bottom[~non_fin_mask])
        
 
        #v_2[non_fin_mask]=arb[:,1:2,:][non_fin_mask]
        #v_3[non_fin_mask]=arb[:,2:3,:][non_fin_mask]
        """
        if(non_fin_mask.sum()>0):

            norm_mu_lens=(normalized_mu**2).sum(axis=1,keepdims=True).sqrt()
        """
            
        all_vs=torch.cat([v_1,v_2,v_3], dim=1)

        return all_vs


    ## jacobian of vs x,y,z axes
    def all_vs_jacobians(self, x, normalized_mu):
        """ 
        The jacobian of the tangent vector calculated in "all_vs"
        """
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

    def basic_exponential_map(self, x, v_unit, v_norm):

        return x*torch.cos(v_norm)+v_unit*torch.sin(v_norm)

    def get_exp_map_and_jacobian(self, x, potential_pars):
        """
        Calculates the exponential map and its jacobian in one go, since Jacobian always requires to calculate the exponential map anyway.
        """

        norm=((potential_pars[:,:3,:]**2).sum(axis=1, keepdim=True)).sqrt()
        normalized_mu=potential_pars[:,:3,:]/norm
        
        if(normalized_mu.shape[0]==1):
            normalized_mu=normalized_mu.repeat(x.shape[0], 1,1)

        vs_jac=self.all_vs_jacobians(x, normalized_mu)
        vs=self.all_vs(x,normalized_mu)

        if(self.exp_map_type=="exponential"):
            log_weights=potential_pars[:,3:4,:]
            weights=torch.softmax(log_weights, dim=2)
            
            x_times_mu=(x[:,:,None]*normalized_mu).sum(axis=1, keepdims=True)

            scaling_beta=potential_pars[:,4:5,:].exp()

            
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
            joint_vec=(projections*vs).sum(axis=2)

            ####### len of the the joint vec (grad phi norm)

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

            total_exp_map_jac=first+second

            total_exp_map=x*torch.cos(grad_phi_norm)+final_vec*torch.sin(grad_phi_norm)

            ## get a basis within the tangent plane to project onto

         
            second_tangent_vec=torch.cross(x,final_vec, dim=1)

            tangent_basis=torch.cat([final_vec.unsqueeze(2), second_tangent_vec.unsqueeze(2)], dim=2)

            #####

            half_projected_jacobian=torch.bmm(total_exp_map_jac, tangent_basis)
            
            projected_jacobian=torch.bmm(half_projected_jacobian.permute(0,2,1), half_projected_jacobian)

        else:
            raise NotImplementedError()

        ## return exponential map, the Jacobian^2 projected onto the tangent plane (3X3 Jacobian -> 2X3 Jacobian), and the full Jacobian^2 (3x3 Jacobian)
        return total_exp_map, projected_jacobian, total_exp_map_jac

  

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        #if(self.higher_order_cylinder_parametrization):
        #    print("extra params MLP", extra_inputs[0,self.num_mlp_params-self.total_euclidean_pars:self.num_mlp_params])
        ## input structure: 0-num_amortization_params -> MLP  , num_amortizpation_params-end: -> moebius trafo
        [x,log_det]=inputs
        
       
        potential_pars=self.potential_pars.to(x)
        if(extra_inputs is not None):
            potential_pars=potential_pars+extra_inputs.reshape(x.shape[0], self.potential_pars.shape[1], self.potential_pars.shape[2])
        """
        def f(tt):

            return inverse_bisection_n_newton_sphere(self.get_exp_map_and_jacobian, self.all_vs, self.basic_exponential_map, tt, potential_pars[1:2] )
        """

        if(self.always_parametrize_in_embedding_space==False):
            # embedding to intrinsic
           
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)
           

        if(self.natural_direction):


            result=inverse_bisection_n_newton_sphere(self.get_exp_map_and_jacobian, self.all_vs, self.basic_exponential_map, x, potential_pars )

            _, jac_squared, _=self.get_exp_map_and_jacobian(result, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)
           
            log_det=log_det-0.5*slog_det
        else:
            result, jac_squared, _=self.get_exp_map_and_jacobian(x, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)
            
            
            log_det=log_det+0.5*slog_det

        ## sqrt of det(jacobian^T * jacobian)
        
        #res=self.eucl_to_spherical_embedding(result)

        if(self.always_parametrize_in_embedding_space==False):
            # embedding to intrinsic
            result, log_det=self.eucl_to_spherical_embedding(result, log_det)

        return result, log_det, None

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
        
        [x,log_det]=inputs
        
        if(self.always_parametrize_in_embedding_space==False):
            # s2 flow is defined in embedding space,
            #x, log_det=self.eucl_to_spherical_embedding(x, log_det)

            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        potential_pars=self.potential_pars.to(x)

        if(extra_inputs is not None):
            potential_pars=potential_pars+extra_inputs.reshape(x.shape[0], self.potential_pars.shape[1], self.potential_pars.shape[2])

        if(self.natural_direction):
            result, jac_squared, _=self.get_exp_map_and_jacobian(x, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)

            log_det=log_det+0.5*slog_det

        else:
            result=inverse_bisection_n_newton_sphere(self.get_exp_map_and_jacobian, self.all_vs, self.basic_exponential_map, x, potential_pars )

            _, jac_squared, _=self.get_exp_map_and_jacobian(result, potential_pars)
            sign, slog_det=torch.slogdet(jac_squared)
           
            log_det=log_det-0.5*slog_det

        if(self.always_parametrize_in_embedding_space==False):
            result, log_det=self.eucl_to_spherical_embedding(result, log_det)

        return result, log_det

    def _init_params(self, params):

        assert(len(params)== (self.num_potential_pars*self.num_components))

        self.potential_pars.data=params.reshape(1, self.num_potential_pars, self.num_components)


    def _get_desired_init_parameters(self):

      
        gaussian_init=torch.randn((self.num_potential_pars*self.num_components))

        return gaussian_init


    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Implemented by Euclidean sublayers.
        """

        potential_pars=self.potential_pars
        if(extra_inputs is not None):
            potential_pars=potential_pars+extra_inputs

        param_dict[extra_prefix+"potential_pars"]=potential_pars.data