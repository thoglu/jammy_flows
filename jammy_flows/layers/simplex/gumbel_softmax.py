import torch
from torch import nn
import collections
import numpy

from . import simplex_base

import torch.distributions as tdist

normal_dist=tdist.Normal(0, 1)

class gumbel_softmax(simplex_base.simplex_base):
    def __init__(self, 
                 dimension, 
                 use_permanent_parameters=False,
                 always_parametrize_in_embedding_space=0,
                 project_from_gauss_to_simplex=0):
        
        super().__init__(dimension=dimension, use_permanent_parameters=use_permanent_parameters, always_parametrize_in_embedding_space=always_parametrize_in_embedding_space ,project_from_gauss_to_simplex=project_from_gauss_to_simplex)


        if(use_permanent_parameters):
            self.log_tau = nn.Parameter(torch.randn(1).type(torch.double).unsqueeze(0))
            self.log_probs=nn.Parameter(torch.randn(self.dimension+1).type(torch.double).unsqueeze(0))

        self.total_param_num+=dimension+2

        self.inverse_function_type="inormal_partly_precise"
        self.pade_approximation_bound=0.5e-7

        ## constant used for pade approximation of inverse gaussian CDF
        self.pade_const_a=0.147

    def gumbel_log_pdf_quantities(self, x, calculate_pdf=True):


        log_pdf=None
        if(calculate_pdf):

            log_pdf = -x-torch.exp(-x)


        log_cdf=-torch.exp(-x)

        # sf = 1-exp(-exp(-x))
        ## for x >> 10 or so, sf = 1- (1-exp(-x)) = exp(-x)
        ## -> log(sf) = -x
        
        #large_x_mask=torch.exp(-(torch.exp(-x)))>0.99999
        large_x_mask=x>5
        safe_sf=-x

      

        ## all x values that are not that large (~large_x_mask) must be calculated exactly
        log_sf=torch.masked_scatter(input=safe_sf, mask=~large_x_mask, source=torch.log(1.0-(torch.exp(-torch.exp(-x[~large_x_mask]))) ) )
        

        return log_cdf, log_sf, log_pdf


    def sigmoid_inv_error_pass_given_cdf_sf(self, log_cdf_l, log_sf_l):

       
        if(self.inverse_function_type=="isigmoid"):

            ## super easy inverse function which can be written in terms of log_cdf and log_sf, which makes it numerically stable!

         
            return -log_sf_l+log_cdf_l


        else:

            cdf_l=torch.exp(log_cdf_l)

            if("partly" in self.inverse_function_type):

                cdf_mask = ((cdf_l > self.pade_approximation_bound) & (cdf_l < 1 - (self.pade_approximation_bound))).double()
                ## intermediate CDF values
                cdf_l_good = cdf_l * cdf_mask + 0.5 * (1. - cdf_mask)
                return_val = normal_dist.icdf(cdf_l_good)

                if(self.inverse_function_type=="inormal_partly_crude"):
                     ## crude approximation beyond limits
                    total_factor=torch.sqrt(-2.0* (log_sf_l+log_cdf_l))-0.4717
                   

                elif(self.inverse_function_type=="inormal_partly_precise"):

                    
                    a=self.pade_const_a
                    c=2.0/(numpy.pi*a)
                    ln_fac=log_cdf_l+log_sf_l+numpy.log(4.0)

                    combined=c+ln_fac/2.0
                    
                    ## make sure argument is positive for square root (numerical imprecision can rarely lead to slightly negative values, even though that should not happen)
                    pos_entry=2.0*(torch.sqrt((combined)**2-ln_fac/a)-(combined))
                    pos_entry[pos_entry<=0]=0.0

                    #mask_neg=(cdf_l<=0.5).double()

                    total_factor=torch.sqrt(pos_entry)

                    ## flip signs
                    #total_factor=(-1.0*total_factor)*mask_neg+(1.0-mask_neg)*total_factor

                   
                ## very HIGH CDF values
                cdf_mask_right = (cdf_l >= 1. - (self.pade_approximation_bound)).double()
                cdf_l_bad_right_log = (total_factor) * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
                return_val += (cdf_l_bad_right_log)*cdf_mask_right

                ## very LOW CDF values
                cdf_mask_left = (cdf_l <= self.pade_approximation_bound).double()
                cdf_l_bad_left_log = (total_factor) * cdf_mask_left + (-1.) * (1. - cdf_mask_left)
                return_val += (-1.0*cdf_l_bad_left_log)*cdf_mask_left

              
              
                return return_val

            else:
               
                ## full pade approximation of erfinv
                a=self.pade_const_a
                c=2.0/(numpy.pi*a)
                ln_fac=log_cdf_l+log_sf_l+numpy.log(4.0)

                combined=c+ln_fac/2.0

                ## make sure argument is positive for square root (numerical imprecision can rarely lead to slightly negative values, even though that should not happen)
                pos_entry=2.0*(torch.sqrt((combined)**2-ln_fac/a)-(combined))
                pos_entry[pos_entry<=0]=0.0

                total_factor=torch.sqrt(pos_entry)

                mask_neg=(cdf_l<=0.5).double()

                return (-1.0*total_factor)*mask_neg+(1.0-mask_neg)*total_factor

    def sigmoid_inv_error_pass_log_derivative_given_cdf_sf(self, log_cdf_l, log_sf_l, log_pdf):

        if(self.inverse_function_type=="isigmoid"):

            ## super easy inverse function which can be written in terms of log_cdf and log_sf, which makes it numerically stable!
            lse_cat=torch.cat([-log_sf_l[:,:,None],-log_cdf_l[:,:,None]], axis=-1)
            lse_sum=torch.logsumexp(lse_cat, axis=-1)

            return lse_sum+log_pdf
        else:

            cdf_l=torch.exp(log_cdf_l)
            

            if("partly" in self.inverse_function_type):

                cdf_mask = ((cdf_l > self.pade_approximation_bound) & (cdf_l < 1 - (self.pade_approximation_bound))).double()
                cdf_l_good = cdf_l * cdf_mask + 0.5 *( 1.-cdf_mask)
                derivative=cdf_mask*(numpy.log((numpy.sqrt(2*numpy.pi)))+torch.erfinv(2*cdf_l_good-1.0)**2+log_pdf)

                total_factor=0
                if(self.inverse_function_type=="inormal_partly_crude"):
                     ## crude approximation beyond limits
                    ln_fac=log_cdf_l+log_sf_l
                    total_factor=-0.5*torch.log(-2.0*(ln_fac))-log_sf_l-log_cdf_l

                  
                elif(self.inverse_function_type=="inormal_partly_precise"):

                    a=self.pade_const_a
                    c=2.0/(numpy.pi*a)
                    ln_fac=log_cdf_l+log_sf_l+numpy.log(4.0)

                    F=ln_fac/2.0+c

                    F_2=torch.sqrt(F**2-ln_fac/a)
                    
                    log_numerator=torch.log((-1.0)*(F-1.0/a-F_2))
                    
                   
                    log_denominator=0.5*numpy.log(8)+0.5*(torch.log(F_2-F))+torch.log(F_2)
                
                    log_total=log_numerator-log_denominator

                    
                    total_factor=log_total-log_sf_l-log_cdf_l


                    mask_neg=(cdf_l<=0.5).double()
                    extra_plus_minus_factor=torch.log((1.0-2*cdf_l)*mask_neg+(-1.0+2*cdf_l)*(1-mask_neg))

                    total_factor=total_factor+extra_plus_minus_factor

                    ######
                    
                    bad_deriv_mask=( (cdf_l>0.49999) & (cdf_l < 0.50001))
                    
                    total_factor=total_factor.masked_fill(bad_deriv_mask, numpy.log(2.506628))

                        
                cdf_mask_right = (cdf_l >= 1. - (self.pade_approximation_bound)).double()
                cdf_l_bad_right_log = total_factor * cdf_mask_right  -0.5*(1.0-cdf_mask_right)
                derivative += cdf_mask_right*(cdf_l_bad_right_log+log_pdf)
                # 3) Step3: invert BAD small CDF
                cdf_mask_left = (cdf_l <= self.pade_approximation_bound).double()
                cdf_l_bad_left_log = total_factor * cdf_mask_left  -0.5*(1.0-cdf_mask_left)
                derivative+=cdf_mask_left*(cdf_l_bad_left_log+log_pdf)

                return derivative

            else:

        
                a=self.pade_const_a
                c=2.0/(numpy.pi*a)
                ln_fac=log_cdf_l+log_sf_l+numpy.log(4.0)

                F=ln_fac/2.0+c

                # at cdf values of 0.5 the derivative is computationally unstable, avoid this region, and fix derivative to ~ 2.506628 which is the approximate numerical value
                ## Check e.g. wolfram alpha:
                ## https://www.wolframalpha.com/input/?i=derivative+of+sqrt%28+2*+%28+++sqrt%28++++%282%2F%280.147*pi%29+%2B+ln%284*x-4*x**2%29+%2F2.0+%29**2-+ln%284*x-4*x**2%29%2F0.147++%29+-++%282%2F%280.147*pi%29+%2B+ln%284*x-4*x**2%29+%2F2.0+%29++++++%29+%29+++++at+x%3D0.4995+++++++
                full_deriv_mask=( (cdf_l<0.49999) | (cdf_l > 0.50001))

                assert( (-ln_fac[full_deriv_mask].detach()<=0).sum()==0)

                return_derivs=torch.ones_like(log_cdf_l)*numpy.log(2.506628)+log_pdf

                F_2=torch.sqrt(F**2-ln_fac/a)
                
                log_numerator=torch.log((-1.0)*(F-1.0/a-F_2))

                
                log_denominator=0.5*numpy.log(8)+0.5*(torch.log(F_2-F))+torch.log(F_2)

                log_total=log_numerator-log_denominator
                
                mask_neg=(cdf_l<=0.5).double()
                extra_plus_minus_factor=(1.0-2*cdf_l)*mask_neg+(-1.0+2*cdf_l)*(1-mask_neg)
              
                return_derivs[full_deriv_mask]=(log_total-log_cdf_l-log_sf_l+log_pdf+torch.log(extra_plus_minus_factor))[full_deriv_mask]


                return return_derivs

    def inv_flow_mapping(self, inputs, extra_inputs=None):
        """
        From target to base. We only transform the d-dimensional simplex coordinates, not the last d+1 th embedding cooridnate. It is used
        however in the logdet calculation to correctly calculate the PDF on the simplex manifold.
        """
        assert(inputs[0].shape[1]==self.dimension), "Require d-dimensional input for d simplex in embedded in d+1 dimension!"
        [x, log_det]=inputs

        print("inputt x")
        print(x)

        if(extra_inputs is not None):
            
            extra_input_counter=0

            log_tau=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+1], [x.shape[0] , 1])
            extra_input_counter+=1

            log_probs=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension+1], [x.shape[0] , self.dimension+1])
        else:
            log_tau=self.log_tau.to(x) 
            log_probs=self.log_probs.to(x)

       
        d_plus_1_input=1.0-x.sum(axis=1, keepdims=True)

        log_d_plus_1_joined=torch.log(torch.cat([x, d_plus_1_input], dim=1))
        
        log_det=log_det+(self.dimension*log_tau-log_d_plus_1_joined.sum(axis=-1,keepdims=True)).sum(axis=-1)

      
        transformed_x=torch.exp(log_tau)*(log_d_plus_1_joined[:,:-1]-torch.log(d_plus_1_input))
            
   
        ## at this stage we arrived at the shifted gumbel distribution .. 
        ## shift to normal gumbel
        # g_i = u_i-logp_i + logp_k + g_k

        normal_gumbel=transformed_x-log_probs[:,:-1]+log_probs[:,-1:]

      
        ## now transform from gumbel back to unit gaussian
        """
        standard_normal=torch.erfinv(2*torch.exp(-torch.exp(-(normal_gumbel)))-1.0)*numpy.sqrt(2)

        print("stan normal", standard_normal)

        log_uniform=torch.log(0.5*(1.0+torch.erf(standard_normal/numpy.sqrt(2))))
        print("log uniform", log_uniform)
        log_d_g_d_z=-torch.log(-log_uniform)-log_uniform-  0.5*numpy.log(2*numpy.pi)-0.5*(standard_normal**2)
        
        ## we go the other way here so subtract
        log_det=log_det-log_d_g_d_z.sum(axis=-1)

        print("log dets am ende", log_det)

        """

        log_cdf, log_sf, log_pdf=self.gumbel_log_pdf_quantities(normal_gumbel,calculate_pdf=True)

        standard_normal=self.sigmoid_inv_error_pass_given_cdf_sf(log_cdf, log_sf)

        delta_logdet=self.sigmoid_inv_error_pass_log_derivative_given_cdf_sf(log_cdf, log_sf, log_pdf)
      
        log_det=log_det+delta_logdet.sum(axis=-1)

        return standard_normal, log_det

    def flow_mapping(self, inputs, extra_inputs=None):
        """
        From base to target
        """

        [z, log_det]=inputs

        if(extra_inputs is not None):
            
            extra_input_counter=0

            log_tau=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+1], [z.shape[0] , 1])
            extra_input_counter+=1

            log_probs=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension+1], [z.shape[0] , self.dimension+1])
        else:
            log_tau=self.log_tau.to(z) 
            log_probs=self.log_probs.to(z)

        # first from gaussian to centered gumbel

        log_uniform=torch.log(0.5*(1.0+torch.erf(z/numpy.sqrt(2))))

        normal_gumbel_coords=-torch.log(-log_uniform)

        #erf_arg=(1.0+torch.erf(z/numpy.sqrt(2)))

        log_d_g_d_z=-torch.log(-log_uniform)-log_uniform-0.5*numpy.log(2*numpy.pi)-0.5*(z**2)
        
        log_det=log_det+log_d_g_d_z.sum(axis=-1)

        ## now shift to shifted gumbel

        shifted_gumbel=normal_gumbel_coords+log_probs[:,:-1]-log_probs[:,-1:]

        args=shifted_gumbel/torch.exp(log_tau)

        print(args.shape)

        zeros=torch.zeros((args.shape[0],1),dtype=torch.float64)
        cat_res=torch.cat([zeros, args], dim=1)


        lse=torch.logsumexp( cat_res, axis=1, keepdims=True)

        ## final coordinates
        new_coords_log=args-lse
        d_plus_1_coord_log=-lse

        all_coords_log=torch.cat([new_coords_log, d_plus_1_coord_log], dim=1)

        all_coords=torch.exp(all_coords_log)

        ## 
        log_det_factor=self.dimension*log_tau-all_coords_log.sum()
        log_det=log_det-log_det_factor.sum(axis=-1)

       

        return new_coords_log.exp(), log_det


    def init_params(self, params):

        assert(len(params)==self.total_param_num)

        self._init_params(params)

    

    #############################################################################

    ## implement the following by specific euclidean child layers

    def _init_params(self, params):

        counter=0
        self.log_tau.data=torch.reshape(params[counter:counter+1], [1,1])

        counter+=1
        self.log_probs.data=torch.reshape(params[counter:counter+self.dimension+1], [1, self.dimension+1])

    def _get_desired_init_parameters(self):
        
        desired_param_vec=[]

        ## log_tau
        desired_param_vec.append(torch.ones(1,dtype=torch.float64)*0.0)

        ## log_probs
        desired_param_vec.append(torch.ones(self.dimension+1,dtype=torch.float64)*0.0)

       
        return torch.cat(desired_param_vec)

   


    