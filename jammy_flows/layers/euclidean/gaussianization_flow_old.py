import torch
from torch import nn
import numpy

from .. import bisection_n_newton as bn
from .. import layer_base
from ... import extra_functions

from . import euclidean_base

import math
import torch.nn.functional as F
import torch.distributions as tdist
import scipy.linalg
from scipy.optimize import minimize
import time
normal_dist=tdist.Normal(0, 1)

import pylab

def generate_log_function_bounded_in_logspace(min_val_normal_space=1, max_val_normal_space=10, center_around_zero=False):
    
    ## min and max values are in normal space -> must be positive
    assert(min_val_normal_space > 0)

    ln_max=numpy.log(max_val_normal_space)
    ln_min=numpy.log(min_val_normal_space)

    ## this shift makes the function equivalent to a normal exponential for small values
    center_val=ln_max

    ## can also center around zero (it will be centered in exp space, not in log space)
    if(center_around_zero==False):
        center_val=0.0

    def f(x):

        res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-x+center_val).unsqueeze(-1)], dim=-1)

        first_term=ln_max-torch.logsumexp(res, dim=-1, keepdim=True)

        return torch.logsumexp( torch.cat([first_term, torch.ones_like(first_term)*ln_min], dim=-1), dim=-1)

    return f


class gf_block_old(euclidean_base.euclidean_base):
    def __init__(self, 
                 dimension, 
                 num_kde=5, 
                 num_householder_iter=-1, 
                 use_permanent_parameters=False, 
                 fit_normalization=0, 
                 inverse_function_type="inormal_partly_precise", 
                 model_offset=0, 
                 softplus_for_width=0,
                 width_smooth_saturation=1,
                 lower_bound_for_widths=0.01,
                 upper_bound_for_widths=100,
                 clamp_widths=0,
                 regulate_normalization=0,
                 add_skewness=0):
        """
        Modified version of official implementation in hhttps://github.com/chenlin9/Gaussianization_Flows (https://arxiv.org/abs/2003.01941). Fixes numerical issues with bisection inversion due to more efficient newton iterations, added offsets, and allows 
        to use reparametrization trick for VAEs due to Newton iterations.
        Parameters:
        dimension (int): dimension of the PDF
        num_kde (int): number of KDE s in the one-dimensional PDF
        num_householder_iter (int): if <=0, no householder transformation is performed. If positive, it defines the number of parameters in householder transformations.
        use_permanent_parameters (float): If permantent parameters are used (no depnendence on other input), or if input is used to define the parameters (conditional pdf).
        mapping_approximation (str): One of "partly_crude", "partly_precise", "full_pade". Partly_pade_crude is implemented in the original repository, but has numerical issues.
        It is recommended to use "partly_precise" or "full_pade".
        """
        super().__init__(dimension=dimension, use_permanent_parameters=use_permanent_parameters, model_offset=model_offset)
        self.init = False

        assert(lower_bound_for_widths>0.0)
        self.hs_min=lower_bound_for_widths

        ## defines maximum width - None -> no maximum width .. only used for exponential width function to cap high values
        self.hs_max=None

        if(upper_bound_for_widths > 0):
            self.hs_max=upper_bound_for_widths

            ### clamp at three times the logarithm to upper bound log_width .. more than enough for whole range
            self.log_width_max_to_clamp=numpy.log(self.hs_max)*3.0

        self.width_smooth_saturation=width_smooth_saturation
        if(self.width_smooth_saturation):
            assert(self.hs_max is not None), "We require a maximum saturation level for smooth saturation!"

        ## clamp at a hundreths of the smallest len allowed width hs_min (yields approximately clamp value below, as long as hs_max >> hs_min)
        self.log_width_min_to_clamp=numpy.log(0.01*self.hs_min)

        self.clamp_widths=clamp_widths

        self.inverse_function_type=inverse_function_type
      
        assert(self.inverse_function_type=="inormal_partly_crude" or self.inverse_function_type=="inormal_partly_precise" or  self.inverse_function_type=="inormal_full_pade" or  self.inverse_function_type=="isigmoid")

        ## p-value after which to switch to pade approximation
        self.pade_approximation_bound=0.5e-7

        ## constant used for pade approximation of inverse gaussian CDF
        self.pade_const_a=0.147

        if num_householder_iter == -1:
            self.householder_iter = dimension #min(dimension, 10)
        else:
            self.householder_iter = num_householder_iter
       

        self.use_householder=True
        if(self.householder_iter==0):
           
            self.use_householder=False

        self.dimension = dimension
        self.num_kde = num_kde
        
        ## initialization from Gaussianization flow paper
        bandwidth = (4. * numpy.sqrt(math.pi) / ((math.pi ** 4) * num_kde)) ** 0.2
        self.init_bandwidth=numpy.log(bandwidth)


        ## ROTATIONS
        self.num_householder_params=0

        if self.use_householder:
            if(use_permanent_parameters):
                self.vs = nn.Parameter(
                    torch.randn(self.householder_iter, dimension).type(torch.double).unsqueeze(0)
                )
            else:
                self.vs = torch.zeros(self.householder_iter, dimension).type(torch.double).unsqueeze(0) 

            self.num_householder_params=self.householder_iter*self.dimension


        ## means

        if use_permanent_parameters:
            self.datapoints = nn.Parameter(torch.randn(self.num_kde, self.dimension).type(torch.double).unsqueeze(0))
        else:
            self.datapoints = torch.zeros(self.num_kde, self.dimension).type(torch.double).unsqueeze(0)#.to(device)

        self.num_params_datapoints=self.num_kde*self.dimension

        
        
        ## handling of normalization

        self.fit_normalization=fit_normalization
        self.regulate_normalization=regulate_normalization

        self.normalization_regulator = None
        if(self.fit_normalization):
            if(self.regulate_normalization):
                ## bound normalization into a range that spans roughly ~ 100 . .we dont want huge discrepancies in normalization
                ## this serves as a stabilizer during training compared to no free-floating normalization, but at the same time
                ## avoids near zero normalizations which can also lead to unwanted side effects

                self.normalization_regulator=generate_log_function_bounded_in_logspace(min_val_normal_space=1, max_val_normal_space=100)


        ###### handling of widths

        self.softplus_for_width=softplus_for_width

        if(self.softplus_for_width):
            ## softplus
            if(clamp_widths):
                upper_clamp=None
                if(self.hs_max is not None):
                    # clamp upper bound with exact hs_max value
                    upper_clamp=numpy.log(self.hs_max)
                self.exp_like_function_linear=lambda x: torch.nn.functional.softplus(torch.clamp(x, min=self.log_width_min_to_clamp, max=upper_clamp))+self.hs_min
            else:
                self.exp_like_function_linear=lambda x: torch.nn.functional.softplus(x)+self.hs_min
            
            self.use_linear_exp_like_function=True

        else:
            ## exponential-type width relation
            if(self.width_smooth_saturation == 0):
                ## normal, infinetly growing exponential
                if(self.clamp_widths):
                    # clamp upper bound with exact hs_max value
                    upper_clamp=None
                    if(self.hs_max is not None):
                        upper_clamp=numpy.log(self.hs_max)
                    self.exp_like_function_linear=lambda x: torch.exp(torch.clamp(x, min=self.log_width_min_to_clamp, max=upper_clamp))+self.hs_min
                else:
                    self.exp_like_function_linear=lambda x: torch.exp(x)+self.hs_min

                self.use_linear_exp_like_function=True
            else:
                ## exponential function at beginning but flattens out at hs_max -> no infinite growth
                ## numerically stable via logsumexp .. clamping should not be necessary, but can be done to damp down large gradients
                ## in weird regions of parameter space
                
                ln_hs_max=numpy.log(self.hs_max)
                ln_hs_min=numpy.log(self.hs_min)

                if(self.clamp_widths):

                    def exp_like_fn(x):

                        res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-torch.clamp(x, min=self.log_width_min_to_clamp, max=self.log_width_max_to_clamp)+ln_hs_max).unsqueeze(-1)], dim=-1)

                        first_term=ln_hs_max-torch.logsumexp(res, dim=-1, keepdim=True)

                        return torch.logsumexp( torch.cat([first_term, torch.ones_like(first_term)*ln_hs_min], dim=-1), dim=-1)

                else:
                    def exp_like_fn(x):

                        res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-x+ln_hs_max).unsqueeze(-1)], dim=-1)

                        first_term=ln_hs_max-torch.logsumexp(res, dim=-1, keepdim=True)

                        return torch.logsumexp( torch.cat([first_term, torch.ones_like(first_term)*ln_hs_min], dim=-1), dim=-1)

                self.exp_like_function_log=exp_like_fn

                self.use_linear_exp_like_function=False

        ## width parameters
        if(use_permanent_parameters):
            self.log_hs = nn.Parameter(
                torch.ones(num_kde, dimension).type(torch.double).unsqueeze(0) * numpy.log(bandwidth)
            )
        else:
            self.log_hs = torch.zeros(num_kde, dimension).type(torch.double).unsqueeze(0)
            

        # normalization parameters
        self.log_kde_weights = torch.zeros(self.num_kde, self.dimension).type(torch.double).unsqueeze(0)#.to(device)
        if(fit_normalization):

            if(use_permanent_parameters):
                self.log_kde_weights = nn.Parameter(torch.randn(num_kde, self.dimension).type(torch.double).unsqueeze(0))


            self.total_param_num+=self.num_params_datapoints

        ### covers means+widths+householder
        self.total_param_num+=self.num_params_datapoints*2+self.num_householder_params
        #####################


        #### skewness
        self.add_skewness=add_skewness

        ## shape to B X KDE index dim X dimension
        self.skew_exponents=torch.DoubleTensor([1.0]).view(1,1,1)
        self.skew_signs=torch.DoubleTensor([1.0])

        if(self.add_skewness):

            self.skew_signs=torch.ones( (1,self.num_kde,1)).type(torch.double)

            num_negative=int(float(self.num_kde)/2.0)

            ## half of the KDEs use a flipped prescription
            self.skew_signs[:,num_negative:,:]=-1.0

            if(use_permanent_parameters):
                self.skew_exponents = nn.Parameter(
                    torch.randn(self.num_kde, dimension).type(torch.double).unsqueeze(0)
                )

            else:
                self.skew_exponents = torch.zeros(self.num_kde, dimension).type(torch.double).unsqueeze(0) 

            ## with 0.1 and 9.0 the function maps 0 to 0 approximately -> 0 -> 1 in normal exponent space, the starting point we want
            self.exponent_regulator=generate_log_function_bounded_in_logspace(min_val_normal_space=0.1, max_val_normal_space=9.0, center_around_zero=True)
            self.total_param_num+=self.num_params_datapoints

    def logistic_kernel_log_cdf(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):
       
        if(self.use_linear_exp_like_function):
            hs = self.exp_like_function_linear(log_widths)
        else:
            log_hs = self.exp_like_function_log(log_widths)
            hs=torch.exp(log_hs)

        ## data comes in ## Batchsize X dimension
        ## unsqueeze at 1 unsqueezes the dimension along the different KDEs for the data x
        ## -> Batchsize X KDE index per dim X dimension
        x_unsqueezed=x.unsqueeze(1)
        
        ## we sum over the KDE index dimension (1)

        if(self.add_skewness):

            log_cdfs=torch.zeros( (x.shape[0], datapoints.shape[1], datapoints.shape[2]), dtype=torch.double)

            pos_mask=skew_signs[0,:,0]>0
      
            log_cdfs.masked_scatter_(pos_mask[None,:,None], - skew_exponents[:,pos_mask,:]*F.softplus(-(x_unsqueezed - datapoints[:,pos_mask,:]) / hs[:,pos_mask,:]))
                
            log_cdfs.masked_scatter_(~pos_mask[None,:,None], extra_functions.log_one_plus_exp_x_to_a_minus_1((x_unsqueezed - datapoints[:,~pos_mask,:]) / hs[:,~pos_mask,:], skew_exponents[:,~pos_mask,:]))

            log_cdfs = log_cdfs+log_norms - torch.logsumexp(log_norms, dim=1, keepdim=True)

        else:
            log_cdfs = - F.softplus(-(x_unsqueezed - datapoints) / hs) + \
                       log_norms - torch.logsumexp(log_norms, dim=1, keepdim=True)

        log_cdf = torch.logsumexp(log_cdfs, dim=1)

      
        return log_cdf

    def logistic_kernel_log_sf(self, x, datapoints, log_widths,log_norms, skew_exponents, skew_signs):
       
        if(self.use_linear_exp_like_function):
            hs = self.exp_like_function_linear(log_widths)
        else:
            log_hs = self.exp_like_function_log(log_widths)
            hs=torch.exp(log_hs)
    
        x_unsqueezed=x.unsqueeze(1)

        if(self.add_skewness):

            ## essentially the inverse procedure from the CDF ...
            log_sfs=torch.zeros( (x.shape[0], datapoints.shape[1], datapoints.shape[2]), dtype=torch.double)

            pos_mask=skew_signs[0,:,0]>0
            log_sfs.masked_scatter_(pos_mask[None,:,None], extra_functions.log_one_plus_exp_x_to_a_minus_1(-(x_unsqueezed - datapoints[:,pos_mask,:]) / hs[:,pos_mask,:], skew_exponents[:,pos_mask,:]))


            log_sfs.masked_scatter_(~pos_mask[None,:,None], - skew_exponents[:,~pos_mask,:]*F.softplus((x_unsqueezed - datapoints[:,~pos_mask,:]) / hs[:,~pos_mask,:]))
            
           
            log_sfs = log_sfs+log_norms - torch.logsumexp(log_norms, dim=1, keepdim=True)

        else:
            log_sfs = -(x_unsqueezed - datapoints) / hs - \
                  F.softplus(-(x_unsqueezed- datapoints) / hs) + \
                  log_norms - torch.logsumexp(log_norms, dim=1, keepdim=True)

        

    
        log_sf = torch.logsumexp(log_sfs, dim=1)

     
        return log_sf

    def logistic_kernel_log_pdf(self, x, datapoints, log_widths,log_norms, skew_exponents, skew_signs):
        
       
        if(self.use_linear_exp_like_function):
            hs = self.exp_like_function_linear(log_widths)
            log_hs=torch.log(hs)
        else:
            log_hs = self.exp_like_function_log(log_widths)
            hs=torch.exp(log_hs)

      
        x_unsqueezed=x.unsqueeze(1)

        log_pdfs = -skew_signs*(x_unsqueezed - datapoints) / hs - log_hs + torch.log(skew_exponents) - \
                    (skew_exponents+1.0) * F.softplus(-skew_signs*(x_unsqueezed - datapoints) / hs) + \
                   log_norms - torch.logsumexp(log_norms, dim=1, keepdim=True)

        
        log_pdf = torch.logsumexp(log_pdfs, dim=1)
    
      
        return log_pdf

    def logistic_kernel_pdf(self, x, datapoints, log_widths,log_norms, skew_exponents, skew_signs):
        
        log_pdf=self.logistic_kernel_log_pdf(x,datapoints,log_widths,log_norms, skew_exponents, skew_signs)
        
        pdf = torch.exp(log_pdf)


        return pdf

    def logistic_kernel_cdf(self, x, datapoints, log_widths,log_norms, skew_exponents,skew_signs):
        # Using bandwidth formula

        log_cdf=self.logistic_kernel_log_cdf(x,datapoints,log_widths,log_norms, skew_exponents, skew_signs)
        
        cdf = torch.exp(log_cdf)

        return cdf


    def compute_householder_matrix(self, vs, device=torch.device("cpu")):

        Q = torch.eye(self.dimension, device=device).type(torch.double).unsqueeze(0).repeat(vs.shape[0], 1,1)
       
        for i in range(self.householder_iter):
        
            v = vs[:,i].reshape(-1,self.dimension, 1).to(device)
            
            v = v / v.norm(dim=1).unsqueeze(-1)

            Qi = torch.eye(self.dimension, device=device).type(torch.double).unsqueeze(0) - 2 * torch.bmm(v, v.permute(0, 2, 1))

            Q = torch.bmm(Q, Qi)

        return Q

   
    def sigmoid_inv_error_pass(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):

       
        log_cdf_l = self.logistic_kernel_log_cdf(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs)  # log(CDF)
        log_sf_l = self.logistic_kernel_log_sf(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs)  # log(1-CDF)
        
       
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

    def sigmoid_inv_error_pass_derivative(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):
        
        """
        #datapoints = self.datapoints

        #cdf_l = self.logistic_kernel_cdf(x, datapoints, log_widths)
        log_cdf_l = self.logistic_kernel_log_cdf(x, datapoints,log_widths,log_norms)  # log(CDF)
        

        log_sf_l = self.logistic_kernel_log_sf(x, datapoints,log_widths,log_norms)  # log(1-CDF)
        log_pdf = self.logistic_kernel_log_pdf(x, datapoints,log_widths,log_norms)  
        
        if(self.inverse_function_type=="isigmoid"):

            ## super easy inverse function which can be written in terms of log_cdf and log_sf, which makes it numerically stable!
            lse_cat=torch.cat([-log_sf_l[:,:,None],-log_cdf_l[:,:,None]], axis=-1)
            lse_sum=torch.logsumexp(lse_cat, axis=-1)

            return torch.exp(lse_sum+log_pdf)

        else:

            cdf_l=torch.exp(log_cdf_l)

            if("partly" in self.inverse_function_type):

                cdf_mask = ((cdf_l > self.pade_approximation_bound) & (cdf_l < 1 - (self.pade_approximation_bound))).double()
                cdf_l_good = cdf_l * cdf_mask + 0.5 *( 1.-cdf_mask)
                derivative=cdf_mask*(numpy.sqrt(2*numpy.pi)*torch.exp(torch.erfinv(2*cdf_l_good-1.0)**2+log_pdf))

                total_factor=0

                if(self.inverse_function_type=="inormal_partly_crude"):
                     ## crude approximation beyond limits
                    ln_fac=log_cdf_l+log_sf_l
                    total_factor=-0.5*torch.log(-2.0*(ln_fac))

                elif(self.inverse_function_type=="inormal_partly_precise"):

                    a=self.pade_const_a
                    c=2.0/(numpy.pi*a)
                    ln_fac=log_cdf_l+log_sf_l+numpy.log(4.0)

                    F=ln_fac/2.0+c

                    F_2=torch.sqrt(F**2-ln_fac/a)
                    
                    log_numerator=torch.log((-1.0)*(F-1.0/a-F_2))
                    
                   
                    log_denominator=0.5*numpy.log(8)+0.5*(torch.log(F_2-F))+torch.log(F_2)
                
                    log_total=log_numerator-log_denominator

                    mask_neg=(cdf_l<=0.5).double()
                    extra_plus_minus_factor=(1.0-2*cdf_l)*mask_neg+(-1.0+2*cdf_l)*(1-mask_neg)

                    total_factor=log_total
                    bad_deriv_mask=( (cdf_l>0.49999) & (cdf_l < 0.50001))

                    ## pseudoe value .. should actually never happen, just evades NANs below

                    total_factor[bad_deriv_mask]=10000000


                cdf_mask_right = (cdf_l >= 1. - (self.pade_approximation_bound)).double()
                cdf_l_bad_right_log = total_factor * cdf_mask_right  -0.5*(1.0-cdf_mask_right)

                derivative += cdf_mask_right*torch.exp(cdf_l_bad_right_log-log_sf_l+log_pdf)
                # 3) Step3: invert BAD small CDF
                #if(x.shape[0]>48000):
                #    print("DERIV2", derivative[33457])
                cdf_mask_left = (cdf_l <= self.pade_approximation_bound).double()
                cdf_l_bad_left_log = total_factor * cdf_mask_left  -0.5*(1.0-cdf_mask_left)
                derivative+=cdf_mask_left*torch.exp(cdf_l_bad_left_log-log_cdf_l+log_pdf)

              
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

                #assert( (-ln_fac.detach()<=0).sum()==0)

                assert( (-ln_fac[full_deriv_mask].detach()<=0).sum()==0)

                return_derivs=torch.ones_like(x)*2.506628*torch.exp(log_pdf)

                F_2=torch.sqrt(F**2-ln_fac/a)

                log_numerator=torch.log((-1.0)*(F-1.0/a-F_2))
               
                log_denominator=0.5*numpy.log(8)+0.5*(torch.log(F_2-F))+torch.log(F_2)
            
                log_total=log_numerator-log_denominator

                mask_neg=(cdf_l<=0.5).double()
                extra_plus_minus_factor=(1.0-2*cdf_l)*mask_neg+(-1.0+2*cdf_l)*(1-mask_neg)

                return_derivs[full_deriv_mask]=(torch.exp(log_total-log_cdf_l-log_sf_l+log_pdf)*extra_plus_minus_factor)[full_deriv_mask]

                return return_derivs

        """

        return torch.exp(self.sigmoid_inv_error_pass_log_derivative(x,datapoints, log_widths,log_norms, skew_exponents, skew_signs))

    def sigmoid_inv_error_pass_log_derivative(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):

        #datapoints = self.datapoints

        #cdf_l = self.logistic_kernel_cdf(x, datapoints, log_widths)
        log_cdf_l = self.logistic_kernel_log_cdf(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs)  # log(CDF)
        
        log_sf_l = self.logistic_kernel_log_sf(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs)  # log(1-CDF)
        log_pdf = self.logistic_kernel_log_pdf(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs)  
        
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


                    ##########

                    
                    """
                    mask_neg=(cdf_l<0.5)
                    total_factor=total_factor.masked_scatter(mask_neg, total_factor[mask_neg]+torch.log((1.0-2*cdf_l[mask_neg])))

                    mask_pos=(cdf_l>0.5)
                    total_factor=total_factor.masked_scatter(mask_pos, total_factor[mask_pos]+torch.log((-1.0+2*cdf_l[mask_pos])))
                    """

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

                return_derivs=torch.ones_like(x)*numpy.log(2.506628)+log_pdf

                F_2=torch.sqrt(F**2-ln_fac/a)
                
                log_numerator=torch.log((-1.0)*(F-1.0/a-F_2))

                
                log_denominator=0.5*numpy.log(8)+0.5*(torch.log(F_2-F))+torch.log(F_2)

                log_total=log_numerator-log_denominator
                
                mask_neg=(cdf_l<=0.5).double()
                extra_plus_minus_factor=(1.0-2*cdf_l)*mask_neg+(-1.0+2*cdf_l)*(1-mask_neg)
              
                return_derivs[full_deriv_mask]=(log_total-log_cdf_l-log_sf_l+log_pdf+torch.log(extra_plus_minus_factor))[full_deriv_mask]


                #######


                return return_derivs


    def _flow_mapping(self, inputs, extra_inputs=None, verbose=False, lower=-1e5, upper=1e5): 
        
        [z, log_det]=inputs


        device=z.device

        extra_input_counter=0

        if self.use_householder:
            this_vs=self.vs.to(z)

          
            if(extra_inputs is not None):

              
                this_vs=this_vs+torch.reshape(extra_inputs[:,:self.num_householder_params], [z.shape[0], self.vs.shape[1], self.vs.shape[2]])

                
                extra_input_counter+=self.num_householder_params
        
            rotation_matrix = self.compute_householder_matrix(this_vs, device=z.device)

            if(rotation_matrix.shape[0]<z.shape[0]):
                if(rotation_matrix.shape[0]==1):
                    rotation_matrix=rotation_matrix.repeat(z.shape[0], 1,1)
                else:
                    raise Exception("something went wrong with first dim of rot matrix!")

            
        this_datapoints=self.datapoints.to(z)
        this_hs=self.log_hs.to(z)
        this_log_norms=self.log_kde_weights.to(z)
        this_skew_signs=self.skew_signs.to(z)
        
        if(extra_inputs is not None):
            
            this_datapoints=this_datapoints+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [z.shape[0] , self.datapoints.shape[1],  self.datapoints.shape[2]])
            extra_input_counter+=self.num_params_datapoints

            this_hs=this_hs+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [z.shape[0], self.datapoints.shape[1], self.datapoints.shape[2]])
            extra_input_counter+=self.num_params_datapoints

            if(self.fit_normalization):
                this_log_norms=this_log_norms+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [z.shape[0], self.datapoints.shape[1], self.datapoints.shape[2]])
                extra_input_counter+=self.num_params_datapoints

        if(self.fit_normalization and self.regulate_normalization):
            ## regulate log-normalizations

            this_log_norms=self.normalization_regulator(this_log_norms)


        ## skewnewss
        this_skew_exponent=self.skew_exponents.to(z)

        if(self.add_skewness):
            if(extra_inputs is not None):
                this_skew_exponent=this_skew_exponent+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [z.shape[0], self.datapoints.shape[1], self.datapoints.shape[2]])
            
            this_skew_exponent=torch.exp(self.exponent_regulator(this_skew_exponent))
        ## if we fit normalization ... 

        res=bn.inverse_bisection_n_newton(self.sigmoid_inv_error_pass, self.sigmoid_inv_error_pass_derivative, z, this_datapoints, this_hs, this_log_norms, this_skew_exponent, this_skew_signs, min_boundary=lower, max_boundary=upper, num_bisection_iter=25, num_newton_iter=20)
        
     
        log_deriv=self.sigmoid_inv_error_pass_log_derivative(res, this_datapoints, this_hs,this_log_norms, this_skew_exponent, this_skew_signs)

        log_det=log_det-log_deriv.sum(axis=-1)

        if self.use_householder:
         
            res = torch.bmm(rotation_matrix, res.unsqueeze(-1)).squeeze(-1)
       
        return res, log_det

    def _get_desired_init_parameters(self):

        ## householder params / means of kdes / log_widths of kdes / normalizations (if fit normalization)

        desired_param_vec=[]

        ## householder
        if(self.num_householder_params > 0):
            desired_param_vec.append(torch.randn(self.householder_iter*self.dimension))

        ## means
        desired_param_vec.append(torch.randn(self.num_kde*self.dimension))

        ## widths
        desired_param_vec.append(torch.ones(self.num_kde*self.dimension)*self.init_bandwidth)

        ## normalization
        if(self.fit_normalization):
            desired_param_vec.append(torch.ones(self.num_kde*self.dimension))

        ## normalization
        if(self.add_skewness):
            desired_param_vec.append(torch.zeros(self.num_kde*self.dimension))

        return torch.cat(desired_param_vec)

    def _init_params(self, params):

        counter=0
        if self.use_householder:
           
              
            self.vs.data=torch.reshape(params[:self.num_householder_params], [1, self.householder_iter,self.dimension])

            counter+=self.num_householder_params

        
        self.datapoints.data=torch.reshape(params[counter:counter+self.num_params_datapoints], [1,self.num_kde, self.dimension])
        counter+=self.num_params_datapoints

        self.log_hs.data=torch.reshape(params[counter:counter+self.num_params_datapoints], [1,self.num_kde, self.dimension])
        counter+=self.num_params_datapoints

        if(self.fit_normalization):
            self.log_kde_weights.data=torch.reshape(params[counter:counter+self.num_params_datapoints], [1,self.num_kde, self.dimension])
            counter+=self.num_params_datapoints

        if(self.add_skewness):
            self.skew_exponents.data=torch.reshape(params[counter:counter+self.num_params_datapoints], [1,self.num_kde, self.dimension])
            counter+=self.num_params_datapoints


    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x, log_det] = inputs

        extra_input_counter=0
        if self.use_householder:
            this_vs=self.vs.to(x)
            if(extra_inputs is not None):
                

                this_vs=this_vs+torch.reshape(extra_inputs[:,:self.num_householder_params], [x.shape[0], self.vs.shape[1], self.vs.shape[2]])

                
                extra_input_counter+=self.num_householder_params

            rotation_matrix = self.compute_householder_matrix(this_vs, device=x.device)

            
            if(rotation_matrix.shape[0]<x.shape[0]):
                if(rotation_matrix.shape[0]==1):
                    rotation_matrix=rotation_matrix.repeat(x.shape[0], 1,1)
                else:
                    raise Exception("something went wrong with first dim of rot matrix!")

            x = torch.bmm(rotation_matrix.permute(0,2,1), x.unsqueeze(-1)).squeeze(-1)  # uncomment

        #############################################################################################
        # Compute inverse CDF
        #############################################################################################

        this_datapoints=self.datapoints.to(x)
        this_hs=self.log_hs.to(x)
        this_log_norms=self.log_kde_weights.to(x)
        this_skew_signs=self.skew_signs.to(x)

        if(extra_inputs is not None):
          
            this_datapoints=this_datapoints+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [x.shape[0] , self.datapoints.shape[1],  self.datapoints.shape[2]])
            extra_input_counter+=self.num_params_datapoints

            this_hs=this_hs+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [x.shape[0], self.datapoints.shape[1], self.datapoints.shape[2]])
            extra_input_counter+=self.num_params_datapoints


            if(self.fit_normalization):
                this_log_norms=this_log_norms+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [x.shape[0], self.datapoints.shape[1], self.datapoints.shape[2]])
                extra_input_counter+=self.num_params_datapoints

        if(self.fit_normalization and self.regulate_normalization):
            ## regulate log-normalizations

            this_log_norms=self.normalization_regulator(this_log_norms)

        this_skew_exponent=self.skew_exponents.to(x)

        if(self.add_skewness):
            if(extra_inputs is not None):
                this_skew_exponent=this_skew_exponent+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [x.shape[0], self.datapoints.shape[1], self.datapoints.shape[2]])
            
            this_skew_exponent=torch.exp(self.exponent_regulator(this_skew_exponent))

        log_det=log_det+self.sigmoid_inv_error_pass_log_derivative(x, this_datapoints, this_hs,this_log_norms, this_skew_exponent, this_skew_signs).sum(axis=-1)

    
        if(torch.isnan(log_det).sum()>0):
            print("x bef ", x[80:100])
            print("LOG DET AFTER ", log_det[80:100])
            print("datapoints ", this_datapoints[80:100])
            print("this hs", this_hs[80:100])
            print("this log norms", this_log_norms[80:100])


        x=self.sigmoid_inv_error_pass(x, this_datapoints, this_hs,this_log_norms, this_skew_exponent, this_skew_signs)

        if(torch.isnan(log_det).sum()>0):
            print("new x ", x)
            print("nan issue")
            raise Exception()

        
        return x, log_det#, cur_datapoints_update

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 


        extra_input_counter=0

        if self.use_householder:
            this_vs=self.vs.reshape(1,-1)
          
            if(extra_inputs is not None):
                this_vs=this_vs+extra_inputs[:,:self.num_householder_params]

                extra_input_counter+=self.num_householder_params
            
            param_dict[extra_prefix+"vs"]=this_vs.data

        ## reshape as 2d tensor
        this_datapoints=self.datapoints.reshape(1,-1)
        this_hs=self.log_hs.reshape(1,-1)
        this_log_norms=self.log_kde_weights.reshape(1,-1)
        this_skew_exponents=self.skew_exponents.reshape(1,-1)

        if(extra_inputs is not None):
            
            this_datapoints=this_datapoints+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints]
            extra_input_counter+=self.num_params_datapoints

            this_hs=this_hs+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints]
            extra_input_counter+=self.num_params_datapoints

            if(self.fit_normalization):
                this_log_norms=this_log_norms+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints]

            if(self.add_skewness):
                this_log_exponents=this_log_norms+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints]

        param_dict[extra_prefix+"means"]=this_datapoints.data
        param_dict[extra_prefix+"log_widths"]=this_hs.data

        if(self.fit_normalization):
            param_dict[extra_prefix+"log_norms"]=this_log_norms.data

        if(self.add_skewness):
            param_dict[extra_prefix+"exponents"]=this_skew_exponents.data


## transformations

def get_loss_fn(target_matrix, num_householder_iter=-1):
    """
    ## Finds orthogonal householder parameters to match a target rotation matrix
    """
    dim=target_matrix.shape[0]

    def compute_matching_distance(a):

        gblock=gf_block_old(dim, num_householder_iter=num_householder_iter)

        hh_pars=torch.from_numpy(numpy.reshape(a, gblock.vs.shape))
        mat=gblock.compute_householder_matrix(hh_pars).squeeze(0).detach().numpy()

        test_vec=numpy.ones(dim)
        test_vec/=numpy.sqrt((test_vec**2).sum())

        v1=numpy.matmul(mat,test_vec)
        v2=numpy.matmul(target_matrix,test_vec)

        return -(v1*v2).sum()

    return compute_matching_distance

        
def find_init_pars_of_chained_gf_blocks_old(layer_list, data):

    ## given an input *data_inits*, this function tries to initialize the gf block parameters
    ## to best match the data intis
    
    cur_data=data

    """
    cur_data[:,0]*=0.05

    cx=numpy.cos(0.8)
    cy=numpy.sin(0.8)

    rotation_matrix=torch.Tensor([[cx,-cy],[cy,cx]]).unsqueeze(0).type_as(data)
    rotation_matrix=rotation_matrix.repeat(cur_data.shape[0], 1,1)

    cur_data=torch.bmm(rotation_matrix, cur_data.unsqueeze(-1)).squeeze(-1)

    cur_data[:,0]+=100.0
    """
    dim=data.shape[1]

    all_layers_params=[]

    with torch.no_grad():
        ## traverse layers in reversed order
        for layer_ind, cur_layer in enumerate(layer_list[::-1]):

            ## param order .. householder / datapoints / width / normaliaztion
            param_list=[]
            """
            fig=pylab.figure()

            xs=cur_data[:,0]
            ys=cur_data[:,1]

            pylab.plot(xs,ys, color="k", lw=0.0, marker="o", ms=3.0)
            exact_normal_pts=numpy.random.normal(size=cur_data.shape)
            pylab.plot(exact_normal_pts[:,0], exact_normal_pts[:,1], color="red", lw=.0, marker="o", ms=3.0)
            pylab.savefig("layer_large_%s_%d.png" % (name, layer_ind))

            pylab.gca().set_xlim(-4,4)
            pylab.gca().set_ylim(-4,4)

            pylab.savefig("layer_small_%s_%d.png" % (name, layer_ind))
            """
            ## subtract means first if necessary

            if(cur_layer.model_offset):

                means=cur_data.mean(axis=0,keepdim=True)
               
                param_list.append(means.squeeze(0))

                cur_data=cur_data-means

            if(cur_layer.use_householder):

                ## find householder params that correspond to orthogonal transformation of svd of X^T*X (PCA data matrix) if low dimensionality
                this_vs=0

                ## USE PCA for first layer to get major correlation out of the way
                if(cur_layer.dimension<30 and layer_ind==0):

                    data_matrix=torch.matmul(cur_data.T, cur_data)

                    evalues, evecs=scipy.linalg.eig(data_matrix)

                    l, sigma, r=scipy.linalg.svd(data_matrix)
                    
                    loss_fn=get_loss_fn(r, num_householder_iter=cur_layer.householder_iter)

                    start_vec=numpy.random.normal(size=dim*dim)

                    ## fit a matrix via householder parametrization such that it fits the target orthogonal matrix V^* from SVD of X^T*X (PCA data Matrix)
                    res=minimize(loss_fn, start_vec)

                    param_list.append(torch.from_numpy(res["x"]))
                    this_vs=torch.from_numpy(res["x"])
                    
                else:

                    this_vs=torch.randn(cur_layer.dimension*cur_layer.householder_iter)
                    param_list.append(this_vs)

                gblock=gf_block_old(dim, num_householder_iter=cur_layer.householder_iter)

                hh_pars=this_vs.reshape(gblock.vs.shape)
                rotation_matrix=gblock.compute_householder_matrix(hh_pars)
                rotation_matrix=rotation_matrix.repeat(cur_data.shape[0], 1,1)
                ## inverted matrix
                cur_data = torch.bmm(rotation_matrix.permute(0,2,1), cur_data.unsqueeze(-1)).squeeze(-1)

     
            num_kde=cur_layer.num_kde

            assert(num_kde<100)
            #percentiles_to_use=numpy.linspace(0,100,num_kde+2)[1:-1]
            ## use all percentiles for KDE
            percentiles_to_use=numpy.linspace(0,100,num_kde)#[1:-1]
            percentiles=torch.from_numpy(numpy.percentile(cur_data.detach().numpy(), percentiles_to_use, axis=0))

         
            ## add means
            param_list.append(percentiles.flatten())
            

            #pts=numpy.linspace(-20,20,200)
            #x, datapoints, log_widths,log_norms

         
            quarter_diffs=percentiles[1:,:]-percentiles[:-1,:]
            min_perc_diff=quarter_diffs.min(axis=0, keepdim=True)[0]

        
            ## this seems to be optimized settings for num_kde=20
            bw=numpy.log(min_perc_diff*1.5)
            bw=torch.ones_like(percentiles[None,:,:])*bw

           
            flattened_bw=bw.flatten()
            #############
            """
            fig=pylab.figure()

            for x in cur_data:
                pylab.gca().axvline(x[0],color="black")
            log_yvals=cur_layer.logistic_kernel_log_pdf(torch.from_numpy(pts)[:,None], percentiles[None,:,0:1], bw[:,:,0:1], torch.ones_like(percentiles[None,:,0:1]))
            yvals=log_yvals.exp().detach().numpy()
            pylab.gca().plot(pts, yvals, color="green")

            pylab.savefig("test_kde_0.png")


            fig=pylab.figure()

            for x in cur_data:
                pylab.gca().axvline(x[1],color="black")
            log_yvals=cur_layer.logistic_kernel_log_pdf(torch.from_numpy(pts)[:,None], percentiles[None,:,1:2], bw[:,:,1:2], torch.ones_like(percentiles[None,:,1:2]))
            yvals=log_yvals.exp().detach().numpy()
            pylab.gca().plot(pts, yvals, color="green")

            pylab.savefig("test_kde_1.png")

            print("CUR PARAMS", param_list)
            ##########

            """
            
            param_list.append(torch.flatten(bw))


            ## widths

            if(cur_layer.fit_normalization):

                ## norms

                param_list.append(torch.ones_like(flattened_bw))

            this_skewness_exponent=1.0
            this_skewness_signs=1.0
            if(cur_layer.add_skewness):


                ## store zeros (log_exponents) in params
                param_list.append(torch.zeros_like(flattened_bw))

                this_skewness_exponent=cur_layer.exponent_regulator(torch.zeros_like(bw)).exp()
                this_skewness_signs=cur_layer.skew_signs

            all_layers_params.append(torch.cat(param_list))

            ## transform params according to CDF_norm^-1(CDF_KDE)

            #gblock=gf_block(dim, num_householder_iter=cur_layer.householder_iter)
            cur_data=cur_layer.sigmoid_inv_error_pass(cur_data, percentiles[None,:,:], bw, torch.ones_like(bw), this_skewness_exponent, this_skewness_signs)




    all_layers_params=torch.cat(all_layers_params[::-1])

    return all_layers_params



