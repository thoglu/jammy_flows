import torch
from torch import nn
import numpy

from .. import bisection_n_newton as bn
from .. import layer_base
from ... import extra_functions
from .. import matrix_fns
from . import euclidean_base
from .. import spline_fns

import math
import torch.nn.functional as F
import torch.distributions as tdist
import scipy.linalg
from scipy.optimize import minimize
import time
normal_dist=tdist.Normal(0, 1)
import itertools

import pylab

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


class gf_block(euclidean_base.euclidean_base):
    def __init__(self,
                 dimension, 
                 nonlinear_stretch_type="classic",
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
                 lower_bound_for_norms=1,
                 upper_bound_for_norms=10,
                 center_mean=0,
                 clamp_widths=0,
                 regulate_normalization=0,
                 add_skewness=0,
                 rotation_mode="householder"):
        """
        Gaussianization flow: Symbol "g"

        Implements https://arxiv.org/abs/2003.01941 as a modified version of the official implementation in https://github.com/chenlin9/Gaussianization_Flows. Fixes numerical issues with bisection inversion due to more efficient newton iterations, which also makes samples differentiable + fixes numerical issues with inverse CDF parametrization. Also allows 
        for amortization (the original implementation is meant only for non-conditional PDFs). Further adds multiple rotation parametrizations, logistic skewness option, and an rq-spline option for the non-linear mapping.

        Parameters:

            dimension (int): Dimension of the PDF. Set by *jammy_flows.pdf*.
            nonlinear_stretch_type (str): One of ["classic", "rq_splines"]. Default is classic, but also allows to replace non-linear mapping with rational quadratic splines.
            num_kde (int): Number of KDE Kernels in each one-dimensional transformation.
            num_householder_iter (int): if <=0, no householder transformation is performed. If positive, it defines the number of parameters in householder transformations.
            fit_normalization (int): If set, also fits the normalization.
            inverse_function_type (str): One of ["partly_crude", "partly_precise", "full_pade", "isgimoid"]. "Partly_crude" is implemented in the original repository (https://github.com/chenlin9/Gaussianization_Flows), but has numerical issues. The option "isigmoid" "is an easier inverse CDF that looses exact CDF identification but gains some speed and is numerically stable. It is recommended to use "isigmoid" or "partly_precise".
            model_offset (int): Set by *jammy_flows.pdf*.
            softplus_for_width (int): If set, uses *soft_plus* to predict width. Otherwhise exponential function.
            width_smooth_saturaiton (int): If set, saturates the width symmetrically at the lower and upper end.
            lower_bound_for_widths (float): Determines the lower width bound for each KDE element.
            upper_bound_for_widths (float): Determines the upper width bound for each KDE element.
            lower_bound_for_norms (float): Determines the lower norm bound for each KDE element.
            upper_bound_for_norms (float): Determines the upper norm bound for each KDE element.
            clamp_widths (int): If set, clamps widths above threshold.
            regulate_normalization (int): If set, regulates with a lower bound of 1 and upper bound of 100 (fixed) similar to width regulation.
            add_skewness (int): If set, adds skewness to the basis functions in order to make flow more flexible.
            rotation_mode (str): One of different rotation parametrizations:

                    | - **householder**: Default implementation from the original paper.
                    | - **angles**: Parametrized via "Givens rotation"
                    | - **triangular_combination**: Parametrization of an upper and lower triangular matrix.
                    | - **cayley**: "Cayley representation" of rotations.

        """
        super().__init__(dimension=dimension, use_permanent_parameters=use_permanent_parameters, model_offset=model_offset)
        self.init = False

        self.nonlinear_stretch_type=nonlinear_stretch_type

        # norms
        self.lower_bound_for_norms=lower_bound_for_norms
        self.upper_bound_for_norms=upper_bound_for_norms

        assert(lower_bound_for_widths>0.0)
        self.width_min=lower_bound_for_widths
        ## defines maximum width - None -> no maximum width .. only used for exponential width function to cap high values
        self.width_max=None

        if(upper_bound_for_widths > 0):
            self.width_max=upper_bound_for_widths

            ### clamp at three times the logarithm to upper bound log_width .. more than enough for whole range
            self.log_width_max_to_clamp=numpy.log(self.width_max)*3.0

        ## doing a smooth (min-max bound) width regularization?
        self.width_smooth_saturation=width_smooth_saturation
        if(self.width_smooth_saturation):
            assert(self.width_max is not None), "We require a maximum saturation level for smooth saturation!"

        ## clamp at a hundreths of the smallest len allowed width width_min (yields approximately clamp value below, as long as width_max >> width_min)
        self.log_width_min_to_clamp=numpy.log(0.01*self.width_min)

        ## clamping widths?
        self.clamp_widths=clamp_widths

        ## inverse function type - 
        self.inverse_function_type=inverse_function_type
        assert(self.inverse_function_type=="inormal_partly_crude" or self.inverse_function_type=="inormal_partly_precise" or  self.inverse_function_type=="inormal_full_pade" or  self.inverse_function_type=="isigmoid")

        #### constants related to inverse Gaussian CDF
        ## p-value after which to switch to pade approximation
        self.pade_approximation_bound=0.5e-7

        ## constant used for pade approximation of inverse gaussian CDF
        self.pade_const_a=0.147
        #########################################################

        ## dimension of target space
        self.dimension = dimension

        self.fit_normalization=fit_normalization
        self.regulate_normalization=regulate_normalization
        self.add_skewness=add_skewness

        #######################################

        ## Householder rotations
        self.rotation_mode=rotation_mode
        if(self.rotation_mode=="triangular_combination"):

            ## diagonal + lower/upper (uni) triangular matrices
            num_triangle_params=int(self.dimension-1+ (self.dimension*(self.dimension-1)))
            self.num_triangle_params=num_triangle_params

            self.total_param_num+=num_triangle_params
            if(use_permanent_parameters):

                if(self.dimension>1):
                    self.triangle_trafo_pars = nn.Parameter(
                        torch.randn(num_triangle_params).unsqueeze(0)
                    )
           
        elif(self.rotation_mode=="householder"):
            if num_householder_iter == -1:
                self.householder_iter = dimension #min(dimension, 10)
            else:
                self.householder_iter = num_householder_iter

            self.use_householder=True
            if(self.householder_iter==0):
               
                self.use_householder=False


            self.num_householder_params=0

            if self.use_householder:
                if(use_permanent_parameters):
                    self.vs = nn.Parameter(
                        torch.randn(self.householder_iter, dimension).unsqueeze(0)
                    )
                #else:
                #    self.vs = torch.zeros(self.householder_iter, dimension).type(torch.double).unsqueeze(0) 

                self.num_householder_params=self.householder_iter*self.dimension

            self.total_param_num+=self.num_householder_params

        elif(self.rotation_mode=="angles"):
            self.num_angle_pars=0

            if(self.dimension>1):
                self.num_angle_pars=int((self.dimension*(self.dimension-1)/2))

                self.total_param_num+=self.num_angle_pars

                #assert(self.dimension==2), "requires 2 dims at the moment"

                if(use_permanent_parameters):
                    self.angle_pars=nn.Parameter(torch.randn((1, self.num_angle_pars)))

        elif(self.rotation_mode=="cayley"):
            self.num_cayley_pars=0

            if(self.dimension>1):
                self.num_cayley_pars=1
                self.num_cayley_pars=self.num_cayley_pars

                self.total_param_num+=self.num_cayley_pars

                assert(self.dimension==2), "Cayley requires 2 dims at the moment"

                if(use_permanent_parameters):
                    self.cayley_pars=nn.Parameter(torch.randn((1, 1)))


        ## number of KDE components per dimension
        self.num_kde = num_kde

        ## number of total params for a given mean/width ...
        self.num_params_datapoints=self.num_kde*self.dimension

        ## initialization from Gaussianization flow paper for widths
        bandwidth = (4. * numpy.sqrt(math.pi) / ((math.pi ** 4) * num_kde)) ** 0.2
        self.init_log_width=numpy.log(bandwidth)

        self.center_mean=center_mean


        if(self.nonlinear_stretch_type=="classic"):
            ## means
            if use_permanent_parameters:
                self.kde_means = nn.Parameter(torch.randn(self.num_kde-self.center_mean, self.dimension).unsqueeze(0))

            ## increase params per (kde-center_mean)*dim
            self.total_param_num+=(self.num_kde-self.center_mean)*self.dimension
            self.total_param_num_means=(self.num_kde-self.center_mean)*self.dimension

            #######################################

            ## widths
            #self.kde_log_widths = torch.zeros(num_kde, dimension).type(torch.double).unsqueeze(0)
            if(use_permanent_parameters):
                self.kde_log_widths = nn.Parameter(
                    torch.ones(num_kde, dimension).unsqueeze(0) * self.init_log_width
                )

            ## increase params per kde*dim
            self.total_param_num+=self.num_params_datapoints

            ## softplus for width?
            self.softplus_for_width=softplus_for_width

            ## we only want specific settings in the new implementation, but keep the older implementation for now
            #assert(self.softplus_for_width==False)
            #assert(self.width_smooth_saturation>0)
            #assert(self.clamp_widths==0)

            ## setup up width regularization
            if(self.softplus_for_width):
                ## softplus
                if(clamp_widths):
                    upper_clamp=None
                    if(self.width_max is not None):
                        # clamp upper bound with exact width_max value
                        upper_clamp=numpy.log(self.width_max)
                    self.width_regulator=lambda x: torch.log(torch.nn.functional.softplus(torch.clamp(x, min=self.log_width_min_to_clamp, max=upper_clamp))+self.width_min)
                else:
                    self.width_regulator=lambda x: torch.log(torch.nn.functional.softplus(x)+self.width_min)
                
                

            else:
                ## exponential-type width relation
                if(self.width_smooth_saturation == 0):
                    ## normal, infinetly growing exponential
                    if(self.clamp_widths):
                        # clamp upper bound with exact width_max value
                        upper_clamp=None
                        if(self.width_max is not None):
                            upper_clamp=numpy.log(self.width_max)
                        self.width_regulator=lambda x: torch.log(torch.exp(torch.clamp(x, min=self.log_width_min_to_clamp, max=upper_clamp))+self.width_min)
                    else:
                        self.width_regulator=lambda x: torch.log(torch.exp(x)+self.width_min)

                else:
                    ## exponential function at beginning but flattens out at width_max -> no infinite growth
                    ## numerically stable via logsumexp .. clamping should not be necessary, but can be done to damp down large gradients
                    ## in weird regions of parameter space
                    
                    ln_width_max=numpy.log(self.width_max)
                    ln_width_min=numpy.log(self.width_min)

                    if(self.clamp_widths):

                        def exp_like_fn(x):

                            res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-torch.clamp(x, min=self.log_width_min_to_clamp, max=self.log_width_max_to_clamp)+ln_width_max).unsqueeze(-1)], dim=-1)

                            first_term=ln_width_max-torch.logsumexp(res, dim=-1, keepdim=True)

                            return torch.logsumexp( torch.cat([first_term, torch.ones_like(first_term)*ln_width_min], dim=-1), dim=-1)

                    else:

                        exp_like_fn=generate_log_function_bounded_in_logspace(self.width_min, self.width_max, center=True)

                    self.width_regulator=exp_like_fn

                    
            #######################################

            # normalization parameters
            #self.kde_log_weights = torch.zeros(self.num_kde, self.dimension).type(torch.double).unsqueeze(0)#.to(device)
            if(fit_normalization):

                if(use_permanent_parameters):
                    self.kde_log_weights = nn.Parameter(torch.randn(num_kde, self.dimension).unsqueeze(0))


                self.total_param_num+=self.num_params_datapoints


            

            self.normalization_regulator = None
            if(self.fit_normalization):
                if(self.regulate_normalization):
                    ## bound normalization into a range that spans roughly ~ 100 . .we dont want huge discrepancies in normalization
                    ## this serves as a stabilizer during training compared to no free-floating normalization, but at the same time
                    ## avoids near zero normalizations which can also lead to unwanted side effects

                    self.normalization_regulator=generate_log_function_bounded_in_logspace(min_val_normal_space=self.lower_bound_for_norms, max_val_normal_space=self.upper_bound_for_norms)

            #######################################

            

            ## shape to B X KDE index dim X dimension
            self.kde_log_skew_exponents=torch.Tensor([0.0]).view(1,1,1)
            self.kde_skew_signs=torch.Tensor([1.0])

            if(self.add_skewness):

                self.kde_skew_signs=torch.ones( (1,self.num_kde,1))

                num_negative=int(float(self.num_kde)/2.0)

                ## half of the KDEs use a flipped prescription
                self.kde_skew_signs[:,num_negative:,:]=-1.0

                if(use_permanent_parameters):
                    self.kde_log_skew_exponents = nn.Parameter(
                        torch.randn(self.num_kde, dimension).unsqueeze(0)
                    )
                
                ## with 0.1 and 9.0 the function maps 0 to 0 approximately -> 0 -> 1 in normal exponent space, the starting point we want
                self.exponent_regulator=generate_log_function_bounded_in_logspace(min_val_normal_space=0.1, max_val_normal_space=9.0, center=True)
                self.total_param_num+=self.num_params_datapoints

        elif(self.nonlinear_stretch_type=="rq_splines"):

            if(use_permanent_parameters):
                self.log_widths = nn.Parameter(torch.randn(self.dimension, self.num_kde).unsqueeze(0))
                self.log_heights = nn.Parameter(torch.randn(self.dimension, self.num_kde).unsqueeze(0))
                self.log_derivatives = nn.Parameter(torch.randn(self.dimension, self.num_kde+1).unsqueeze(0))
                self.boundary_points=nn.Parameter(torch.randn(self.dimension, 4).unsqueeze(0))  

           
            self.total_param_num+=(self.num_kde*self.dimension)*2+(self.num_kde+1)*self.dimension

            # add 4*dimension boundary pts
            self.total_param_num+=(4*self.dimension)

        else:
            raise Exception("Unknown non linear stretch type: %s" % self.nonlinear_stretch_type)



    def logistic_kernel_log_pdf_quantities(self, x, means, log_widths, log_norms, log_skew_exponents, skew_signs, calculate_pdf=True):

        ## the widths can actually be a non log
       
        ## shape to B X KDE index dim X dimension

        widths=torch.exp(log_widths)

        x_unsqueezed=x.unsqueeze(1)

        common_x_argument=(x_unsqueezed - means) / widths

        skew_exponents=torch.exp(log_skew_exponents)


        #### pdf part

        individual_normalizers=log_norms - torch.logsumexp(log_norms, dim=1, keepdim=True)

        log_pdf=None
        if(calculate_pdf):

            log_pdfs = -skew_signs*common_x_argument - log_widths + log_skew_exponents - \
                        (skew_exponents+1.0) * F.softplus(-skew_signs*common_x_argument)+ individual_normalizers

            log_pdf=torch.logsumexp(log_pdfs, dim=1)

        ## CDF/SF are more complicated with skewness
        if(self.add_skewness):
            ## differentiate CDF/SF for +/- skewed distributions (they mirror each other)

            log_cdfs=torch.zeros( (x.shape[0], means.shape[1], means.shape[2]), dtype=x.dtype, device=x.device)

            pos_mask=skew_signs[0,:,0]>0
        
            ## positive cdfs
            log_cdfs.masked_scatter_(pos_mask[None,:,None], - skew_exponents[:,pos_mask,:]*F.softplus(-common_x_argument[:,pos_mask,:]))
            ## negative cdfs
            log_cdfs.masked_scatter_(~pos_mask[None,:,None], extra_functions.log_one_plus_exp_x_to_a_minus_1(common_x_argument[:,~pos_mask,:], skew_exponents[:,~pos_mask,:]))

            log_cdfs=log_cdfs+individual_normalizers

            ####

            log_sfs=torch.zeros( (x.shape[0], means.shape[1], means.shape[2]), dtype=x.dtype, device=x.device)

            ## positive sfs
            log_sfs.masked_scatter_(pos_mask[None,:,None], extra_functions.log_one_plus_exp_x_to_a_minus_1(-common_x_argument[:,pos_mask,:], skew_exponents[:,pos_mask,:]))

            ## negative sfs
            log_sfs.masked_scatter_(~pos_mask[None,:,None], - skew_exponents[:,~pos_mask,:]*F.softplus(common_x_argument[:,~pos_mask,:]))
            
           
            log_sfs = log_sfs+individual_normalizers

        else:

            log_sfs = -common_x_argument - F.softplus(-common_x_argument) + individual_normalizers

            log_cdfs = - F.softplus(-common_x_argument) + individual_normalizers
                       

        log_sf=torch.logsumexp(log_sfs, dim=1)
        log_cdf=torch.logsumexp(log_cdfs, dim=1)

        return log_cdf, log_sf, log_pdf


    def compute_householder_matrix(self, vs):

        Q = torch.eye(self.dimension, device=vs.device).type(vs.dtype).unsqueeze(0).repeat(vs.shape[0], 1,1)
       
        for i in range(self.householder_iter):
        
            v = vs[:,i].reshape(-1,self.dimension, 1).to(vs.device)
            
            v = v / v.norm(dim=1).unsqueeze(-1)

            Qi = torch.eye(self.dimension, device=vs.device).type(vs.dtype).unsqueeze(0) - 2 * torch.bmm(v, v.permute(0, 2, 1))

            Q = torch.bmm(Q, Qi)

        return Q

   
    def sigmoid_inv_error_pass_w_params(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):

        log_cdf_l, log_sf_l, _=self.logistic_kernel_log_pdf_quantities(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs, calculate_pdf=False)  
      
        return self.sigmoid_inv_error_pass_given_cdf_sf(log_cdf_l, log_sf_l)

    def sigmoid_inv_error_pass_given_cdf_sf(self, log_cdf_l, log_sf_l):

       
        if(self.inverse_function_type=="isigmoid"):

            ## super easy inverse function which can be written in terms of log_cdf and log_sf, which makes it numerically stable!

          
            return -log_sf_l+log_cdf_l


        else:
          
            cdf_l=torch.exp(log_cdf_l)

            if("partly" in self.inverse_function_type):

                cdf_mask = ((cdf_l > self.pade_approximation_bound) & (cdf_l < 1 - (self.pade_approximation_bound))).type(log_cdf_l.dtype)
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
                cdf_mask_right = (cdf_l >= 1. - (self.pade_approximation_bound)).type(log_cdf_l.dtype)
                cdf_l_bad_right_log = (total_factor) * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
                return_val += (cdf_l_bad_right_log)*cdf_mask_right

                ## very LOW CDF values
                cdf_mask_left = (cdf_l <= self.pade_approximation_bound).type(log_cdf_l.dtype)
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

                mask_neg=(cdf_l<=0.5).type(log_cdf_l.dtype)

                return (-1.0*total_factor)*mask_neg+(1.0-mask_neg)*total_factor

    
    def sigmoid_inv_error_pass_log_derivative_w_params(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):

        log_cdf, log_sf, log_pdf=self.logistic_kernel_log_pdf_quantities(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs, calculate_pdf=True)  

        return self.sigmoid_inv_error_pass_log_derivative_given_cdf_sf(log_cdf, log_sf, log_pdf)
    
    def sigmoid_inv_error_pass_log_derivative_given_cdf_sf(self, log_cdf_l, log_sf_l, log_pdf):

        if(self.inverse_function_type=="isigmoid"):

            ## super easy inverse function which can be written in terms of log_cdf and log_sf, which makes it numerically stable!
            lse_cat=torch.cat([-log_sf_l[:,:,None],-log_cdf_l[:,:,None]], axis=-1)
            lse_sum=torch.logsumexp(lse_cat, axis=-1)

            return lse_sum+log_pdf
        else:

            cdf_l=torch.exp(log_cdf_l)
            

            if("partly" in self.inverse_function_type):

                cdf_mask = ((cdf_l > self.pade_approximation_bound) & (cdf_l < 1 - (self.pade_approximation_bound))).type(log_cdf_l.dtype)
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


                    mask_neg=(cdf_l<=0.5).type(log_cdf_l.dtype)
                    extra_plus_minus_factor=torch.log((1.0-2*cdf_l)*mask_neg+(-1.0+2*cdf_l)*(1-mask_neg))

                    total_factor=total_factor+extra_plus_minus_factor

                    ######
                    
                    bad_deriv_mask=( (cdf_l>0.49999) & (cdf_l < 0.50001))
                    
                    total_factor=total_factor.masked_fill(bad_deriv_mask, numpy.log(2.506628))

                        
                cdf_mask_right = (cdf_l >= 1. - (self.pade_approximation_bound)).type(log_cdf_l.dtype)
                cdf_l_bad_right_log = total_factor * cdf_mask_right  -0.5*(1.0-cdf_mask_right)
                derivative += cdf_mask_right*(cdf_l_bad_right_log+log_pdf)
                # 3) Step3: invert BAD small CDF
                cdf_mask_left = (cdf_l <= self.pade_approximation_bound).type(log_cdf_l.dtype)
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

    def sigmoid_inv_error_pass_combined_val_n_log_derivative(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):
        """
        Used by inverse flow to compactly calculate both quantities.
        """
        log_cdf, log_sf, log_pdf=self.logistic_kernel_log_pdf_quantities(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs, calculate_pdf=True)  

        new_val=self.sigmoid_inv_error_pass_given_cdf_sf(log_cdf, log_sf)

        log_deriv=self.sigmoid_inv_error_pass_log_derivative_given_cdf_sf(log_cdf, log_sf, log_pdf)

        return new_val, log_deriv

    def sigmoid_inv_error_pass_combined_val_n_normal_derivative(self, x, datapoints, log_widths, log_norms, skew_exponents, skew_signs):
        """
        Used by Newton iterations (requires derivative instead of log-derivative).
        """
        log_cdf, log_sf, log_pdf=self.logistic_kernel_log_pdf_quantities(x, datapoints,log_widths,log_norms, skew_exponents, skew_signs, calculate_pdf=True)  

        new_val=self.sigmoid_inv_error_pass_given_cdf_sf(log_cdf, log_sf)

        log_deriv=self.sigmoid_inv_error_pass_log_derivative_given_cdf_sf(log_cdf, log_sf, log_pdf)

        return new_val, log_deriv.exp()

    

    def _obtain_usable_flow_params(self, x, extra_inputs=None):
        """
        Checks if to use fixed (permanent) parameters of the flow or defines them via extra_inputs. Returns
        the actual parameters of the flow layer (besides householder parameters which are treated extra).

        x: batch of data -> x.shape[0] = batch_size
        """

        extra_input_counter=0

        rotation_params=None

        if(self.rotation_mode=="triangular_combination"):

            if(self.dimension>1):

                num_params_per_mat=int(self.dimension*(self.dimension-1)/2)

                if(extra_inputs is None):
                    left_triangular_pars=self.triangle_trafo_pars[:,:num_params_per_mat].to(x)
                    diagonal_pars=self.triangle_trafo_pars[:,num_params_per_mat:num_params_per_mat+self.dimension-1].to(x)
                    right_triangular_pars=self.triangle_trafo_pars[:,num_params_per_mat+self.dimension-1:2*num_params_per_mat+self.dimension-1].to(x)

                else:
                    left_triangular_pars=extra_inputs[:,:num_params_per_mat]
                    diagonal_pars=extra_inputs[:,num_params_per_mat:num_params_per_mat+self.dimension-1]
                    right_triangular_pars=extra_inputs[:,num_params_per_mat+self.dimension-1:2*num_params_per_mat+self.dimension-1]
                        
                    extra_input_counter+=self.num_triangle_params

                rotation_params=(left_triangular_pars, diagonal_pars, right_triangular_pars)

        elif(self.rotation_mode=="householder"):
           
            if self.use_householder:
                
                if(extra_inputs is None):
                    this_vs=self.vs.to(x)
                else:
                   
                    this_vs=torch.reshape(extra_inputs[:,:self.num_householder_params], [-1, self.householder_iter, self.dimension])

                    
                    extra_input_counter+=self.num_householder_params

                ## rotation_params is actually a matrix
                rotation_params = self.compute_householder_matrix(this_vs)

        elif(self.rotation_mode=="angles"):
            # implemented as givens rotations
            if(self.dimension>1):

                if(extra_inputs is None):

                    rotation_params=self.angle_pars.to(x)

                else:
                    rotation_params=extra_inputs[:, :self.num_angle_pars]
                    
                extra_input_counter+=self.num_angle_pars

                combi_list=[]

                for a, b in itertools.combinations(numpy.arange(self.dimension), 2):
                    combi_list.append( (a,b) )

                base_matrix=torch.eye(self.dimension).to(rotation_params).unsqueeze(0).repeat(rotation_params.shape[0],1,1)

                prev_matrix=base_matrix
                
                for ind, combi in enumerate(combi_list):

                    new_matrix=base_matrix.clone()

                    new_matrix[:, combi[0], combi[0]]=torch.cos(rotation_params[:, ind])
                    new_matrix[:, combi[1], combi[1]]=new_matrix[:, combi[0], combi[0]]
                    new_matrix[:, combi[0], combi[1]]=torch.sin(rotation_params[:, ind])
                    new_matrix[:, combi[1], combi[0]]=-new_matrix[:, combi[0], combi[1]]

                    prev_matrix=torch.bmm(new_matrix, prev_matrix)
               
                rotation_params=prev_matrix

        elif(self.rotation_mode=="cayley"):

            if(self.dimension>1):
                if(extra_inputs is None):

                    rotation_params=self.cayley_pars.to(x)

                else:
                    rotation_params=extra_inputs[:, :self.num_cayley_pars]

                extra_input_counter+=self.num_cayley_pars
                mult_fac=1.0/(1+rotation_params**2)
                rot_matrix=torch.diag_embed( ((1-rotation_params**2)*mult_fac).repeat(1,2))
                rot_matrix[:,0:1,1:2]=(-2*rotation_params*mult_fac).unsqueeze(-1)
                rot_matrix[:,1:2,0:1]=(2*rotation_params*mult_fac).unsqueeze(-1)

                rotation_params=rot_matrix

        if(self.nonlinear_stretch_type=="classic"):

            kde_log_skew_exponents=self.kde_log_skew_exponents.to(x)
            kde_skew_signs=self.kde_skew_signs.to(x)

            if(extra_inputs is None):

                kde_means=self.kde_means.to(x)
                kde_log_widths=self.kde_log_widths.to(x)
                if(self.fit_normalization):
                    kde_log_weights=self.kde_log_weights.to(x)
                else:
                    kde_log_weights=torch.zeros_like(kde_log_widths)
                
            else:
                ## skipping householder params
                #extra_input_counter=self.num_householder_params

                kde_means=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.total_param_num_means], [x.shape[0] , self.num_kde-self.center_mean,  self.dimension])
                extra_input_counter+=self.total_param_num_means

                kde_log_widths=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [x.shape[0] , self.num_kde,  self.dimension])
                extra_input_counter+=self.num_params_datapoints

                if(self.fit_normalization):
                    kde_log_weights=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [x.shape[0] , self.num_kde,  self.dimension])
                    extra_input_counter+=self.num_params_datapoints
                else:
                    kde_log_weights=torch.zeros(x.shape[0], self.num_kde, self.dimension).to(x)

                if(self.add_skewness):

                    kde_log_skew_exponents=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints], [x.shape[0] , self.num_kde,  self.dimension])
            
            ## transform width

            kde_log_widths=self.width_regulator(kde_log_widths)

            ## transform normalization if necessary
            if(self.fit_normalization and self.regulate_normalization):

                ## regulate log-normalizations if desired
                kde_log_weights=self.normalization_regulator(kde_log_weights)

            if(self.center_mean):
                #calculate last mean based on other weights and previous means
                normal_weights=kde_log_weights.exp()

                new_mean=-(kde_means*normal_weights[:,:-1,:]).sum(axis=1, keepdims=True)/normal_weights[:,-1:,:]
               
                kde_means=torch.cat([kde_means, new_mean], dim=1)


            ## transform skewness if necessary
            if(self.add_skewness):
        
                kde_log_skew_exponents=self.exponent_regulator(kde_log_skew_exponents)


            return (kde_means, kde_log_widths, kde_log_weights, kde_log_skew_exponents, kde_skew_signs), rotation_params
        else:
            # nonlinear stretch type is based on rq_splines

            if(extra_inputs is None):
                log_widths = self.log_widths.to(x)
                log_heights = self.log_heights.to(x)
                log_derivatives = self.log_derivatives.to(x)
                boundary_points = self.boundary_points.to(x)

            else:

                log_widths=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*self.num_kde], [-1 , self.dimension, self.num_kde])
                extra_input_counter+=self.dimension*self.num_kde

                log_heights=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*self.num_kde], [-1 , self.dimension, self.num_kde])
                extra_input_counter+=self.dimension*self.num_kde

                log_derivatives=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*(self.num_kde+1)], [-1 ,  self.dimension,self.num_kde+1])
                extra_input_counter+=self.dimension*(self.num_kde+1)

                boundary_points=torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*4], [-1 , self.dimension, 4])
                extra_input_counter+=self.dimension*4


            
            ## older parametrization .. does not seem work well with gradients
            #left=boundary_points[:,:,0:1]
            #right=boundary_points[:,:,1:2]

            #new_left=torch.where(left<right, left, right)
            #new_right=torch.where(left<right, right, left)

            #bottom=boundary_points[:,:,2:3]
            #top=boundary_points[:,:,3:4]
                
            #new_bottom=torch.where(bottom<top, bottom, top)
            #new_top=torch.where(bottom<top, top, bottom)
           
            ## TODO: remove hardcoding of minimum widths
            min_abs_width=0.5

            new_left=boundary_points[:,:,0:1]
            new_right=new_left+torch.exp(boundary_points[:,:,1:2])+min_abs_width

            new_bottom=boundary_points[:,:,2:3]
            new_top=new_bottom+torch.exp(boundary_points[:,:,3:4])+min_abs_width

            return (log_widths, log_heights, log_derivatives, new_left,new_right,new_bottom,new_top), rotation_params

    def _flow_mapping(self, inputs, extra_inputs=None, verbose=False, lower=-1e5, upper=1e5): 
        
        [z, log_det]=inputs

        device=z.device

        flow_params, rotation_params=self._obtain_usable_flow_params(z, extra_inputs=extra_inputs)
       
        if(self.nonlinear_stretch_type=="classic"):
           
            res=bn.inverse_bisection_n_newton_joint_func_and_grad(self.sigmoid_inv_error_pass_w_params, self.sigmoid_inv_error_pass_combined_val_n_normal_derivative, z, flow_params[0], flow_params[1],flow_params[2],flow_params[3],flow_params[4], min_boundary=lower, max_boundary=upper, num_bisection_iter=25, num_newton_iter=20)
            log_deriv=self.sigmoid_inv_error_pass_log_derivative_w_params(res, flow_params[0], flow_params[1], flow_params[2], flow_params[3], flow_params[4])

            log_det=log_det-log_deriv.sum(axis=-1)
           
        elif(self.nonlinear_stretch_type=="rq_splines"):
           
            res, log_deriv=spline_fns.rational_quadratic_spline_with_linear_extension(z.unsqueeze(-1), 
                                                                       flow_params[0],
                                                                        flow_params[1],
                                                                        flow_params[2],
                                                                        left=flow_params[3],
                                                                        right=flow_params[4],
                                                                        bottom=flow_params[5],
                                                                        top=flow_params[6],
                                                                        inverse=True)

            res=res.squeeze(-1)
          
            log_det=log_det+log_deriv.squeeze(-1).sum(axis=-1)

        if(self.rotation_mode=="triangular_combination"):
          
            if(self.dimension>1):
                left,middle,right=rotation_params

                zero_entries=torch.zeros(self.dimension).to(res).unsqueeze(0)

                trafo_matrix_right, _=matrix_fns.obtain_lower_triangular_matrix_and_logdet(self.dimension, log_diagonal_entries=zero_entries, lower_triangular_entries=right, upper_triangular=True)
                trafo_matrix_left, _=matrix_fns.obtain_lower_triangular_matrix_and_logdet(self.dimension, log_diagonal_entries=zero_entries, lower_triangular_entries=left)
                diag=torch.cat([middle, -middle.sum(axis=1, keepdims=True)], dim=1)

                if(trafo_matrix_right.shape[0]<z.shape[0]):
                    if(trafo_matrix_right.shape[0]==1):
                        trafo_matrix_right=trafo_matrix_right.repeat(z.shape[0], 1,1)
                        trafo_matrix_left=trafo_matrix_left.repeat(z.shape[0], 1,1)
                    else:
                        raise Exception("something went wrong with first dim of rot matrix!")
                
                res = torch.bmm(trafo_matrix_right, res.unsqueeze(-1)).squeeze(-1)
                
                res=res*torch.exp(diag)
               
                res = torch.bmm(trafo_matrix_left, res.unsqueeze(-1)).squeeze(-1)
               
        elif(self.rotation_mode=="householder"):
            if self.use_householder:

                if(rotation_params.shape[0]<z.shape[0]):
                    if(rotation_params.shape[0]==1):
                        rotation_params=rotation_params.repeat(z.shape[0], 1,1)
                    else:
                        raise Exception("something went wrong with first dim of rot matrix!")

                res = torch.bmm(rotation_params, res.unsqueeze(-1)).squeeze(-1)
        
        elif(self.rotation_mode=="angles" or self.rotation_mode=="cayley"):

            if(self.dimension>1):
                if(rotation_params is not None):
                    if(rotation_params.shape[0]<z.shape[0]):
                        if(rotation_params.shape[0]==1):
                            rotation_params=rotation_params.repeat(z.shape[0], 1,1)
                        else:
                            raise Exception("something went wrong with first dim of rot matrix!")

                res = torch.bmm(rotation_params, res.unsqueeze(-1)).squeeze(-1)

        return res, log_det

    



    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x, log_det] = inputs

        #############################################################################################
        # Compute inverse CDF
        #############################################################################################
        flow_params, rotation_params=self._obtain_usable_flow_params(x, extra_inputs=extra_inputs)

        if(self.rotation_mode=="triangular_combination"):

            if(self.dimension>1):
                left,middle,right=rotation_params

                zero_entries=torch.zeros(self.dimension).to(x).unsqueeze(0)

                inverse_trafo_matrix_right, _=matrix_fns.obtain_inverse_lower_triangular_matrix_and_logdet(self.dimension, log_diagonal_entries=zero_entries, lower_triangular_entries=right, upper_triangular=True)
                inverse_trafo_matrix_left, _=matrix_fns.obtain_inverse_lower_triangular_matrix_and_logdet(self.dimension, log_diagonal_entries=zero_entries, lower_triangular_entries=left)
                diag=torch.cat([middle, -middle.sum(axis=1, keepdims=True)], dim=1)

                if(inverse_trafo_matrix_right.shape[0]<x.shape[0]):
                    if(inverse_trafo_matrix_right.shape[0]==1):
                        inverse_trafo_matrix_right=inverse_trafo_matrix_right.repeat(x.shape[0], 1,1)
                        inverse_trafo_matrix_left=inverse_trafo_matrix_left.repeat(x.shape[0], 1,1)
                    else:
                        raise Exception("something went wrong with first dim of rot matrix!")

               
                x = torch.bmm(inverse_trafo_matrix_left, x.unsqueeze(-1)).squeeze(-1)
              
                x=x/torch.exp(diag)

                x = torch.bmm(inverse_trafo_matrix_right, x.unsqueeze(-1)).squeeze(-1)
            
        elif(self.rotation_mode=="householder"):
            if self.use_householder:

                if(rotation_params.shape[0]<x.shape[0]):
                    if(rotation_params.shape[0]==1):
                        rotation_params=rotation_params.repeat(x.shape[0], 1,1)
                    else:
                        raise Exception("something went wrong with first dim of rot matrix!")

                x = torch.bmm(rotation_params.permute(0,2,1), x.unsqueeze(-1)).squeeze(-1)

        elif(self.rotation_mode=="angles" or self.rotation_mode=="cayley"):

            if(self.dimension>1):
                if(rotation_params.shape[0]<x.shape[0]):
                    if(rotation_params.shape[0]==1):
                        rotation_params=rotation_params.repeat(x.shape[0], 1,1)
                    else:
                        raise Exception("something went wrong with first dim of rot matrix!")

                x = torch.bmm(rotation_params.permute(0,2,1), x.unsqueeze(-1)).squeeze(-1)

        ###############################

        if(self.nonlinear_stretch_type=="classic"):
         
            x, log_deriv=self.sigmoid_inv_error_pass_combined_val_n_log_derivative(x, *flow_params)

            log_det=log_det+log_deriv.sum(axis=-1)

        elif(self.nonlinear_stretch_type=="rq_splines"):
            x, log_deriv=spline_fns.rational_quadratic_spline_with_linear_extension(x.unsqueeze(-1), 
                                                                       flow_params[0],
                                                                        flow_params[1],
                                                                        flow_params[2],
                                                                        left=flow_params[3],
                                                                        right=flow_params[4],
                                                                        bottom=flow_params[5],
                                                                        top=flow_params[6],
                                                                        inverse=False)

            """
            testz=torch.linspace(-2.5, 2.1, 1000).unsqueeze(-1).unsqueeze(-1)

            fig=pylab.figure()

            ax=fig.add_subplot(221)
            ax2=fig.add_subplot(222)

            for do_inv in [False, True]:
                
                

              
                res, log_deriv=spline_fns.rational_quadratic_spline_with_linear_extension(testz, 
                                                                           flow_params[0][:,0:1,:],
                                                                            flow_params[1][:,0:1,:],
                                                                            flow_params[2][:,0:1,:],
                                                                            left=flow_params[3][:,0:1,:],
                                                                            right=flow_params[4][:,0:1,:],
                                                                            bottom=flow_params[5][:,0:1,:],
                                                                            top=flow_params[6][:,0:1,:],
                                                                            inverse=do_inv)
                
            
            
                ax.plot(testz.squeeze(-1).squeeze(-1).numpy(), res.detach().squeeze(-1).squeeze(-1).numpy(), color="k" if do_inv else "r")

                

                ax2.plot(testz.squeeze(-1).squeeze(-1).numpy(), log_deriv.detach().exp().squeeze(-1).squeeze(-1).numpy(), color="k" if do_inv else "r")

            ax2.semilogy()


            pylab.savefig("test.png")
            sys.exit(-1)
            """
            


            x=x.squeeze(-1)

            log_det=log_det+log_deriv.squeeze(-1).sum(axis=-1)
        
        return x, log_det

    def _get_desired_init_parameters(self):

        ## householder params / means of kdes / log_widths of kdes / normalizations (if fit normalization)

        desired_param_vec=[]

        ## householder
        if(self.rotation_mode=="triangular_combination"):
            desired_param_vec.append(torch.zeros(self.num_triangle_params))
        elif(self.rotation_mode=="householder"):
            if(self.num_householder_params > 0):
                desired_param_vec.append(torch.randn(self.householder_iter*self.dimension))

        elif(self.rotation_mode=="angles"):

            desired_param_vec.append(torch.zeros(self.num_angle_pars))

        elif(self.rotation_mode=="cayley"):
            desired_param_vec.append(torch.zeros(self.num_cayley_pars))

        if(self.nonlinear_stretch_type=="classic"):
            ## means
            desired_param_vec.append(torch.randn(self.total_param_num_means))

            ## widths
            desired_param_vec.append(torch.ones(self.num_kde*self.dimension)*self.init_log_width)
           
            ## normalization
            if(self.fit_normalization):
                desired_param_vec.append(torch.ones(self.num_kde*self.dimension))

            ## normalization
            if(self.add_skewness):
                desired_param_vec.append(torch.zeros(self.num_kde*self.dimension))
        else:

            # log_widths
            desired_param_vec.append(torch.ones(self.num_kde*self.dimension))

            # log heights
            desired_param_vec.append(torch.ones(self.num_kde*self.dimension))

            # log derivatives
            # 0.54135 corresponds to 1 if input to soft_plus
            desired_param_vec.append(torch.ones((self.num_kde+1)*self.dimension)*0.54135)

            # boundary

            desired_param_vec.append(torch.Tensor(self.dimension*[-1.0, 1.0, -1.0, 1.0]))
            
         
        return torch.cat(desired_param_vec)

    def _init_params(self, params):

        counter=0

        if(self.rotation_mode=="triangular_combination"):
            self.triangle_trafo_pars.data=torch.reshape(params[:self.num_triangle_params], [1, self.num_triangle_params])

            counter+=self.num_triangle_params
        elif(self.rotation_mode=="householder"):
            if self.use_householder:
               
                self.vs.data=torch.reshape(params[:self.num_householder_params], [1, self.householder_iter,self.dimension])

                counter+=self.num_householder_params

        elif(self.rotation_mode=="angles"):

            if(self.dimension>1):
                self.angle_pars.data=torch.reshape(params[:self.num_angle_pars], [1,self.num_angle_pars])
                counter+=self.num_angle_pars

        elif(self.rotation_mode=="cayley"):
            if(self.dimension>1):
                self.cayley_pars.data=torch.reshape(params[:, self.num_cayley_pars], [1, self.num_cayley_pars])
        
        if(self.nonlinear_stretch_type=="classic"):
            # classic gaussianization flow
            self.kde_means.data=torch.reshape(params[counter:counter+self.total_param_num_means], [1,self.num_kde-self.center_mean, self.dimension])
            counter+=self.total_param_num_means

            self.kde_log_widths.data=torch.reshape(params[counter:counter+self.num_params_datapoints], [1,self.num_kde, self.dimension])
            counter+=self.num_params_datapoints

            if(self.fit_normalization):
                self.kde_log_weights.data=torch.reshape(params[counter:counter+self.num_params_datapoints], [1,self.num_kde, self.dimension])
                counter+=self.num_params_datapoints

            if(self.add_skewness):
                self.kde_log_skew_exponents.data=torch.reshape(params[counter:counter+self.num_params_datapoints], [1,self.num_kde, self.dimension])
                counter+=self.num_params_datapoints
        else:
            # rq_splines
            self.log_widths.data=torch.reshape(params[counter:counter+self.num_kde*self.dimension], [1,self.dimension,self.num_kde])
            counter+=self.num_kde*self.dimension

            self.log_heights.data=torch.reshape(params[counter:counter+self.num_kde*self.dimension], [1,self.dimension,self.num_kde])
            counter+=self.num_kde*self.dimension

            self.log_derivatives.data=torch.reshape(params[counter:counter+(self.num_kde+1)*self.dimension], [1,self.dimension,(self.num_kde+1)])
            counter+=(self.num_kde+1)*self.dimension

            self.boundary_points.data=torch.reshape(params[counter:counter+4*self.dimension], [1,self.dimension,4])
            counter+=4*self.dimension

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Debugging function that puts current flow parameters along with their name into "param_dict".
        """

        extra_input_counter=0

        ### rotation stuff
        if(self.rotation_mode=="triangular_combination"):
            if(self.dimension>1):
                if(extra_inputs is None):
                    param_dict[extra_prefix+"trianglepars"]=self.triangle_trafo_pars.data
                else:
                    param_dict[extra_prefix+"trianglepars"]=extra_inputs[:,:self.num_triangle_params]

                    extra_input_counter+=self.num_triangle_params
        elif(self.rotation_mode=="householder"):
            if self.use_householder:
                this_vs=self.vs.reshape(1, -1)
              
                if(extra_inputs is not None):
                    this_vs=this_vs+extra_inputs[:,:self.num_householder_params]

                    extra_input_counter+=self.num_householder_params
                
                param_dict[extra_prefix+"vs"]=this_vs.data

                this_vs=torch.reshape(this_vs, [-1, self.dimension, self.dimension])

                this_vs_determinant=torch.det(self.compute_householder_matrix(this_vs))

                param_dict[extra_prefix+"hh_det"]=this_vs_determinant

        elif(self.rotation_mode=="angles"):
            if(self.dimension>1):
                if(extra_inputs is None):
                    param_dict[extra_prefix+"anglepars"]=self.angle_pars.data
                else:
                    param_dict[extra_prefix+"anglepars"]=extra_inputs[:,:self.num_angle_pars]

                    extra_input_counter+=self.num_angle_pars
        elif(self.rotation_mode=="cayley"):
            if(self.dimension>1):
                if(extra_inputs is None):
                    param_dict[extra_prefix+"cayleypars"]=self.cayley_pars.data
                else:
                    param_dict[extra_prefix+"cayleypars"]=extra_inputs[:,:self.num_cayley_pars]

                    extra_input_counter+=self.num_cayley_pars

        if(self.nonlinear_stretch_type=="classic"):
            ### non rotation stuff
            kde_log_skew_exponents=self.kde_log_skew_exponents

            if(extra_inputs is None):

                kde_means=self.kde_means
                kde_log_widths=self.kde_log_widths
                kde_log_weights=self.kde_log_weights
                
            else:
                ## skipping householder params
                

                kde_means=extra_inputs[:,extra_input_counter:extra_input_counter+self.total_param_num_means].reshape(-1, self.num_kde-self.center_mean, self.dimension)
                extra_input_counter+=self.total_param_num_means

                kde_log_widths=extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints].reshape(-1, self.num_kde, self.dimension)
                extra_input_counter+=self.num_params_datapoints

                if(self.fit_normalization):
                    kde_log_weights=extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints].reshape(-1,self.num_kde, self.dimension)
                    extra_input_counter+=self.num_params_datapoints

                if(self.add_skewness):

                    kde_log_skew_exponents=extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_datapoints].reshape(-1,self.num_kde, self.dimension)
            

            param_dict[extra_prefix+"means"]=kde_means.data
            param_dict[extra_prefix+"log_widths"]=kde_log_widths.data

            if(self.fit_normalization):
                param_dict[extra_prefix+"log_norms"]=kde_log_weights.data

            if(self.add_skewness):
                param_dict[extra_prefix+"exponents"]=kde_log_skew_exponents.data
        else:

            if(extra_inputs is None):

                log_widths=self.log_widths
                log_heights=self.log_heights
                log_derivatives=self.log_derivatives
                boundary_points=self.boundary_points
            else:
                log_widths=extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*self.num_kde].reshape(-1, self.dimension, self.num_kde)
                extra_input_counter+=self.dimension*self.num_kde

                log_heights=extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*self.num_kde].reshape(-1, self.dimension, self.num_kde)
                extra_input_counter+=self.dimension*self.num_kde

                log_derivatives=extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*(self.num_kde+1)].reshape(-1, self.dimension, self.num_kde+1)
                extra_input_counter+=self.dimension*(self.num_kde+1)

                boundary_points=extra_inputs[:,extra_input_counter:extra_input_counter+self.dimension*4].reshape(-1, self.dimension, 4)
            
            param_dict[extra_prefix+"log_widths"]=log_widths.data
            param_dict[extra_prefix+"log_heights"]=log_heights.data
            param_dict[extra_prefix+"log_derivatives"]=log_derivatives.data
            param_dict[extra_prefix+"boundary_points"]=boundary_points.data



