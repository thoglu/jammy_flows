import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from .. import spline_fns
from . import interval_base



"""
This layer implements neural spline flows (https://arxiv.org/abs/1906.04032) via monotonic rational-quadratic splines. 
It is an adaptation from the offical github implementation at https://github.com/bayesiains/nsf.
"""


"""
def unconstrained_rational_quadratic_spline(inputs,
                                            unnormalized_widths,
                                            unnormalized_heights,
                                            unnormalized_derivatives,
                                            inverse=False,
                                            tails='linear',
                                            tail_bound=1.,
                                            rel_min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            rel_min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        rel_min_bin_width=rel_min_bin_width,
        rel_min_bin_height=rel_min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet
"""



class rational_quadratic_spline(interval_base.interval_base):
    def __init__(self, 
                 dimension, 
                 num_basis_functions=10, 
                 euclidean_to_interval_as_first=0, 
                 use_permanent_parameters=0, 
                 low_boundary=0, 
                 high_boundary=1.0, 
                 min_width=1e-4, 
                 min_height=1e-4, 
                 min_derivative=1e-4,
                 fix_boundary_derivatives=-1.0,
                 smooth_second_derivative=0,
                 restrict_max_min_width_height_ratio=-1.0):
        """
        Rational-quadartic spline layer: Symbol “r"

        Also known as Neural spline flows - https://arxiv.org/abs/1906.04032. Adapted code from pyro implementation.
        
        Parameters:
            num_basis_functions (int): Number of spline elements.
            low_boundary (float): Lower boundary of the inverval.
            high_boundary (float): Higher boundary of the interval.
            min_width (float): Minimum width of a spline element.
            min_height (float): Minimum height of a spline element.
            min_derivative (float): Minimum derivative of a spline element. Only used if *smooth_second_derivative* is 0.
            fix_boundary_derivatives (float): If larger than 0, determines the value of the boundary derivatives
            smooth_second_derivative (int): Determines if second derivatives should be smooth. Only works for 2 basis function currently. Ignores *min_derivative* if positive.
            restrict_max_min_width_height_ratio (float): Maximum relative between maximum/miniumum of widths/heights.. negative will not constrain this quantity. 

        """
        super().__init__(dimension=dimension, euclidean_to_interval_as_first=euclidean_to_interval_as_first, use_permanent_parameters=use_permanent_parameters, low_boundary=low_boundary, high_boundary=high_boundary)
        
        ## add parameters from this layer (on top of householder params defined in the class parent)
        
        self.num_basis_functions=num_basis_functions

        ## widths of intervals in rational spline
        if(use_permanent_parameters):
            self.rel_log_widths=nn.Parameter(torch.randn(self.num_basis_functions).type(torch.double).unsqueeze(0))
            
        ## heights of intervals in rational spline
        if(use_permanent_parameters):
            self.rel_log_heights=nn.Parameter(torch.randn(self.num_basis_functions).type(torch.double).unsqueeze(0))
        
        ## derivatives
        self.fix_boundary_derivatives=fix_boundary_derivatives
        self.deriv_num_bd_subtraction=0
        self.boundary_log_derivs_fixed_value=None
        if(self.fix_boundary_derivatives>0.0):
            ## recalculate what we would need to feed the spline to correctly get 1.0 via *SoftPlus*
            self.boundary_log_derivs_fixed_value=np.log(np.exp(self.fix_boundary_derivatives-min_derivative)-1.0)
        
        self.smooth_second_derivative=smooth_second_derivative
        if(self.smooth_second_derivative==1):
            assert(self.num_basis_functions==2), "Only support 2 basis functions for smooth derivative!"
            #assert(self.fix_boundary_derivatives>0.0), "For smooth derivatives we need to define boundary derivatives!"

            if(self.fix_boundary_derivatives>0.0):
                self.deriv_num_bd_subtraction=num_basis_functions+1 # subtract all derivs
            else:
                self.deriv_num_bd_subtraction=num_basis_functions-1 # subtract all derivs but 2
        else:
            if(self.fix_boundary_derivatives>0.0):
                ## we only need to define this if we have no smooth derivatives
                self.deriv_num_bd_subtraction=2

                assert(self.fix_boundary_derivatives>min_derivative)

        self.num_derivative_params=num_basis_functions+1-self.deriv_num_bd_subtraction
                
        ## only use derivatives if necessary
        if(use_permanent_parameters):
            if( self.num_derivative_params>0):
                self.rel_log_derivatives=nn.Parameter(torch.randn(self.num_basis_functions+1-self.deriv_num_bd_subtraction).type(torch.double).unsqueeze(0))
        
        ## subtract number of derivative parameters as we need
        self.total_param_num+=3*self.num_basis_functions+1-self.deriv_num_bd_subtraction

        ## minimum values to which relative logarithmic values are added with softmax
     
        self.min_width=min_width
        self.min_height=min_height
        self.min_derivative=min_derivative
        self.restrict_max_min_width_height_ratio=restrict_max_min_width_height_ratio

    def _flow_mapping(self, inputs, extra_inputs=None): 
        
        [x, log_det]=inputs

        if(self.use_permanent_parameters):
            widths=self.rel_log_widths.to(x)
            heights=self.rel_log_heights.to(x)
            derivatives=None
            if(self.num_derivative_params>0):
                derivatives=self.rel_log_derivatives.to(x)

        else:
            
            assert( extra_inputs is not None), "Conditional PDF.. require *extra_inputs*"
            assert( (extra_inputs.shape[0]==x.shape[0]) or (extra_inputs.shape[0]==1) ), ("Extra inputs must be Tensor of shape B X .. (B=Batch size) or 1 X .. (broadcasting).. got for first dimension instead : ", extra_inputs.shape[0])

            widths=extra_inputs[:,:self.num_basis_functions]#.reshape(extra_inputs.shape[0], self.rel_log_widths.shape[1])
            heights=extra_inputs[:,self.num_basis_functions:2*self.num_basis_functions]#.reshape(extra_inputs.shape[0], self.rel_log_heights.shape[1])
            
            if(self.num_derivative_params>0):
                derivatives=extra_inputs[:,2*self.num_basis_functions:]#.reshape(extra_inputs.shape[0], self.rel_log_derivatives.shape[1])
        
        if(self.smooth_second_derivative==0):

            if(self.fix_boundary_derivatives>0):
                ## add fixed log derivatives to first and last of derivatives tensor .. derivatives must exist here
                
                assert(self.deriv_num_bd_subtraction==2)

                first_and_last=torch.ones(derivatives.shape[:-1]+(1,)).to(x)*self.boundary_log_derivs_fixed_value

                derivatives=torch.cat([first_and_last, derivatives, first_and_last], dim=-1)

            x, log_det_update=spline_fns.rational_quadratic_spline(x, 
                                                             widths, 
                                                             heights, 
                                                             derivatives, 
                                                             inverse=False, 
                                                             left=self.low_boundary, 
                                                             right=self.high_boundary, 
                                                             bottom=self.low_boundary, 
                                                             top=self.high_boundary, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             min_derivative=self.min_derivative,
                                                             restrict_max_min_width_height_ratio=self.restrict_max_min_width_height_ratio
                                                             )
        else:

            if(self.fix_boundary_derivatives>0):

                first_and_last=torch.ones(widths.shape[:-1]+(2,)).to(x)*self.boundary_log_derivs_fixed_value
            else:
                assert(self.num_derivative_params==2 and derivatives is not None)

                first_and_last=derivatives

            x, log_det_update=spline_fns.rational_quadratic_spline_smooth(x, 
                                                             widths, 
                                                             heights, 
                                                             unnormalized_boundary_derivatives=first_and_last,
                                                             inverse=False, 
                                                             left=self.low_boundary, 
                                                             right=self.high_boundary, 
                                                             bottom=self.low_boundary, 
                                                             top=self.high_boundary, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             min_derivative=self.min_derivative,
                                                             restrict_max_min_width_height_ratio=self.restrict_max_min_width_height_ratio
                                                             )
       
        log_det_new=log_det+log_det_update.sum(axis=-1)
        
        return x, log_det_new

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x, log_det]=inputs

        if(self.use_permanent_parameters):
            widths=self.rel_log_widths.to(x)
            heights=self.rel_log_heights.to(x)
            derivatives=None
            if(self.num_derivative_params>0):
                derivatives=self.rel_log_derivatives.to(x)

        else:
            
            assert( extra_inputs is not None), "Conditional PDF.. require *extra_inputs*"
            assert( (extra_inputs.shape[0]==x.shape[0]) or (extra_inputs.shape[0]==1) ), ("Extra inputs must be Tensor of shape B X .. (B=Batch size) or 1 X .. (broadcasting).. got for first dimension instead : ", extra_inputs.shape[0])

            widths=extra_inputs[:,:self.num_basis_functions]#.reshape(extra_inputs.shape[0], self.rel_log_widths.shape[1])
            heights=extra_inputs[:,self.num_basis_functions:2*self.num_basis_functions]#.reshape(extra_inputs.shape[0], self.rel_log_heights.shape[1])
            
            if(self.num_derivative_params>0):
                derivatives=extra_inputs[:,2*self.num_basis_functions:]#.reshape(extra_inputs.shape[0], self.rel_log_derivatives.shape[1])
        
        if(self.smooth_second_derivative==0):

            if(self.fix_boundary_derivatives>0):
                ## add fixed log derivatives to first and last of derivatives tensor .. derivatives must exist here
                
                assert(self.deriv_num_bd_subtraction==2)

                first_and_last=torch.ones(derivatives.shape[:-1]+(1,)).to(x)*self.boundary_log_derivs_fixed_value

                derivatives=torch.cat([first_and_last, derivatives, first_and_last], dim=-1)

            x, log_det_update=spline_fns.rational_quadratic_spline(x, 
                                                             widths, 
                                                             heights, 
                                                             derivatives, 
                                                             inverse=True, 
                                                             left=self.low_boundary, 
                                                             right=self.high_boundary, 
                                                             bottom=self.low_boundary, 
                                                             top=self.high_boundary, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             min_derivative=self.min_derivative,
                                                             restrict_max_min_width_height_ratio=self.restrict_max_min_width_height_ratio
                                                             )
        else:

            if(self.fix_boundary_derivatives>0):

                first_and_last=torch.ones(widths.shape[:-1]+(2,)).to(x)*self.boundary_log_derivs_fixed_value

            else:
                assert(self.num_derivative_params==2 and derivatives is not None)

                first_and_last=derivatives
                
            x, log_det_update=spline_fns.rational_quadratic_spline_smooth(x, 
                                                             widths, 
                                                             heights, 
                                                             unnormalized_boundary_derivatives=first_and_last,
                                                             inverse=True, 
                                                             left=self.low_boundary, 
                                                             right=self.high_boundary, 
                                                             bottom=self.low_boundary, 
                                                             top=self.high_boundary, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             min_derivative=self.min_derivative,
                                                             restrict_max_min_width_height_ratio=self.restrict_max_min_width_height_ratio
                                                             )
       
        log_det_new=log_det+log_det_update.sum(axis=-1)
      
        return x, log_det_new


    def _get_desired_init_parameters(self):

        ## widths

        desired_param_vec=[]

        ## 0.54 as start value in log space seems to result in a rather flat spline (defined by the log-derivatives in particular)
        desired_param_vec.append(torch.ones(self.num_basis_functions*3+1-self.deriv_num_bd_subtraction)*0.54)

        return torch.cat(desired_param_vec)

    

    def _init_params(self, params):

        counter=0
        self.rel_log_widths.data[0,:]=params[counter:counter+self.num_basis_functions]    
        counter+=self.num_basis_functions
        
        self.rel_log_heights.data[0,:]=params[counter:counter+self.num_basis_functions]
        counter+=self.num_basis_functions

        if(self.num_derivative_params>0):
            self.rel_log_derivatives.data[0,:]=params[counter:counter+self.num_derivative_params]

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 

        extra_input_counter=0

        if(self.use_permanent_parameters):
            widths=self.rel_log_widths
            heights=self.rel_log_heights
            if(self.num_derivative_params>0):
                derivatives=self.rel_log_derivatives
        else:
            assert(extra_inputs is not None)
            
            widths=extra_inputs[:,:self.num_basis_functions]
            heights=extra_inputs[:,self.num_basis_functions:2*self.num_basis_functions]
            if(self.num_derivative_params>0):
                derivatives=extra_inputs[:,2*self.num_basis_functions:]

        param_dict[extra_prefix+"widths"]=widths
        param_dict[extra_prefix+"heights"]=heights

        if(self.smooth_second_derivative==0):
            param_dict[extra_prefix+"derivatives"]=derivatives

    
       