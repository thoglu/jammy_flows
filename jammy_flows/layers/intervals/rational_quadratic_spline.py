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
    def __init__(self, dimension, num_basis_elements=10, euclidean_to_interval_as_first=0, use_permanent_parameters=0, low_boundary=0, high_boundary=1.0, min_width=1e-4, min_height=1e-4, min_derivative=1e-4):
    
        super().__init__(dimension=dimension, euclidean_to_interval_as_first=euclidean_to_interval_as_first, use_permanent_parameters=use_permanent_parameters, low_boundary=low_boundary, high_boundary=high_boundary)
        
        ## add parameters from this layer (on top of householder params defined in the class parent)
        
        self.num_basis_elements=num_basis_elements

        ## widths of intervals in rational spline
        if(use_permanent_parameters):
            self.rel_log_widths=nn.Parameter(torch.randn(self.num_basis_elements).type(torch.double).unsqueeze(0))
            
        else:
            self.rel_log_widths=torch.zeros(self.num_basis_elements).type(torch.double).unsqueeze(0)
      
        ## heights of intervals in rational spline
        if(use_permanent_parameters):
            self.rel_log_heights=nn.Parameter(torch.randn(self.num_basis_elements).type(torch.double).unsqueeze(0))
            
        else:
            self.rel_log_heights=torch.zeros(self.num_basis_elements).type(torch.double).unsqueeze(0)

        ## derivatives
        if(use_permanent_parameters):
            self.rel_log_derivatives=nn.Parameter(torch.randn(self.num_basis_elements+1).type(torch.double).unsqueeze(0))
            
        else:
            self.rel_log_derivatives=torch.zeros(self.num_basis_elements+1).type(torch.double).unsqueeze(0)

        self.total_param_num+=3*self.num_basis_elements+1

        ## minimum values to which relative logarithmic values are added with softmax
     
        self.min_width=min_width
        self.min_height=min_height
        self.min_derivative=min_derivative




    def _flow_mapping(self, inputs, extra_inputs=None): 
        
        [x, log_det]=inputs

      
        widths=self.rel_log_widths.to(x)
        heights=self.rel_log_heights.to(x)
        derivatives=self.rel_log_derivatives.to(x)

        if(extra_inputs is not None):
            
            assert( (extra_inputs.shape[0]==x.shape[0]) or (extra_inputs.shape[0]==1) ), ("Extra inputs must be of shape B X .. (B=Batch size) or 1 X .. (broadcasting).. got for first dimension instead : ", extra_inputs.shape[0])

            widths=extra_inputs[:,:self.num_basis_elements]#.reshape(extra_inputs.shape[0], self.rel_log_widths.shape[1])
            heights=extra_inputs[:,self.num_basis_elements:2*self.num_basis_elements]#.reshape(extra_inputs.shape[0], self.rel_log_heights.shape[1])
            derivatives=extra_inputs[:,2*self.num_basis_elements:]#.reshape(extra_inputs.shape[0], self.rel_log_derivatives.shape[1])
            

       
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
                                                         min_derivative=self.min_derivative
                                                         )
        
       
        log_det_new=log_det+log_det_update.sum(axis=-1)
        
        return x, log_det_new

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x, log_det]=inputs

        widths=self.rel_log_widths.to(x)
        heights=self.rel_log_heights.to(x)
        derivatives=self.rel_log_derivatives.to(x)

      

        if(extra_inputs is not None):

            assert( (extra_inputs.shape[0]==x.shape[0]) or (extra_inputs.shape[0]==1) ), ("Extra inputs must be of shape B X .. (B=Batch size) or 1 X .. (broadcasting).. got for first dimension instead : ", extra_inputs.shape[0])

            widths=extra_inputs[:,:self.num_basis_elements]#.reshape(extra_inputs.shape[0], self.rel_log_widths.shape[1])
            heights=extra_inputs[:,self.num_basis_elements:2*self.num_basis_elements]#.reshape(extra_inputs.shape[0], self.rel_log_heights.shape[1])
            derivatives=extra_inputs[:,2*self.num_basis_elements:]#.reshape(extra_inputs.shape[0], self.rel_log_derivatives.shape[1])
            
           
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
                                                         min_derivative=self.min_derivative
                                                         )
       
        log_det_new=log_det+log_det_update.sum(axis=-1)
      
        return x, log_det_new


    def _get_desired_init_parameters(self):

        ## widths

        desired_param_vec=[]

        ## 0.54 as start value in log space seems to result in a rather flat spline (defined by the log-derivatives in particular)
        desired_param_vec.append(torch.ones(self.num_basis_elements*3+1)*0.54)

        return torch.cat(desired_param_vec)

    

    def _init_params(self, params):

        counter=0
        self.rel_log_widths.data[0,:]=params[counter:counter+self.num_basis_elements]    

        counter+=self.num_basis_elements
        self.rel_log_heights.data[0,:]=params[counter:counter+self.num_basis_elements]

        counter+=self.num_basis_elements
        self.rel_log_derivatives.data[0,:]=params[counter:counter+self.num_basis_elements+1]

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 


        extra_input_counter=0

        widths=self.rel_log_widths
        heights=self.rel_log_heights
        derivatives=self.rel_log_derivatives

        if(extra_inputs is not None):
            
            widths=widths+extra_inputs[:,:self.num_basis_elements]
            heights=heights+extra_inputs[:,self.num_basis_elements:2*self.num_basis_elements]
            derivatives=derivatives+extra_inputs[:,2*self.num_basis_elements:]


        param_dict[extra_prefix+"widths"]=widths
        param_dict[extra_prefix+"heights"]=heights
        param_dict[extra_prefix+"derivatives"]=derivatives

    
       