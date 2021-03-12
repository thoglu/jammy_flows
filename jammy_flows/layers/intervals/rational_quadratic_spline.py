import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from . import interval_base

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

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
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
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
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet
"""



class rational_quadratic_spline(interval_base.interval_base):
    def __init__(self, dimension, num_basis_elements=10, euclidean_to_interval_as_first=0, use_permanent_parameters=0, low_boundary=0, high_boundary=1.0, min_width=1e-3, min_height=1e-3, min_derivative=1e-3):
    
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
            self.rel_log_derivatives=nn.Parameter(torch.randn(self.num_basis_elements).type(torch.double).unsqueeze(0))
            
        else:
            self.rel_log_derivatives=torch.zeros(self.num_basis_elements).type(torch.double).unsqueeze(0)

        self.total_param_num+=3*self.num_basis_elements

        ## minimum values to which relative logarithmic values are added with softmax
     
        self.min_width=min_width
        self.min_height=min_height
        self.min_derivative=min_derivative


    def rational_quadratic_spline(self,
                              inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=1e-3,
                              min_bin_height=1e-3,
                              min_derivative=1e-3):
        print("inputs ", inputs)
        print(inputs.max())
        print(inputs.min())
        if torch.min(inputs) < left or torch.max(inputs) > right:
            raise Exception("outside boundaries in rational-spline flow!")

        num_bins = unnormalized_widths.shape[-1]

        if min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if inverse:
            bin_idx = searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = searchsorted(cumwidths, inputs)[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if inverse:
            a = (((inputs - input_cumheights) * (input_derivatives
                                                 + input_derivatives_plus_one
                                                 - 2 * input_delta)
                  + input_heights * (input_delta - input_derivatives)))
            b = (input_heights * input_derivatives
                 - (inputs - input_cumheights) * (input_derivatives
                                                  + input_derivatives_plus_one
                                                  - 2 * input_delta))
            c = - input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                         * theta_one_minus_theta)
            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                         + 2 * input_delta * theta_one_minus_theta
                                                         + input_derivatives * (1 - root).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return outputs, -logabsdet
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (input_delta * theta.pow(2)
                                         + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                         * theta_one_minus_theta)
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                         + 2 * input_delta * theta_one_minus_theta
                                                         + input_derivatives * (1 - theta).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return outputs, logabsdet

    def _flow_mapping(self, inputs, extra_inputs=None,): 
        
        [x, log_det]=inputs

        widths=self.rel_log_widths.to(x)
        heights=self.rel_log_heights.to(x)
        derivatives=self.rel_log_derivatives.to(x)

        if(extra_inputs is not None):
            widths=widths+extra_inputs[:,self.num_basis_elements].reshape(x.shape[0], self.rel_log_widths.shape[1])
            heights=heights+extra_inputs[:,self.num_basis_elements].reshape(x.shape[0], self.rel_log_heights.shape[1])
            derivatives=derivatives+extra_inputs[:,self.num_basis_elements].reshape(x.shape[0], self.rel_log_derivatives.shape[1])
        
        print("before the actual mapping ")
        print(x)
        x, log_det_update=self.rational_quadratic_spline(x, 
                                                         widths, 
                                                         heights, 
                                                         derivatives, 
                                                         inverse=False, 
                                                         left=self.low_boundary, 
                                                         right=self.high_boundary, 
                                                         bottom=self.low_boundary, 
                                                         top=self.high_boundary, 
                                                         min_bin_width=self.min_width,
                                                         min_bin_height=self.min_height,
                                                         min_derivative=self.min_derivative
                                                         )

        log_det+=log_det_update

        return x, log_det

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        [x, log_det]=inputs

        widths=self.rel_log_widths.to(x)
        heights=self.rel_log_heights.to(x)
        derivatives=self.rel_log_derivatives.to(x)

        if(extra_inputs is not None):
            widths=widths+extra_inputs[:,self.num_basis_elements].reshape(x.shape[0], self.rel_log_widths.shape[1])
            heights=heights+extra_inputs[:,self.num_basis_elements].reshape(x.shape[0], self.rel_log_heights.shape[1])
            derivatives=derivatives+extra_inputs[:,self.num_basis_elements].reshape(x.shape[0], self.rel_log_derivatives.shape[1])
        
        x, log_det_update=self.rational_quadratic_spline(x, 
                                                         widths, 
                                                         heights, 
                                                         derivatives, 
                                                         inverse=True, 
                                                         left=self.low_boundary, 
                                                         right=self.high_boundary, 
                                                         bottom=self.low_boundary, 
                                                         top=self.high_boundary, 
                                                         min_bin_width=self.min_width,
                                                         min_bin_height=self.min_height,
                                                         min_derivative=self.min_derivative
                                                         )

        log_det+=log_det_update

        return x, log_det


    def _get_desired_init_parameters(self):

        ## widths

        desired_param_vec=[]

        desired_param_vec.append(torch.ones(self.num_basis_elements*3))

        return torch.cat(desired_param_vec)

    def _init_params(self, params):

        counter=0
        self.rel_log_widths.data=params[counter:counter+self.num_basis_elements]    

        counter+=self.num_basis_elements
        self.rel_log_widths.data=params[counter:counter+self.num_basis_elements]

        counter+=self.num_basis_elements
        self.rel_log_widths.data=params[counter:counter+self.num_basis_elements]