import torch
from torch import nn
from torch.nn import functional as F

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs >= bin_locations,
        dim=-1,
        keepdims=True
    ) - 1

def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              rel_min_bin_width=1e-3,
                              rel_min_bin_height=1e-3,
                              min_derivative=1e-3):

        
        if torch.min(inputs) < left or torch.max(inputs) > right:
           
            raise Exception("outside boundaries in rational-spline flow! (min/max (%.2f/%.2f), allowed: (%.2f/%.2f)" % (torch.min(inputs), torch.max(inputs), left, right))

        num_bins = unnormalized_widths.shape[-1]

        if rel_min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if rel_min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')
        
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = rel_min_bin_width + (1 - rel_min_bin_width * num_bins) * widths
     
        cumwidths = torch.cumsum(widths, dim=-1)
        
        cumwidths = F.pad(cumwidths, pad=(1,0), mode='constant', value=0.0)

        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]
        
        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = rel_min_bin_height + (1 - rel_min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]


        
        if inverse:
            bin_idx = searchsorted(cumheights, inputs)#[..., None]
        else:
            bin_idx = searchsorted(cumwidths, inputs)#[..., None]
        
        if(cumwidths.shape[0]==1 and bin_idx.shape[0]>1):
          repeats=[bin_idx.shape[0]]+(len(cumwidths.shape)-1)*[1]
          
          cumwidths=cumwidths.repeat(repeats)
          widths=widths.repeat(repeats)
          heights=heights.repeat(repeats)
          cumheights=cumheights.repeat(repeats)
          derivatives=derivatives.repeat(repeats)
        
        input_cumwidths = cumwidths.gather(-1, bin_idx)#[..., 0]


        
        input_bin_widths = widths.gather(-1, bin_idx)#[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)#[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)#[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)#[..., 0]
        

        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)#[..., 0]
       
        input_heights = heights.gather(-1, bin_idx)#[..., 0]

      
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

def rational_quadratic_spline_extended_to_line(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=torch.DoubleTensor([[[0.0]]]), right=torch.DoubleTensor([[[1.0]]]), bottom=torch.DoubleTensor([[[0.0]]]), top=torch.DoubleTensor([[[1.0]]]),
                              rel_min_bin_width=1e-3,
                              rel_min_bin_height=1e-3,
                              min_derivative=1e-3):

        ## this function only works for 2-d inputs currently (batch-size X dim)
       
        # batchsize X dim 
        assert(len(inputs.shape)==2)

        # all parameters must come in 1 X dim X num_bins or batchsize X dim X num_bins/num_splines (3-tensor)
        assert(len(unnormalized_widths.shape)==3)
        assert(len(unnormalized_heights.shape)==3)
        assert(len(unnormalized_derivatives.shape)==3)

        assert(len(left.shape)==3)
        assert(len(right.shape)==3)
        assert(len(top.shape)==3)
        assert(len(bottom.shape)==3)

        
        """
        if torch.min(inputs) < left or torch.max(inputs) > right:
           
            raise Exception("outside boundaries in rational-spline flow! (min/max (%.2f/%.2f), allowed: (%.2f/%.2f)" % (torch.min(inputs), torch.max(inputs), left, right))
        """

        num_bins = unnormalized_widths.shape[-1]

        if rel_min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal rel bin width too large for the number of bins')
        if rel_min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal rel bin height too large for the number of bins')
        
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = rel_min_bin_width + (1 - rel_min_bin_width * num_bins) * widths
     
        cumwidths = torch.cumsum(widths, dim=-1)
        
        cumwidths = F.pad(cumwidths, pad=(1,0), mode='constant', value=0.0)

        cumwidths = (right - left) * cumwidths + left
        #cumwidths[..., 0] = left
        #cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = rel_min_bin_height + (1 - rel_min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        #cumheights[..., 0] = bottom
        #cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

       
        
        if inverse:
            bin_idx = searchsorted(cumheights, inputs.unsqueeze(-1))#[..., None]
        else:
            bin_idx = searchsorted(cumwidths, inputs.unsqueeze(-1))#[..., None]


        """
        if(cumwidths.shape[0]==1 and bin_idx.shape[0]>1):
          
          cumwidths=cumwidths.repeat(bin_idx.shape[0], 1)
          widths=widths.repeat(bin_idx.shape[0], 1)
          heights=heights.repeat(bin_idx.shape[0], 1)
          cumheights=cumheights.repeat(bin_idx.shape[0], 1)
          derivatives=derivatives.repeat(bin_idx.shape[0], 1)
        """   

       
        input_cumwidths = cumwidths.gather(-1, bin_idx)#[..., 0]

        
        input_bin_widths = widths.gather(-1, bin_idx)#[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)#[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)#[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)#[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)#[..., 0]
 
        input_heights = heights.gather(-1, bin_idx)#[..., 0]

      
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