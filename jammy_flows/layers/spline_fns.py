import torch
from torch import nn
from torch.nn import functional as F

import numpy

sympy=None
try:
    import sympy
except:
    print("Sympy not installed!")

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs >= bin_locations,
        dim=-1,
        keepdims=True
    ) - 1


def return_safe_angle_within_2pi(x, safety_margin=1e-7):
    """
    Restricts the angle to not hit 0 or pi exactly.
    """

    used_safety_margin=safety_margin
    if(safety_margin is None):
        if(x.dtype==torch.float32):
            used_safety_margin=1e-7
        elif(x.dtype==torch.float64):
            used_safety_margin=1e-10
    
    upper_bound=2*numpy.pi-safety_margin
    lower_bound=0.0+safety_margin

    small_mask=x<lower_bound
    large_mask=x>upper_bound
   
    ret=torch.where(small_mask, lower_bound, x)
    ret=torch.where(large_mask, upper_bound, ret)

    return ret

def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              rel_min_bin_width=1e-3,
                              rel_min_bin_height=1e-3,
                              min_derivative=1e-3,
                              restrict_max_min_width_height_ratio=-1.0):

        
        if torch.min(inputs) < left or torch.max(inputs) > right:
           
            raise Exception("outside boundaries in rational-spline flow! (min/max (%.2f/%.2f), allowed: (%.2f/%.2f)" % (torch.min(inputs), torch.max(inputs), left, right))

        num_bins = unnormalized_widths.shape[-1]

        if rel_min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if rel_min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')
        
        ## calculate parameter transformation of allowed range

        ## (min_allowed)/((tot-1)*(max_allowed) = 1.0/actual_ratio
        ## actual_ratio= (tot-1)*max_allowed/min_allowed
        ## log(actual_ratio) = log(tot-1)+log(max_allowed)-log(min_allowed)
        ## --> symmetric: log(max_allowed)=-log(min_allowed)
        ## --> log(max_allowed)=(log(actual_ratio)-log(tot-1))/2.0

        used_unnormalized_widths=unnormalized_widths
        used_unnormalized_heights=unnormalized_heights

        ## check for restriction in width/heigth ratios
        if(restrict_max_min_width_height_ratio>0.0):
            ln_max_allowed=(numpy.log(restrict_max_min_width_height_ratio)-numpy.log(num_bins-1))/2.0
            
            assert(ln_max_allowed>0), "Allowed max/min ratio for widths/heights is too small.. %.3e .. require at least %.3e" % (restrict_max_min_width_height_ratio, numpy.exp(numpy.log(num_bins-1)/2.0))
            used_unnormalized_widths=2.0*F.sigmoid(unnormalized_widths)*ln_max_allowed-ln_max_allowed
            used_unnormalized_heights=2.0*F.sigmoid(unnormalized_heights)*ln_max_allowed-ln_max_allowed
            

        widths = F.softmax(used_unnormalized_widths, dim=-1)
        widths = rel_min_bin_width + (1 - rel_min_bin_width * num_bins) * widths
     
        cumwidths = torch.cumsum(widths, dim=-1)
        
        cumwidths = F.pad(cumwidths, pad=(1,0), mode='constant', value=0.0)

        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]
       
        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(used_unnormalized_heights, dim=-1)
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

def rational_quadratic_spline_with_linear_extension(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=torch.DoubleTensor([[[0.0]]]), right=torch.DoubleTensor([[[1.0]]]), bottom=torch.DoubleTensor([[[0.0]]]), top=torch.DoubleTensor([[[1.0]]]),
                              rel_min_bin_width=1e-3,
                              rel_min_bin_height=1e-3,
                              min_derivative=1e-3):

        
        assert(len(inputs.shape)==3)
        assert(inputs.shape[2]==1)

        # all parameters must come in 1 X dim X num_bins or batchsize X dim X num_bins/num_splines (3-tensor)
        assert(len(unnormalized_widths.shape)==3)
        assert(len(unnormalized_heights.shape)==3)
        assert(len(unnormalized_derivatives.shape)==3)

        assert(len(left.shape)==3)
        assert(len(right.shape)==3)
        assert(len(top.shape)==3)
        assert(len(bottom.shape)==3)
        
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
        
        #cumwidths[..., 0:1] = left
        #cumwidths[..., -1:] = right

        #sys.exit(-1)

        
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]


        derivatives = min_derivative + F.softplus(unnormalized_derivatives)
       
        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = rel_min_bin_height + (1 - rel_min_bin_height * num_bins) * heights

       
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        #cumheights[..., 0:1] = bottom
        #cumheights[..., -2:-1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if inverse:
            bin_idx = searchsorted(cumheights, inputs,eps=0.0)#[..., None]
        else:
            bin_idx = searchsorted(cumwidths, inputs,eps=0.0)#[..., None]

        ## keep bin idx sane
        bin_idx=torch.where(bin_idx<0, 0, bin_idx)
        bin_idx=torch.where(bin_idx>=heights.shape[-1], heights.shape[-1]-1, bin_idx)

        if(cumwidths.shape[0]==1 and bin_idx.shape[0]>1):

          repeats=[bin_idx.shape[0]]+(len(cumwidths.shape)-1)*[1]
          
          cumwidths=cumwidths.repeat(repeats)
          widths=widths.repeat(repeats)
          heights=heights.repeat(repeats)
          cumheights=cumheights.repeat(repeats)
          derivatives=derivatives.repeat(repeats)

          repeats=[bin_idx.shape[0]]+(len(left.shape)-1)*[1]

          left=left.repeat(repeats)
          right=right.repeat(repeats)
          top=top.repeat(repeats)
          bottom=bottom.repeat(repeats)

        
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
            #assert (discriminant >= 0).all(), (inputs[discriminant<0], input_cumwidths[discriminant<0], input_cumheights[discriminant<0], input_bin_widths[discriminant<0], input_heights[discriminant<0],bin_idx[discriminant<0],discriminant[discriminant<0], a[discriminant<0], b[discriminant<0], c[discriminant<0], a,b,c )

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                         * theta_one_minus_theta)
            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                         + 2 * input_delta * theta_one_minus_theta
                                                         + input_derivatives * (1 - root).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            ## fill in linear bits
            left_offset=cumwidths[...,0:1]-cumheights[..., 0:1]/derivatives[...,0:1]
            outputs=torch.where(inputs<=bottom, inputs/derivatives[...,0:1]+left_offset, outputs)

            right_offset=cumwidths[...,-1:]-cumheights[..., -1:]/derivatives[...,-1:]
            outputs=torch.where(inputs>=top, inputs/derivatives[...,-1:]+right_offset, outputs)

            final_logabsdet=-logabsdet
            final_logabsdet=torch.where(inputs<=bottom, -torch.log(derivatives[...,0:1]), final_logabsdet)
            final_logabsdet=torch.where(inputs>=top, -torch.log(derivatives[...,-1:]), final_logabsdet)

            return outputs, final_logabsdet
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

            ## fill in linear bits
            
            left_offset=cumheights[...,0:1]-cumwidths[..., 0:1]*derivatives[...,0:1]

            outputs=torch.where(inputs<=left, inputs*derivatives[...,0:1]+left_offset, outputs)

            right_offset=cumheights[...,-1:]-cumwidths[..., -1:]*derivatives[...,-1:]
            outputs=torch.where(inputs>=right, inputs*derivatives[...,-1:]+right_offset, outputs)
            
            logabsdet=torch.where(inputs<=left, torch.log(derivatives[...,0:1]), logabsdet)
            logabsdet=torch.where(inputs>=right, torch.log(derivatives[...,-1:]), logabsdet)

            return outputs, logabsdet


def rational_quadratic_spline_smooth(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_boundary_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              rel_min_bin_width=1e-3,
                              rel_min_bin_height=1e-3,
                              min_derivative=1e-3,
                              restrict_max_min_width_height_ratio=-1.0,
                              solution_index=0):


        if torch.min(inputs) < left or torch.max(inputs) > right:
           
            raise Exception("outside boundaries in rational-spline flow! (min/max (%.20e/%.20e), allowed: (%.6e/%.6e)" % (torch.min(inputs), torch.max(inputs), left, right))

        num_bins = unnormalized_widths.shape[-1]

        if rel_min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if rel_min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')
        
        assert(unnormalized_boundary_derivatives.shape[-1]==2), unnormalized_boundary_derivatives.shape

        ## calculate parameter transformation of allowed range

        ## (min_allowed)/((tot-1)*(max_allowed) = 1.0/actual_ratio
        ## actual_ratio= (tot-1)*max_allowed/min_allowed
        ## log(actual_ratio) = log(tot-1)+log(max_allowed)-log(min_allowed)
        ## --> symmetric: log(max_allowed)=-log(min_allowed)
        ## --> log(max_allowed)=(log(actual_ratio)-log(tot-1))/2.0

        used_unnormalized_widths=unnormalized_widths
        used_unnormalized_heights=unnormalized_heights

        ## check for restriction in width/heigth ratios
        if(restrict_max_min_width_height_ratio>0.0):
            ln_max_allowed=(numpy.log(restrict_max_min_width_height_ratio)-numpy.log(num_bins-1))/2.0
            
            assert(ln_max_allowed>0), "Allowed max/min ratio for widths/heights is too small.. %.3e .. require at least %.3e" % (restrict_max_min_width_height_ratio, numpy.exp(numpy.log(num_bins-1)/2.0))
            used_unnormalized_widths=2.0*F.sigmoid(unnormalized_widths)*ln_max_allowed-ln_max_allowed
            used_unnormalized_heights=2.0*F.sigmoid(unnormalized_heights)*ln_max_allowed-ln_max_allowed
        
        
        widths = F.softmax(used_unnormalized_widths, dim=-1)
        widths = rel_min_bin_width + (1 - rel_min_bin_width * num_bins) * widths
       
        cumwidths = torch.cumsum(widths, dim=-1)
        ## padding should be done effectively outside later
        cumwidths = F.pad(cumwidths, pad=(1,0), mode='constant', value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]
        
        heights = F.softmax(used_unnormalized_heights, dim=-1)
        heights = rel_min_bin_height + (1 - rel_min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)

        ## padding should be done effectively outside later
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        boundary_derivatives=min_derivative + F.softplus(unnormalized_boundary_derivatives)
        # if we have more than 1 basis function we need to look at correlations
        if(widths.shape[-1]>1):

            if(widths.shape[-1]==2):

                height_sum=heights[...,:-1]+heights[...,1:]
                lower_height_prob=heights[...,:-1]/height_sum
                higher_height_prob=heights[...,1:]/height_sum

                height_product=heights[...,:-1]*heights[...,1:]

                ####
                #### TODO: check order of boundary derivatives here!!!!!
                neg_p_half=0.5*( lower_height_prob* (  (heights[...,1:]/widths[...,1:]) - boundary_derivatives[...,1:]) + higher_height_prob * ( (heights[...,:-1]/widths[...,:-1]) - boundary_derivatives[...,:-1]) )
                
                q=-(heights[...,:-1]*heights[...,1:])*(lower_height_prob*( 1.0 /widths[...,:-1]**2) + higher_height_prob*( 1.0 /widths[...,1:]**2))
               
                if(solution_index==0):
                    res=neg_p_half+(neg_p_half**2-q).sqrt()
                else:
                    res=neg_p_half-(neg_p_half**2-q).sqrt()

                
                derivatives=torch.cat([boundary_derivatives[...,:1], res, boundary_derivatives[...,1:]], dim=-1)

            elif(widths.shape[-1]==3):
                ## symmetric solution only
                ## make sure index 0 == index 2
                #assert( (widths[...,0]!=widths[...,2]).sum()==0), (widths[...,0], widths[...,2], widths[...,0]-widths[...,2])
                #assert( (heights[...,0]!=heights[...,2]).sum()==0), (heights[...,0], heights[...,2],heights[...,0]-heights[...,2])

                w_1=widths[...,0:1]
                w_2=widths[...,1:2]
                h_1=heights[...,0:1]
                h_2=heights[...,1:2]

                common_denominator=w_1*w_2*(2*h_1+h_2)

                p=h_2*(boundary_derivatives[...,:1]*w_1*w_2-h_1*(w_1+w_2))/common_denominator
                q=-h_1*h_2*(h_1*w_2**2+h_2*w_1**2)/(common_denominator*w_1*w_2)

                neg_p_half=-p/2.0

                # always + solution
                res=neg_p_half+torch.sqrt(neg_p_half**2-q)

               
                # add two symmetric derivatives in the middle
                derivatives=torch.cat([boundary_derivatives[...,:1], res, res, boundary_derivatives[...,1:]], dim=-1)

            else:

                raise NotImplementedError()
        else:
            derivatives=boundary_derivatives
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

            return outputs, -logabsdet#, (widths, heights, None)
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
           
            return outputs, logabsdet#, (widths, heights,None)


def rational_quadratic_spline_smooth_circular(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              inverse=False,
                              rel_min_bin_width=1e-3,
                              rel_min_bin_height=1e-3,
                              restrict_max_min_width_height_ratio=-1.0,
                              shift_to_middle=True):

        
        left=0.0
        right=2*numpy.pi
        bottom=0.0
        top=2*numpy.pi

        if torch.min(inputs) < left or torch.max(inputs) > right:
           
            raise Exception("outside boundaries in rational-spline flow! (min/max (%.20e/%.20e), allowed: (%.6e/%.6e)" % (torch.min(inputs), torch.max(inputs), left, right))

        num_bins = unnormalized_widths.shape[-1]

        if rel_min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if rel_min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')
        
      
        used_unnormalized_widths=unnormalized_widths
        used_unnormalized_heights=unnormalized_heights

        ## check for restriction in width/heigth ratios
        if(restrict_max_min_width_height_ratio>0.0):
            ln_max_allowed=(numpy.log(restrict_max_min_width_height_ratio)-numpy.log(num_bins-1))/2.0
            
            assert(ln_max_allowed>0), "Allowed max/min ratio for widths/heights is too small.. %.3e .. require at least %.3e" % (restrict_max_min_width_height_ratio, numpy.exp(numpy.log(num_bins-1)/2.0))
            used_unnormalized_widths=2.0*F.sigmoid(unnormalized_widths)*ln_max_allowed-ln_max_allowed
            used_unnormalized_heights=2.0*F.sigmoid(unnormalized_heights)*ln_max_allowed-ln_max_allowed
        
        
        widths = F.softmax(used_unnormalized_widths, dim=-1)
        widths = rel_min_bin_width + (1 - rel_min_bin_width * num_bins) * widths
       
        cumwidths = torch.cumsum(widths, dim=-1)
        ## padding should be done effectively outside later
        cumwidths = F.pad(cumwidths, pad=(1,0), mode='constant', value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]
        
        heights = F.softmax(used_unnormalized_heights, dim=-1)
        heights = rel_min_bin_height + (1 - rel_min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)

        ## padding should be done effectively outside later
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        assert(widths.shape[-1]==2)
        assert(heights.shape[-1]==2)

        # calculate derivatives
        # they are the same for all know points (3) for circular smooth splines

        w1=widths[...,:1]
        w2=widths[...,1:]

        h1=heights[...,:1]
        h2=heights[...,1:]

        h_prod=h1*h2
        w_prod=w1*w2

        sqrt_fac=torch.sqrt(  h_prod*(8*((h2*w1)**2+(h1*w2)**2)+(9*(w1+w2)**2-16*w_prod)*h_prod)   )

        denom=4*(h1+h2)*w_prod

        res=(h_prod*(w1+w2)+sqrt_fac)/denom

        derivatives=torch.cat([res, res, res], dim=-1)

        ## calculate corrective shift
        corrective_factor=0.0
        if(shift_to_middle):
            w1mx=-numpy.pi+w1/2.0
            w1mx_p_w2=w1mx+w2

            nom=h2*w1mx*(w1mx*h1-res*w1*w1mx_p_w2)
            den=h1*w2**2+2*(h1-res*w1)*w1mx*w1mx_p_w2

            corrective_factor=2*numpy.pi-(h1+nom/den)
            
        ##################################

        used_inputs=inputs
        
        if(shift_to_middle):
            if(inverse):
                used_inputs=inputs-corrective_factor
                used_inputs=torch.where(used_inputs<0.0, used_inputs+2*numpy.pi, used_inputs)

            else:
                
                used_inputs=inputs-(numpy.pi-widths[...,0:1]/2.0)
                used_inputs=torch.where(used_inputs <0.0, used_inputs+2*numpy.pi, used_inputs)

        if inverse:
            bin_idx = searchsorted(cumheights, used_inputs)#[..., None]
        else:
            bin_idx = searchsorted(cumwidths, used_inputs)#[..., None]
        
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
            a = (((used_inputs - input_cumheights) * (input_derivatives
                                                 + input_derivatives_plus_one
                                                 - 2 * input_delta)
                  + input_heights * (input_delta - input_derivatives)))
            b = (input_heights * input_derivatives
                 - (used_inputs - input_cumheights) * (input_derivatives
                                                  + input_derivatives_plus_one
                                                  - 2 * input_delta))
            c = - input_delta * (used_inputs - input_cumheights)

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

            if(shift_to_middle):
                outputs=outputs+(numpy.pi-widths[...,0:1]/2.0)
                outputs=torch.where(outputs>2*numpy.pi, outputs-2*numpy.pi, outputs)

                ## take care of boundaries if necessary
                outputs=torch.where(inputs==0.0, 0.0, outputs)
                outputs=torch.where(inputs==2*numpy.pi,2*numpy.pi, outputs)
                
            return outputs, -logabsdet#, (widths, heights, None)
        else:
            theta = (used_inputs - input_cumwidths) / input_bin_widths
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

            if(shift_to_middle):
                
                outputs=outputs+corrective_factor
                outputs=torch.where(outputs>2*numpy.pi, outputs-2*numpy.pi, outputs)

                ## take care of boundaries if necessary
                outputs=torch.where(inputs==0.0, 0.0, outputs)
                outputs=torch.where(inputs==2*numpy.pi,2*numpy.pi, outputs)
               
            return outputs, logabsdet#, (widths, heights,None)