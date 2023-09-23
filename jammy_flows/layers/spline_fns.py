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
        
        cumwidths = F.pad(cumwidths, pad=(1,0), mode='constant', value=0.0)

        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]
        
        
        heights = F.softmax(used_unnormalized_heights, dim=-1)
        heights = rel_min_bin_height + (1 - rel_min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
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
                
                derivatives=torch.cat([boundary_derivatives[...,:-1], res, boundary_derivatives[...,1:]], dim=-1)

            elif(widths.shape[-1]==3):

                x=sympy.symbols("x")
                y=sympy.symbols("y")

                h_1=sympy.symbols("h_1")
                h_2=sympy.symbols("h_2")
                h_3=sympy.symbols("h_3")

                w_1=sympy.symbols("w_1")
                w_2=sympy.symbols("w_2")
                w_3=sympy.symbols("w_3") 

                lb=sympy.symbols("l_b") # lower deriv
                ub=sympy.symbols("u_b") # upper deriv

                i=x**2+x*(h_1*( (h_2/w_2)-lb) + h_2 * ( (h_1/w_1)-y))/(h_1+h_2) - (h_1*h_2/(h_1+h_2))*((h_1*w_2**2+h_2*w_1**2)/(w_1**2 *w_2**2))
                j=y**2+y*(h_2*( (h_3/w_3)-x) + h_3 * ( (h_2/w_2)-ub))/(h_2+h_3) - (h_2*h_3/(h_2+h_3))*((h_2*w_3**2+h_3*w_2**2)/(w_2**2 *w_3**2))
                
                G=sympy.groebner([i,j],x,y)

                e=G[1].coeff("y", 0)
                d=G[1].coeff("y", 1)
                c=G[1].coeff("y", 2)
                b=G[1].coeff("y", 3)
                a=G[1].coeff("y", 4)

                assert(widths.shape[0]==1)

                ############################### replace widths and heights
                replacements=[(w_1, float(widths[0,0].cpu().detach().item())), (w_2, float(widths[0,1].cpu().detach().item())), (w_3, float(widths[0,2].cpu().detach().item()))]
                replacements.extend([(h_1, float(heights[0,0].cpu().detach().item())), (h_2, float(heights[0,1].cpu().detach().item())), (h_3, float(heights[0,2].cpu().detach().item()))])

                ## upper / lower bound replacement
                replacements.extend([(lb, float(boundary_derivatives[0,0].cpu().detach().item())), (ub, float(boundary_derivatives[0,1].cpu().detach().item()))])
                
                a_=a.subs(replacements)
                b_=b.subs(replacements)
                c_=c.subs(replacements)
                d_=d.subs(replacements)
                e_=e.subs(replacements)
                    
                ## also replace first polynomial
                first_poly_=G[0].subs(replacements)
                
                p_=(8*a_*c_-3*b_**2)/(8*a_**2)
                q_=(b_**3-4*a_*b_*c_+8*d_*a_**2)/(8*a_**3)
                del0_=c_**2-3*b_*d_+12*a_*e_
                del1_=2*c_**3-9*b_*c_*d_+27*e_*b_**2+27*a_*d_**2-72*a_*c_*e_
                discriminant_=(del1_**2-4*del0_**3)/-27
                
                phi=numpy.arccos(float(del1_/(2*numpy.sqrt(float(del0_)**3))))
                
                S=0.5*numpy.sqrt(float( -(2.0/3.0)*p_+(2.0/(3.0*a_))*numpy.sqrt(float(del0_))*numpy.cos(phi/3.0)))
                
                sqrt_term_pos=numpy.sqrt(float(-4.0*S**2-2*p_+q_/S))
                sqrt_term_neg=numpy.sqrt(float(-4.0*S**2-2*p_-q_/S))
                
                first_y=(-b_/(4*a_))-S+0.5*sqrt_term_pos
                second_y=(-b_/(4*a_))-S-0.5*sqrt_term_pos
                third_y=(-b_/(4*a_))+S+0.5*sqrt_term_neg
                fourth_y=(-b_/(4*a_))+S-0.5*sqrt_term_neg
                
                rpls=first_poly_.subs("y", first_y)
                first_x=sympy.solve(rpls, "x")[0]
                
                rpls=first_poly_.subs("y", second_y)
                second_x=sympy.solve(rpls, "x")[0]
                
                rpls=first_poly_.subs("y", third_y)
                third_x=sympy.solve(rpls, "x")[0]
                
                rpls=first_poly_.subs("y", fourth_y)
                fourth_x=sympy.solve(rpls, "x")[0]

                if(solution_index==0):
                    solution_vec=torch.Tensor([first_x,first_y])[None,:]
                    
                elif(solution_index==1):
                    solution_vec=torch.Tensor([second_x,second_y])[None,:]
                  
                elif(solution_index==2):
                    solution_vec=torch.Tensor([third_x,third_y])[None,:]
                  
                elif(solution_index==3):
                    solution_vec=torch.Tensor([fourth_x,fourth_y])[None,:]
                
                derivatives=torch.cat([boundary_derivatives[...,:-1], solution_vec, boundary_derivatives[...,1:]], dim=-1)

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