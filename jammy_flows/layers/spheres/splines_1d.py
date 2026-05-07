import torch
from torch import nn
import numpy

from . import sphere_base
from .. import spline_fns

class spline_1d(sphere_base.sphere_base):
    def __init__(self, dimension=1, 
                       euclidean_to_sphere_as_first=True, 
                       add_rotation=1, 
                       natural_direction=1,
                       use_permanent_parameters=False,  
                       num_basis_functions=2,
                       min_width=1e-4, 
                       min_height=1e-4, 
                       min_derivative=1e-4,
                       fix_boundary_derivatives=-1.0,
                       smooth_second_derivative=0,
                       #restrict_max_min_width_height_ratio=-1.0,
                       fix_first_width_n_height_to_zero=0,
                       also_fix_second_width_to_zero=0,
                       independent_width_height_parametrization=0):

        """
        A circular spline variant. Symbol: "o"

        Follows https://arxiv.org/abs/2002.02428. Still experimental and has not really been tested.
    
        """
        super().__init__(dimension=1, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, add_rotation=add_rotation, use_permanent_parameters=use_permanent_parameters)
        
        if(dimension!=1):
            raise Exception("The moebius flow is defined for dimension 1, but dimension %d is handed over" % (dimension))
    
        self.natural_direction=natural_direction
        self.fix_boundary_derivatives=fix_boundary_derivatives

        self.num_basis_functions=num_basis_functions
        self.num_width_params=num_basis_functions
        self.num_height_params=num_basis_functions

        self.fix_first_width_n_height_to_zero=fix_first_width_n_height_to_zero
        self.also_fix_second_width_to_zero=also_fix_second_width_to_zero

        if(self.fix_first_width_n_height_to_zero):
            self.num_width_params=num_basis_functions-1
            self.num_height_params=num_basis_functions-1

            ## second width has same length as first width
            if(self.also_fix_second_width_to_zero>0):
                self.num_width_params-=1

            

        ## widths of intervals in rational spline
        if(use_permanent_parameters):
            self.rel_log_widths=nn.Parameter(torch.randn(self.num_width_params).type(torch.double).unsqueeze(0))
            
        ## heights of intervals in rational spline
        if(use_permanent_parameters):
            self.rel_log_heights=nn.Parameter(torch.randn(self.num_height_params).type(torch.double).unsqueeze(0))
        
        ## derivatives
       
        self.deriv_num_bd_subtraction=0

                
        self.smooth_second_derivative=smooth_second_derivative
        if(self.smooth_second_derivative==1):
            assert( (self.num_basis_functions==2) ), "Only support 2 basis functions for smooth derivative!"
            
            self.deriv_num_bd_subtraction=3 # subtract all derivs.. all derivs will be predicted
           
        else:
            if(self.fix_boundary_derivatives>0.0):
                ## we only need to define this if we have no smooth derivatives
                self.deriv_num_bd_subtraction=2
                assert(self.fix_boundary_derivatives>min_derivative), "Fixed boundary derivative should be larger than min derivative!"

                # boundary derivatives fixed value = 1 if used
                self.boundary_log_derivs_fixed_value=numpy.log(numpy.exp(self.fix_boundary_derivatives-min_derivative)-1.0)

            else:
                self.deriv_num_bd_subtraction=1 # if we fit boundary derivative, it must be the same at 0 and 2pi so subtract one

        self.num_derivative_params=num_basis_functions+1-self.deriv_num_bd_subtraction

        if(self.smooth_second_derivative):
            if(self.num_basis_functions==3):
                ## subtract another width/height pair as we are in a symmetric situation
                self.num_width_params-=1
                self.num_height_params-=1
                
        ## only use derivatives if necessary
        if(use_permanent_parameters):
            if( self.num_derivative_params>0):
                self.rel_log_derivatives=nn.Parameter(torch.randn(self.num_derivative_params).type(torch.double).unsqueeze(0))
        
        ## subtract number of derivative parameters as we need
        self.total_param_num+=self.num_width_params+self.num_height_params+self.num_derivative_params #3*self.num_basis_functions+1-self.deriv_num_bd_subtraction

        ## minimum values to which relative logarithmic values are added with softmax
     
        self.min_width=min_width
        self.min_height=min_height
        self.min_derivative=min_derivative
       
        self.independent_width_height_parametrization=independent_width_height_parametrization

    def _inv_flow_mapping(self, inputs, extra_inputs=None, extra_inputs_base=None, sf_extra=None):

        [x, log_det]=inputs

        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        ## make sure we stay in range
        x=spline_fns.return_safe_angle_within_2pi(x)
        #x=torch.where(x>=2*numpy.pi, 2*numpy.pi, x)
        #x=torch.where(x<0.0, 0.0, x)

        if(self.use_permanent_parameters):
            widths=self.rel_log_widths.to(x)
            heights=self.rel_log_heights.to(x)
            derivatives=None
            if(self.num_derivative_params>0):
                derivatives=self.rel_log_derivatives.to(x)

        else:
            
            assert( extra_inputs is not None), "Conditional PDF.. require *extra_inputs*"
            assert( (extra_inputs.shape[0]==x.shape[0]) or (extra_inputs.shape[0]==1) ), ("Extra inputs must be Tensor of shape B X .. (B=Batch size) or 1 X .. (broadcasting).. got for first dimension instead : ", extra_inputs.shape[0])

            widths=extra_inputs[:,:self.num_width_params]#.reshape(extra_inputs.shape[0], self.rel_log_widths.shape[1])
            heights=extra_inputs[:,self.num_width_params:self.num_width_params+self.num_height_params]#.reshape(extra_inputs.shape[0], self.rel_log_heights.shape[1])
            
            if(self.num_derivative_params>0):
                derivatives=extra_inputs[:,self.num_width_params+self.num_height_params:]#.reshape(extra_inputs.shape[0], self.rel_log_derivatives.shape[1])
        
        ## check if have first width/height fixed to 0
        if(self.fix_first_width_n_height_to_zero):

            zero_add=torch.zeros_like(heights[:,0:1])

            heights=torch.cat([zero_add, heights], dim=1)

            if(self.also_fix_second_width_to_zero):
                widths=torch.cat([zero_add, zero_add, widths], dim=1)
            else:
                widths=torch.cat([zero_add, widths], dim=1)

        ## check for relative scaling
        if(self.independent_width_height_parametrization):
            heights=widths+heights

        use_inverse=True
        if(self.natural_direction==0):
            use_inverse=False

        if(self.smooth_second_derivative==0):

            if(self.fix_boundary_derivatives>0.0):
                first_and_last=torch.ones(derivatives.shape[:-1]+(1,)).to(x)*self.boundary_log_derivs_fixed_value

                derivatives=torch.cat([first_and_last, derivatives, first_and_last], dim=-1)
            else:
                ## copy over first element to the end
                derivatives=torch.cat([derivatives, derivatives[:,0:1]], dim=-1)

            x, log_det_update=spline_fns.rational_quadratic_spline(x, 
                                                             widths, 
                                                             heights, 
                                                             derivatives, 
                                                             inverse=use_inverse, 
                                                             left=0.0, 
                                                             right=2*numpy.pi, 
                                                             bottom=0.0, 
                                                             top=2*numpy.pi, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             min_derivative=self.min_derivative
                                                             )
        else:

            x, log_det_update=spline_fns.rational_quadratic_spline_smooth_circular(x, 
                                                             widths, 
                                                             heights, 
                                                             inverse=use_inverse, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             shift_to_middle=True
                                                             )
       
        log_det_new=log_det+log_det_update.sum(axis=-1)
        
        ## make sure we stay in range
        x=spline_fns.return_safe_angle_within_2pi(x)
        #x=torch.where(x>=2*numpy.pi, 2*numpy.pi, x)
        #x=torch.where(x<0.0, 0.0, x)

        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        return x, log_det_new


    def _flow_mapping(self, inputs, extra_inputs=None, extra_inputs_base=None, sf_extra=None):
        
        
        [x, log_det]=inputs

        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        ## make sure we stay in range
        x=torch.where(x>=2*numpy.pi, 2*numpy.pi, x)
        x=torch.where(x<0.0, 0.0, x)

        if(self.use_permanent_parameters):
            widths=self.rel_log_widths.to(x)
            heights=self.rel_log_heights.to(x)
            derivatives=None
            if(self.num_derivative_params>0):
                derivatives=self.rel_log_derivatives.to(x)

        else:
            
            assert( extra_inputs is not None), "Conditional PDF.. require *extra_inputs*"
            assert( (extra_inputs.shape[0]==x.shape[0]) or (extra_inputs.shape[0]==1) ), ("Extra inputs must be Tensor of shape B X .. (B=Batch size)", x.shape[0], "or 1 X .. (broadcasting).. got for first dimension instead : ", extra_inputs.shape[0])

            widths=extra_inputs[:,:self.num_width_params]#.reshape(extra_inputs.shape[0], self.rel_log_widths.shape[1])
            heights=extra_inputs[:,self.num_width_params:self.num_width_params+self.num_height_params]#.reshape(extra_inputs.shape[0], self.rel_log_heights.shape[1])
            
            if(self.num_derivative_params>0):
                derivatives=extra_inputs[:,self.num_width_params+self.num_height_params:]#.reshape(extra_inputs.shape[0], self.rel_log_derivatives.shape[1])
        
        if(self.fix_first_width_n_height_to_zero):

            zero_add=torch.zeros_like(heights[:,0:1])

            heights=torch.cat([zero_add, heights], dim=1)

            if(self.also_fix_second_width_to_zero):
                widths=torch.cat([zero_add, zero_add, widths], dim=1)
            else:
                widths=torch.cat([zero_add, widths], dim=1)

        ## check for relative scaling
        if(self.independent_width_height_parametrization):

            heights=widths+heights

        use_inverse=False
        if(self.natural_direction==0):
            use_inverse=True

        if(self.smooth_second_derivative==0):

            if(self.fix_boundary_derivatives>0.0):

                first_and_last=torch.ones(derivatives.shape[:-1]+(1,)).to(x)*self.boundary_log_derivs_fixed_value

                derivatives=torch.cat([first_and_last, derivatives, first_and_last], dim=-1)
            else:
                ## copy over first element to the end
                derivatives=torch.cat([derivatives, derivatives[:,0:1]], dim=-1)

            x, log_det_update=spline_fns.rational_quadratic_spline(x, 
                                                             widths, 
                                                             heights, 
                                                             derivatives, 
                                                             inverse=use_inverse, 
                                                             left=0.0, 
                                                             right=2*numpy.pi, 
                                                             bottom=0.0, 
                                                             top=2*numpy.pi, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             min_derivative=self.min_derivative
                                                             )
        else:

            x, log_det_update=spline_fns.rational_quadratic_spline_smooth_circular(x, 
                                                             widths, 
                                                             heights, 
                                                             inverse=use_inverse, 
                                                             rel_min_bin_width=self.min_width,
                                                             rel_min_bin_height=self.min_height,
                                                             shift_to_middle=True
                                                             )
       
        log_det_new=log_det+log_det_update.sum(axis=-1)

        ## make sure we stay in range
        x=torch.where(x>=2*numpy.pi, 2*numpy.pi, x)
        x=torch.where(x<0.0, 0.0, x)

        if(self.always_parametrize_in_embedding_space):
            # embedding to intrinsic
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)
        
        return x, log_det_new

    def _get_desired_init_parameters(self):

        ## widths

        desired_param_vec=[]

        ## 0.54 as start value in log space seems to result in a rather flat spline (defined by the log-derivatives in particular)
        if(self.smooth_second_derivative):  
            desired_param_vec.append(torch.zeros(self.num_width_params+self.num_height_params+self.num_derivative_params))
        else:
            desired_param_vec.append(torch.ones(self.num_width_params+self.num_height_params+self.num_derivative_params)*0.54)
        
        return torch.cat(desired_param_vec)

    def _init_params(self, params):

        counter=0
        self.rel_log_widths.data[0,:]=params[counter:counter+self.num_width_params]    
        counter+=self.num_width_params
        
        self.rel_log_heights.data[0,:]=params[counter:counter+self.num_height_params]
        counter+=self.num_height_params

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
            
            widths=extra_inputs[:,:self.num_width_params]
            heights=extra_inputs[:,self.num_width_params:self.num_width_params+self.num_height_params]
            if(self.num_derivative_params>0):
                derivatives=extra_inputs[:,self.num_width_params+self.num_height_params:]

        param_dict[extra_prefix+"widths"]=widths
        param_dict[extra_prefix+"heights"]=heights

        if(self.smooth_second_derivative==0):
            param_dict[extra_prefix+"derivatives"]=derivatives
        

