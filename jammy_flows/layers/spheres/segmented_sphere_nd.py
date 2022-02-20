import torch
from torch import nn
import numpy

from . import sphere_base
from . import moebius_1d
from ..bisection_n_newton import inverse_bisection_n_newton
from ...amortizable_mlp import AmortizableMLP
from ...extra_functions import list_from_str
from ..euclidean.gaussianization_flow import gf_block, find_init_pars_of_chained_gf_blocks
from ..euclidean.polynomial_stretch_flow import psf_block
from ..euclidean.euclidean_do_nothing import euclidean_do_nothing
from ..intervals.interval_do_nothing import interval_do_nothing
from ..intervals.rational_quadratic_spline import rational_quadratic_spline

import sys
import os
import copy

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class segmented_sphere_nd(sphere_base.sphere_base):
    def __init__(self, 
                 dimension, 
                 euclidean_to_sphere_as_first=True, 
                 use_extra_householder=True, 
                 use_permanent_parameters=False, 
                 use_moebius_xyz_parametrization=True, 
                 num_basis_functions=5, 
                 zenith_type_layers="g", 
                 max_rank=20, 
                 hidden_dims="64", 
                 subspace_mapping="logistic", 
                 higher_order_cylinder_parametrization=False):

        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_extra_householder=use_extra_householder, use_permanent_parameters=use_permanent_parameters, higher_order_cylinder_parametrization=higher_order_cylinder_parametrization)
        
        if(dimension==1):
            raise Exception("The moebius flow should be used for dimension 1!")
        
        ## If this is not the only s2-flow, and not the first in the column, switch off shortcut via cylinder for coverage
        if(self.euclidean_to_sphere_as_first==False):
            self.higher_order_cylinder_parametrization=False
        #if(higher_order_cylinder_parametrization==False):
       #     raise Exception("Cylinder parametrization is required! Switching it off is legacy behavior and less stable.")

        ## a moebius layer
        self.moebius_trafo=moebius_1d.moebius(1, euclidean_to_sphere_as_first=False, use_extra_householder=False, natural_direction=0, use_permanent_parameters=use_permanent_parameters, use_moebius_xyz_parametrization=use_moebius_xyz_parametrization, num_basis_functions=num_basis_functions)
        self.num_s1_pars=self.moebius_trafo.total_param_num
        self.total_param_num+=self.num_s1_pars

        self.subspace_mapping=subspace_mapping

        self.zenith_type_layer_list=nn.ModuleList()
        self.num_parameter_list=[]

        self.flow_dict = dict()

        ### euclidean layers - for these layers the input (0 - pi) has to be transformed to a real line first

        self.flow_dict["g"] = dict()
        self.flow_dict["g"]["module"] = gf_block
        self.flow_dict["g"]["type"] = "e"
        self.flow_dict["g"]["kwargs"] = dict()
        self.flow_dict["g"]["kwargs"]["fit_normalization"] = 1
        self.flow_dict["g"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["g"]["kwargs"]["num_householder_iter"] = -1
        self.flow_dict["g"]["kwargs"]["num_kde"] = 10
        self.flow_dict["g"]["kwargs"]["inverse_function_type"] = "isigmoid"

        ### the next two are not real parameters, and normally deleted
        #self.flow_dict["g"]["kwargs"]["replace_first_sigmoid_with_icdf"]=0
        #self.flow_dict["g"]["kwargs"]["skip_model_offset"]=0


        self.flow_dict["g"]["kwargs"]["softplus_for_width"]=0 # use softplus instead of exp to transform log_width -> width
        self.flow_dict["g"]["kwargs"]["upper_bound_for_widths"]=100 # define an upper bound for the value of widths.. -1 = no upper bound
        self.flow_dict["g"]["kwargs"]["lower_bound_for_widths"]=0.01 # define a lower bound for the value of widths
        self.flow_dict["g"]["kwargs"]["clamp_widths"]=0
        self.flow_dict["g"]["kwargs"]["width_smooth_saturation"]=1 # 


        self.flow_dict["p"] = dict()
        self.flow_dict["p"]["module"] = psf_block
        self.flow_dict["p"]["type"] = "e"
        self.flow_dict["p"]["kwargs"] = dict()
        self.flow_dict["p"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["p"]["kwargs"]["num_householder_iter"] = -1
        self.flow_dict["p"]["kwargs"]["num_transforms"] = 3

        self.flow_dict["x"] = dict()
        self.flow_dict["x"]["module"] = euclidean_do_nothing
        self.flow_dict["x"]["type"] = "e"
        self.flow_dict["x"]["kwargs"]=dict()

        ### interval layers - for these layers, the input (0-pi) only has to be transformed to 0-1 via a cos-transformation

        self.flow_dict["r"] = dict()
        self.flow_dict["r"]["module"] = rational_quadratic_spline
        self.flow_dict["r"]["type"] = "i"
        self.flow_dict["r"]["kwargs"] = dict()
        self.flow_dict["r"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["r"]["kwargs"]["num_basis_elements"]=5
        self.flow_dict["r"]["kwargs"]["low_boundary"] = 0
        self.flow_dict["r"]["kwargs"]["high_boundary"] = 1.0

        self.flow_dict["z"] = dict()
        self.flow_dict["z"]["module"] = interval_do_nothing
        self.flow_dict["z"]["type"] = "i"
        self.flow_dict["z"]["kwargs"]=dict()


        self.zenith_type_layer_defs=zenith_type_layers

        if("z" in self.zenith_type_layer_defs or "r" in self.zenith_type_layer_defs):
            self.subspace_is_euclidean=False
            assert("x" not in self.zenith_type_layer_defs)
            assert("g" not in self.zenith_type_layer_defs)
            assert("p" not in self.zenith_type_layer_defs)
        if("g" in self.zenith_type_layer_defs or "p" in self.zenith_type_layer_defs or "x" in self.zenith_type_layer_defs):
            self.subspace_is_euclidean=True
            assert("z" not in self.zenith_type_layer_defs)
            assert("r" not in self.zenith_type_layer_defs)
            
        self.total_euclidean_pars=0

        ## if there are no euclidean layers the transformation does not exist, might use empty layer then
        assert(len(self.zenith_type_layer_defs)>0)

        for layer_ind, layer_type in enumerate(self.zenith_type_layer_defs):
           
            this_kwargs = copy.deepcopy(self.flow_dict[layer_type]["kwargs"])

            ## overwrite permanent parameters if desired or necessary
            #if self.force_permanent_parameters:
            this_kwargs["use_permanent_parameters"] = 0

         
            ## this flow is a spherical flow, so the first layer should also project from plane to sphere or vice versa

            self.zenith_type_layer_list.append(
                self.flow_dict[layer_type]["module"](self.dimension-1, **this_kwargs)
            )

            #self.num_parameter_list.append(self.zenith_type_layer_list[-1].total_param_num)

            #self.total_param_num+=self.num_parameter_list[-1]
            self.total_euclidean_pars+=self.zenith_type_layer_list[-1].total_param_num

        self.num_mlp_params=0

        #print("total eucl params to predict ,", self.total_euclidean_pars)

        if(self.total_euclidean_pars>0):
            self.amortized_mlp=AmortizableMLP(2, hidden_dims, self.total_euclidean_pars, low_rank_approximations=max_rank, use_permanent_parameters=use_permanent_parameters,svd_mode="smart")
            self.num_mlp_params=self.amortized_mlp.num_amortization_params
            self.total_param_num+=self.num_mlp_params
        else:
            self.amortized_mlp=None
            
    def to_subspace(self, x, log_det, sf_extra=None):
        """
        Maps the zenith-type dimension to an appropriate subspace for further transformation.
        """

        if(self.subspace_mapping=="logistic"):

            # Cylinder parametrization uses a logarithmic parametrization to represent values arbitrarily close to boundaries in a differentiable fashion.
            if(self.higher_order_cylinder_parametrization):

                if(self.subspace_is_euclidean):
                    lcyl=x
                    sfcyl=sf_extra

                    res=lcyl-sfcyl
                    #print("to subspace ", lcyl, sfcyl)
                    #log_det-=(res-2.0*lsexp).sum(axis=-1)
                    log_det=log_det+(-lcyl-sfcyl).sum(axis=-1)

                    return res, log_det
                else:

                    ### we go to the subspace .. x is the encoded log-cdf values, sf_extra is the encoded log(sf) value
                    ### no log-det change needed, just taking exponential to be in proper [0,1] range

                    return torch.exp(x), log_det


            else:
              
                ### just a linear trafo
                #scaled=x/numpy.pi
                #log_det-=numpy.log(numpy.pi)
                ###
       
                ## FIXME .. this is only correct for 2-d!!
                scaled=-torch.cos(x)/2.0+1.0/2.0
                log_det=log_det+torch.log(torch.sin(x[:,0])/2.0)
                ##########

                good_scaled=(scaled>0) & (scaled < 1.0)
                scaled=torch.where(scaled<=0.0, 1e-8, scaled)
                scaled=torch.where(scaled>=1.0, scaled-1e-8,scaled)

                #scaled=good_scaled*scaled+(scaled==0)*(scaled+1e-3)+(scaled==1.0)*(scaled-1e-3)

                if(self.subspace_is_euclidean):
                    inv=(torch.log(scaled)-torch.log(1.0-scaled))
               
                    log_det=log_det+(-torch.log(scaled)-torch.log(1.0-scaled)).sum(axis=-1)

                    return inv, log_det
                else:
                    return scaled, log_det
        else:
            raise Exception("Unknown subspace mapping", self.subspace_mapping)

    def from_subspace(self, x, log_det):
        """
        Maps the real/interval subspace back to the orginal zenith-type space.
        """

        if(self.subspace_mapping=="logistic"):

            sf_extra=None
            if(self.higher_order_cylinder_parametrization):
                
                if(self.subspace_is_euclidean):
                    lsexp=torch.cat( [torch.zeros_like(x).to(x)[:,:,None], x[:,:,None]] , dim=2).logsumexp(dim=2)
                    log_det=log_det+(x-2.0*lsexp).sum(axis=-1)

                    lncyl=x-lsexp
                    sf_extra=-lsexp
                    #print("from eucl subspace ", lncyl, sf_extra)
                    return lncyl, log_det, sf_extra
                else:
                    ## we are already in subspace of [0,1] .. no log-det change needed
                    ## lncyl takes log of lowe boundary, sf_extra takes log of higher_boundary (1-x),i.e. the survival function
                    lncyl=torch.log(x)
                    sf_extra=torch.log(1.0-x)

                    return lncyl, log_det, sf_extra

            else:
                    
                if(self.subspace_is_euclidean):
                    lsexp=torch.cat( [torch.zeros_like(x).to(x)[:,:,None], x[:,:,None]] , dim=2).logsumexp(dim=2)
                
                    log_det=log_det+(x-2.0*lsexp).sum(axis=-1)

                    x=1.0/(1.0+torch.exp(-x))

                #### linear trafo
                #res*=numpy.pi
                #log_det+=numpy.log(numpy.pi)
                ####

              
                ## FIXME .. this is only correct for 2-d!!
                
                #log_det+=-0.5*torch.log(res-res**2)[:,0]

                x=torch.acos(1.0-2*x)
                log_det=log_det-torch.log(torch.sin(x[:,0])/2.0)
                
                return x, log_det, None
        else:
            raise Exception("Unknown subspace mapping", self.subspace_mapping)

    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        #if(self.higher_order_cylinder_parametrization):
        #    print("extra params MLP", extra_inputs[0,self.num_mlp_params-self.total_euclidean_pars:self.num_mlp_params])
        ## input structure: 0-num_amortization_params -> MLP  , num_amortizpation_params-end: -> moebius trafo
        [x,log_det]=inputs

        sf_extra=None

        if(self.always_parametrize_in_embedding_space):
            print("-------> Warning: Cylinder-based 2-sphere flow is not recommended to use embedding space paremetrization between layers!")

        ## this implementation requires another sin(theta) factor here
        ## 
        #log_det=log_det-torch.log(torch.sin(x[:,0]))

        
        if(self.always_parametrize_in_embedding_space):
           
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)

        moebius_extra_inputs=None
        if(extra_inputs is not None):
            
            moebius_extra_inputs=extra_inputs[:,self.num_mlp_params:]
        
        #print("moeb bef",x[:,self.dimension-1:] )
        xm,log_det=self.moebius_trafo.inv_flow_mapping([x[:,self.dimension-1:],log_det], extra_inputs=moebius_extra_inputs)
        #xm=x[:,self.dimension-1:]
        #print("moeb af",xm )
        potential_eucl=x[:,:self.dimension-1]
        
        ## parameter that holds possibly infinitesimal boundary of cylinder height parametrization
        

        if(self.higher_order_cylinder_parametrization):
            ## maps [0,pi] to [0,1]
            cos_coords=0.5-0.5*torch.cos(x[:, :self.dimension-1])

            ## small angle approximation of cosine for these values
            small_mask=x[:, :self.dimension-1]<1e-4
            large_mask=x[:, :self.dimension-1]>(numpy.pi-1e-4)
            
            #ln_cyl=torch.log(cos_coords)
            #ln_cyl[small_mask]=2*torch.log(x[:, :self.dimension-1][small_mask])-numpy.log(4.0)
            ln_cyl=torch.where(small_mask, 2*torch.log(x[:, :self.dimension-1])-numpy.log(4.0), torch.log(cos_coords))

            sf_extra=torch.where(large_mask, 2*torch.log(numpy.pi-x[:, :self.dimension-1])-numpy.log(4.0), torch.log(1.0-cos_coords))
            #sf_extra=torch.log(1.0-cos_coords)
            #sf_extra[large_mask]=2*torch.log(numpy.pi-x[:, :self.dimension-1][large_mask])-numpy.log(4.0)

            potential_eucl=ln_cyl
            ## ddx = sin(x)/2
            log_det=log_det+0.5*(ln_cyl+sf_extra).sum(axis=-1)

            #print("_inv flow cyl ln / sf", ln_cyl, sf_extra)
        if(len(self.zenith_type_layer_list)>0):

            eucl_x, log_det=self.to_subspace(potential_eucl,log_det, sf_extra=sf_extra)
           
            ## loop through all layers in each pdf and transform "this_target"

            if(self.num_mlp_params>0):
                amortized_inputs=extra_inputs
                if(extra_inputs is not None):
                    amortized_inputs=extra_inputs[:,:self.num_mlp_params]

              
                ## apply MLP that takes as input moebius output
                eucl_layer_pars=self.amortized_mlp(self.moebius_trafo._embedding_conditional_return(x[:,self.dimension-1:]), amortized_inputs)

                extra_param_counter = 0
                for l, layer in list(enumerate(self.zenith_type_layer_list)):
                    
                  
                    this_extra_params = eucl_layer_pars[:, extra_param_counter : extra_param_counter + layer.total_param_num]

                    eucl_x, log_det = layer.inv_flow_mapping([eucl_x, log_det], extra_inputs=this_extra_params)

                    extra_param_counter += layer.total_param_num

         
            potential_eucl, log_det, sf_extra=self.from_subspace(eucl_x, log_det)

        ## combine zylinder
        op=torch.cat([potential_eucl, xm],dim=1)

        ##  if we are the first layer we dont want to embed and pass on sf_extra
        if(self.always_parametrize_in_embedding_space):
            op, log_det=self.spherical_to_eucl_embedding(op, log_det)
        
        

        return op, log_det, sf_extra

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
        
        if(self.always_parametrize_in_embedding_space):
            print("-------> Warning: Cylinder-based 2-sphere flow is not recommended to use embedding space paremetrization between layers!")
        [x,log_det]=inputs

        moebius_extra_inputs=None
        if(extra_inputs is not None):
            
            moebius_extra_inputs=extra_inputs[:,self.num_mlp_params:]

        if(self.always_parametrize_in_embedding_space):
            x, log_det=self.eucl_to_spherical_embedding(x, log_det)
        
        #print("forw moeb ", x[:,self.dimension-1:])
        xm,log_det=self.moebius_trafo.flow_mapping([x[:,self.dimension-1:],log_det], extra_inputs=moebius_extra_inputs)
        #print("forw moeb af", xm)
        potential_eucl=x[:,:self.dimension-1]

        ### 

        if(len(self.zenith_type_layer_list)>0):

            
            eucl_x, log_det=self.to_subspace(potential_eucl, log_det, sf_extra=sf_extra)
            ## loop through all layers in each pdf and transform "this_target"

           
            if(self.num_mlp_params>0):
                amortized_inputs=extra_inputs
                if(extra_inputs is not None):
                    amortized_inputs=extra_inputs[:,:self.num_mlp_params]

                ## apply MLP that takes as input moebius output
                eucl_layer_pars=self.amortized_mlp(self.moebius_trafo._embedding_conditional_return(xm), amortized_inputs)
                extra_param_counter = 0


                for l, layer in reversed(list(enumerate(self.zenith_type_layer_list))):
                    
                    this_extra_params = None
                    

                    this_extra_params = eucl_layer_pars[:, self.total_euclidean_pars-extra_param_counter-layer.total_param_num : self.total_euclidean_pars-extra_param_counter ]

                    
                    eucl_x, log_det = layer.flow_mapping([eucl_x, log_det], extra_inputs=this_extra_params)
                   
                    extra_param_counter += layer.total_param_num

            potential_eucl, log_det, sf_extra=self.from_subspace(eucl_x, log_det)

           
        if(self.higher_order_cylinder_parametrization):

            ln_cyl=potential_eucl

            ## ddx = sin(x)/2
            log_det=log_det-0.5*(ln_cyl+sf_extra).sum(axis=-1)

            potential_eucl=torch.acos(1.0-2.0*ln_cyl.exp())
        
        ## stitch cylinder back to together
        op=torch.cat([potential_eucl, xm],dim=1)

        if(self.always_parametrize_in_embedding_space):
            op, log_det=self.spherical_to_eucl_embedding(op, log_det)

        #log_det=log_det+torch.log(torch.sin(op[:,0]))

        return op, log_det

    def _init_params(self, params):

        ## first parameters are MLP / second part is the moebius flow
 
        assert(len(params)== (self.num_mlp_params+self.num_s1_pars))

        if(self.num_mlp_params>0):
            mlp_params=params[:self.num_mlp_params]

            bias_mlp_pars=mlp_params[-self.total_euclidean_pars:]

            self.amortized_mlp.initialize_uvbs(init_b=bias_mlp_pars)

            ## 

        moebius_pars=params[self.num_mlp_params:]

        self.moebius_trafo.init_params(moebius_pars)   

    def _get_desired_init_parameters(self):

        ## first parameters are MLP / second part is the moebius flow
        
        ## the parameters consist of the weights of the MLP and the biases afterwards
        ## The weights should be suppressed (gaussian init weighted down)
        ## The bias should be set to match a good euclidean flow init (i.e. Gaussianization flow)

      
        gaussian_init=torch.randn((self.num_mlp_params+self.num_s1_pars))/10000.0

        #return gaussian_init

        desired_euclidean_pars=[]

        gf_init=True

        for l in self.zenith_type_layer_defs:
            if(l!="g"):
                gf_init=False

        
        if(gf_init):

            ## flat data between 0 to pi in every dim
            pseudo_data=torch.rand((1000, self.dimension-1)).type(torch.float64)

            ## get the corresponding distribution in the euclidean space
        

            ## use that distribution as initilization to GF flow
            desired_euclidean_pars=find_init_pars_of_chained_gf_blocks(self.zenith_type_layer_list, pseudo_data,householder_inits="random")

        else:

            ## get basic rough init...
            this_list=[]
            for l in self.zenith_type_layer_list:
                this_list.append(l.get_desired_init_parameters())

            desired_euclidean_pars=torch.cat(this_list)


        gaussian_init[-self.total_euclidean_pars-self.num_s1_pars:-self.num_s1_pars]=desired_euclidean_pars
        gaussian_init[-self.num_s1_pars:]=self.moebius_trafo.get_desired_init_parameters()

     
        return gaussian_init
    
       

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 

        moebius_extra_inputs=None

        if(extra_inputs is not None):
            
            moebius_extra_inputs=extra_inputs[:,self.num_mlp_params:]
        
        ## we dont care about log_det for layer structure
        ld=0.0
       
        xm,_=self.moebius_trafo.flow_mapping([previous_x[:,self.dimension-1:],ld], extra_inputs=moebius_extra_inputs)
        self.moebius_trafo.obtain_layer_param_structure(param_dict, extra_inputs=moebius_extra_inputs)
        #print("forw moeb af", xm)
        #potential_eucl=x[:,:self.dimension-1]

        ### 

        if(len(self.zenith_type_layer_list)>0):

            
            #eucl_x, log_det=self.to_subspace(potential_eucl, log_det, sf_extra=sf_extra)
            ## loop through all layers in each pdf and transform "this_target"

           
            if(self.num_mlp_params>0):
                amortized_inputs=extra_inputs
                if(extra_inputs is not None):
                    amortized_inputs=extra_inputs[:,:self.num_mlp_params]
                    param_dict[extra_prefix+"uvb_pars"]=amortized_inputs
                else:
                    param_dict[extra_prefix+"uvb_pars"]=self.amortized_mlp.u_v_b_pars

                ## apply MLP that takes as input moebius output
                eucl_layer_pars=self.amortized_mlp(self.moebius_trafo._embedding_conditional_return(xm), amortized_inputs)
                extra_param_counter = 0


                for l, layer in reversed(list(enumerate(self.zenith_type_layer_list))):
                    
                    this_extra_params = None
                    
                    this_extra_params = eucl_layer_pars[:, self.total_euclidean_pars-extra_param_counter-layer.total_param_num : self.total_euclidean_pars-extra_param_counter ]

                    #eucl_x, log_det = layer.flow_mapping([eucl_x, log_det], extra_inputs=this_extra_params)

                    #layer.obtain_layer_param_structure(param_dict, extra_inputs=this_extra_params, extra_prefix=extra_prefix+"%.2d" % l)
                   
                    extra_param_counter += layer.total_param_num

            #potential_eucl, log_det, sf_extra=self.from_subspace(eucl_x, log_det)

           
       