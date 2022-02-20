import torch
from torch import nn
import collections
import numpy

from .. import layer_base

class simplex_base(layer_base.layer_base):
    def __init__(self, dimension=1, use_permanent_parameters=False, always_parametrize_in_embedding_space=0, project_from_gauss_to_simplex=0):
        
        super().__init__(dimension=dimension, always_parametrize_in_embedding_space=always_parametrize_in_embedding_space)

        self.use_permanent_parameters=use_permanent_parameters

        self.project_from_gauss_to_simplex=project_from_gauss_to_simplex

        ## M defines the projection onto canonical simplex from base simplex
        ## See https://arxiv.org/pdf/2008.05456.pdf
        self.M=torch.zeros((self.dimension,self.dimension+1), dtype=torch.float64)
        self.M[:,0]=-1.0
        self.M[:,1:]=torch.eye(self.dimension, dtype=torch.float64)

        ## M_reverse = M^t*(M*M^t)**-1  = (1/(d+1)) * C where C is d+1 X d, has d on the diagonal in the lower symmetric (dxd) part and -1 everywhere else
        ## projects from the canonical simples back to base simplex
        self.M_reverse=torch.ones((self.dimension+1,self.dimension), dtype=torch.float64)*-1.0

        for ind in range(self.dimension):
            self.M_reverse[1+ind,ind]=self.dimension

        self.M_reverse*=1.0/(1.0+self.dimension)

        ###############

        self.canonical_one_hot=torch.zeros(self.dimension+1, dtype=torch.float64)
        self.canonical_one_hot[0]=1.0


    def gauss_to_non_uniform_box(self, inputs, use_gauss_projection=True):
        """
        
        """
        [x, log_det]=inputs

        if(use_gauss_projection):

            log_det_factor=-(0.5*(x)**2)-0.5*numpy.log(2.0*numpy.pi)
            log_det=log_det+log_det_factor.sum(axis=-1)

            x=0.5*(1.0+torch.erf(x/numpy.sqrt(2)))


        ## skew inside box

        ## skew inside box to yield flat distrbution on simplex (only works exactly for up to 3-simplex)
        if(x.shape[1]>1):
            x=x.clone()
            x[:,:-1]=1.0-(1.0-x[:,:-1])**0.5
            log_det=log_det-0.5*torch.log(1.0-x[:,:-1]).sum(axis=-1)-numpy.log(2)

        return x, log_det

    def non_uniform_box_to_gauss(self, inputs, use_gauss_projection=True):

        [x, log_det]=inputs

        ####
        ## make box straight first
        res=torch.zeros_like(x)

        if(res.shape[1]>1):
               
            log_det=log_det+torch.log(1.0-x[:,:-1]).sum(axis=-1)+numpy.log(2)

            res[:,:-1]=1.0-(1.0-x[:,:-1])**2

        ## we dont want to mess with inputs so we clone it
        res[:,-1]=x[:,-1].clone()

        ##  from straight box to gauss
        if(use_gauss_projection):
            res=numpy.sqrt(2.0)*torch.erfinv(2.0*res-1.0)

            log_det_factor=-(0.5*(res)**2)-0.5*numpy.log(2.0*numpy.pi)

            log_det=log_det-log_det_factor.sum(axis=-1)

        return res, log_det

    ####

    def non_uniform_box_to_base_simplex(self, inputs):
        """
        Transform from box to the (base) simplex aligned with the coordinate axes. This is not the canonical standard simplex. In 4-d, e.g. this base simplex is bounded by the vertices
        (0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1),
        """

        [x, log_det]=inputs

        res=x.clone()
        
        for ind in range(x.shape[1]):
           
            if(ind>0):
                
                log_det=log_det+torch.log(1.0-x[:,:ind]).sum(axis=-1)

                res[:,ind]=x[:,ind]*torch.prod(1.0-x[:,:ind], dim=1)

        return res, log_det

    def base_simplex_to_non_uniform_box(self, inputs):
        """
        Transform from the base simplex to box. This is not the canonical standard simplex. In 4-d, e.g. this base simplex is bounded by the vertices
        (0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1),
        """

        [res, log_det]=inputs

        new_res=res.clone()

        ### from ground simplex to box
        
        for ind in range(res.shape[1]):
            
            #if(ind==0):
            #    new_res[:,ind]=res[:,ind].clone()
            if(ind>0):
                
                new_res[:,ind]=res[:,ind]/(1.0-torch.sum(res[:,:ind], dim=1))

                log_det=log_det-torch.log(1.0-torch.sum(res[:,:ind], axis=-1))
            
        return new_res, log_det


    def base_simplex_to_canonical_simplex(self, inputs):

        [x, log_det]=inputs

        ## (1,0,0,...)+ 1X2 * 2XN

        mm_result=self.canonical_one_hot+torch.matmul(x, self.M)

        ## area increases by sqrt(dim+1)
        log_det=log_det+0.5*numpy.log(self.dimension+1)

        return mm_result, log_det

    def canonical_simplex_to_base_simplex(self, inputs):

        [x, log_det]=inputs

        ## (1,0,0,...)+ 1X2 * 2XN

        mm_result=torch.matmul(x-self.canonical_one_hot, self.M_reverse)

        ## area decreases by sqrt(dim+1)
        log_det=log_det-0.5*numpy.log(self.dimension+1)

        return mm_result, log_det

    #def canonical_simplex_to_base_simplex(self, inputs):

    def inv_flow_mapping(self, inputs, extra_inputs=None,use_gauss_projection=True):
        
        [res, log_det] = inputs
        """
        if(force_embedding_coordinates):

            assert(force_intrinsic_coordinates==False)
            assert(res.shape[1]==(self.dimension+1))
            ## check if we are in base simplex, if so embed in canonical simplex because it is forced
            if(self.always_parametrize_in_embedding_space==False):
                # typically this flow takes intrinsic coordinates, but we overwrite this here...
                res, log_det=self.canonical_simplex_to_base_simplex([res, log_det])

        elif(force_intrinsic_coordinates):
            assert(res.shape[1]==self.dimension)
            ## check if we are in canonical simplex, if so project to base simplex because it is forced
            if(self.always_parametrize_in_embedding_space):
                res, log_det=self.base_simplex_to_canonical_simplex([res, log_det])
        else:

            if(self.always_parametrize_in_embedding_space):
                assert(res.shape[1]==(self.dimension+1))
            else:
                assert(res.shape[1]==self.dimension)
        """

        res, log_det=self._inv_flow_mapping([res, log_det], extra_inputs=extra_inputs)
        
        if(self.project_from_gauss_to_simplex):

            if(self.always_parametrize_in_embedding_space):
                res, log_det=self.canonical_simplex_to_base_simplex([res, log_det])
            
            res, log_det=self.base_simplex_to_non_uniform_box([res, log_det])
         
            ### from skewed box to straight box
          
            res, log_det=self.non_uniform_box_to_gauss([res, log_det], use_gauss_projection=use_gauss_projection)

        return res, log_det

    def flow_mapping(self, inputs, extra_inputs=None, force_embedding_coordinates=False, force_intrinsic_coordinates=False):

        [res, log_det]=inputs

        if(self.project_from_gauss_to_simplex):
            uniform, log_det=self.gauss_to_non_uniform_box(inputs)

            res, log_det=self.non_uniform_box_to_base_simplex([uniform, log_det])
        
            if(self.always_parametrize_in_embedding_space):
                res, log_det=self.base_simplex_to_canonical_simplex([res, log_det])

            
        ### all flow mappings happen at the base simplex
        res, log_det= self._flow_mapping([res, log_det], extra_inputs=extra_inputs)

        """
        if(force_embedding_coordinates):

            assert(force_intrinsic_coordinates==False)

            ## check if we are in base simplex, if so embed in canonical simplex because it is forced
            if(res.shape[1]==self.dimension):
                res, log_det=self.base_simplex_to_canonical_simplex([res, log_det])

        elif(force_intrinsic_coordinates):

            ## check if we are in canonical simplex, if so project to base simplex because it is forced
            if(res.shape[1]==(self.dimension+1)):
                res, log_det=self.canonical_simplex_to_base_simplex([res, log_det])
        """

        return res, log_det

    def get_desired_init_parameters(self):

        ## offset + specific layer params

        par_list=[]
    
        par_list.append(self._get_desired_init_parameters())

        return torch.cat(par_list)

    def init_params(self, params):

        assert(len(params)==self.total_param_num)

   
        self._init_params(params)

    def _embedding_conditional_return(self, x):
        ## make a dim+1 dimensional vector by summing the d+1 dimension to 1

        ## embed in higher dimension
        if(x.shape[1]==self.dimension):
            x,_=self.base_simplex_to_canonical_simplex([x,0.0])

        return x

    def _embedding_conditional_return_num(self): 
        return self.dimension+1

    def transform_target_space(self, x, log_det=0.0, transform_from="default", transform_to="embedding"):
        
        currently_intrinsic=True
        if(transform_from=="default"):
            if(self.always_parametrize_in_embedding_space):
                assert(x.shape[1]==(self.dimension+1))
                currently_intrinsic=False
            else:
                assert(x.shape[1]==self.dimension)

        elif(transform_from=="intrinsic"):
            assert(x.shape[1]==self.dimension)

        elif(transform_from=="embedding"):
            assert(x.shape[1]==(self.dimension+1))
            currently_intrinsic=False


        if(transform_to=="default"):
            if(self.always_parametrize_in_embedding_space):
                if(currently_intrinsic):
                    new_x, new_ld=self.base_simplex_to_canonical_simplex([x,log_det])

                    return new_x, new_ld
                else:
                    return x, log_det
            else:
                if(currently_intrinsic):
                    return x, log_det
                else:
                    new_x, new_ld=self.canonical_simplex_to_base_simplex([x,log_det])
                    return new_x, new_ld

        elif(transform_to=="intrinsic"):
            if(currently_intrinsic):
                return x, log_det
            else:
                new_x, new_ld=self.canonical_simplex_to_base_simplex([x,log_det])
                return new_x, log_det
        elif(transform_to=="embedding"):
            if(currently_intrinsic):
                new_x, new_ld=self.base_simplex_to_canonical_simplex([x,log_det])

                return new_x, new_ld
            else:
                return x, log_det

    def _get_layer_base_dimension(self):
        """ 
        Usually this is just the dimension .. if we work in embedding space and do not project, base space is actually dim+1
        """

        if(self.always_parametrize_in_embedding_space==True and self.project_from_gauss_to_simplex==False):
            return self.dimension+1

        else:
            return self.dimension

    #############################################################################

    ## implement the following by specific euclidean child layers

    def _init_params(self, params):
        raise NotImplementedError
    def _get_desired_init_parameters(self):
        raise NotImplementedError

    def _inv_flow_mapping(self, inputs, extra_inputs=None):
        raise NotImplementedError

    def _flow_mapping(self, inputs, extra_inputs=None):
        raise NotImplementedError

    def obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        
        self._obtain_layer_param_structure(param_dict, extra_inputs=extra_inputs, previous_x=previous_x, extra_prefix=extra_prefix)


    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 
        """ 
        Implemented by Euclidean sublayers.
        """
     
        raise NotImplementedError




    