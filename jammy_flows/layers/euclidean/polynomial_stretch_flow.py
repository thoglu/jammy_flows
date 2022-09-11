import torch
from torch import nn
import numpy
from .. import bisection_n_newton as bn
from .. import layer_base
from . import euclidean_base

import math
import torch.nn.functional as F
import torch.distributions as tdist


normal_dist=tdist.Normal(0, 1)

def obtain_bounded_variable_fn(min_val=0, max_val=1):

    difference=max_val-min_val

    log_difference=numpy.log(difference)

    def fn(x):

        res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-x).unsqueeze(-1)], dim=-1)

        result=torch.exp(log_difference-torch.logsumexp(res, dim=-1))+min_val

        return result

    return fn

class psf_block(euclidean_base.euclidean_base):
    def __init__(self, 
                 dimension, 
                 num_transforms=3, 
                 num_householder_iter=-1, 
                 use_permanent_parameters=False, 
                 model_offset=0, 
                 exact_mode=True):
        """ 
        Polynomial stretch flow - Symbol: "p"

        A polynomial flow that uses *abs(x)* to make arbitrary polynomials work as flow-mappings. Has interesting structure but does not seem to work very well in practice.
        """
        super().__init__(dimension=dimension, use_permanent_parameters=use_permanent_parameters, model_offset=model_offset)

        ##############

        if num_householder_iter == -1:
            self.householder_iter = dimension #min(dimension, 10)
        else:
            self.householder_iter = num_householder_iter

        self.use_householder=True
        if(self.householder_iter==0):
            self.use_householder=False

        #elf.layer = layer
        self.dimension = dimension
        self.num_transforms = num_transforms

        self.width_min=0.1
        self.exp_min=0.1

        self.exact_mode=exact_mode

        #self.num_params_per_transform_per_item=self.num_transforms*self.dimension
        self.num_params_per_item=num_transforms*self.dimension

        ## 5 param types * dimension * number of transforms
        self.total_transform_params=self.num_params_per_item*5

        ## transformation functions for widths and the exponent
        self.width_transform=obtain_bounded_variable_fn(min_val=0.01, max_val=10)
        self.log_exponent_transform=obtain_bounded_variable_fn(min_val=-3, max_val=3)

        init_log_value=-0.1053605156 # 0.9

        if use_permanent_parameters:
            self.log_widths1 = nn.Parameter(torch.ones(num_transforms, self.dimension).type(torch.double).unsqueeze(0)*init_log_value)
        else:
            self.log_widths1 = torch.zeros(num_transforms, self.dimension).type(torch.double).unsqueeze(0)#.to(device)


        if use_permanent_parameters:
            self.log_widths2 = nn.Parameter(torch.ones(num_transforms, self.dimension).type(torch.double).unsqueeze(0)*init_log_value)
        else:
            self.log_widths2 = torch.zeros(num_transforms, self.dimension).type(torch.double).unsqueeze(0)

        #self.means1 = nn.Parameter(torch.ones(num_transforms, self.dimension).type(torch.double).unsqueeze(0)*0.1)
      
        if use_permanent_parameters:
            self.means1 = nn.Parameter(torch.ones(num_transforms, self.dimension).type(torch.double).unsqueeze(0)*0.1)
        else:
            self.means1 = torch.zeros(num_transforms, self.dimension).type(torch.double).unsqueeze(0)#.to(device)
        
        
        if use_permanent_parameters:
            self.means2 = nn.Parameter(torch.ones(num_transforms, self.dimension).type(torch.double).unsqueeze(0)*0.1)
        else:
            self.means2 = torch.zeros(num_transforms, self.dimension).type(torch.double).unsqueeze(0)#.to(device)

        if use_permanent_parameters:
            self.log_exponent = nn.Parameter(torch.ones(num_transforms, self.dimension).type(torch.double).unsqueeze(0)*init_log_value)
        else:
            self.log_exponent = torch.zeros(num_transforms, self.dimension).type(torch.double).unsqueeze(0)#.to(device)
    
        
     
        self.num_householder_params=0

        if self.use_householder:
            
            if(use_permanent_parameters):
                self.vs = nn.Parameter(
                    torch.randn(self.householder_iter, dimension).type(torch.double).unsqueeze(0)
                )
            else:
                self.vs = torch.zeros(self.householder_iter, dimension).type(torch.double).unsqueeze(0) 

            self.num_householder_params=self.householder_iter*self.dimension

        self.total_param_num+=self.total_transform_params+self.num_householder_params

    
    
    def fw_loop(self, x, means1, means2, widths1, widths2, exponents, tr_iter):

        xp=(x-means1[:,tr_iter,:])/widths1[:,tr_iter,:]

        x=(torch.sign(xp)*(torch.sign(xp)*xp)**(exponents[:,tr_iter,:]))*widths2[:,tr_iter,:]+means2[:,tr_iter,:]

        return x

    def fw_loop_derivative(self, x, means1, means2, widths1, widths2, exponents, tr_iter):
     
        xp=(x-means1[:,tr_iter,:])/widths1[:,tr_iter,:]
        
        x=((exponents[:,tr_iter,:])/widths1[:,tr_iter,:])*torch.sign(xp)*torch.sign(xp)*((torch.sign(xp)*xp)**(exponents[:,tr_iter,:]-1))*widths2[:,tr_iter,:] 

        return x


    def rw_loop(self, x, means1, means2, widths1, widths2, exponents, tr_iter):

        xp=(x-means2[:,tr_iter,:])/widths2[:,tr_iter,:]

   
        x=(torch.sign(xp)*(torch.sign(xp)*xp)**(1.0/exponents[:,tr_iter,:]))*widths1[:,tr_iter,:]+means1[:,tr_iter,:]

        return x

    def rw_loop_derivative(self, x, means1, means2, widths1, widths2, exponents, tr_iter):
     
        xp=(x-means2[:,tr_iter,:])/widths2[:,tr_iter,:]

        x= ((1.0/exponents[:,tr_iter,:])/widths2[:,tr_iter,:])*torch.sign(xp)*torch.sign(xp)*((torch.sign(xp)*xp)**(1.0/exponents[:,tr_iter,:]-1))*widths1[:,tr_iter,:] 

        return x


    def compute_householder_matrix(self, vs, device=torch.device("cpu")):

        Q = torch.eye(self.dimension, device=device).type(torch.double).unsqueeze(0).repeat(vs.shape[0], 1,1)
       
        for i in range(self.householder_iter):
        
            v = vs[:,i].reshape(-1,self.dimension, 1).to(device)
            
            v = v / v.norm(dim=1).unsqueeze(-1)

            Qi = torch.eye(self.dimension, device=device).type(torch.double).unsqueeze(0) - 2 * torch.bmm(v, v.permute(0, 2, 1))

            Q = torch.bmm(Q, Qi)

        return Q

 


    def _flow_mapping(self, inputs, extra_inputs=None, verbose=False, lower=-1e5, upper=1e5): 

        [x, log_det] = inputs

        extra_input_counter=0
        if self.use_householder:
            this_vs=self.vs
            if(extra_inputs is not None):
                

                this_vs=this_vs+torch.reshape(extra_inputs[:,:self.num_householder_params], [x.shape[0], self.vs.shape[1], self.vs.shape[2]])

                
                extra_input_counter+=self.num_householder_params

            rotation_matrix = self.compute_householder_matrix(this_vs, device=x.device)

            if(rotation_matrix.shape[0]!=x.shape[0]):
                if(rotation_matrix.shape[0]<x.shape[0]):
                    rotation_matrix=rotation_matrix.repeat(x.shape[0], 1,1)
                else:
                    raise Exception("rotation matrix first dim is larger than data first dim (batch dim) .. taht should never happen!")


        log_widths1=self.log_widths1.to(x)
        log_widths2=self.log_widths2.to(x)

        means1=self.means1.to(x)
        means2=self.means2.to(x)

        log_exponent=self.log_exponent.to(x)

        if(extra_inputs is not None):
          
            log_widths1=log_widths1+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.log_widths1.shape[1],  self.log_widths1.shape[2]])
            extra_input_counter+=self.num_params_per_item

            log_widths2=log_widths2+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.log_widths2.shape[1],  self.log_widths2.shape[2]])
            extra_input_counter+=self.num_params_per_item

            means1=means1+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.means1.shape[1],  self.means1.shape[2]])
            extra_input_counter+=self.num_params_per_item

            means2=means2+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.means2.shape[1],  self.means2.shape[2]])
            extra_input_counter+=self.num_params_per_item

            log_exponent=log_exponent+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.log_exponent.shape[1],  self.log_exponent.shape[2]])
            extra_input_counter+=self.num_params_per_item

        """
        widths1 = torch.exp(log_widths1)+self.width_min
        widths2 = torch.exp(log_widths2)+self.width_min
        exponents=torch.exp(log_exponent)+self.exp_min
        """
        
        widths1=self.width_transform(log_widths1)
        widths2=self.width_transform(log_widths2)
        exponents=torch.exp(self.log_exponent_transform(log_exponent))
        

        for tr_iter in range(self.num_transforms):

            if(self.exact_mode==True):
                log_det=log_det+torch.log(self.fw_loop_derivative(x, means1, means2, widths1, widths2, exponents, tr_iter)).sum(axis=1)
               
                x=self.fw_loop(x, means1, means2, widths1, widths2, exponents, tr_iter)
            else:

                ## use older bisection method as it is only for debugging
                x=bn.inverse_bisection_n_newton_slow(self.rw_loop, self.rw_loop_derivative, x, means1, means2, widths1, widths2, exponents, tr_iter, min_boundary=lower, max_boundary=upper, num_bisection_iter=25, num_newton_iter=20)
                log_deriv=torch.log(self.rw_loop_derivative(x, means1, means2, widths1, widths2, exponents, tr_iter))

                #print("new positions ..", res)
                log_det=log_det-log_deriv.sum(axis=-1)

            """

            xp=(x-means1[:,tr_iter,:])/widths1[:,tr_iter,:]

            x=(torch.sign(xp)*(torch.sign(xp)*xp)**(exponents[:,tr_iter,:]))*widths2[:,tr_iter,:]+means2[:,tr_iter,:]

            log_det+=torch.log( ((exponents[:,tr_iter,:])/widths1[:,tr_iter,:])*torch.sign(xp)*torch.sign(xp)*((torch.sign(xp)*xp)**(exponents[:,tr_iter,:]-1))*widths2[:,tr_iter,:] ).sum(axis=1)
            """
       
        if self.use_householder:
         
            x = torch.bmm(rotation_matrix, x.unsqueeze(-1)).squeeze(-1)
        
        return x, log_det


    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        
        [x, log_det] = inputs

        log_widths1=self.log_widths1.to(x)
        log_widths2=self.log_widths2.to(x)

        means1=self.means1.to(x)
        means2=self.means2.to(x)

        log_exponent=self.log_exponent.to(x)

        extra_input_counter=0
        if self.use_householder:
            this_vs=self.vs

          
            if(extra_inputs is not None):

                #print("extra inputs ", extra_inputs[:,:self.num_householder_params])
                this_vs=this_vs+torch.reshape(extra_inputs[:,:self.num_householder_params], [x.shape[0], self.vs.shape[1], self.vs.shape[2]])
                #print(this_vs)

                
                extra_input_counter+=self.num_householder_params
        
            rotation_matrix = self.compute_householder_matrix(this_vs, device=x.device)

            if(rotation_matrix.shape[0]!=x.shape[0]):
                if(rotation_matrix.shape[0]<x.shape[0]):
                    rotation_matrix=rotation_matrix.repeat(x.shape[0], 1,1)
                else:
                    raise Exception("rotation matrix first dim is larger than data first dim (batch dim) .. taht should never happen!")
                
            x = torch.bmm(rotation_matrix.permute(0,2,1), x.unsqueeze(-1)).squeeze(-1)  # uncomment

        
        if(extra_inputs is not None):
            
            log_widths1=log_widths1+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.log_widths1.shape[1],  self.log_widths1.shape[2]])
            extra_input_counter+=self.num_params_per_item

            log_widths2=log_widths2+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.log_widths2.shape[1],  self.log_widths2.shape[2]])
            extra_input_counter+=self.num_params_per_item

            means1=means1+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.means1.shape[1],  self.means1.shape[2]])
            extra_input_counter+=self.num_params_per_item

            means2=means2+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.means2.shape[1],  self.means2.shape[2]])
            extra_input_counter+=self.num_params_per_item

            log_exponent=log_exponent+torch.reshape(extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item], [x.shape[0] , self.log_exponent.shape[1],  self.log_exponent.shape[2]])
            extra_input_counter+=self.num_params_per_item


        """
        widths1 = torch.exp(log_widths1)+self.width_min
        widths2 = torch.exp(log_widths2)+self.width_min
        exponents=torch.exp(log_exponent)+self.exp_min
        """
        
        widths1=self.width_transform(log_widths1)
        widths2=self.width_transform(log_widths2)
        exponents=torch.exp(self.log_exponent_transform(log_exponent))
        

        for tr_iter in reversed(range(self.num_transforms)):
            
            log_det=log_det+torch.log(self.rw_loop_derivative(x, means1, means2, widths1, widths2, exponents, tr_iter)).sum(axis=1)
           
            x=self.rw_loop(x, means1, means2, widths1, widths2, exponents, tr_iter)

        return x, log_det

    def _get_desired_init_parameters(self):

        desired_param_vec=[]

        ## logwidth1, logwidth2, means1, means2, log_exponent, (householder)

        ## householder
        if(self.num_householder_params > 0):
            desired_param_vec.append(torch.randn(self.householder_iter*self.dimension))

        ## log-width 1
        #desired_param_vec.append(torch.ones(self.num_params_per_item)*-0.1053605156)
        desired_param_vec.append(torch.ones(self.num_params_per_item)*0.0)

        ## log-width 2
        #desired_param_vec.append(torch.ones(self.num_params_per_item)*-0.1053605156)
        desired_param_vec.append(torch.ones(self.num_params_per_item)*0.0)
        ## mean1
        desired_param_vec.append(torch.randn(self.num_params_per_item))

        ## mean2
        desired_param_vec.append(torch.randn(self.num_params_per_item))

        ## log_exponent
        #desired_param_vec.append(torch.ones(self.num_params_per_item)*-0.1053605156)
        desired_param_vec.append(torch.ones(self.num_params_per_item)*0.0)

        return torch.cat(desired_param_vec)
        
    def _init_params(self, params):
       
        counter=0
        if self.use_householder:
           
              
            self.vs.data=torch.reshape(params[:self.num_householder_params], [1, self.householder_iter, self.dimension])

            counter+=self.num_householder_params

        self.log_widths1.data=torch.reshape(params[counter:counter+self.num_params_per_item], [1 , self.log_widths1.shape[1],  self.log_widths1.shape[2]])
        counter+=self.num_params_per_item

        self.log_widths2.data=torch.reshape(params[counter:counter+self.num_params_per_item], [1 , self.log_widths2.shape[1],  self.log_widths2.shape[2]])
        counter+=self.num_params_per_item

        self.means1.data=torch.reshape(params[counter:counter+self.num_params_per_item], [1 , self.means1.shape[1],  self.means1.shape[2]])
        counter+=self.num_params_per_item

        self.means2.data=torch.reshape(params[counter:counter+self.num_params_per_item], [1 , self.means2.shape[1],  self.means2.shape[2]])
        counter+=self.num_params_per_item

        self.log_exponent.data=torch.reshape(params[counter:counter+self.num_params_per_item], [1 , self.log_exponent.shape[1],  self.log_exponent.shape[2]])
        counter+=self.num_params_per_item

    def _obtain_layer_param_structure(self, param_dict, extra_inputs=None, previous_x=None, extra_prefix=""): 

        extra_input_counter=0

        if self.use_householder:
            this_vs=self.vs

            if(extra_inputs is not None):
                

                this_vs=extra_inputs[:,:self.num_householder_params]

                
                extra_input_counter+=self.num_householder_params


            param_dict[extra_prefix+"hh_params"]=this_vs.data

        log_widths1=self.log_widths1.data
        log_widths2=self.log_widths2.data

        means1=self.means1.data
        means2=self.means2.data

        log_exponent=self.log_exponent

        if(extra_inputs is not None):
          
            log_widths1=log_widths1+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item]
            extra_input_counter+=self.num_params_per_item

            log_widths2=log_widths2+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item]
            extra_input_counter+=self.num_params_per_item

            means1=means1+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item]
            extra_input_counter+=self.num_params_per_item

            means2=means2+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item]
            extra_input_counter+=self.num_params_per_item

            log_exponent=log_exponent+extra_inputs[:,extra_input_counter:extra_input_counter+self.num_params_per_item]
            extra_input_counter+=self.num_params_per_item

            ##########

        param_dict[extra_prefix+"log_widths1"]=log_widths1.data
        param_dict[extra_prefix+"log_widths2"]=log_widths2.data

        param_dict[extra_prefix+"means1"]=means1.data
        param_dict[extra_prefix+"means2"]=means2.data

        param_dict[extra_prefix+"log_exponent"]=log_exponent.data
    



