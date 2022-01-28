from torch import nn
import torch
import numpy
import time

from .extra_functions import NONLINEARITIES, list_from_str

class AmortizableMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, highway_mode=0, low_rank_approximations=0, nonlinearity="tanh",  use_permanent_parameters=True, svd_mode="smart", precise_mlp_structure=dict()):
        """
        input dim: input dimension of the MLP
        output dim: output dimension of the MLP
        hidden_dims: depending on "extended" mode, defines hidden dimensions
        highway_mode: 0 - like a normal MLP
                       1 - An extra Linear function is added to the final result, adding effectively a skip connection from input to output with additionaly weight multiplication
                       2 - Hidden dimensions define hidden dims of 1-hidden layer MLPs that are interleaved and always added to the previous layer. They depend on the previous layer and the first layer input,
                           and likely perform better when output_dim < input_dim. 

                       3 - 
        """             
        super(AmortizableMLP, self).__init__()

        self.input_dim=input_dim
        self.output_dim=output_dim
        self.highway_mode=highway_mode
        self.total_low_rank_approximations=low_rank_approximations
        self.use_permanent_parameters=use_permanent_parameters
        self.nonlinearity=nonlinearity
        self.svd_mode=svd_mode

        if(len(precise_mlp_structure.keys())>0):
            self.sub_mlp_structures=precise_mlp_structure
            
            if(self.highway_mode>0):
                assert("linear_highway" in self.sub_mlp_structures.keys())
                
        else:


           
            ## setup hidden dims correctly
            if(type(hidden_dims)==str):
                self.hidden_dims=list_from_str(hidden_dims)
            elif(type(hidden_dims)==int):
                self.hidden_dims=[hidden_dims]
            elif(type(hidden_dims)==list):
                self.hidden_dims=hidden_dims
                for i in self.hidden_dims:
                    assert(type(i)==int)
            else:
                raise Exception("Unsupported type ", type(hidden_dims), " for hidden_dims .. can be int/str/list of ints")

            ## check that low rank approximations match hidden dims depending on highway_mode
            num_matrices=len(self.hidden_dims)+1
           
            if(self.highway_mode==0):
                num_matrices=len(self.hidden_dims)+1
            elif(self.highway_mode==1):
                num_matrices=len(self.hidden_dims)+2
            elif(self.highway_mode>=2):
                num_matrices=2*len(self.hidden_dims)+1

            assert(self.highway_mode <=4 and self.highway_mode >=0)
            
            ##########################

            ## setup low rank approximations
            if(type(low_rank_approximations)==int):
                self.total_low_rank_approximations=num_matrices*[low_rank_approximations]
            if(type(low_rank_approximations)==list):
                self.total_low_rank_approximations=low_rank_approximations
                for i in self.total_low_rank_approximations:
                    assert(type(i)==int)
            elif(type(low_rank_approximations)==str):
                self.total_low_rank_approximations=list_from_str(low_rank_approximations)

            # assert len of low rank approximations matches number of matrices
            assert( len(self.total_low_rank_approximations)==num_matrices)

            self.sub_mlp_structures=dict()
            self.sub_mlp_structures["mlp_list"]=[]

            if(self.highway_mode<2):
                ## for no hidden layers or one hidden layer, it makes no sense to have skip connection


                inputs=[self.input_dim]+self.hidden_dims
                outputs=self.hidden_dims+[self.output_dim]

                mlp_dict=dict()
                mlp_dict["inputs"]=inputs
                mlp_dict["outputs"]=outputs
                #mlp_dict["max_ranks"]=[]
                #mlp_dict["used_ranks"]=[]
                mlp_dict["activations"]=[]
                self._fill_activations(mlp_dict, mlp_dict["inputs"])
                
                mlp_dict["low_rank_approximations"]=self.total_low_rank_approximations
                mlp_dict["add_final_bias"]=True
            
                mlp_dict["num_u_s"]=[]
                mlp_dict["num_v_s"]=[]
                mlp_dict["num_b_s"]=[]
                mlp_dict["full_weight_matrix_flags"]=[]
                mlp_dict["sigmas"]=[]
                mlp_dict["svd_mode"]=self.svd_mode

                self.sub_mlp_structures["mlp_list"].append(mlp_dict)

                ## only add linear skip connection if hidden dims > 0
                if(self.highway_mode==1):

                    self.sub_mlp_structures["mlp_list"][0]["low_rank_approximations"]=self.total_low_rank_approximations[:-1]
                    self.sub_mlp_structures["mlp_list"][0]["add_final_bias"]=False

                    ## reset mlp list if no hidden dims are given
                    if(len(self.hidden_dims)==0):
                        self.sub_mlp_structures["mlp_list"]=[]
                    ## generate an extra linear layer

                    mlp_dict=dict()
                    mlp_dict["inputs"]=[self.input_dim]
                    mlp_dict["outputs"]=[self.output_dim]
                    #mlp_dict["max_ranks"]=[]
                    #mlp_dict["used_ranks"]=[]
                    mlp_dict["activations"]=[]
                    self._fill_activations(mlp_dict, mlp_dict["inputs"])

                    mlp_dict["low_rank_approximations"]=self.total_low_rank_approximations[-1:]
                    mlp_dict["add_final_bias"]=True
                
                    mlp_dict["num_u_s"]=[]
                    mlp_dict["num_v_s"]=[]
                    mlp_dict["num_b_s"]=[]
                    mlp_dict["full_weight_matrix_flags"]=[]
                    mlp_dict["sigmas"]=[]
                    mlp_dict["svd_mode"]=self.svd_mode

                    self.sub_mlp_structures["linear_highway"]=mlp_dict

            else:
                num_mlp_matrices=num_matrices-1
                ## iterative adds by 1-hidden layer MLPs

                assert(num_mlp_matrices==2*len(self.hidden_dims))

                target_dim=self.output_dim

                ## highway_mode=2 -> input to each MLP is the input_dim
                mlp_start_dim=self.input_dim

                if(self.highway_mode==3):
                    ## input to each MLP is the target output
                    mlp_start_dim=self.output_dim

                elif(self.highway_mode==4):
                    ## input to each MLP is the joint input and output dim
                    mlp_start_dim=self.input_dim+self.output_dim

                for ind in range(len(self.hidden_dims)):

                    mlp_dict=dict()

                    ## the first MLP is always taking the input dim as input
                    if(ind==0):
                        mlp_dict["inputs"]=[self.input_dim, self.hidden_dims[ind]]
                    else:
                        mlp_dict["inputs"]=[mlp_start_dim, self.hidden_dims[ind]]
                    mlp_dict["outputs"]=[self.hidden_dims[ind], target_dim]
                    #mlp_dict["max_ranks"]=[]
                    #mlp_dict["used_ranks"]=[]
                    mlp_dict["activations"]=[]
                    self._fill_activations(mlp_dict, mlp_dict["inputs"])

                    mlp_dict["low_rank_approximations"]=self.total_low_rank_approximations[ind*2:ind*2+2]
                    mlp_dict["add_final_bias"]=False
                
                    mlp_dict["num_u_s"]=[]
                    mlp_dict["num_v_s"]=[]
                    mlp_dict["num_b_s"]=[]
                    mlp_dict["full_weight_matrix_flags"]=[]
                    mlp_dict["sigmas"]=[]
                    mlp_dict["svd_mode"]=self.svd_mode

                    self.sub_mlp_structures["mlp_list"].append(mlp_dict)

                ## the linear part
                mlp_dict=dict()
                mlp_dict["inputs"]=[self.input_dim]
                mlp_dict["outputs"]=[self.output_dim]
                #mlp_dict["max_ranks"]=[]
                #mlp_dict["used_ranks"]=[]
                mlp_dict["activations"]=[]
                self._fill_activations(mlp_dict, mlp_dict["inputs"])
                mlp_dict["low_rank_approximations"]=self.total_low_rank_approximations[-1:]
                mlp_dict["add_final_bias"]=True
            
                mlp_dict["num_u_s"]=[]
                mlp_dict["num_v_s"]=[]
                mlp_dict["num_b_s"]=[]
                mlp_dict["full_weight_matrix_flags"]=[]
                mlp_dict["sigmas"]=[]
                mlp_dict["svd_mode"]=self.svd_mode

                self.sub_mlp_structures["linear_highway"]=mlp_dict

        num_amortization_params=0

        for sub_mlp_index in range(len(self.sub_mlp_structures["mlp_list"])):

            this_num_params=self.initialize_uv_structure(self.sub_mlp_structures["mlp_list"][sub_mlp_index])
            num_amortization_params+=this_num_params

        if("linear_highway" in self.sub_mlp_structures.keys()):
            this_num_params=self.initialize_uv_structure(self.sub_mlp_structures["linear_highway"])
            num_amortization_params+=this_num_params

        # the number of parameters is the total number of all MLP params
        self.num_amortization_params=num_amortization_params

        ## the actual parameter vector that holds a compressed representation of the MLP
        if(use_permanent_parameters):
          
            self.u_v_b_pars=nn.Parameter(torch.randn(self.num_amortization_params).type(torch.double).unsqueeze(0)) 

            self.initialize_uvbs()

        else:
           
            self.u_v_b_pars=torch.zeros(self.num_amortization_params).unsqueeze(0)

        #######################
        #print("first 10 uvb after init .. ", self.u_v_b_pars[0,:10])

    def _fill_activations(self, mlp_def, input):

        for ind in range(len(input)):

            if(ind==(len(input)-1)):
                mlp_def["activations"].append(lambda x: x)
            else:
                #print("nonlinear act ...........")
                mlp_def["activations"].append(NONLINEARITIES[self.nonlinearity])

    def initialize_uv_structure(self, mlp_def):
            
        num_amortization_params=0

        max_ranks=[]
        used_ranks=[]

        #print("MLP DEF ", mlp_def)

        for ind in range(len(mlp_def["inputs"])):

            max_ranks.append(min(mlp_def["inputs"][ind], mlp_def["outputs"][ind]))

            if(mlp_def["low_rank_approximations"][ind]>0):
                used_ranks.append(min(max_ranks[ind], mlp_def["low_rank_approximations"][ind]))
            else:
                if(mlp_def["svd_mode"]=="naive"):
                    # append 0
                    used_ranks.append(0)
                else:
                    ## smart mode just uses the max rank
                    used_ranks.append(max_ranks[ind])

            if(mlp_def["svd_mode"]=="naive"):
                    
                if(used_ranks[ind]>0):

                    mlp_def["num_u_s"].append(used_ranks[ind]*mlp_def["outputs"][ind])
                    mlp_def["num_v_s"].append(used_ranks[ind]*mlp_def["inputs"][ind])
                    num_amortization_params+=(used_ranks[ind]*mlp_def["inputs"][ind]+used_ranks[ind]*mlp_def["outputs"][ind])
                    
                    mlp_def["full_weight_matrix_flags"].append(0)
                else:

                    mlp_def["num_u_s"].append(mlp_def["outputs"][ind]*mlp_def["inputs"][ind])
                    mlp_def["num_v_s"].append(0)
                    num_amortization_params+=(mlp_def["outputs"][ind]*mlp_def["inputs"][ind])
                    
                    mlp_def["full_weight_matrix_flags"].append(1)

                
            elif(mlp_def["svd_mode"]=="smart"):
                ## smart mode takes a full matrix if the low-rank approximation has more parameters
                
                max_num_pars=(mlp_def["inputs"][ind]*mlp_def["outputs"][ind])

                # smart flag = 0 -> standard low rank approximation for this matrix
                if( ((used_ranks[ind]*(mlp_def["inputs"][ind]+mlp_def["outputs"][ind]) ) < max_num_pars) and (mlp_def["low_rank_approximations"][ind]>0) ):
                    mlp_def["num_u_s"].append(used_ranks[ind]*mlp_def["outputs"][ind])
                    mlp_def["num_v_s"].append(used_ranks[ind]*mlp_def["inputs"][ind])
                    num_amortization_params+=(used_ranks[ind]*mlp_def["inputs"][ind]+used_ranks[ind]*mlp_def["outputs"][ind])
                    mlp_def["full_weight_matrix_flags"].append(0)
                else:
                    # smart_flag = 1 -> full matrix (all parameters of the full matrix are stored in the *v* vector)
                    mlp_def["num_u_s"].append(mlp_def["inputs"][ind]*mlp_def["outputs"][ind])
                    mlp_def["num_v_s"].append(0)
                    num_amortization_params+=(mlp_def["inputs"][ind]*mlp_def["outputs"][ind])
                    mlp_def["full_weight_matrix_flags"].append(1)

            elif(mlp_def["svd_mode"]=="explicit_svd"):

                raise NotImplementedError()
                """
                mlp_def["num_u_s"].append(used_ranks[ind]*mlp_def["inputs"][ind])
                mlp_def["num_v_s"].append(used_ranks[ind]*mlp_def["outputs"][ind])
                mlp_def["sigmas"].append(used_ranks[ind])
            
                num_amortization_params+=(used_ranks[ind]*mlp_def["inputs"][ind]+used_ranks[ind]*mlp_def["outputs"][ind]+mlp_def["outputs"][ind]+used_ranks[ind])
                """
            else:
                raise Exception("unknown svd mode", self.svd_mode)

            
            ## no activation in first mapping if full linear connection
            ## this will allow for a full linear mapping between input and output
            #if(ind==0 and full_linear_connection):
            #    add_activation=False
            
            ## no activation in last mapping (first if len=1, i.e. no hidden layer)
            if(ind==(len(mlp_def["inputs"])-1)):

                #mlp_def["activations"].append(lambda x: x)

                if(mlp_def["add_final_bias"]):
                    mlp_def["num_b_s"].append(mlp_def["outputs"][ind])
                else:
                    mlp_def["num_b_s"].append(0)
                #print("linear act ->>>>>>>>>")
            else:
                #print("nonlinear act ...........")
                #mlp_def["activations"].append(NONLINEARITIES[self.nonlinearity])
                mlp_def["num_b_s"].append(mlp_def["outputs"][ind])

            num_amortization_params+=mlp_def["num_b_s"][-1]
            
            
        mlp_def["max_ranks"]=max_ranks
        mlp_def["used_ranks"]=used_ranks
        mlp_def["num_params"]=num_amortization_params

        return num_amortization_params

          
    def initialize_uvbs(self, init_b=None):

        init_tensor=torch.randn(self.num_amortization_params, dtype=torch.float64).unsqueeze(0)
    
        index=0
        #print("IN INIT ", init_b)
        for mlp_def in self.sub_mlp_structures["mlp_list"]:

            for ind in range(len(mlp_def["inputs"])):

                ## this layer is not low-rank aproximated, use kaiming init
                if(mlp_def["full_weight_matrix_flags"][ind]==1):

                    ## init weights
                    nn.init.kaiming_uniform_(init_tensor[:,index:index+mlp_def["num_u_s"][ind]], a=numpy.sqrt(5))
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(init_tensor[:,index:index+mlp_def["num_u_s"][ind]])
                    bound = 1 / numpy.sqrt(fan_in)
                    
                    ## init biases
                    #bs=nn.Parameter(torch.randn(outputs[ind]))
                    if(mlp_def["num_b_s"][ind]>0):
                        nn.init.uniform_(init_tensor[:,index+mlp_def["num_u_s"][ind]:index+mlp_def["num_u_s"][ind]+mlp_def["num_b_s"][ind]], -bound, bound)

                ## increase index, also for layers that are low-rank approximated
                index+=mlp_def["num_u_s"][ind]+mlp_def["num_v_s"][ind]+mlp_def["num_b_s"][ind]

        ## the linear function is at the very end if it is there
        if("linear_highway" in self.sub_mlp_structures.keys()):

            mlp_def=self.sub_mlp_structures["linear_highway"]

            nn.init.kaiming_uniform_(init_tensor[:,index:index+mlp_def["num_u_s"][0]], a=numpy.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(init_tensor[:,index:index+mlp_def["num_u_s"][0]])
            bound = 1 / numpy.sqrt(fan_in)

            nn.init.uniform_(init_tensor[:,index+mlp_def["num_u_s"][0]:index+mlp_def["num_u_s"][0]+mlp_def["num_b_s"][0]], -bound, bound)

        init_tensor=init_tensor/1000.0


        ## initialize last b pars if desired
        if(init_b is not None):

            relevant_mlp=self.sub_mlp_structures["mlp_list"][-1]

            if("linear_highway" in self.sub_mlp_structures.keys()):

                relevant_mlp=self.sub_mlp_structures["linear_highway"]

            init_tensor[0,-relevant_mlp["num_b_s"][-1]:]=init_b

        if(self.use_permanent_parameters):
            self.u_v_b_pars.data=init_tensor
        else:
            ## return desired init for amortization purposes
            return init_tensor.squeeze(0)


    def _adaptive_matmul(self, matrix, vec):


        if(vec.dim()==2):
            #res1=torch.matmul(matrix, vec)
            ## A_inputs,outputs * v_outputs
            ## A_i,j * v_j
            ret=torch.einsum("...ij,...j",matrix,vec)
            
            #ret=torch.bmm(matrix, vec.unsqueeze(-1)).squeeze(-1)
        elif(vec.dim()==3):

            ret=vec.matmul(matrix.permute(0,2,1))
        else:
            this_dim=vec.dim()
            extra_broadcast=this_dim-3
            slices=[slice(None,None)]+[None]*extra_broadcast+[slice(None,None),slice(None,None)]
           
            ret=vec.matmul( (matrix.permute(0,2,1))[slices])
        return ret

    def apply_amortized_mlp(self, mlp_def, prev_argument, params):

        amortization_params=params
        prev=prev_argument
        batch_size=prev_argument.shape[0]
        
        
        for ind in range(len(mlp_def["outputs"])):
            #print("MLP ", ind)
            nonlinear=0

            this_rank=mlp_def["used_ranks"][ind]
        
            this_u, amortization_params=amortization_params[:, :mlp_def["num_u_s"][ind]], amortization_params[:, mlp_def["num_u_s"][ind]:]
            this_v, amortization_params=amortization_params[:, :mlp_def["num_v_s"][ind]], amortization_params[:, mlp_def["num_v_s"][ind]:]
            this_b, amortization_params=amortization_params[:, :mlp_def["num_b_s"][ind]], amortization_params[:, mlp_def["num_b_s"][ind]:]

            if(mlp_def["svd_mode"]=="smart" or mlp_def["svd_mode"]=="naive"):
                
                ## the low-rank decomposition would actualy take more parameters than the full matrix .. just do a standard full matrix product
                if(mlp_def["full_weight_matrix_flags"][ind]):
                    # no svd decomposition, the whole weight matrix is stored in the "u" vector

                 
                    A=this_u.view(-1, mlp_def["outputs"][ind], mlp_def["inputs"][ind])
                    
                  
                    nonlinear=self._adaptive_matmul(A, prev)

                else:
                    #print("ADVANCED MAT")
                    ## we do the standard svd decomposition (without proper normalization) .. normalization is implicit in the u/v definition


                    this_u=this_u.view(this_u.shape[0],  int(this_u.shape[1]/this_rank), this_rank) # U
                    this_v=this_v.view(this_v.shape[0], this_rank, int(this_v.shape[1]/this_rank)) # V^T

                    res_intermediate=self._adaptive_matmul(this_v, prev)
                    nonlinear=self._adaptive_matmul(this_u, res_intermediate)

            elif(self.svd_mode=="explicit_svd"):
                
                ## code not working anymore - has to be rewritten eventually
                raise NotImplementedError()
                """
                this_u=this_u.view(batch_size, int(this_u.shape[1]/this_rank), this_rank) # U
                this_v=this_v.view(batch_size, this_rank,int(this_v.shape[1]/this_rank)) # V^T

                #print(amortization_params)
                this_sigmas, amortization_params=amortization_params[:, :mlp_def["sigmas"][ind]], amortization_params[:, mlp_def["sigmas"][ind]:]
                this_sigmas=NONLINEARITIES["softplus"](this_sigmas.view(i.shape[0], this_rank))+torch.ones_like(this_sigmas)*0.1
                
                #print("SIGMAS", this_sigmas[0])
                res=torch.diag_embed(this_sigmas)
             
                A=torch.bmm(this_u, res)
                
                A=torch.bmm(A, this_v)
                """

            ## add bias

            if(mlp_def["num_b_s"][ind]>0):
                #print("have bias")
                bias_broadcast=nonlinear.dim()-this_b.dim()
                assert (bias_broadcast >= 0)

                slices=[slice(None,None)]+[None]*bias_broadcast+[slice(None,None)]
                nonlinear=nonlinear+this_b[slices]
            #else:
                #print("NO BIAS", mlp_def)
            prev=mlp_def["activations"][ind](nonlinear)

        #print(amortization_params.shape)

        return prev, amortization_params


    def forward(self, i, extra_inputs=None):

        amortization_params=self.u_v_b_pars.to(i)

        
        if(extra_inputs is not None):

            if(self.use_permanent_parameters):
                raise Exception("MLP uses permanent parameters but extra inputs are given in forward. This is not allowed!")

            assert(extra_inputs.shape[1]==self.num_amortization_params), ("Extra inputs dimension (%d) does not match number of amortization params of MLP (%d) " % (extra_inputs.shape[1], self.num_amortization_params))

            amortization_params=extra_inputs

        ## we only work with batched inputs
        ## currently dim=2
        #assert i.dim() > 1
        assert(i.dim()==2)

        prev=0.0

        ## if the linear MLP is defined, it contains the bias, and it is defined to have the parameters in the very end of the amortization_params
        ## this way it is straightforward to initalize the very last params as the bias vec
        if(self.highway_mode>0):
            
            linear_def=self.sub_mlp_structures["linear_highway"]

            linear_amortization_params=amortization_params[:,-linear_def["num_params"]:]
            amortization_params=amortization_params[:,:-linear_def["num_params"]]

            linear_result, _= self.apply_amortized_mlp(linear_def, i, linear_amortization_params)
            #print("FINISHED LINEAR APPLICATION ....")
            assert(_.shape[1]==0)

            prev=linear_result


        if(self.highway_mode<2):
            
            ## number of activations counts all mappings, even if there is no nonlinearity (activation in this case is just a 1-to-1 mapping)
            if(len(self.sub_mlp_structures["mlp_list"])>0):
                ## only when hidden dims are given is the mlp_list filled
                mlp_def=self.sub_mlp_structures["mlp_list"][0]
               
                nonlinear, amortization_params=self.apply_amortized_mlp(mlp_def, i, amortization_params)
                
                prev=prev+nonlinear
        else:
            
            ## thie first MLP in highway mode >= 2 is always taking the input

            if(len(self.sub_mlp_structures["mlp_list"]) > 0):
                first_mlp=self.sub_mlp_structures["mlp_list"][0]
                #print("applying the first MLP ", first_mlp)
                nonlinear, amortization_params=self.apply_amortized_mlp(first_mlp, i, amortization_params)

                if(self.highway_mode==2):
                    
                    next_input=i

                elif(self.highway_mode==3):
                    next_input=prev+nonlinear

                elif(self.highway_mode==4):
                    next_input=torch.cat([i, prev+nonlinear], dim=1)

                prev=prev+nonlinear

                ## the other MLP's inputs depend on the type of highway_mode
                for mlp_def in self.sub_mlp_structures["mlp_list"][1:]:
                    #print("OTHER ", )
                    nonlinear, amortization_params=self.apply_amortized_mlp(mlp_def, next_input, amortization_params)

                    ## set next input
                    if(self.highway_mode==2):
                        next_input=i

                    elif(self.highway_mode==3):
                        next_input=prev+nonlinear

                    elif(self.highway_mode==4):
                        next_input=torch.cat([i, prev+nonlinear], dim=1)

                    ### add nonlinear to previous result
                    prev=prev+nonlinear

        return prev