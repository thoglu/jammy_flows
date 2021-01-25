from torch import nn
import torch
import numpy

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}

def list_from_str(spec):
    if(spec==""):
        return []
        
    return list(tuple(map(int, spec.split("-"))))


class AmortizableMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, low_rank_approximations=0, skip_connections=False, skip_first_to_last_if_similar_dim=False, nonlinearity="tanh",  use_permanent_parameters=True, svd_mode="smart"):
        super(AmortizableMLP, self).__init__()

        self.input_dim=input_dim
        
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

        if(type(low_rank_approximations)==int):
            self.low_rank_approximations=(len(self.hidden_dims)+1)*[low_rank_approximations]
        if(type(low_rank_approximations)==list):
            self.low_rank_approximations=low_rank_approximations
            for i in self.low_rank_approximations:
                assert(type(i)==int)
        elif(type(low_rank_approximations)==str):
            self.low_rank_approximations=list_from_str(low_rank_approximations)

        assert( len(self.low_rank_approximations)==(len(self.hidden_dims)+1) )


        self.output_dim=output_dim
       
        self.use_permanent_parameters=use_permanent_parameters

        self.skip_connections=skip_connections

        self.svd_mode=svd_mode
       
        self.skip_first_to_last_if_similar_dim=skip_first_to_last_if_similar_dim

        
        self.maxranks=[]
        self.used_ranks=[]
        self.activations=[]

        ## for no hidden layers or one hidden layer, it makes no sense to have skip connection
        inputs=[self.input_dim]+self.hidden_dims
        outputs=self.hidden_dims+[self.output_dim]
       
        self.inputs=inputs
        self.outputs=outputs

        self.full_skip_connection=False

        num_hidden_layers=len(self.hidden_dims)

        if(self.inputs[0]==self.outputs[-1]):
            if(num_hidden_layers>0):
                if(skip_first_to_last_if_similar_dim):
                    if(skip_connections):
                        self.full_skip_connection=True

        self.mappings=nn.ModuleList()
        self.identities=[]

        ## contains number of us /vs/and bs needed in each step
        self.u_s=[]
        self.v_s=[]
        self.b_s=[]
        self.smart_flags=[]
        self.sigmas=[]

        num_amortization_params=0

        for ind, input in enumerate(inputs):
            self.maxranks.append(min(inputs[ind], outputs[ind]))
            if(self.low_rank_approximations[ind]>0):
                self.used_ranks.append(min(self.maxranks[ind], self.low_rank_approximations[ind]))
            else:
                self.used_ranks.append(self.maxranks[ind])

            self.b_s.append(self.outputs[ind])

            if(self.svd_mode=="naive"):
                print("naive")
                self.u_s.append(self.used_ranks[ind]*self.outputs[ind])
                self.v_s.append(self.used_ranks[ind]*self.inputs[ind])
                num_amortization_params+=(self.used_ranks[ind]*inputs[ind]+self.used_ranks[ind]*outputs[ind]+outputs[ind])
            elif(self.svd_mode=="smart"):

                max_num_pars=(self.inputs[ind]*self.outputs[ind])

                if( (self.used_ranks[ind]*(self.inputs[ind]+self.outputs[ind]) ) < max_num_pars ):
                    self.u_s.append(self.used_ranks[ind]*self.outputs[ind])
                    self.v_s.append(self.used_ranks[ind]*self.inputs[ind])
                    num_amortization_params+=(self.used_ranks[ind]*inputs[ind]+self.used_ranks[ind]*outputs[ind]+outputs[ind])
                    self.smart_flags.append(0)
                else:
                    
                    self.u_s.append(self.inputs[ind]*self.outputs[ind])
                    self.v_s.append(0)
                    num_amortization_params+=(inputs[ind]*outputs[ind]+outputs[ind])
                    self.smart_flags.append(1)
            elif(self.svd_mode=="explicit_svd"):

                self.u_s.append(self.used_ranks[ind]*self.inputs[ind])
                self.v_s.append(self.used_ranks[ind]*self.outputs[ind])
                self.sigmas.append(self.used_ranks[ind])
            
                num_amortization_params+=(self.used_ranks[ind]*inputs[ind]+self.used_ranks[ind]*outputs[ind]+outputs[ind]+self.used_ranks[ind])
            else:
                print("unknown svd mode", self.svd_mode)
                sys.exit(-1)
            

            

            add_activation=True
            
            ## no activation in first mapping if full linear connection
            ## this will allow for a full linear mapping between input and output
            #if(ind==0 and full_linear_connection):
            #    add_activation=False

            ## no activation in last mapping (first if len=1, i.e. no hidden layer)
            if(ind==(len(inputs)-1)):
                add_activation=False

            if(add_activation):
                self.activations.append(NONLINEARITIES[nonlinearity])
            else:
                self.activations.append(lambda x: x)
            
            use_identity=0
            if(outputs[ind]==inputs[ind]):
                if(skip_connections):
                    use_identity=1

            self.identities.append(use_identity)
            
        self.num_amortization_params=num_amortization_params

        ## the actual parameter vector that holds a compressed representation of the MLP
        if(use_permanent_parameters):
          
            self.u_v_b_pars=nn.Parameter(torch.randn(self.num_amortization_params).type(torch.double).unsqueeze(0)) 

            ## kaiming init ?
            
            if(True):
                index=0

                for ind in range(len(self.activations)):

                    ## this layer is not low-rank aproximated, use kaiming init
                    if(self.smart_flags[ind]==1):

                      
                        ## init weights
                        nn.init.kaiming_uniform_(self.u_v_b_pars[index:index+self.u_s[ind]], a=numpy.sqrt(5))
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.u_v_b_pars[index:index+self.u_s[ind]])
                        bound = 1 / numpy.sqrt(fan_in)
                        
                        ## init biases
                        #bs=nn.Parameter(torch.randn(outputs[ind]))
                        nn.init.uniform_(self.u_v_b_pars[index+self.u_s[ind]:index+self.u_s[ind]+self.b_s[ind]], -bound, bound)

                    ## increase index, also for layers that are low-rank approximated
                    index+=self.u_s[ind]+self.v_s[ind]+self.b_s[ind]
            

        else:
           
            self.u_v_b_pars=torch.zeros(self.num_amortization_params).unsqueeze(0)

        #######################
        #print("first 10 uvb after init .. ", self.u_v_b_pars[0,:10])

    def initialize_uvbs(self, init_b=None):

        if(self.use_permanent_parameters):
          
            index=0

            for ind in range(len(self.activations)):

                ## this layer is not low-rank aproximated, use kaiming init
                if(self.smart_flags[ind]==1):

                    ## init weights

                    nn.init.kaiming_uniform_(self.u_v_b_pars.data[:,index:index+self.u_s[ind]], a=numpy.sqrt(5))
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.u_v_b_pars.data[:,index:index+self.u_s[ind]])
                    bound = 1 / numpy.sqrt(fan_in)
                    
                    ## init biases
                    #bs=nn.Parameter(torch.randn(outputs[ind]))
                    nn.init.uniform_(self.u_v_b_pars.data[:,index+self.u_s[ind]:index+self.u_s[ind]+self.b_s[ind]], -bound, bound)


                ## increase index, also for layers that are low-rank approximated
                index+=self.u_s[ind]+self.v_s[ind]+self.b_s[ind]

            self.u_v_b_pars.data/=1000.0

            if(init_b is not None):

                self.u_v_b_pars.data[0,-self.b_s[-1]:]=init_b


            #print("first 10 uvb after explicit init .. ", self.u_v_b_pars[0,:10])


    def _adaptive_matmul(self, matrix, vec):

        if(vec.dim()==2):
            ret=torch.bmm(matrix, vec.unsqueeze(-1)).squeeze(-1)
        elif(vec.dim()==3):

            ret=vec.matmul(matrix.permute(0,2,1))
        else:
            this_dim=vec.dim()
            extra_broadcast=this_dim-3
            slices=[slice(None,None)]+[None]*extra_broadcast+[slice(None,None),slice(None,None)]
           
            ret=vec.matmul( (matrix.permute(0,2,1))[slices])
        return ret

    def forward(self, i, extra_inputs=None):


        amortization_params=self.u_v_b_pars.to(i)

        if(extra_inputs is not None):

            if(self.use_permanent_parameters):
                raise Exception("MLP uses permanent parameters but extra inputs are given in forward. This is not allowed!")
            assert(len(extra_inputs[0])==self.num_amortization_params)

            ## amortization only works currently if input is 2 dimensional (batch_dim X emb_dim)
            assert(i.dim()==2)

            amortization_params=amortization_params+torch.reshape(extra_inputs, [i.shape[0], self.num_amortization_params])
        else:
            amortization_params=amortization_params.repeat(i.shape[0], 1)

        prev=i

        ## we only work with batched inputs
        assert prev.dim() > 1
        
        ## number of activations counts all mappings, even if there is no nonlinearity (activation in this case is just a 1-to-1 mapping)
        for ind in range(len(self.activations)):
            #print("IND ", ind)
            nonlinear=0

            this_rank=self.used_ranks[ind]


            unsqueezed_weights=0
            unsqueezed_bias=0
        
          
            this_u, amortization_params=amortization_params[:, :self.u_s[ind]], amortization_params[:, self.u_s[ind]:]
            this_v, amortization_params=amortization_params[:, :self.v_s[ind]], amortization_params[:, self.v_s[ind]:]
            this_b, amortization_params=amortization_params[:, :self.b_s[ind]], amortization_params[:, self.b_s[ind]:]

            """
            if(self.svd_mode=="naive"):
                sys.exit(-1)
                this_u=this_u.view(i.shape[0], int(this_u.shape[1]/this_rank), this_rank) # U
                this_v=this_v.view(i.shape[0], this_rank,int(this_v.shape[1]/this_rank)) # V^T
                    
                A=torch.bmm(this_u, this_v)
            """ 
                
            if(self.svd_mode=="smart"):
                
                ## the low-rank decomposition would actualy take more parameters than the full matrix .. just do a standard full matrix product
                if(self.smart_flags[ind]):
                    
                    A=this_u.view(i.shape[0], self.outputs[ind], self.inputs[ind])
                    """
                    if(prev.dim()==2):
                        nonlinear=torch.bmm(A, prev.unsqueeze(-1)).squeeze(-1)
                    elif(prev.dim()==3):

                        nonlinear=prev.matmul(A.permute(0,2,1))
                    else:
                        this_dim=prev.dim()
                        extra_broadcast=this_dim-3
                        slices=[slice(None,None)]+[None]*extra_broadcast+[slice(None,None),slice(None,None)]
                       
                        nonlinear=prev.matmul( (A.permute(0,2,1))[slices])
                    """

                    nonlinear=self._adaptive_matmul(A, prev)

                else:
                    ## we do the standard svd decomposition (without proper normalization) .. normalization is implicit in the u/v definition
                    this_u=this_u.view(i.shape[0],  int(this_u.shape[1]/this_rank), this_rank) # U
                    this_v=this_v.view(i.shape[0], this_rank, int(this_v.shape[1]/this_rank)) # V^T

                    #nonlinear=torch.bmm(this_v, prev.unsqueeze(-1))
                    #nonlinear=torch.bmm(this_u, nonlinear).squeeze(-1)

                    res_intermediate=self._adaptive_matmul(this_v, prev)
                    nonlinear=self._adaptive_matmul(this_u, res_intermediate)

            elif(self.svd_mode=="explicit_svd"):
                
                ## code not working anymore
                raise NotImplementedError()

                this_u=this_u.view(i.shape[0], int(this_u.shape[1]/this_rank), this_rank) # U
                this_v=this_v.view(i.shape[0], this_rank,int(this_v.shape[1]/this_rank)) # V^T

                #print(amortization_params)
                this_sigmas, amortization_params=amortization_params[:, :self.sigmas[ind]], amortization_params[:, self.sigmas[ind]:]
                this_sigmas=NONLINEARITIES["softplus"](this_sigmas.view(i.shape[0], this_rank))+torch.ones_like(this_sigmas)*0.1
                
                #print("SIGMAS", this_sigmas[0])
                res=torch.diag_embed(this_sigmas)
             
                A=torch.bmm(this_u, res)
                
                A=torch.bmm(A, this_v)

            ## add bias
            bias_broadcast=nonlinear.dim()-this_b.dim()
            assert (bias_broadcast >= 0)

            slices=[slice(None,None)]+[None]*bias_broadcast+[slice(None,None)]
            nonlinear=nonlinear+this_b[slices]

            nonlinear=self.activations[ind](nonlinear)

            #print(self.identities)
            if(self.identities[ind]):
                ## add skip connection
                prev=nonlinear+prev
            else:
                # no skip connection
                prev=nonlinear

      
        ## skip connection from beginning to end?
        if(self.full_skip_connection):
            prev=i+prev

       
        return prev