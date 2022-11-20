import collections
import numpy
import copy
import sys

import torch
from torch import nn

from typing import Union

from . import default
from .. import amortizable_mlp
from ..extra_functions import list_from_str, NONLINEARITIES

def copy_attributes(objfrom, objto, names):
    for n in names:
        assert(hasattr(objfrom, n)), "'%s' is not an attribute of *from class*" % n
        
        v = getattr(objfrom, n)
        setattr(objto, n, v)

class fully_amortized_pdf(nn.Module):

    def __init__(
        self,
        pdf_defs, 
        flow_defs, 
        options_overwrite=dict(),
        conditional_input_dim=None,
        inner_mlp_dims_sub_pdfs="128",
        inner_mlp_ranks=10,
        inner_mlp_highway_mode=1,
        amortization_mlp_dims="128",
        amortization_mlp_use_custom_mode=False,
        amortization_mlp_ranks=0,
        amortization_mlp_highway_mode=0,
        predict_log_normalization=False,
        skip_mlp_initialization=False
    ):  
        """
        A fully amortized PDF, where in contrast to the standard autoregressive conditional PDF, also the whole autoregressive transformation, including MLPs, is amortized.

        See documentation for an exact structural difference. Accessed via *jammy_flows.fully_amortized_pdf*.
        
        Parameters:
            pdf_defs (str): String of characters describing the joint PDF structure: Sub-space structure is spearated by "+".
                            Example: "e2+s1+s2", describes a joint PDF over a 2-dimensional euclidean space, a 1-sphere and a 2-sphere: a joint 5-dimensional PDF.
            
            flow_defs (str): A string, that describes how each conditional subfflow defined in "pdfs_defs" is structured in terms of normalizing-flow layers.
                             Example: "gg+m+n" to describe a layer structure compatible with the other example "e2+s1+s2". Two letters mean two consecutive applications of a certain flow layer, 3 letters three etc.
                             Each layer holds their own parameters.
            
            options_overwrite (dict): Dictionary to overwrite default options of individual flow layers.

            conditional_input_dim (None/int): Conditional input dimension if a conditional PDF.

            inner_mlp_dims_sub_pdfs (str/list(str)): Hidden structure of MLP for each sub-manifold. 
            inner_mlp_ranks (int/str): The maximum rank allowed for the inner custom mlp mappings. 
            inner_mlp_highway_mode (int): The highway mode used for the custom MLP. See *amortizable_mlp* class for more details.
        
            amortization_mlp_dims (str/list(str)): Hidden structure of the amortization MLP.
            amortization_mlp_use_custom_mode (bool): Indicates, whether custom mode should be used for all MLPs in the autoregressive linking process.
            amortization_mlp_ranks (int/str): The maximum rank allowed for the amortization MLP. See *amortizable_mlp* class for more details.
            amortization_mlp_highway_mode (int): The highway mode used for the amortization MLP. See *amortizable_mlp* class for more details.
            predict_log_normalization (bool): Predict log-mean of Poisson distribution.
            skip_mlp_initialization (bool): Indicates, whether to skip MLP inits entirely. Can be used for custom MLP initialization.
          
        """

        super().__init__()

        self.conditional_input_dim=conditional_input_dim
        self.use_amortizable_mlp=amortization_mlp_use_custom_mode


        self.pdf_to_amortize=default.pdf(pdf_defs,
                                       flow_defs,
                                       options_overwrite=options_overwrite,
                                       conditional_input_dim=None,
                                       hidden_mlp_dims_sub_pdfs=inner_mlp_dims_sub_pdfs,
                                       predict_log_normalization=False,
                                       use_custom_low_rank_mlps=True,
                                       rank_of_mlp_mappings_sub_pdfs=inner_mlp_ranks,
                                       custom_mlp_highway_mode=inner_mlp_highway_mode,
                                       amortize_everything=True,
                                       skip_mlp_initialization=skip_mlp_initialization
                                       )

        # mirror some attributes
        copy_attributes(self.pdf_to_amortize, self, ["pdf_defs_list", "total_target_dim"])


        mlp_hidden_dims=list_from_str(amortization_mlp_dims)

        if(self.use_amortizable_mlp):

            

            self.amortization_mlp=amortizable_mlp.AmortizableMLP(conditional_input_dim, 
                                                      mlp_hidden_dims, 
                                                      self.pdf_to_amortize.total_number_amortizable_params, 
                                                      low_rank_approximations=amortization_mlp_ranks, 
                                                      use_permanent_parameters=True, # 
                                                      highway_mode=amortization_mlp_highway_mode, 
                                                      svd_mode="smart")

            self.total_param_num=self.amortization_mlp.num_amortization_params

        else:
            
            par_counter=0

            mlp_in_dims = [conditional_input_dim] + mlp_hidden_dims
            mlp_out_dims = mlp_hidden_dims + [self.pdf_to_amortize.total_number_amortizable_params]

            nn_list = []
            for i in range(len(mlp_in_dims)):
               
                l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                nn_list.append(l)
                
                if i < (len(mlp_in_dims) - 1):
                    nn_list.append(NONLINEARITIES["tanh"])
                
                par_counter+=mlp_in_dims[i]*mlp_out_dims[i]+mlp_out_dims[i]
            
            self.amortization_mlp=torch.nn.Sequential(*nn_list)

            self.total_param_num=par_counter


        self.double()

        assert(predict_log_normalization == False), "TODO: Still need to implement log normalization prediction here."

        if(skip_mlp_initialization==False):
            self.init_params()


    def forward(self, 
                x, 
                conditional_input=None, 
                force_embedding_coordinates=False, 
                force_intrinsic_coordinates=False):
        """
        Calculates log-probability at the target *x*. Also returns some other quantities that are calculated as a consequence.

        Parameters:

            x (Tensor): Target position to calculate log-probability at. Must be of shape (B,D), where B = batch dimension.
            conditional_input (Tensor/None): Amortization input for conditional PDFs. If given, must be of shape (B,A), where A is the conditional input dimension defined in __init__.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the input *x*.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the input *x*. 
        
        Returns:

            Tensor
                Log-probability, shape = (B,)

            Tensor
                Log-probability at base distribution, shape = (B,)

            Tensor
                Position at base distribution, shape = (B,D)

        """

        assert(conditional_input is not None), "This is by design a conditional PDF .. we require conditional input!"

        all_flow_params=self.amortization_mlp(conditional_input)    

        return self.pdf_to_amortize(x, amortization_parameters=all_flow_params, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)


    def sample(self, 
               conditional_input=None, 
               samplesize=1,  
               seed=None, 
               allow_gradients=False, 
               force_embedding_coordinates=False, 
               force_intrinsic_coordinates=False):

        """ 
        Samples from the (conditional) PDF. 

        Parameters:
            conditional_input (Tensor/None): Of shape N x D where N is the batch size and D the input space dimension if given. Else None.
            samplesize (int): Samplesize.
            seed (None/int):
            allow_gradients (bool): If False, does not propagate gradients and saves memory by not building the graph. Off by default, so has to be switched on for training.
            force_embedding_coordinates (bool): Enforces embedding coordinates for the sample.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates for the sample.
        
        Returns:

            Tensor
                Sample in target space.
            Tensor
                Sample in base space.
            Tensor
                Log-pdf evaluation in target space
            Tensor
                Log-pdf evaluation in base space

        """

        assert(conditional_input is not None), "This is by design a conditional PDF .. we require conditional input!"

        all_flow_params=self.amortization_mlp(conditional_input) 

        return self.pdf_to_amortize.sample(amortization_parameters=all_flow_params, 
                                           seed=seed,
                                           allow_gradients=allow_gradients,
                                           force_embedding_coordinates=force_embedding_coordinates, 
                                           force_intrinsic_coordinates=force_intrinsic_coordinates)



    def init_params(self, data=None, damping_factor=1000.0):

        global_amortization_init=self.pdf_to_amortize.init_params(data=data, damping_factor=damping_factor)

        ## initialize MLP with global desired init
        if(self.use_amortizable_mlp):
            self.amortization_mlp.initialize_uvbs(fix_final_bias=global_amortization_init, prev_damping_factor=damping_factor)

        else:
            # initialize all layers with default kaiming init and additional damping
            # then set final bias to desired init
            for internal_layer in self.amortization_mlp:
                
                # test if this is a real Linear layer or a nonlinearity
                if(hasattr(internal_layer, "weight")):

                    # only initialize if a Linear layer
                    nn.init.kaiming_uniform_(internal_layer.weight.data, a=numpy.sqrt(5))
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(internal_layer.weight.data)
                    bound = 1 / numpy.sqrt(fan_in)

                    nn.init.uniform_(internal_layer.bias.data, -bound, bound)
                    
                    internal_layer.weight.data/=damping_factor
                    internal_layer.bias.data/=damping_factor
                
            # finally overwrite bias to be equivalent to desired parameters at initialization
            
            self.amortization_mlp[-1].bias.data=global_amortization_init.data

    def count_parameters(self, verbose=False):

        """
            Counts parameters of the model. It does not matter, if all paramters are amortized or not, will always return the same.
            
            Parameters:
                verbose (bool): Prints out number of parameters. Differentiates amortization from non-amortization params.
            
            Returns:
                int
                    Number of parameters (incl. amortization params).
        """

        if(verbose):
            print("Amoritized PDF param count: \n\
                   target PDF pars predicted (not real): %d \n\
                   Total PDF (MLP) pars: %d" % (self.pdf_to_amortize.total_number_amortizable_params, 
                                                self.total_param_num))

        return self.total_param_num




