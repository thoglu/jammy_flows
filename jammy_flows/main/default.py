import torch
from torch import nn

from ..flow_options import check_flow_option, obtain_default_options, obtain_overall_flow_info
from ..extra_functions import list_from_str, NONLINEARITIES, recheck_sampling, find_init_pars_of_chained_blocks, _calculate_coverage
from ..amortizable_mlp import AmortizableMLP


import collections
import numpy
import copy
import sys
import scipy
import math
import time

try:
    import healpy
except:
    print("Cannot use healpy functionality. Install healpy, if you need to do entropy scanning!")

from scipy.special import iv, i0, i1
import scipy.linalg

from typing import Union

## used to peek into param generator which can be empty
def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first

class pdf(nn.Module):

    def __init__(
        self,
        pdf_defs, 
        flow_defs, 
        options_overwrite=dict(),
        conditional_input_dim=None,
        amortization_mlp_dims="128",
        predict_log_normalization=False,
        join_poisson_and_pdf_description=False,
        hidden_mlp_dims_poisson="128",
        rank_of_mlp_mappings_poisson=0,
        amortization_mlp_use_custom_mode=False,
        amortization_mlp_ranks=0,
        amortization_mlp_highway_mode=0,
        amortize_everything=False,
        use_as_passthrough_instead_of_pdf=False,
        skip_mlp_initialization=False
    ):  
        """
        The main class of the project that defines a pytorch normalizing-flow PDF.
        The two main actions are evaluating the log-probability and sampling. Accessed via *jammy_flows.pdf*.

        Parameters:
            pdf_defs (str): String of characters describing the joint PDF structure: Sub-space structure is spearated by "+".
                            Example: "e2+s1+s2", describes a joint PDF over a 2-dimensional euclidean space, a 1-sphere and a 2-sphere: a joint 5-dimensional PDF.
            
            flow_defs (str): A string, that describes how each conditional subfflow defined in "pdfs_defs" is structured in terms of normalizing-flow layers.
                             Example: "gg+m+n" to describe a layer structure compatible with the other example "e2+s1+s2". Two letters mean two consecutive applications of a certain flow layer, 3 letters three etc.
                             Each layer holds their own parameters.
            
            options_overwrite (dict): Dictionary to overwrite default options of individual flow layers.

            conditional_input_dim (None/int/list(int)): Conditional input dimension if a conditional PDF. If a list is given, defines conditional input for each sub-pdf. None to define non-conditional PDF.

            amortization_mlp_dims (str/list(str)): Hidden structure of MLP for each sub-manifold. 

            predict_log_normalization (bool): Predict log-mean of Poisson distribution

            joint_poisson_and_pdf_description (bool): If predicting log-mean, predict it together with other flow parameters if this is a conditional PDF. Only makes sense when conditional_input_dim is not None.
    
            hidden_mlp_dims_poisson (str/list(str)): If the log-mean is predicted by its own MLP, defines the hidden structure of the MLP.
            
            amortization_mlp_use_custom_mode (bool): Use custom AmortizableMLP class instead of default chained Linear layers.

            rank_of_mlp_mappings_poisson (int): Max rank of custom Poisson predictor MLP matrices.
    
            amortization_mlp_highway_mode (int): Connectivity mode for custom MLPs if used.
    
            amortize_everything (bool): Indicates whether all parameters, including the MLPs, should be amortized.

            use_as_passthrough_instead_of_pdf (bool): Indicates, whether the class acts as a PDF, or only as a flow mapping function of the overall autoregressive flow.
            
            skip_mlp_initialization (bool): Indicates, whether to skip MLP inits entirely. Can be used for custom MLP initialization.
    

        """
        super().__init__()

        self.amortization_mlp_use_custom_mode=amortization_mlp_use_custom_mode
        self.predict_log_normalization=predict_log_normalization
        self.join_poisson_and_pdf_description=join_poisson_and_pdf_description
        self.amortization_mlp_highway_mode=amortization_mlp_highway_mode
        self.amortize_everything=amortize_everything

        if(self.amortize_everything):
            assert(self.predict_log_normalization==False), "Log Poisson prediction works only without full amortization in the default PDF. It can be used in the *fully_amortized_pdf*!"
        
        self.use_as_passthrough_instead_of_pdf=use_as_passthrough_instead_of_pdf
        self.skip_mlp_initialization=skip_mlp_initialization

        ## holds total number of params for amortization - only used if "amortize_everything" set to True
        self.total_number_amortizable_params=None

        if(self.amortize_everything):
            assert(self.amortization_mlp_use_custom_mode), "Amortizing all MLPs requires custom MLPs."
            self.total_number_amortizable_params=0
    
        self.read_model_definition(pdf_defs, 
                                   flow_defs, 
                                   options_overwrite, 
                                   conditional_input_dim, 
                                   amortization_mlp_dims, 
                                   amortization_mlp_ranks)
       
        self.init_flow_structure()
        
        self.hidden_mlp_dims_poisson=hidden_mlp_dims_poisson
        self.rank_of_mlp_mappings_poisson=rank_of_mlp_mappings_poisson

        self.init_encoding_structure()
       
        # add self reference for specific layers
        for subflow_index, subflow_description in enumerate(self.pdf_defs_list):

            layer_descr=self.flow_defs_list[subflow_index]

            for layer_index, layer in enumerate(self.layer_list[subflow_index]):

                ## sphere charts
                if(layer_descr[layer_index]=="c"):

                    layer.set_variables_from_parent(self)

        
        ## initialize params
        self.init_params()

    def read_model_definition(self, 
                              pdf_defs, 
                              flow_defs, 
                              options_overwrite,
                              conditional_input_dim,
                              amortization_mlp_dims,
                              amortization_mlp_ranks):
        
        # list of the pdf defs (e.g. e2 for 2-d Euclidean) for each subpdf
        # i.e. e2+s2 will yield 2 entries, one for each manifold
        self.pdf_defs_list = pdf_defs.split("+")

        # respective flow functions for each submanifold
        self.flow_defs_list = flow_defs.split("+")

        ## dictionary holding the options used by the flows of this pdf
        self.flow_opts = dict()

        #top_flow_def_keys=[k for k in options_overwrite.keys()]

        ## loop through flow defs and initialize sub-manifold specific options
        for ind, cur_flow_defs in enumerate(self.flow_defs_list):

            this_iter_flow_abbrvs=cur_flow_defs

            self.flow_opts[ind]=[]

            cur_flow_index=0

            for flow_abbrv in this_iter_flow_abbrvs:

                ## first copy default options
                self.flow_opts[ind].append(obtain_default_options(flow_abbrv))

                assert(len(self.flow_opts[ind])==(cur_flow_index+1))
                ## make sure default options are consistent
                for opt in self.flow_opts[ind][-1].keys():
                    check_flow_option(flow_abbrv, opt, self.flow_opts[ind][-1][opt])

                ## refine a specific flow within a specific sub-manifold if necessary 
                found_specific=False
                overwrote_default=False     
                for k in options_overwrite.keys():

                    if(type(k)==tuple):

                        assert(type(k[0])==int and type(k[1])==int), "Require 2 ints for tuple-based flow definition! The first indexes the sub-manifold, the second the flow within the manifold."

                        assert( (k[0]>=0) and (k[0]<len(self.flow_defs_list))), "Index of detailed options is outside allowed range of defined autoregressive structure."
                        
                        if(k[0] != ind):
                            continue

                        if(k[1] != cur_flow_index):
                            continue

                        assert(len(options_overwrite[k])==1), "We have detailed flow definition per item, require length of 1 here."
                        
                        found_specific=True

                        for detail_abbrv in options_overwrite[k].keys():

                            assert(detail_abbrv==flow_abbrv)

                            for detail_opt in options_overwrite[k][detail_abbrv].keys():

                                print("sub-manifold (%d - %s - %s) and intra-manifold flow (%d - %s) options overwrite " % (ind, self.pdf_defs_list[ind], cur_flow_defs, cur_flow_index,flow_abbrv ), detail_opt, " with ", options_overwrite[k][detail_abbrv][detail_opt])
                                overwrote_default=True

                                check_flow_option(flow_abbrv, detail_opt, options_overwrite[k][detail_abbrv][detail_opt])

                                self.flow_opts[ind][-1][detail_opt]=options_overwrite[k][detail_abbrv][detail_opt]
                
                if(found_specific==False):
                    ## refine specific sub manifold defs if necessary
                    for k in options_overwrite.keys():

                        if(type(k)==int):

                            assert( (k>=0) and (k<len(self.flow_defs_list))), "Index of detailed options is outside allowed range of defined autoregressive structure."
                            
                            if(k != ind):
                                continue

                            for detail_abbrv in options_overwrite[k].keys():

                                if(detail_abbrv == flow_abbrv):
                                    found_specific=True
                                    
                                    for detail_opt in options_overwrite[k][detail_abbrv].keys():

                                        print("sub-manifold (%d - %s - %s) and intra-manifold flow (%d - %s) options overwrite " % (ind, self.pdf_defs_list[ind],cur_flow_defs, cur_flow_index,flow_abbrv ), detail_opt, " with ", options_overwrite[k][detail_abbrv][detail_opt])
                                        overwrote_default=True

                                        check_flow_option(flow_abbrv, detail_opt, options_overwrite[k][detail_abbrv][detail_opt])

                                        self.flow_opts[ind][-1][detail_opt]=options_overwrite[k][detail_abbrv][detail_opt]
                    
                if(found_specific==False):

                    ## first check general flow defs
                    for k in options_overwrite.keys():

                        if(k==flow_abbrv):

                            for detail_opt in options_overwrite[k].keys():
                                print("sub-manifold (%d - %s - %s) and intra-manifold flow (%d - %s) options overwrite " % (ind, self.pdf_defs_list[ind], cur_flow_defs, cur_flow_index,flow_abbrv ), detail_opt, " with ", options_overwrite[k][detail_opt])
                                overwrote_default=True

                                check_flow_option(flow_abbrv, detail_opt, options_overwrite[k][detail_opt])

                                self.flow_opts[ind][-1][detail_opt]=options_overwrite[k][detail_opt]


                if(overwrote_default==False):
                    print("sub-manifold (%d - %s - %s) and intra-manifold flow (%d - %s) - using *default* options" % (ind, self.pdf_defs_list[ind], cur_flow_defs, cur_flow_index,flow_abbrv ))

                cur_flow_index+=1

        if(len(self.pdf_defs_list)!=len(self.flow_defs_list)):
            raise Exception("PDF defs list has to be same length as flow defs list, but ... ", self.pdf_defs_list, self.flow_defs_list)
        
        ### now define input stuff
        self.conditional_input_dim=conditional_input_dim

        ### single encoding tensor
        self.encoding_type="single"

        if(type(self.conditional_input_dim)==list):
            for ci in self.conditional_input_dim:
                assert(type(ci)==int)
            # one encoding tensor per sub-pdf
            self.encoding_type="multi"
            
        ## define internal mlp mapping dims as list
        self.amortization_mlp_dims=amortization_mlp_dims
        if(type(self.amortization_mlp_dims)==str):
            self.amortization_mlp_dims=[self.amortization_mlp_dims]*len(self.pdf_defs_list)
        elif(type(self.amortization_mlp_dims)!=list):
            raise Exception("Hidden MLP dimensions must be defined either str or list, received ", type(self.amortization_mlp_dims))
        
        self.amortization_mlp_ranks=amortization_mlp_ranks
        if(type(self.amortization_mlp_ranks)==int):
            self.amortization_mlp_ranks=[self.amortization_mlp_ranks]*len(self.pdf_defs_list)
        elif(type(self.amortization_mlp_ranks)==str):
            self.amortization_mlp_ranks=[self.amortization_mlp_ranks]*len(self.pdf_defs_list)
        elif(type(self.amortization_mlp_ranks)!=list):
            raise Exception("Rank of MLP sub pdfs has to defined as an int or list type!")
     
        ## sub pdf hidden mlp dims must be equal to number of sub pdfs (or sub-pdfs-1 if no encoder present)
        required_hidden_mlp_dims_len=len(self.pdf_defs_list)
        if(required_hidden_mlp_dims_len>0):
            if(len(self.amortization_mlp_dims)!=required_hidden_mlp_dims_len):
                raise Exception("hidden mlp dimension definitions for sub pdfs is wrong length (%d) .. requires length (%d)" %(len(self.amortization_mlp_dims), required_hidden_mlp_dims_len))

        self.layer_list = nn.ModuleList()

        ## might force to use permanent parameters for flows which have no input and are not conditional
        self.force_permanent_parameters_in_first_subpdf = 0

        if self.conditional_input_dim is None:
            if(self.amortize_everything==False):
                self.force_permanent_parameters_in_first_subpdf = 1

    def set_use_embedding_parameters_flag(self, usement_flag, sub_pdf_index=None):
        """
        Resets the default parameter embedding structure. This defines how tensors are to be given to the PDF without
        extra flags like *force_embedding_coordinates* or *force_intrinsic_coordinates*.

        Parameters:

            usement_Flag (bool): Defines if selected sub manifold should switch to embedding coordinates (True) or intrinsic coordinates (False).
            sub_pdf_index (int/None): The index of of the manifold for which to set the flag. If None, sets flag for all.
        """
        assert( (usement_flag==True or usement_flag==False) )

        for ind, ll in enumerate(self.layer_list):

            check_col=False
            if(sub_pdf_index==None):
               
                check_col=True
            else:
                if(ind==sub_pdf_index):
                    check_col=True

            if(check_col):
                for l in ll:
                   
                    l.always_parametrize_in_embedding_space=usement_flag

        ## update the embedding structure with the new information
        self.update_embedding_structure()
    
    def init_flow_structure(self):

        self.num_parameter_list=[]

        ## loop through flows and define layers according to flow definitions
        ## also count number of parameters of each layer

        flow_info=obtain_overall_flow_info()

        for subflow_index, subflow_description in enumerate(self.pdf_defs_list):

            ## append a collection for this subflow which will hold the number of parameters of each layer in the sub-flow
            self.num_parameter_list.append([])

            self.layer_list.append(nn.ModuleList())

            this_num_layers=len(self.flow_defs_list[subflow_index])
          
            for layer_ind, layer_type in enumerate(self.flow_defs_list[subflow_index]):
                if(flow_info[layer_type]["type"]!=subflow_description[0]):
                    raise Exception("layer type ", layer_type, " is not compatible with flow type ", subflow_description)
                  
                this_kwargs = copy.deepcopy(self.flow_opts[subflow_index][layer_ind])

                ## overwrite permanent parameters if desired or necessary
                if(self.force_permanent_parameters_in_first_subpdf and (subflow_index==0)):
                    this_kwargs["use_permanent_parameters"] = 1
                else:
                    this_kwargs["use_permanent_parameters"] = 0
                
                if("s" in subflow_description):
                    ## this flow is a spherical flow, so the first layer should also project from plane to sphere or vice versa
                    if(layer_ind==0 and self.use_as_passthrough_instead_of_pdf==False):
                        this_kwargs["euclidean_to_sphere_as_first"]=1
                    else:
                        this_kwargs["euclidean_to_sphere_as_first"]=0

                elif("i" in subflow_description):
                    ## this flow is an interval flow, so the first layer should also project from real line to interval or vice versa

                    interval_boundaries=subflow_description.split("_")[1:]

                    ## overwrite the boundary parameters
                    if(len(interval_boundaries)==0):
                        this_kwargs["low_boundary"]=0.0
                        this_kwargs["high_boundary"]=1.0
                    else:

                        this_kwargs["low_boundary"]=float(interval_boundaries[0])
                        this_kwargs["high_boundary"]=float(interval_boundaries[1])
                     
                    if(layer_ind==0 and self.use_as_passthrough_instead_of_pdf==False):
                        this_kwargs["euclidean_to_interval_as_first"]=1
                    else:
                        this_kwargs["euclidean_to_interval_as_first"]=0
                elif("a" in subflow_description):

                    if(layer_ind==0 and self.use_as_passthrough_instead_of_pdf==False):
                        this_kwargs["project_from_gauss_to_simplex"]=1
                    else:
                        this_kwargs["project_from_gauss_to_simplex"]=0

                elif("e" in subflow_description):
                    if(layer_type!="x"):
                        if(layer_ind==(this_num_layers-1) and this_kwargs["skip_model_offset"]==0):
                            
                            this_kwargs["model_offset"]=1
                        elif(layer_ind==0):
                            if(layer_type=="g" or layer_type=="h"):
                                if(this_kwargs["replace_first_sigmoid_with_icdf"]>0 and this_kwargs["inverse_function_type"]=="isigmoid"):
                                    this_kwargs["inverse_function_type"]="inormal_partly_precise"

                ## this is not a real parameter to pass on to the flow layers - delete it
                if("skip_model_offset" in this_kwargs):
                    del this_kwargs["skip_model_offset"]
                ## we dont want to pass this to layer
                if(layer_type=="g" or layer_type=="h"):
                    del this_kwargs["replace_first_sigmoid_with_icdf"]

                    
                self.layer_list[subflow_index].append(
                    flow_info[layer_type]["module"](int(subflow_description.split("_")[0][1:]), **this_kwargs)
                )

                # add parameters for the very first layer to total amortizable_params

                self.num_parameter_list[subflow_index].append(self.layer_list[subflow_index][-1].get_total_param_num())

        ## add log-normalization prediction
        self.log_normalization=None

        if(self.predict_log_normalization):

            assert(len(self.pdf_defs_list)==1), "You chose to predict log-lambda, which is only allowed with a single sub-pdf (no autoregressive structure). \
                                                 For autoregressive PDFs with log-lambda prediction, use fully amortized PDFs."

            if self.force_permanent_parameters_in_first_subpdf:
                self.log_normalization=nn.Parameter(torch.randn(1).unsqueeze(0))
            else:
                self.log_normalization=torch.zeros(1).unsqueeze(0)

        self.update_embedding_structure()

    def update_embedding_structure(self):

        self.target_dims_intrinsic=[]
        self.target_dims_embedded=[]
        self.target_dims=[]

        # sub-flow indices for the input dimension to describe target-dim for this sub-flow

        # intrinsic coordinates
        self.target_dim_indices_intrinsic=[]

        # embedding coordinates
        self.target_dim_indices_embedded=[]

        # actually used and potentially mixed coordinates
        self.target_dim_indices=[]

        # base dim indices can be different
        self.base_dim_indices=[]

        
        ## the total dimension if all manifold layers use intrinsic coordinates
        total_dim_intrinsic=0

        ## the total dimension if all manifold layers embed coordinates
        total_dim_embedded=0

        ## the total dimension if we use setting sspecified in the layer ops - some layers can be in embedding coords, others not
        total_dim=0
        total_base_dim=0

        for ll in self.layer_list:

            cur_target_dims_intrinsic=[]
            cur_target_dims_embedded=[]
            cur_base_dims=[]

            use_embedding_dim=False

            for layer in ll:

                layer_target_dim_intrinsic=layer.get_layer_intrinsic_target_dimension()
                layer_target_dim_embedded=layer.get_layer_embedded_target_dimension()
                layer_base_dim=layer.get_layer_base_dimension()

                cur_target_dims_intrinsic.append(layer_target_dim_intrinsic)
                cur_target_dims_embedded.append(layer_target_dim_embedded)
                cur_base_dims.append(layer_base_dim)
                
                if(layer.always_parametrize_in_embedding_space == True):
                    use_embedding_dim=True


            ## target dimensions of all layers of a given subspace must match 
            for lind in range(len(cur_target_dims_intrinsic)-1):
                assert(cur_target_dims_intrinsic[lind]==cur_target_dims_intrinsic[lind+1])

            self.target_dims_intrinsic.append(cur_target_dims_intrinsic[-1])
            self.target_dims_embedded.append(cur_target_dims_embedded[-1])

            if(use_embedding_dim):
                self.target_dims.append(cur_target_dims_embedded[-1])
            else:
                self.target_dims.append(cur_target_dims_intrinsic[-1])

            #self.base_dims_list.append(cur_base_dims)

            ## the base dim of the first layer is really the important one for the total base dim of sub-PFDs
            self.base_dim_indices.append((total_base_dim, total_base_dim+cur_base_dims[0]))
            total_base_dim+=cur_base_dims[0]

            self.target_dim_indices_intrinsic.append((total_dim_intrinsic, total_dim_intrinsic+self.target_dims_intrinsic[-1]))
            total_dim_intrinsic+=self.target_dims_intrinsic[-1]

            self.target_dim_indices_embedded.append((total_dim_embedded, total_dim_embedded+self.target_dims_embedded[-1]))
            total_dim_embedded+=self.target_dims_embedded[-1]

            self.target_dim_indices.append((total_dim, total_dim+self.target_dims[-1]))
            total_dim+=self.target_dims[-1]

        # target dims
        self.total_target_dim_intrinsic=total_dim_intrinsic
        self.total_target_dim_embedded=total_dim_embedded
        self.total_target_dim=total_dim

        # base dim
        self.total_base_dim=total_base_dim
      


    def init_encoding_structure(self):

        self.mlp_predictors=nn.ModuleList()
        self.log_normalization_mlp=None

        if(self.skip_mlp_initialization==False):

            prev_extra_input_num=0

            if(self.join_poisson_and_pdf_description):
                if(len(self.pdf_defs_list)>1):
                    raise Exception("A common poisson log-lambda and flow parameter prediction is currently only supported for a PDF that has a single flow (no autoregressive structure) for simplicity! .. number of autoregressive parts here: ", len(self.pdf_defs_list))
                if(self.conditional_input_dim is None):
                    raise Exception("Flow does not depend on conditional input .. please set 'join_poisson_and_pdf_description' to False, currently True")
           
            for pdf_index, pdf in enumerate(self.pdf_defs_list):


                if(pdf_index==0 and self.conditional_input_dim is None):
                  
                    self.mlp_predictors.append(None)

                    prev_extra_input_num+=self.layer_list[pdf_index][-1]._embedding_conditional_return_num()

                    ### we amortize everything and have no encoder (data_summary_dim=None) -> the first layer column is also amortized
                    if(self.amortize_everything):
                        ##
                        self.total_number_amortizable_params+=sum(self.num_parameter_list[0])
                        
                        if(self.predict_log_normalization and self.join_poisson_and_pdf_description==False):
                            self.total_number_amortizable_params+=1

                    continue

                
                if(pdf_index>0):
                    tot_num_pars_this_pdf=0

                    tot_num_pars_this_pdf=sum(self.num_parameter_list[pdf_index])
        
                    ## if this sub pdf has no parameters, we do not need to define an MLP
                    if(tot_num_pars_this_pdf==0):
                   
                        self.mlp_predictors.append(None)
                        prev_extra_input_num+=self.layer_list[pdf_index][-1]._embedding_conditional_return_num()

                        continue

              
                # calculate the number of output parameters for the mlp
                num_predicted_pars=sum(self.num_parameter_list[pdf_index])
        
                # add log normalization as a single output parameters
                if(self.predict_log_normalization):
                    if(pdf_index==0 and self.join_poisson_and_pdf_description):
                        num_predicted_pars+=1

                if(num_predicted_pars==0):
                    self.mlp_predictors.append(None)
                    prev_extra_input_num+=self.layer_list[pdf_index][-1]._embedding_conditional_return_num()
                    continue

                ## take previous dimensions as input
                this_summary_dim=prev_extra_input_num

                # also add input summary dimensions
                if(self.conditional_input_dim is not None):
                    if(type(self.conditional_input_dim)==int):
                        this_summary_dim+=self.conditional_input_dim
                    else:
                        this_summary_dim+=self.conditional_input_dim[pdf_index]
                
                if(self.amortization_mlp_use_custom_mode):

                    these_hidden_dims=list_from_str(self.amortization_mlp_dims[pdf_index])
                    
                    custom_mlp=AmortizableMLP(this_summary_dim, these_hidden_dims, num_predicted_pars, low_rank_approximations=self.amortization_mlp_ranks[pdf_index], use_permanent_parameters=self.amortize_everything==False, highway_mode=self.amortization_mlp_highway_mode, svd_mode="smart")
                    
                    if(self.amortize_everything):
                        self.total_number_amortizable_params+=custom_mlp.num_amortization_params

                    self.mlp_predictors.append(custom_mlp)

                else:

                    mlp_in_dims = [this_summary_dim] + list_from_str(self.amortization_mlp_dims[pdf_index])
                    mlp_out_dims = list_from_str(self.amortization_mlp_dims[pdf_index]) + [num_predicted_pars]
                   
                    nn_list = []
                    for i in range(len(mlp_in_dims)):
                       
                        l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                        nn_list.append(l)
                        
                        if i < (len(mlp_in_dims) - 1):
                            nn_list.append(NONLINEARITIES["tanh"])
                        
                    
                    self.mlp_predictors.append(torch.nn.Sequential(*nn_list))

               
                prev_extra_input_num+=self.layer_list[pdf_index][-1]._embedding_conditional_return_num()
                
            if(self.predict_log_normalization):

                if(self.conditional_input_dim is not None):

                    ## we only have the encoding summary dim.. poisson mean does not depend on the other PDF pars
                    ## only generate a Poisson MLP if poisson log-lambda and other flow parameters are to be predicted by separate MLPs
                    if(self.join_poisson_and_pdf_description==False):

                        assert(self.amortize_everything==False), "Separate poisson log-lambda predictor not implemented with 'amortize_everything' currently"
                        
                        num_predicted_pars=1

                        this_summary_dim=self.conditional_input_dim
                        if(type(self.conditional_input_dim)==list):
                            this_summary_dim=self.conditional_input_dim[0]

                        if(self.amortization_mlp_use_custom_mode):
                           
                            self.log_normalization_mlp=AmortizableMLP(this_summary_dim, self.hidden_mlp_dims_poisson, num_predicted_pars, low_rank_approximations=self.rank_of_mlp_mappings_poisson, use_permanent_parameters=True, highway_mode=self.amortization_mlp_highway_mode, svd_mode="smart")

                        else:

                            mlp_in_dims = [this_summary_dim] + list_from_str(self.amortization_mlp_dims[pdf_index])
                            mlp_out_dims = list_from_str(self.amortization_mlp_dims[pdf_index]) + [num_predicted_pars]

                            nn_list = []
                            for i in range(len(mlp_in_dims)):
                               
                                l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                                nn_list.append(l)

                                if i < (len(mlp_in_dims) - 1):
                                    nn_list.append(NONLINEARITIES["tanh"])
                                else:
                                    ## initialize some weights
                                    ## TODO.. change hard coded damping factor here
                                    nn_list[-1].weight.data/=1000.0

                                    nn_list[-1].bias.data[0]=-1.0
                                    
                            self.log_normalization_mlp=torch.nn.Sequential(*nn_list)

        else:
            # If we chose to not initilaize MLPs here, make sure we do not separately also predict log normalization
            # Externel initializations should predict the Poisson and PDF description jointly.
            if(self.predict_log_normalization):
                assert(self.join_poisson_and_pdf_description==True)

    def count_parameters(self, verbose=False):
        """
            Counts parameters of the model. It does not matter, if all paramters are amortized or not, will always return the same.
            
            Parameters:
                verbose (bool): Prints out number of parameters. Differentiates amortization from non-amortization params.
            
            Returns:
                int
                    Number of parameters (incl. amortization params).
        """
        n_enc = 0
        n_enc_nograd = 0

        if(verbose):
            print("--------------------------------------------")
            print("Counting Conditional PDF Pars ... ")
            print("--------------------------------------------")

        mlp_pars = 0
        mlp_pars_nograd = 0

        ## MLP summary to flow part
        if(verbose):
            print("<<-- (2) MLP summary to flow / encoder mappings -->>")
        
        for ind, mlp_predictor in enumerate(self.mlp_predictors):
            if(verbose):
                print("--------------")
                print("MLP predictor: ", ind)
            this_mlp=0
            this_mlp_nograd=0
            if mlp_predictor is not None:

                for p in mlp_predictor.parameters():
                    if p.requires_grad:
                        mlp_pars += numpy.prod(p.size())
                        this_mlp += numpy.prod(p.size())
                    else:
                        
                        this_mlp_nograd += numpy.prod(p.size())
                        mlp_pars_nograd += numpy.prod(p.size())
            if(verbose):
                print(" %d / (nograd) %d " % (this_mlp, this_mlp_nograd))
        
        if(self.log_normalization_mlp is not None):
            for p in self.log_normalization_mlp.parameters():
                if p.requires_grad:
                    mlp_pars += numpy.prod(p.size())
                    this_mlp += numpy.prod(p.size())
                else:
                    
                    this_mlp_nograd += numpy.prod(p.size())
                    mlp_pars_nograd += numpy.prod(p.size())
            if(verbose):
                print(" %d / (nograd) %d " % (this_mlp, this_mlp_nograd))

        if(verbose):
            print("total MLP predictor pars: %d / (nograd) %d " % (mlp_pars, mlp_pars_nograd))
            print("--------------------------------------------")
        ### Flow structure part
        tot_layer_pars = 0
        tot_layer_pars_nograd = 0

        if(verbose):
            print("<<-- (3) Flow structure -->>")
        
        for ind, pdf_def in enumerate(self.pdf_defs_list):

            if(verbose):
                print("pdf type: ", pdf_def)#
                print("layer structure: ", self.flow_defs_list[ind])

            for layer_ind, l in enumerate(self.layer_list[ind]):
                this_layer_pars = 0
                this_layer_pars_nograd = 0

                for p in l.parameters():
                    if p.requires_grad:
                        this_layer_pars += numpy.prod(p.size())
                    else:
                        this_layer_pars_nograd += numpy.prod(p.size())

                if(verbose):
                    print(
                        " -- layer %d - %d / (nograd) %d / (internal crosscheck: %d)"
                        % (layer_ind, this_layer_pars, this_layer_pars_nograd, self.num_parameter_list[ind][layer_ind])
                    )
                tot_layer_pars += this_layer_pars
                tot_layer_pars_nograd += this_layer_pars_nograd

        if(verbose):
            print(
                "total layer pars sum: %d / (nograd) %d"
                % (tot_layer_pars, tot_layer_pars_nograd)
            )

        tot_pars = n_enc + tot_layer_pars + mlp_pars
        tot_pars_nograd = n_enc_nograd + tot_layer_pars_nograd + mlp_pars_nograd

        if(verbose):
            print("--------------------------------------------")
            print(
                "total Conditional PDF pars: %d / (nograd) %d" % (tot_pars, tot_pars_nograd)
            )
            print("--------------------------------------------")
        return tot_pars

    def log_mean_poisson(self, conditional_input=None, amortization_parameters=None):
        """
        Calculates log-mean Poisson prediction.

        Parameters:
            conditional_input (Tensor/None): Must be tensor of appropriate dimension if conditional PDF. Otherwise None.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.

        Returns:
            Tensor
                log-lambda of Poisson, size (B,1)
        """
        if(self.log_normalization is None):
            raise Exception("This PDF does not predict the log-mean of a Poisson distriution. Initialize with 'predict_log_normalization'=True for this possibility.")

        if(amortization_parameters is not None):
            assert(amortization_parameters.shape[1]==self.total_number_amortizable_params)

        if(conditional_input is None):
            # we have no encoding
            assert(self.amortize_everything==False), "Currently there is no support for the prediction of log-lambda and simultanesouly passing amortization_parameters ('amortize everything').. there is some thinking involved in what to do in this situation so it is not supported at the moment."

            if(amortization_parameters is not None):
                raise Exception("Currently there is no support for the prediction of log-lambda and simultanesouly passing amortization_parameters .. there is some thinking involved in what to do in this situation so it is not supported at the moment.")
            else:
                return self.log_normalization
        else:
            if(self.join_poisson_and_pdf_description):

                extra_mlp_inputs=None
                if(amortization_parameters is not None):
                    assert(type(self.mlp_predictors[0])==AmortizableMLP)
                    assert(self.mlp_predictors[0].use_permanent_parameters==False)

                    num_mlp_params=self.mlp_predictors[0].num_amortization_params

                    extra_mlp_inputs=amortization_parameters[:,:num_mlp_params]

                    return self.mlp_predictors[0](conditional_input, extra_inputs=extra_mlp_inputs)[:,-1:]

                else:
                    ## the last parameter of the first MLP is predicting log-lambda (convention)
                    return self.mlp_predictors[0](conditional_input)[:,-1:]
            else:
                raise NotImplementedError("This way of independenly predicting the normalization (from all other parameters) is outdated!")
                return self.log_normalization_mlp(conditional_input)
        
    def all_layer_inverse(self, 
                          x, 
                          log_det, 
                          data_summary, 
                          amortization_parameters=None, 
                          force_embedding_coordinates=False, 
                          force_intrinsic_coordinates=False):
        """
        Performs the autoregressive (IAF) backward normalizing-flow mapping of all sub-manifold flows.

        Parameters:
            x (Tensor): Target Input.
            log_det (Tensor): Input log-Det. Soon Deprecated.
            data_summary (Tensor/None): Holds summary information for the conditional PDF. Otherwise None.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the input x for this inverse mapping.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the input x for this inverse mapping.

        Returns: 
            Tensor
                Position in base space after inverse mapping.
            Tensor
                Log-Det factors of this inverse mapping.
        """

        ## make sure we transform to default settings (potentially mixed) if we force embedding/intrinsic coordinates
        if(force_embedding_coordinates):
            assert(x.shape[1]==self.total_target_dim_embedded), (x.shape[1], self.total_target_dim_embedded)

            x, log_det=self.transform_target_space(x, log_det, transform_from="embedding", transform_to="default")
        elif(force_intrinsic_coordinates):
            assert(x.shape[1]==self.total_target_dim_intrinsic)

            x, log_det=self.transform_target_space(x, log_det, transform_from="intrinsic", transform_to="default")
        else:
            assert(x.shape[1]==self.total_target_dim), (x.shape[1], self.total_target_dim)


        extra_conditional_input=[]
        base_targets=[]

        individual_logps=dict()

        extra_params = None

        if(amortization_parameters is not None):

            assert(amortization_parameters.shape[1]==self.total_number_amortizable_params)
            
        amort_param_counter=0

        for pdf_index, pdf_layers in enumerate(self.layer_list):

            extra_param_counter = 0
            this_pdf_type=self.pdf_defs_list[pdf_index]

            if(self.mlp_predictors[pdf_index] is not None):

                ## mlp preditors can be None for unresponsive layers like x/y
                if(data_summary is not None):

                    if(type(data_summary)==list):
                        this_data_summary=data_summary[pdf_index]
                    else:
                        this_data_summary=data_summary

                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)
                    
                    if(amortization_parameters is not None):
                        num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                        extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                        amort_param_counter+=num_amortization_params

                    else:
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                   
                else:

                    ## data summary is None (non-conditional pdf) .. just encode previous dims
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        if(amortization_parameters is not None):
                            num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                            extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                            amort_param_counter+=num_amortization_params

                        else:
                            extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                        
                    else:
                        raise Exception("FORWARD: extra conditional input is empty but required for encoding!")

                if(self.predict_log_normalization):
                    if( (pdf_index==0) and self.join_poisson_and_pdf_description):
                        extra_params=extra_params[:,:-1]


            else:
                ## we amortize everything with amortization_parameters .. including the first layer column if there is no encoder
                if(self.amortize_everything):
                    assert(amortization_parameters is not None)

                    tot_num_params=0
                    for l in self.layer_list[0]:
                        tot_num_params+=l.get_total_param_num()

                    if(tot_num_params>0):
                        extra_params=amortization_parameters[:,:tot_num_params]

                        amort_param_counter+=tot_num_params

            this_target=x[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]
            
            ## reverse mapping is required for pdf evaluation
            for l, layer in reversed(list(enumerate(pdf_layers))):

                this_extra_params = None

                if extra_params is not None:

                    if extra_param_counter == 0:
                            this_extra_params = extra_params[:, -layer.total_param_num :]
                    else:

                        this_extra_params = extra_params[
                            :,
                            -extra_param_counter
                            - layer.total_param_num : -extra_param_counter,
                        ]

                
                if(l==(len(pdf_layers)-1)):
                    # force embedding or intrinsic coordinates in the layer that defines the target dimension
                    this_target, log_det = layer.inv_flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                else:

                    this_target, log_det = layer.inv_flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                
                extra_param_counter += layer.total_param_num

            if(False):
                ## stems from debugging purposes, not used currently
                ind_base_eval=this_logp = torch.distributions.MultivariateNormal(
                    torch.zeros_like(this_target).to(x),
                    covariance_matrix=torch.eye(this_target.shape[1]).type_as(x).to(x),
                ).log_prob(this_target)

                ind_logdet=log_det
                

                individual_logps["%.2d_%s" % (pdf_index, this_pdf_type)]=ind_base_eval+ind_logdet
                individual_logps["%.2d_%s_logdet" % (pdf_index, this_pdf_type)]=ind_logdet
                individual_logps["%.2d_%s_base" % (pdf_index, this_pdf_type)]=ind_base_eval

            
            base_targets.append(this_target)

            prev_target=x[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]
            prev_target=pdf_layers[-1]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        base_pos=torch.cat(base_targets, dim=1)

        return base_pos, log_det

    def forward(self, 
                x, 
                conditional_input=None,
                amortization_parameters=None, 
                force_embedding_coordinates=False, 
                force_intrinsic_coordinates=False):
        """
        Calculates log-probability at the target *x*. Also returns some other quantities that are calculated as a consequence.

        Parameters:

            x (Tensor): Target position to calculate log-probability at. Must be of shape (B,D), where B = batch dimension.
            conditional_input (Tensor/list(Tensor)/None): Amortization input for conditional PDFs. If given, must be of shape (B,A), where A is the conditional input dimension defined in __init__. Can also be 
                              a list of tensors, one for each sub-PDF, if *conditional_input_dim* in __init__ is a list of ints.
            amortization_parameters (Tensor/None): If the PDF is fully amortized, defines all the parameters of the PDF. Must be of shape (B,T), where T is the total number of parameters of the PDF.
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
        assert(self.use_as_passthrough_instead_of_pdf == False), "The module is only used as a passthrough of all layers, not as actually evaluating the pdf!"
        if(conditional_input is not None):
            if(type(conditional_input)==list):

                assert(len(self.conditional_input_dim)==len(conditional_input))
                for ci_ind in range(len(self.conditional_input_dim)):
                    assert(self.conditional_input_dim[ci_ind]==conditional_input[ci_ind].shape[1]), "Inputs of conditional input vector do not match with pre-defined input_dims!"

                for ci in conditional_input:
                    assert(x.shape[0]==ci.shape[0]), "Evaluating input x and condititional input shape must be similar!"
                    assert(x.is_cuda==ci.is_cuda), ("input tensor *x* and *conditional_input* are on different devices .. resp. cuda flags: 1) x, 2) conditional_input, 3) pdf model", x.is_cuda, ci.is_cuda, next(self.parameters()).is_cuda)
            else:
                assert(x.shape[0]==conditional_input.shape[0]), "Evaluating input x and condititional input shape must be similar!"
                assert(x.is_cuda==conditional_input.is_cuda), ("input tensor *x* and *conditional_input* are on different devices .. resp. cuda flags: 1) x, 2) conditional_input, 3) pdf model", x.is_cuda, conditional_input.is_cuda, next(self.parameters()).is_cuda)

        tot_log_det = torch.zeros(x.shape[0]).type_as(x)

        base_pos, tot_log_det=self.all_layer_inverse(x, tot_log_det, conditional_input, amortization_parameters=amortization_parameters, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)

        ## must faster calculation based on std normal
        other=torch.distributions.Normal(
            0.0,
            1.0,
        ).log_prob(base_pos)

        log_pdf=other.sum(dim=-1)

        return log_pdf + tot_log_det, log_pdf, base_pos

    def obtain_flow_param_structure(self, 
                                    conditional_input=None, 
                                    predefined_target_input=None, 
                                    seed=None,
                                    dtype=None,
                                    device=None):
        """
        Obtain values of flow parameters for given input along with their name. For debugging and plotting purposes mostly.
        """

        data_type, used_device=self.obtain_current_dtype_n_device()

        if(device is not None):
            used_device=device
        if(dtype is not None):
            data_type=dtype

        data_summary = None
        used_sample_size = 1
        

        this_layer_param_structure=collections.OrderedDict()

        if conditional_input is not None:

            if(type(conditional_input)==list):

                assert(len(self.conditional_input_dim)==len(conditional_input))
                for ci_ind in range(len(self.conditional_input_dim)):
                    assert(self.conditional_input_dim[ci_ind]==conditional_input[ci_ind].shape[1]), "Inputs of conditional input vector do not match with pre-defined input_dims!"


                for ci_ind, ci in enumerate(conditional_input[:-1]):

                    assert(ci.shape[0]==conditional_input[ci_ind].shape[0]), "Conditional input batch sizes do not agree!"
                    assert(ci.dim()==conditional_input[ci_ind].dim()==2), "Conditional inputs must be 2-dimensional tensors."
                    assert(ci.dtype==conditional_input[ci_ind].dtype), "Conditional input types do not match!"
                    assert(ci.device==conditional_input[ci_ind].device), "Conditional input devices do not match!"

                used_sample_size=conditional_input[0].shape[0]
                data_type=conditional_input[0].dtype
                used_device=conditional_input[0].device

            else:
           
                used_sample_size = conditional_input.shape[0]
                data_type = conditional_input.dtype
                used_device = conditional_input.device

            data_summary=conditional_input

        x=None
        log_gauss_evals=0.0
        std_normal_samples=0.0

        if(predefined_target_input is not None):

            x=predefined_target_input

            if(conditional_input is not None):

                ## make sure inputs agree

                if(type(conditional_input)==list):
                    assert(x.shape[0]==conditional_input[0].shape[0])
                    assert(x.dtype==conditional_input[0].dtype)
                    assert(x.device==conditional_input[0].device)
                else:
                    assert(x.shape[0]==conditional_input.shape[0])
                    assert(x.dtype==conditional_input.dtype)
                    assert(x.device==conditional_input.device)

            else:
                data_type=predefined_target_input.dtype
                used_sample_size=predefined_target_input.shape[0]
                used_device=predefined_target_input.device

            log_gauss_evals = torch.distributions.MultivariateNormal(
                torch.zeros(self.total_base_dim).type(data_type).to(used_device),
                covariance_matrix=torch.eye(self.total_base_dim)
                .type(data_type)
                .to(used_device),
            ).log_prob(predefined_target_input)

        else:

            if(seed is not None):
                numpy.random.seed(seed)

            std_normal = numpy.random.normal(size=(used_sample_size, self.total_base_dim))

            std_normal_samples = (
                torch.from_numpy(std_normal).type(data_type).to(used_device)
            )
            log_gauss_evals = torch.distributions.MultivariateNormal(
                torch.zeros(self.total_base_dim).type(data_type).to(used_device),
                covariance_matrix=torch.eye(self.total_base_dim)
                .type(data_type)
                .to(used_device),
            ).log_prob(std_normal_samples)

            x = std_normal_samples

        log_det = torch.zeros(used_sample_size).type(data_type).to(used_device)

        extra_conditional_input=[]
        new_targets=[]

        param_structure=[]

        for pdf_index, pdf_layers in enumerate(self.layer_list):

            this_pdf_type=self.pdf_defs_list[pdf_index]
            this_flow_def=self.flow_defs_list[pdf_index]

            extra_params = None
            if(data_summary is not None and self.mlp_predictors[pdf_index] is not None):
               
                if(type(data_summary)==list):
                    this_data_summary=data_summary[pdf_index]
                else:
                    this_data_summary=data_summary

                if(len(extra_conditional_input)>0):
                    this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)

                extra_params=self.mlp_predictors[pdf_index](this_data_summary)

            else:

                if(self.mlp_predictors[pdf_index] is not None):
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                    else:
                        raise Exception("SAMPLE: extra conditional input is empty but required for encoding!")
     
            if(self.predict_log_normalization):

                if(pdf_index==0 and self.join_poisson_and_pdf_description and extra_params is not None):
                    extra_params=extra_params[:,:-1]

            this_target=x[:,self.base_dim_indices[pdf_index][0]:self.base_dim_indices[pdf_index][1]]

            ## loop through all layers in each pdf and transform "this_target"
            
            extra_param_counter = 0
            for l, layer in list(enumerate(pdf_layers)):

                this_extra_params = None
                

                if extra_params is not None:
                    
                    this_extra_params = extra_params[:, extra_param_counter : extra_param_counter + layer.total_param_num]
                    
                this_param_dict=collections.OrderedDict()

                layer.obtain_layer_param_structure(this_param_dict, extra_inputs=this_extra_params, previous_x=this_target)

                this_layer_param_structure[("%.3d" % pdf_index)+"_"+this_flow_def+".%.3d" % l]=this_param_dict
                #################

                this_target, log_det = layer.flow_mapping([this_target, log_det], extra_inputs=this_extra_params)

                extra_param_counter += layer.total_param_num 
           
            prev_target=this_target

           
            ## return embedding value for next conditional inputs
            prev_target=self.layer_list[pdf_index][-1]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        if (torch.isfinite(x) == 0).sum() > 0:
            raise Exception("nonfinite samples generated .. this should never happen!")

        return this_layer_param_structure

    def sample(self, 
               conditional_input=None, 
               samplesize=1,  
               seed=None, 
               allow_gradients=False, 
               amortization_parameters=None, 
               force_embedding_coordinates=False, 
               force_intrinsic_coordinates=False,
               failsafe_crosscheck_tolerance=None,
               dtype=None,
               device=None):
        """ 
        Samples from the (conditional) PDF. 

        Parameters:
            conditional_input (Tensor/list(Tensor)/None): Tensor of shape B x D where B is the batch size and D the input space dimension if given. Can also be a list of tensors, which must share batch dimensionality. Else None.
            samplesize (int): Samplesize.
            seed (None/int):
            allow_gradients (bool): If False, does not propagate gradients and saves memory by not building the graph. Off by default, so has to be switched on for training.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates for the sample.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates for the sample.
            dtype (torch dtype): Dtype and device are normally inferred by parameters or conditional input. If no parameters are part of the 
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.

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
       
        assert(self.use_as_passthrough_instead_of_pdf == False), "The module is only used as a passthrough of all layers, not as actually evaluating the pdf or sampling from the pdf!"

        if(allow_gradients):

            sample, normal_base_sample, log_pdf_target, log_pdf_base=self._obtain_sample(conditional_input=conditional_input, 
                                                                                         seed=seed, 
                                                                                         samplesize=samplesize, 
                                                                                         amortization_parameters=amortization_parameters, 
                                                                                         force_embedding_coordinates=force_embedding_coordinates, 
                                                                                         force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                         failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                                                                         device=device,
                                                                                         dtype=dtype)


            return sample, normal_base_sample, log_pdf_target, log_pdf_base

        else:   
            with torch.no_grad():
                sample, normal_base_sample, log_pdf_target, log_pdf_base=self._obtain_sample(conditional_input=conditional_input, 
                                                                                             seed=seed, 
                                                                                             samplesize=samplesize, 
                                                                                             amortization_parameters=amortization_parameters, 
                                                                                             force_embedding_coordinates=force_embedding_coordinates, 
                                                                                             force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                             failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                                                                             device=device,
                                                                                             dtype=dtype)
           
            return sample, normal_base_sample, log_pdf_target, log_pdf_base

    def all_layer_forward(self, 
                          x,   
                          log_det,   
                          data_summary, 
                          amortization_parameters=None,
                          force_embedding_coordinates=False, 
                          force_intrinsic_coordinates=False):

        """
        Performs the autoregressive (IAF) forward normalizing-flow mapping of all sub-manifold flows.

        Parameters:
            x (Tensor): Target Input.
            log_det (Tensor): Input log-Det. Soon Deprecated.
            data_summary (Tensor/None): Holds summary information for the conditional PDF. Otherwise None.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the output sample.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the output sample.

        Returns: 
            Tensor
                Position in the target space after sampling.
            
            Tensor
                Log-Det factors of the forward mapping.
        """

        extra_conditional_input=[]
        new_targets=[]

        if(amortization_parameters is not None):

            #assert(amortization_parameters.shape[0]==x.shape[0]), ("batch size of x must agree with batch size of amortization_parameters")
            assert(amortization_parameters.shape[1]==self.total_number_amortizable_params), (amortization_parameters.shape[1], self.total_number_amortizable_params)
        else:
            assert(self.amortize_everything==False)

        amort_param_counter=0

        for pdf_index, pdf_layers in enumerate(self.layer_list):

            this_pdf_type=self.pdf_defs_list[pdf_index]

            ## by default not extra_params for the layers
            extra_params = None

            if(self.mlp_predictors[pdf_index] is not None):

                if(data_summary is not None):
                    # conditional PDF (data_summary!=None) and MLP predictor given
                    if(type(data_summary)==list):
                        this_data_summary=data_summary[pdf_index]
                    else:   
                        this_data_summary=data_summary
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)

                    if(amortization_parameters is not None):
                        num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                        extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                        amort_param_counter+=num_amortization_params

                    else:
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)

               
                else:

                
                    ## no conditional PDF (data_summary==None) but MLP predictor there
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        if(amortization_parameters is not None):
                            num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                            extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                            amort_param_counter+=num_amortization_params

                        else:
                            extra_params=self.mlp_predictors[pdf_index](this_data_summary)

                    else:
                        raise Exception("SAMPLE: extra conditional input is empty but required for encoding!")

                if(self.predict_log_normalization):
                    if( (pdf_index==0) and self.join_poisson_and_pdf_description):
                        extra_params=extra_params[:,:-1]

            else:
                ## we amortize everything with amortization_parameters .. including the first layer if there is no encoder
                if(self.amortize_everything):
                    assert(amortization_parameters is not None)
                    
                    tot_num_params=0
                    for l in self.layer_list[pdf_index]:
                        tot_num_params+=l.get_total_param_num()

                    if(tot_num_params>0):
                        extra_params=amortization_parameters[:,amort_param_counter:amort_param_counter+tot_num_params]
                        amort_param_counter+=tot_num_params

            this_target=x[:,self.base_dim_indices[pdf_index][0]:self.base_dim_indices[pdf_index][1]]

            ## loop through all layers in each pdf and transform "this_target"
            
            extra_param_counter = 0
            for l, layer in list(enumerate(pdf_layers)):
               
                this_extra_params = None
                
                if extra_params is not None:
                    
                    this_extra_params = extra_params[:, extra_param_counter : extra_param_counter + layer.total_param_num]
              
                if(l==(len(pdf_layers)-1)):
                    this_target, log_det = layer.flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                else:
                    this_target, log_det = layer.flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                
                extra_param_counter += layer.total_param_num

            new_targets.append(this_target)

            prev_target=this_target
           
            prev_target=self.layer_list[pdf_index][-1]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        if (torch.isfinite(x) == 0).sum() > 0:
            raise Exception("nonfinite samples generated .. this should never happen!")
        
        x=torch.cat(new_targets, dim=1)

        ## transform to desired output space 
        if(force_embedding_coordinates):

            x, log_det=self.transform_target_space(x, log_det, transform_from="default", transform_to="embedding")
            
        elif(force_intrinsic_coordinates):
            assert(x.shape[1]==self.total_target_dim_intrinsic)
           
            x, log_det=self.transform_target_space(x, log_det, transform_from="default", transform_to="intrinsic")
      
        return x, log_det

    def _obtain_sample(self, 
                       conditional_input=None, 
                       predefined_target_input=None, 
                       samplesize=1, 
                       seed=None, 
                       amortization_parameters=None, 
                       force_embedding_coordinates=False, 
                       force_intrinsic_coordinates=False,
                       failsafe_crosscheck_tolerance=None,
                       dtype=None,
                       device=None):
        """
        Obtains a sample from the Multivariate Standard Normal, evaluates it and passes it through forward machinery. 
        When *predefined_target_input* is given, takes this as a sample.

        Parameters:

            conditional_input (Tensor/None): Input tensor when conditional PDF.
            predefined_target_input (Tensor/None): When given, evaluates the MVN there. Otherwise samples a MVN random variable before.
            samplesize (int):
            seed (None/int):
            amortization_parameters (bool): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the output sample.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the output sample.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.

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

        used_sample_size = samplesize

        ## some crosschecks
        if(conditional_input is not None):
            if(type(conditional_input)==list):

                assert(len(self.conditional_input_dim)==len(conditional_input))
                for ci_ind in range(len(self.conditional_input_dim)):
                    assert(self.conditional_input_dim[ci_ind]==conditional_input[ci_ind].shape[1]), "Inputs of conditional input vector do not match with pre-defined input_dims!"

                for ci_ind, ci in enumerate(conditional_input[:-1]):

                    assert(ci.shape[0]==conditional_input[ci_ind].shape[0]), "Conditional input batch sizes do not agree!"
                    assert(ci.dtype==conditional_input[ci_ind].dtype), "Conditional input types do not match!"
                    assert(ci.device==conditional_input[ci_ind].device), "Conditional input devices do not match!"


        # make sure device is set if amortization is used
        if(self.amortize_everything):
            assert(amortization_parameters is not None)
            used_device=amortization_parameters.device
            used_sample_size=amortization_parameters.shape[0]

            if(conditional_input is not None):
                ## TODO - maybe allow for more flexible shape combinations
                assert(conditional_input.shape[0]==amortization_parameters.shape[0])
                assert(conditional_input.device==amortization_parameters.device)

        elif(conditional_input is not None):

                if(type(conditional_input)==list):
                    used_sample_size = conditional_input[0].shape[0]
                    data_type = conditional_input[0].dtype
                    used_device = conditional_input[0].device
                else:
                    used_sample_size = conditional_input.shape[0]
                    data_type = conditional_input.dtype
                    used_device = conditional_input.device

        else:
            ## if one blindly uses next() on an empty param generator, it throws an error
            data_type, used_device=self.obtain_current_dtype_n_device()

            if(device is not None):
                used_device=device
            if(dtype is not None):
                data_type=dtype

        assert( (data_type is not None) and (used_device is not None)), "DType and/or device is None. This can only happen if layers without any parameters are used. In this case, you have to define dtype and device as keyword arguments!"

        x=None
        log_gauss_evals=0.0
        std_normal_samples=0.0
       
        if(predefined_target_input is not None):

            x=predefined_target_input

            assert(used_device==predefined_target_input.device)

            if(conditional_input is not None):

                ## make sure inputs agree

                if(type(conditional_input)==list):
                    assert(x.shape[0]==conditional_input[0].shape[0])
                    assert(x.dtype==conditional_input[0].dtype)
                    assert(x.device==conditional_input[0].device)
                else:
                    assert(x.shape[0]==conditional_input.shape[0])
                    assert(x.dtype==conditional_input.dtype)
                    assert(x.device==conditional_input.device)

            else:
                data_type=predefined_target_input.dtype
                used_sample_size=predefined_target_input.shape[0]
              
            log_gauss_evals=torch.distributions.Normal(0.0,1.0).log_prob(predefined_target_input).sum(dim=-1)

        else:

            if(seed is not None):
                numpy.random.seed(seed)

            std_normal = numpy.random.normal(size=(used_sample_size, self.total_base_dim))

            std_normal_samples = (
                torch.from_numpy(std_normal).type(data_type).to(used_device)
            )

            log_gauss_evals=torch.distributions.Normal(0.0,1.0).log_prob(std_normal_samples).sum(dim=-1)
            
            x = std_normal_samples

        log_det = torch.zeros(used_sample_size).type(data_type).to(used_device)
        
        new_targets, log_det=self.all_layer_forward(x, log_det, conditional_input, amortization_parameters=amortization_parameters, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)

        ## failsafe crosscheck?

        return_log_pdf=-log_det + log_gauss_evals

        if(failsafe_crosscheck_tolerance):

            assert(predefined_target_input is None), "Failsafe does not work with predefined input!"
            
            new_targets_prop, std_normal_samples_prop, return_log_pdf_prop, log_gauss_evals_prop=recheck_sampling(self, 
                      new_targets,
                      std_normal_samples,
                      return_log_pdf,
                      log_gauss_evals,
                      failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                      conditional_input=conditional_input,
                      amortization_parameters=amortization_parameters,
                      force_embedding_coordinates=force_embedding_coordinates,
                      force_intrinsic_coordinates=force_intrinsic_coordinates,
                      dtype=data_type,
                      device=used_device)

            if(new_targets_prop is not None):
                new_targets=new_targets_prop
                std_normal_samples=std_normal_samples_prop
                return_log_pdf=return_log_pdf_prop
                log_gauss_evals=log_gauss_evals_prop
            

        ## -logdet because log_det in sampling is derivative of forward function d/dx(f), but log_p requires derivative of backward function d/dx(f^-1) whcih flips the sign here
        return new_targets, std_normal_samples, return_log_pdf, log_gauss_evals

    def get_total_embedding_dim(self):
        """Returns embedding dimension of the overall PDF."""
        tot_dim=0
        for pdf_index, pdf_type in enumerate(self.pdf_defs_list):
            ## add the conditional return dimension of the last layer for each sub pdf
            tot_dim+=self.layer_list[pdf_index][-1]._embedding_conditional_return_num()
        
        return tot_dim

    ### 
    def transform_target_into_returnable_params(self, target):
        """ 
        Transforms an input tensor from default to embedding parametrization.

        Parameters:
            target (Tensor): Input tensor in current default parametrization.

        Returns:
            Tensor
                Output tensor in embedding parametrization.

        """

        
        res, _=self.transform_target_space(target)

        return res

    def transform_target_space(self, 
                              target, 
                              log_det=0, 
                              transform_from="default", 
                              transform_to="embedding"):
        """
        Transform the destimation space tensor of the PDF as defined by *transform_from* and *transform_to*, which is the embedding space. (I.e. transform spherical angles into x/y/z pairs and so on when *transform_to* is embedding space.)
        
        Parameters:

            target (Tensor): Tensor to transform.
            log_det (Tensor): log_det Tensor (Soon Deprecated)
            transform_from (str): Coordinates to start with. One of "default"/"intrinsic"/"embedding".
            transform_to (str): Coordinates to end with. One of "default"/"intrinsic"/"embedding".

        Returns:

            Tensor
                 Target in new coordinates.

            
            Tensor
                Any additional log-det added to input log-det. 
        """

        new_target=target

        if(len(target.shape)==1):
            new_target=target.unsqueeze(0)

        ## transforming only makes sense if input has correct target shape
        if(transform_from=="default"):
            assert(new_target.shape[1]==self.total_target_dim)
        elif(transform_from=="intrinsic"):
            assert(new_target.shape[1]==self.total_target_dim_intrinsic)
        elif(transform_from=="embedding"):
            assert(new_target.shape[1]==self.total_target_dim_embedded) 

        potentially_transformed_vals=[]

        index=0
        for pdf_index, pdf_type in enumerate(self.pdf_defs_list):

            if(transform_from=="default"):
                this_dim=self.target_dims[pdf_index]
            elif(transform_from=="intrinsic"):
                this_dim=self.target_dims_intrinsic[pdf_index]
            elif(transform_from=="embedding"):
                this_dim=self.target_dims_embedded[pdf_index]
            
            this_target, log_det=self.layer_list[pdf_index][-1].transform_target_space(new_target[:,index:index+this_dim], log_det=log_det, transform_from=transform_from, transform_to=transform_to)
            
            potentially_transformed_vals.append(this_target)

            index+=this_dim

        potentially_transformed_vals=torch.cat(potentially_transformed_vals, dim=1)

        if(transform_to=="default"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim)
        elif(transform_to=="intrinsic"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_intrinsic), (potentially_transformed_vals.shape, self.total_target_dim_intrinsic)
        elif(transform_to=="embedding"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_embedded), (new_target.shape[1], self.total_target_dim_embedded)  


        if(len(target.shape)==1):
            potentially_transformed_vals=potentially_transformed_vals.squeeze(0)

        return potentially_transformed_vals, log_det

    ########

    def init_params(self, data=None, damping_factor=1000.0, mvn_min_max_sv_ratio=1e-4):
        """
        Initialize params of the normalizing flow such that the different sub flows play nice with each other and the starting distribution is a reasonable one.
        For the Gaussianization flow, data can be used to initilialize the starting distribution such that it roughly follows the data.
            
        Parameters:
            data (None/Tensor): If given a Tensor with target data, Gaussianization Flow subflows can make use of the distribution and initialize such that they follow the distribution.
            damping_factor (float): Weights in final matrices of amortization MLPs are divided by this factor (after already having been initialized) to dampen the impact of previous flow layers and conditional input in the autoregressive amortization structure.

        """

        global_amortization_init=None
        global_amortization_index=0
       
        if(self.amortize_everything):
            global_amortization_init=torch.zeros(self.total_number_amortizable_params)

        with torch.no_grad():
            ## 0) check data
            if(data is not None):
                ## initialization data has to match pdf dimenions
                assert(data.shape[1]==self.total_target_dim), "Initialization with data must match the target dimension of the PDF!"

            ## 1) Find initialization params of all layers - each index corresponds to all flows from a given sub manifold
            params_list=[]
            ## loop through all the layers and get the initializing parameters

            this_dim_index=0

            for subflow_index, subflow_description in enumerate(self.pdf_defs_list):
                
                this_dim=self.target_dims[subflow_index]

                this_layer_list=self.layer_list[subflow_index]

                if("e" in subflow_description):
                    
                    params=find_init_pars_of_chained_blocks(this_layer_list, data[:, this_dim_index:this_dim_index+this_dim] if data is not None else None, mvn_min_max_sv_ratio=mvn_min_max_sv_ratio)

                    params_list.append(params)

                else:
                    
                    this_list=[]
                    for l in this_layer_list:
                        this_list.append(l.get_desired_init_parameters())

                    params_list.append(torch.cat(this_list))

                this_dim_index+=this_dim

            ## 2) Depending on encoding structure, use the init params at appropriate places
       

            if(self.predict_log_normalization):

                if(self.join_poisson_and_pdf_description):
                    if(len(self.mlp_predictors)>1):
                        ## TODO: need to find a good way to predict the log-normaliaztion as a joint parameter when multiple mlp predictors are present
                        raise NotImplementedError


            # loop through all mlps
            for ind, mlp_predictor in enumerate(self.mlp_predictors):
                
                # these are the desired params at initialization for the MLP -> set bias of last MLP layer to these values
                # and make the weights and bias in previous layers very small
               
                these_params=params_list[ind]

                if(len(these_params)>0):

                    if(mlp_predictor is not None):
                       
                        ## the first MLP can predict log_lambda if desired
                        ## attach desired log-lambda to this bunch of params
                        if(self.predict_log_normalization):
                            if(self.join_poisson_and_pdf_description):
                                if(ind==0):
                                    log_lambda_init=0.1
                                    these_params=torch.cat([these_params, torch.Tensor([log_lambda_init]).type(mlp_predictor[-1].bias.data.dtype)])

                        ## custom low-rank MLPs - initialization is done inside the custom MLP class
                        if(type(mlp_predictor)== AmortizableMLP):
                           
                            if(self.amortize_everything):
                                desired_uvb_params=mlp_predictor.obtain_default_init_tensor(fix_final_bias=these_params, prev_damping_factor=damping_factor)
                                num_uvb_pars=mlp_predictor.num_amortization_params
                                global_amortization_init[global_amortization_index:global_amortization_index+num_uvb_pars]=desired_uvb_params
                                global_amortization_index+=num_uvb_pars
                            else:
                                mlp_predictor.initialize_uvbs(fix_final_bias=these_params, prev_damping_factor=damping_factor)

                        else:
                            # initialize all layers
                            for internal_layer in mlp_predictor:
                                
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
                            
                            mlp_predictor[-1].bias.data=these_params.data.type(mlp_predictor[-1].bias.data.dtype)#torch.ones_like(these_params.data)*0.44  # .copy_(torch.randn(these_params.shape))

                    else:
                        ## threre is no MLP - initialize parameters of flows directly
                        tot_param_index=0
                  
                        for layer_ind, layer in enumerate(self.layer_list[ind]):
                            
                            this_layer_num_params=self.layer_list[ind][layer_ind].get_total_param_num()
                           
                            if(self.amortize_everything==False):
                                self.layer_list[ind][layer_ind].init_params(these_params[tot_param_index:tot_param_index+this_layer_num_params])

                            tot_param_index+=this_layer_num_params

                        if(self.amortize_everything):
                            global_amortization_init[global_amortization_index:global_amortization_index+tot_param_index]=these_params
                            global_amortization_index+=tot_param_index

                        if(ind==0 and self.amortize_everything and self.predict_log_normalization):
                            global_amortization_init[global_amortization_index+1]=0.1
                            global_amortization_index+=1
        
        return global_amortization_init

    def coverage(self, 
                target_x,
                conditional_input=None,
                amortization_parameters=None, 
                force_embedding_coordinates=False, 
                force_intrinsic_coordinates=False,
                num_percentile_points=100,
                sub_manifolds=[-1]):
        """
        Calculates coverage for the base distribution via the quantity 2*(log(p(0))-log(p_(z_base))) which should be chi^2 distributed for good coverage.

        Parameters:

            target_x (Tensor): Target positions to calculate coverage with. Must be of shape (B,D), where B = batch dimension.
            conditional_input (Tensor/list(Tensor)/None): Amortization input for conditional PDFs. If given, must be of shape (B,A), where A is the conditional input dimension defined in __init__. Can also be 
                              a list of tensors, one for each sub-PDF, if *conditional_input_dim* in __init__ is a list of ints.
            amortization_parameters (Tensor/None): If the PDF is fully amortized, defines all the parameters of the PDF. Must be of shape (B,T), where T is the total number of parameters of the PDF.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the input *x*.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the input *x*. 
            num_percentile_points (int): At how many points along the chi2 do we want to compare true vs expected coverage?
            sub_manifolds (list(int)): Contains indices of sub-manifolds if coverage should be calculated for onditional PDF of the given sub-manifold. *-1* stands for the total PDF and is the default.

        Returns:

            expected_coverage_probs (Numpy array): Array of expected coverage probabilities.
            actual_coverage_probs (Numpy array): Array of actual coverage probabilities.
            actual_twice_logprob (Numpy array): Array of twice the log-probability difference at the base.
        """

        return_dict=dict()
        return_dict["true"]=dict()
        return_dict["logprob_diffs"]=dict()
   
        expected_coverage_probs=numpy.linspace(0,1.0,num_percentile_points)
        return_dict["expected"]=expected_coverage_probs

        with torch.no_grad():

            _, logp_base, base_points=self.forward(target_x, 
                    conditional_input=conditional_input,
                    amortization_parameters=amortization_parameters, 
                    force_embedding_coordinates=force_embedding_coordinates, 
                    force_intrinsic_coordinates=force_intrinsic_coordinates)

            ## overall coverage
            if(-1 in sub_manifolds):
                true_cov, logprob_diffs=_calculate_coverage(logp_base.cpu().numpy(), self.total_base_dim, expected_coverage_probs)
                
                return_dict["true"]["total"]=true_cov
                return_dict["logprob_diffs"]["total"]=logprob_diffs

            for sm in sub_manifolds:
                if(sm==-1):
                    continue

                if(sm!=-1):
                    assert(sm>=0 and sm<len(self.pdf_defs_list)), ("Sub manifold index %d is invalid" % sm)
             
                sub_logp_base=torch.distributions.Normal(0.0,1.0).log_prob(base_points[:,self.target_dim_indices_intrinsic[sm][0]:self.target_dim_indices_intrinsic[sm][1]]).sum(axis=-1)
               
                true_cov, logprob_diffs=_calculate_coverage(sub_logp_base.cpu().numpy(), self.target_dims_intrinsic[sm], expected_coverage_probs)
            
                return_dict["true"][int(sm)]=true_cov
                return_dict["logprob_diffs"][int(sm)]=logprob_diffs

        return return_dict

       

#### Experimental functions
#### Some of these functions generalize existing functions and will replace them in future release.

    def entropy(self, 
                sub_manifolds=[-1], 
                conditional_input=None,
                force_embedding_coordinates=True, 
                force_intrinsic_coordinates=False,
                samplesize=100,
                failsafe_crosscheck_tolerance=None,
                dtype=None,
                device=None):

        """
        Calculates entropy of the PDF.
    
        Parameters:
            sub_manifolds (list(int)): Contains indices of sub-manifolds if entropy should be calculated for marginal PDF of the given sub-manifold. *-1* stands for the total PDF and is the default.
            conditional_input (Tensor/list(Tensor)/None): If passed defines the input to the PDF.
            force_embedding_coordinates (bool): Forces embedding coordinates in entropy calculation. Should always be true for correct manifold entropies.
            force_intrinsic_coordinates (bool): Forces intrinsic coordinates in entropy calculation. Should always be false for correct manifold entropies.
            samplesize (int): Samplesize to use for entropy approximation.
            failsafe_crosscheck_tolerance (float / None): If set, is used to crosscheck forward/bakckward pass compatability and resample if necessary. Has been introduced for the v flow in particular, so it should not be necessary for other flows.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.

        Returns:
            dict
                Dictionary containing entropy for each index defined in parameter *sub_manifolds*. If *-1* was given in *sub_manifolds*, the resulting entropy is stored under the *total* key.

        """

        data_type, used_device=self.obtain_current_dtype_n_device()

        if(device is not None):
            used_device=device
        if(dtype is not None):
            data_type=dtype

        data_summary = None
     
        ## some crosschecks
        if(conditional_input is not None):
            if(type(conditional_input)==list):

                assert(len(self.conditional_input_dim)==len(conditional_input))
                for ci_ind in range(len(self.conditional_input_dim)):
                    assert(self.conditional_input_dim[ci_ind]==conditional_input[ci_ind].shape[1]), "Inputs of conditional input vector do not match with pre-defined input_dims!"


                for ci_ind, ci in enumerate(conditional_input[:-1]):

                    assert(ci.shape[0]==conditional_input[ci_ind].shape[0]), "Conditional input batch sizes do not agree!"
                    assert(ci.dtype==conditional_input[ci_ind].dtype), "Conditional input types do not match!"
                    assert(ci.device==conditional_input[ci_ind].device), "Conditional input devices do not match!"

        if(force_embedding_coordinates==False):
            print("#### CAUTION: Calculating entropy without forcing embedding coordinates. This might lead to undesired and wrong entropies when using manifold PDFs!#############")
            #raise Exception()
        use_marginal_subdims=False
        ## make sure the settings are self consistent
        for subdim in sub_manifolds:
            if(subdim!=-1):
                assert(subdim>=0 and subdim < len(self.layer_list))
                use_marginal_subdims=True

        batch_size=1

        if conditional_input is not None:

            assert(self.conditional_input_dim is not None)

            if(type(conditional_input)==list):

                data_type = conditional_input[0].dtype
                used_device = conditional_input[0].device
                
                # a list of data summaries for the next functions
                data_summary=[ci.repeat_interleave(samplesize, dim=0) for ci in conditional_input]

                batch_size=conditional_input[0].shape[0]

            else:

                data_type = conditional_input.dtype
                used_device = conditional_input.device
                
                # this behavior is a little differnet than in standard sample .. we sample for every conditional input multiple times
                data_summary=conditional_input.repeat_interleave(samplesize, dim=0)

                batch_size=conditional_input.shape[0]

        else:
            assert(self.conditional_input_dim is None), "We require conditional input, since this is a conditional PDF."

        entropy_dict=dict()

      
        if(use_marginal_subdims==False):
            ## just calculate normal entropy by summing over samples
            
            _, _, log_pdf_dict, _=self.all_layer_forward_individual_subdims_incl_sampling(data_summary, 
                                                                                          samplesize*batch_size, 
                                                                                          failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance, 
                                                                                          force_embedding_coordinates=force_embedding_coordinates, 
                                                                                          force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                          dtype=data_type,
                                                                                          device=used_device)

            entropy_dict["total"]=-(log_pdf_dict["total"]).reshape(-1,samplesize).mean(dim=1)
            
        else:

            ## make sure we go all the way to the end by including -1 if it is not there
            sub_manifolds_here=sub_manifolds
            if(-1 not in sub_manifolds):
                sub_manifolds_here=[-1]+sub_manifolds

            ## transform all base samples forward
            
            targets, _, log_pdf_dict, _=self.all_layer_forward_individual_subdims_incl_sampling(data_summary, 
                                                                                          samplesize*batch_size, 
                                                                                          failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance, 
                                                                                          force_embedding_coordinates=force_embedding_coordinates, 
                                                                                          force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                          dtype=data_type,
                                                                                          device=used_device)

            #targets, log_det_dict_fw=self.all_layer_forward_individual_subdims(std_normal_samples, data_summary, sub_manifolds=sub_manifolds_here, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)
                    

            for sub_mf in sub_manifolds:
                ## also calculate total
                if(-1 == sub_mf):
                    
                    entropy_dict["total"]=-(log_pdf_dict["total"]).reshape(-1, samplesize).mean(dim=1)
                elif(0==sub_mf):
                 
                    entropy_dict[0]=-(log_pdf_dict[0]).reshape(-1, samplesize).mean(dim=1)
                else:

                    max_target_first_index=0
                   
                    for lower_mf in range(sub_mf):

                        if(force_embedding_coordinates):
                            max_target_first_index+=self.target_dims_embedded[lower_mf]
                        elif(force_intrinsic_coordinates):
                            max_target_first_index+=self.target_dims_intrinsic[lower_mf]
                        else:
                            max_target_first_index+=self.target_dims[lower_mf]
                 
                    ## this construction correctly orders both cases of conditional input and no conditional input
                    repeated_targets_first=targets[:, :max_target_first_index].unsqueeze(0).reshape(-1, samplesize, max_target_first_index).repeat(1,samplesize,1)
                    repeated_targets_first=repeated_targets_first.reshape(-1, max_target_first_index)
                    
                    if(force_embedding_coordinates):
                        repeated_final=targets[:, self.target_dim_indices_embedded[sub_mf][0]:self.target_dim_indices_embedded[sub_mf][1]].unsqueeze(0).reshape(-1, samplesize, self.target_dim_indices_embedded[sub_mf][1]-self.target_dim_indices_embedded[sub_mf][0]).repeat_interleave(samplesize, dim=1)
                        repeated_final=repeated_final.reshape(-1, self.target_dim_indices_embedded[sub_mf][1]-self.target_dim_indices_embedded[sub_mf][0])
                    elif(force_intrinsic_coordinates):
                        repeated_final=targets[:, self.target_dim_indices_intrinsic[sub_mf][0]:self.target_dim_indices_intrinsic[sub_mf][1]].unsqueeze(0).reshape(-1, samplesize, self.target_dim_indices_intrinsic[sub_mf][1]-self.target_dim_indices_intrinsic[sub_mf][0]).repeat_interleave(samplesize, dim=1)
                        repeated_final=repeated_final.reshape(-1, self.target_dim_indices_intrinsic[sub_mf][1]-self.target_dim_indices_intrinsic[sub_mf][0])
                    else:
                        repeated_final=targets[:, self.target_dim_indices[sub_mf][0]:self.target_dim_indices[sub_mf][1]].unsqueeze(0).reshape(-1, samplesize, self.target_dim_indices[sub_mf][1]-self.target_dim_indices[sub_mf][0]).repeat_interleave(samplesize, dim=1)
                        repeated_final=repeated_final.reshape(-1, self.target_dim_indices[sub_mf][1]-self.target_dim_indices[sub_mf][0])
                    
                    ########################
                   
                    joint_repeated=torch.cat([repeated_targets_first, repeated_final], dim=1)

                    fillup_difference=int(targets.shape[1])-int(joint_repeated.shape[1])
                    
                    ## fill up the rest (after current sub-mf) with ones so we get "some coordinates" in those to call all_layer_inverse
                    filled_up=torch.cat([joint_repeated, torch.ones(joint_repeated.shape[0], fillup_difference).to(repeated_final)], dim=1)

                    if(data_summary is None):
                        new_base_vals, log_det_dict_individual=self.all_layer_inverse_individual_subdims(filled_up, None, sub_manifolds=[sub_mf], force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)
                    elif(type(data_summary)==list):
                        new_base_vals, log_det_dict_individual=self.all_layer_inverse_individual_subdims(filled_up, [ds.repeat_interleave(samplesize, dim=0) for ds in data_summary], sub_manifolds=[sub_mf], force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)
                    else:
                        new_base_vals, log_det_dict_individual=self.all_layer_inverse_individual_subdims(filled_up, data_summary.repeat_interleave(samplesize, dim=0), sub_manifolds=[sub_mf], force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)


                    this_base_dim=self.base_dim_indices[sub_mf][1]-self.base_dim_indices[sub_mf][0]

                    log_gauss_evals_base=torch.distributions.Normal(0.0,1.0).log_prob(new_base_vals[:, self.base_dim_indices[sub_mf][0]:self.base_dim_indices[sub_mf][1]]).sum(dim=-1)
                        
                    log_probs=(log_gauss_evals_base+log_det_dict_individual[sub_mf]).reshape(-1,samplesize, samplesize)
                    
                    log_probs=torch.logsumexp(log_probs, dim=-1)-numpy.log(float(samplesize))
                    
                    entropy_dict[sub_mf]=-log_probs.mean(dim=1)
                   

        return entropy_dict

    def entropy_iterative(self, 
                sub_manifolds=[-1], 
                conditional_input=None,
                force_embedding_coordinates=True, 
                force_intrinsic_coordinates=False,
                samplesize=100,
                iterative_samplesize=10,
                max_iterative_batchsize=20,
                failsafe_crosscheck_tolerance=None,
                dtype=None,
                device=None,
                return_samples=False,
                verbose=False):

        """
        Calculates entropy of the PDF in an iterative manner. By iterating potentially both over target samples of later sub-pdfs, and over batch items, memory is saved
        and larger samplesizes (> a few 100 - 1000s) can be calculated. Probably only necessary for large sample sizes of later sub-pdfs, not of the first sub-pdf or the total PDF.
    
        Parameters:
            sub_manifolds (list(int)): Contains indices of sub-manifolds if entropy should be calculated for marginal PDF of the given sub-manifold. *-1* stands for the total PDF and is the default.
            conditional_input (Tensor/list(Tensor)/None): If passed defines the input to the PDF.
            force_embedding_coordinates (bool): Forces embedding coordinates in entropy calculation. Should always be true for correct manifold entropies.
            force_intrinsic_coordinates (bool): Forces intrinsic coordinates in entropy calculation. Should always be false for correct manifold entropies.
            samplesize (int): Samplesize to use for entropy approximation.
            iterative_samplesize (int): Number of target PDF samples evaluated simultaneously. Must be a divisor of samplesize.
            max_iterative_batchsize (int): The max number of batch samples evaluated simultaneously. 
            failsafe_crosscheck_tolerance (float / None): If set, is used to crosscheck forward/bakckward pass compatability and resample if necessary. Has been introduced for the v flow in particular, so it should not be necessary for other flows.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.
            return_samples (bool): Return the samples that are generated to calculate the entropy? Samples are returned as B*num_samples X sample_dim, so the effective batch dimension is B*num_samples.
            verbose (bool): Adds some extra prints if given.

        Returns:
            dict
                Dictionary containing entropy for each index defined in parameter *sub_manifolds*. If *-1* was given in *sub_manifolds*, the resulting entropy is stored under the *total* key.
            
            Tensor
                Only returned if *return_samples* is set to True. A tensor that contains the generated samples used to calculate the entropy.
        """

        data_type, used_device=self.obtain_current_dtype_n_device()

        if(device is not None):
            used_device=device
        if(dtype is not None):
            data_type=dtype

        data_summary = None

        assert(samplesize % iterative_samplesize == 0), ("Sample size must be divisble by iterative sample size!", samplesize, iterative_samplesize)

        ## some crosschecks
        if(conditional_input is not None):
            if(type(conditional_input)==list):

                assert(len(self.conditional_input_dim)==len(conditional_input))
                for ci_ind in range(len(self.conditional_input_dim)):
                    assert(self.conditional_input_dim[ci_ind]==conditional_input[ci_ind].shape[1]), "Inputs of conditional input vector do not match with pre-defined input_dims!"


                for ci_ind, ci in enumerate(conditional_input[:-1]):

                    assert(ci.shape[0]==conditional_input[ci_ind].shape[0]), "Conditional input batch sizes do not agree!"
                    assert(ci.dtype==conditional_input[ci_ind].dtype), "Conditional input types do not match!"
                    assert(ci.device==conditional_input[ci_ind].device), "Conditional input devices do not match!"

        if(force_embedding_coordinates==False):
            print("#### CAUTION: Calculating entropy without forcing embedding coordinates. This might lead to undesired and wrong entropies when using manifold PDFs!#############")
            #raise Exception()
        use_marginal_subdims=False
        ## make sure the settings are self consistent
        for subdim in sub_manifolds:
            if(subdim!=-1):
                assert(subdim>=0 and subdim < len(self.layer_list))
                use_marginal_subdims=True


        ## always use marginal subdims
        use_marginal_subdims=True

        # no conditional input means batch size of *1*
        batch_size=1

        if conditional_input is not None:

            assert(self.conditional_input_dim is not None)

            if(type(conditional_input)==list):

                data_type = conditional_input[0].dtype
                used_device = conditional_input[0].device
                
                # a list of data summaries for the next functions
                data_summary=[ci.repeat_interleave(samplesize, dim=0) for ci in conditional_input]

                batch_size=conditional_input[0].shape[0]
            else:

                data_type = conditional_input.dtype
                used_device = conditional_input.device
                
                # this behavior is a little differnet than in standard sample .. we sample for every conditional input multiple times
                data_summary=conditional_input.repeat_interleave(samplesize, dim=0)

                batch_size=conditional_input.shape[0]
        else:
            assert(self.conditional_input_dim is None), "We require conditional input, since this is a conditional PDF."

        entropy_dict=dict()

        
        if(use_marginal_subdims==False):
            ## just calculate normal entropy by summing over samples
            
            #targets, log_det_dict=self.all_layer_forward_individual_subdims(std_normal_samples, data_summary, sub_manifolds=sub_manifolds, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)
            
            targets, _, log_pdf_dict, _=self.all_layer_forward_individual_subdims_incl_sampling(data_summary, 
                                                                                          samplesize*batch_size, 
                                                                                          failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance, 
                                                                                          force_embedding_coordinates=force_embedding_coordinates, 
                                                                                          force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                          dtype=data_type,
                                                                                          device=used_device)

            entropy_dict["total"]=-(log_pdf_dict["total"]).reshape(-1,samplesize).mean(dim=1)
        
        else:

            ## make sure we go all the way to the end by including -1 if it is not there
            sub_manifolds_here=sub_manifolds
            if(-1 not in sub_manifolds):
                sub_manifolds_here=[-1]+sub_manifolds

            ## transform all base samples forward
            
            #targets, log_det_dict_fw=self.all_layer_forward_individual_subdims(std_normal_samples, data_summary, sub_manifolds=sub_manifolds_here, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)
            targets, _, log_pdf_dict, _=self.all_layer_forward_individual_subdims_incl_sampling(data_summary, 
                                                                                          samplesize*batch_size, 
                                                                                          failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance, 
                                                                                          force_embedding_coordinates=force_embedding_coordinates, 
                                                                                          force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                          dtype=data_type,
                                                                                          device=used_device)

            for sub_mf in sub_manifolds:

                if(verbose):
                    print("---> calculating entropy for index ", sub_mf)

                ## also calculate total
                if(-1 == sub_mf):
                    
                    entropy_dict["total"]=-(log_pdf_dict["total"]).reshape(-1, samplesize).mean(dim=1)
                elif(0==sub_mf):
                 
                    entropy_dict[0]=-(log_pdf_dict[0]).reshape(-1, samplesize).mean(dim=1)
                   
                else:

                    max_target_first_index=0
                   
                    for lower_mf in range(sub_mf):

                        if(force_embedding_coordinates):
                            max_target_first_index+=self.target_dims_embedded[lower_mf]
                        elif(force_intrinsic_coordinates):
                            max_target_first_index+=self.target_dims_intrinsic[lower_mf]
                        else:
                            max_target_first_index+=self.target_dims[lower_mf]
                    

                    num_iterative_steps=samplesize//iterative_samplesize

                    num_batch_iterative_steps=math.ceil(float(batch_size)/float(max_iterative_batchsize))
                    if(max_iterative_batchsize>=batch_size):
                        num_batch_iterative_steps=1


                    ## the number of effective items per batch is just the samplesize at this point
                    per_batch_number_of_items=samplesize

                    all_log_probs=[]

                    if(verbose):
                        print("---> number of batch iterative steps: ", num_batch_iterative_steps)
                        print("---> number of iterative steps per batch: ", num_iterative_steps)
                        print("------> total: ", num_batch_iterative_steps*num_iterative_steps)

                    for this_batch_sub_iter in range(num_batch_iterative_steps):
                        
                        
                        log_probs=[]

                        cur_batch_slice=slice(this_batch_sub_iter*max_iterative_batchsize*per_batch_number_of_items,this_batch_sub_iter*max_iterative_batchsize*per_batch_number_of_items+max_iterative_batchsize*per_batch_number_of_items)

                        for cur_step in range(num_iterative_steps):

                            if(verbose):
                                print(this_batch_sub_iter, cur_step)

                            ## this construction correctly orders both cases of conditional input and no conditional input
                            repeated_targets_first=targets[cur_batch_slice, :max_target_first_index].unsqueeze(0).reshape(-1, samplesize, max_target_first_index).repeat(1,iterative_samplesize,1)
                            repeated_targets_first=repeated_targets_first.reshape(-1, max_target_first_index)
                            
                            if(force_embedding_coordinates):
                                repeated_final=targets[cur_batch_slice, self.target_dim_indices_embedded[sub_mf][0]:self.target_dim_indices_embedded[sub_mf][1]].unsqueeze(0).reshape(-1, samplesize, self.target_dim_indices_embedded[sub_mf][1]-self.target_dim_indices_embedded[sub_mf][0])[:,cur_step*iterative_samplesize:cur_step*iterative_samplesize+iterative_samplesize,:].repeat_interleave(samplesize, dim=1)
                                repeated_final=repeated_final.reshape(-1, self.target_dim_indices_embedded[sub_mf][1]-self.target_dim_indices_embedded[sub_mf][0])
                            elif(force_intrinsic_coordinates):
                                repeated_final=targets[cur_batch_slice, self.target_dim_indices_intrinsic[sub_mf][0]:self.target_dim_indices_intrinsic[sub_mf][1]].unsqueeze(0).reshape(-1, samplesize, self.target_dim_indices_intrinsic[sub_mf][1]-self.target_dim_indices_intrinsic[sub_mf][0])[:,cur_step*iterative_samplesize:cur_step*iterative_samplesize+iterative_samplesize,:].repeat_interleave(samplesize, dim=1)
                                repeated_final=repeated_final.reshape(-1, self.target_dim_indices_intrinsic[sub_mf][1]-self.target_dim_indices_intrinsic[sub_mf][0])
                            else:
                                repeated_final=targets[cur_batch_slice, self.target_dim_indices[sub_mf][0]:self.target_dim_indices[sub_mf][1]].unsqueeze(0).reshape(-1, samplesize, self.target_dim_indices[sub_mf][1]-self.target_dim_indices[sub_mf][0])[:,cur_step*iterative_samplesize:cur_step*iterative_samplesize+iterative_samplesize,:].repeat_interleave(samplesize, dim=1)
                                repeated_final=repeated_final.reshape(-1, self.target_dim_indices[sub_mf][1]-self.target_dim_indices[sub_mf][0])
                            
                            assert(repeated_targets_first.shape[0]>0), "Weird, no remaining items.. maybe the // operator behavior changed."
                            
                          
                            ########################
                           
                            joint_repeated=torch.cat([repeated_targets_first, repeated_final], dim=1)

                            fillup_difference=int(targets.shape[1])-int(joint_repeated.shape[1])
                            
                            ## fill up the rest (after current sub-mf) with ones so we get "some coordinates" in those to call all_layer_inverse
                            filled_up=torch.cat([joint_repeated, torch.ones(joint_repeated.shape[0], fillup_difference).to(repeated_final)], dim=1)

                            if(data_summary is None):
                                new_base_vals, log_det_dict_individual=self.all_layer_inverse_individual_subdims(filled_up, None, sub_manifolds=[sub_mf], force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)
                            elif(type(data_summary)==list):
                                new_base_vals, log_det_dict_individual=self.all_layer_inverse_individual_subdims(filled_up, [ds[cur_batch_slice].repeat_interleave(iterative_samplesize, dim=0) for ds in data_summary], sub_manifolds=[sub_mf], force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)
                            else:
                                new_base_vals, log_det_dict_individual=self.all_layer_inverse_individual_subdims(filled_up, data_summary[cur_batch_slice].repeat_interleave(iterative_samplesize, dim=0), sub_manifolds=[sub_mf], force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)


                            this_base_dim=self.base_dim_indices[sub_mf][1]-self.base_dim_indices[sub_mf][0]

                            log_gauss_evals_base=torch.distributions.Normal(0.0,1.0).log_prob(new_base_vals[:, self.base_dim_indices[sub_mf][0]:self.base_dim_indices[sub_mf][1]]).sum(dim=-1)
                            
                           
                            this_log_probs=(log_gauss_evals_base+log_det_dict_individual[sub_mf]).reshape(-1,iterative_samplesize, samplesize).logsumexp(dim=-1)-numpy.log(float(samplesize))
                          
                            log_probs.append(this_log_probs)

                        log_probs=torch.cat(log_probs, dim=1)

                        all_log_probs.append(log_probs)


                    all_log_probs = torch.cat(all_log_probs, dim=0)
                       
                    entropy_dict[sub_mf]=-all_log_probs.mean(dim=1)
                   
        if(return_samples):
            return entropy_dict, targets, log_pdf_dict
        else:
            return entropy_dict

    def all_layer_inverse_individual_subdims(self, 
                                             x, 
                                             data_summary, 
                                             amortization_parameters=None, 
                                             force_embedding_coordinates=False, 
                                             force_intrinsic_coordinates=False,
                                             sub_manifolds=[-1]):


        ## set maximum iter to last sub dimension
        max_iter=0
        log_det_dict=dict()
        ## make sure the settings are self consistent
        for subdim in sub_manifolds:
            if(subdim!=-1):
                
                assert(subdim>=0 and subdim < len(self.layer_list))

                if(subdim>max_iter):
                    max_iter=subdim

                log_det_dict[subdim]=0.0
        
        ## only go as far as required
        if(-1 in sub_manifolds):
            max_iter=len(self.layer_list)-1
            log_det_dict["total"]=0.0

        
        ## make sure we transform to default settings (potentially mixed) if we force embedding/intrinsic coordinates
        if(force_embedding_coordinates):
            assert(x.shape[1]==self.total_target_dim_embedded), (x.shape[1], self.total_target_dim_embedded)
            assert(force_intrinsic_coordinates==False)

            x, log_det=self.transform_target_space_individual_subdims(x, transform_from="embedding", transform_to="default")

            for k in log_det_dict.keys():
                log_det_dict[k]=log_det_dict[k]+log_det[k]

        elif(force_intrinsic_coordinates):
            assert(x.shape[1]==self.total_target_dim_intrinsic)
            assert(force_embedding_coordinates==False)

            x, log_det=self.transform_target_space_individual_subdims(x, transform_from="intrinsic", transform_to="default")

            for k in log_det_dict.keys():
                log_det_dict[k]=log_det_dict[k]+log_det[k]

        else:
            assert(x.shape[1]==self.total_target_dim), (x.shape[1], self.total_target_dim)
        
        ## we shoould not be in default target dim mode
        tot_remaining_dim=0
        for pdf_index in range(max_iter+1):

            tot_remaining_dim+=self.target_dims[pdf_index]
        
        x=x[:,:tot_remaining_dim]

        extra_conditional_input=[]
        base_targets=[]

        individual_logdets=dict()
        total_logdet=0.0

        extra_params = None

        if(amortization_parameters is not None):

            assert(amortization_parameters.shape[1]==self.total_number_amortizable_params)
            
        amort_param_counter=0

        for pdf_index, pdf_layers in enumerate(self.layer_list[:max_iter+1]):

            extra_param_counter = 0
            this_pdf_type=self.pdf_defs_list[pdf_index]

            ## mlp preditors can be None for unresponsive layers like x/y
            if(data_summary is not None and self.mlp_predictors[pdf_index] is not None):
                
                if(type(data_summary)==list):
                    this_data_summary=data_summary[pdf_index]
                else:
                    this_data_summary=data_summary

                if(len(extra_conditional_input)>0):
                    this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)
                
                if(amortization_parameters is not None):
                    num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                    extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                    amort_param_counter+=num_amortization_params

                else:
                    extra_params=self.mlp_predictors[pdf_index](this_data_summary)
               
            else:

                if(self.mlp_predictors[pdf_index] is not None):
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        if(amortization_parameters is not None):
                            num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                            extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                            amort_param_counter+=num_amortization_params

                        else:
                            extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                        
                    else:
                        raise Exception("FORWARD: extra conditional input is empty but required for encoding!")

                else:
                    ## we amortize everything with amortization_parameters .. including the first layer column if there is no encoder
                    if(self.amortize_everything and pdf_index ==0):
                        assert(amortization_parameters is not None)
                        assert(amort_param_counter==0)

                        tot_num_params=0
                        for l in self.layer_list[0]:
                            tot_num_params+=l.get_total_param_num()

                        extra_params=amortization_parameters[:,:tot_num_params]

                        if(self.predict_log_normalization and self.join_poisson_and_pdf_description):
                            tot_num_params+=1

                        amort_param_counter+=tot_num_params


            if(self.predict_log_normalization):
                if(pdf_index==0 and self.join_poisson_and_pdf_description and extra_params is not None):
                    extra_params=extra_params[:,:-1]

            this_target=x[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]

            this_subpdf_log_det=0.0
            
            ## reverse mapping is required for pdf evaluation
            for l, layer in reversed(list(enumerate(pdf_layers))):

                this_extra_params = None

                if extra_params is not None:

                    if extra_param_counter == 0:
                            this_extra_params = extra_params[:, -layer.total_param_num :]
                    else:

                        this_extra_params = extra_params[
                            :,
                            -extra_param_counter
                            - layer.total_param_num : -extra_param_counter,
                        ]

 
                if(l==(len(pdf_layers)-1)):
                    # force embedding or intrinsic coordinates in the layer that defines the target dimension
                    this_target, this_layer_log_det = layer.inv_flow_mapping([this_target, 0.0], extra_inputs=this_extra_params)
                else:

                    this_target, this_layer_log_det = layer.inv_flow_mapping([this_target, 0.0], extra_inputs=this_extra_params)
                
                this_subpdf_log_det=this_subpdf_log_det+this_layer_log_det

                extra_param_counter += layer.total_param_num

            if("total" in log_det_dict.keys()):
                log_det_dict["total"]=log_det_dict["total"]+this_subpdf_log_det

            if(pdf_index in log_det_dict.keys()):
                log_det_dict[pdf_index]=log_det_dict[pdf_index]+this_subpdf_log_det

         
            base_targets.append(this_target)

            ## we are back in "default target mode", so need to transform target for next layer if necessary
            prev_target=x[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]
            prev_target=pdf_layers[-1]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        base_pos=torch.cat(base_targets, dim=1)

        return base_pos, log_det_dict

    def all_layer_forward_individual_subdims_incl_sampling(self, 
                                       data_summary,
                                       total_samplesize,
                                       failsafe_crosscheck_tolerance=None,
                                       force_embedding_coordinates=False, 
                                       force_intrinsic_coordinates=False,
                                       amortization_parameters=None,
                                       dtype=None,
                                       device=None
                                       ):

        std_normal_samples = torch.randn(size=(total_samplesize, self.total_base_dim), dtype=dtype, device=device)
        
        ## save the easy cases in dict
        base_evals_dict=dict()

        ## get all manfiolds + total
        sub_manifolds=list(range(len(self.pdf_defs_list)))+[-1]
      
        for mf_dim in sub_manifolds:
           
            if(mf_dim==-1):
            
                this_mask=slice(0, self.total_base_dim)

            else:

                this_mask=slice(self.base_dim_indices[mf_dim][0], self.base_dim_indices[mf_dim][1])
        
           
            log_gauss_evals=torch.distributions.Normal(0.0,1.0).log_prob(std_normal_samples[:, this_mask]).sum(dim=-1)
                
            if(mf_dim==-1):
                base_evals_dict["total"]=log_gauss_evals
            else:
                base_evals_dict[mf_dim]=log_gauss_evals



        new_targets, logdet_per_manifold=self.all_layer_forward_individual_subdims(std_normal_samples, 
                                                  data_summary, 
                                                  force_embedding_coordinates=force_embedding_coordinates, 
                                                  force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                  sub_manifolds=sub_manifolds,
                                                  amortization_parameters=amortization_parameters)

 
        return_log_pdf=dict()
        for k in logdet_per_manifold:
            return_log_pdf[k]=base_evals_dict[k]-logdet_per_manifold[k]

        if(failsafe_crosscheck_tolerance):
            new_targets_prop, std_normal_samples_prop, return_log_pdf_prop, base_evals_dict_prop=recheck_sampling(self, 
                      new_targets,
                      std_normal_samples,
                      return_log_pdf,
                      base_evals_dict,
                      sub_manifolds=sub_manifolds,
                      failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                      conditional_input=data_summary,
                      amortization_parameters=amortization_parameters,
                      force_embedding_coordinates=force_embedding_coordinates,
                      force_intrinsic_coordinates=force_intrinsic_coordinates,
                      dtype=dtype,
                      device=device)

            if(new_targets_prop is not None):
                new_targets=new_targets_prop
                std_normal_samples=std_normal_samples_prop
                return_log_pdf=return_log_pdf_prop
                base_evals_dict=base_evals_dict_prop


        return new_targets, std_normal_samples, return_log_pdf, base_evals_dict


    def all_layer_forward_individual_subdims(self, 
                                       x,
                                       data_summary,
                                       force_embedding_coordinates=False, 
                                       force_intrinsic_coordinates=False,
                                       sub_manifolds=[-1],
                                       amortization_parameters=None
                                       ):

            

            total_batch_size=x.shape[0]
            if(force_embedding_coordinates):
                assert(force_intrinsic_coordinates!=force_embedding_coordinates), "Embedding and intrinsic coordinates can not be used at the same time!"

            ## set maximum iter to last sub dimension
            max_iter=0

            ## make sure the settings are self consistent
            for subdim in sub_manifolds:
                if(subdim!=-1):
                    
                    assert(subdim>=0 and subdim < len(self.layer_list))

                    if(subdim>max_iter):
                        max_iter=subdim
            
            ## only go as far as required
            if(-1 in sub_manifolds):
                max_iter=len(self.layer_list)-1

            
            extra_conditional_input=[]
            new_targets=[]
            logdet_per_manifold=dict()
            tot_log_det=0.0

            if(amortization_parameters is not None):

                raise Exception("Currently only supported without full amortization")
                #assert(amortization_parameters.shape[0]==x.shape[0]), ("batch size of x must agree with batch size of amortization_parameters")
                assert(amortization_parameters.shape[1]==self.total_number_amortizable_params)

            amort_param_counter=0

            for pdf_index, pdf_layers in enumerate(self.layer_list):

                this_pdf_type=self.pdf_defs_list[pdf_index]

                extra_params = None
                if(data_summary is not None and self.mlp_predictors[pdf_index] is not None):
                    
                    if(type(data_summary)==list):
                        this_data_summary=data_summary[pdf_index]
                    else:
                        this_data_summary=data_summary
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)

                    if(amortization_parameters is not None):
                        num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                        extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                        amort_param_counter+=num_amortization_params

                    else:
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                   

                else:

                    if(self.mlp_predictors[pdf_index] is not None):
                        if(len(extra_conditional_input)>0):
                            this_data_summary=torch.cat(extra_conditional_input, dim=1)
                            
                            if(amortization_parameters is not None):
                                num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                                extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                                amort_param_counter+=num_amortization_params

                            else:
                                extra_params=self.mlp_predictors[pdf_index](this_data_summary)

                        else:
                            raise Exception("SAMPLE: extra conditional input is empty but required for encoding!")

                    else:
                        ## we amortize everything with amortization_parameters .. including the first layer if there is no encoder
                        if(self.amortize_everything and pdf_index ==0):
                            assert(amortization_parameters is not None)
                            assert(amort_param_counter==0)

                            tot_num_params=0
                            for l in self.layer_list[0]:
                                tot_num_params+=l.get_total_param_num()

                            if(self.predict_log_normalization and self.join_poisson_and_pdf_description):
                                tot_num_params+=1

                            extra_params=amortization_parameters[:,:tot_num_params]

                            amort_param_counter+=tot_num_params


         
                if(self.predict_log_normalization):

                    if(pdf_index==0 and self.join_poisson_and_pdf_description):
                        extra_params=extra_params[:,:-1]

                this_target=x[:,self.base_dim_indices[pdf_index][0]:self.base_dim_indices[pdf_index][1]]

                ## loop through all layers in each pdf and transform "this_target"
                
                extra_param_counter = 0

                ## holds either the marginal
                logdet_this_manifold = 0.0

                ## default all layer foward
                for l, layer in list(enumerate(pdf_layers)):
                   
                    this_extra_params = None
                    
                    if extra_params is not None:
                        
                        this_extra_params = extra_params[:, extra_param_counter : extra_param_counter + layer.total_param_num]
                  
                    this_target, this_log_det = layer.flow_mapping([this_target, 0.0], extra_inputs=this_extra_params)
                    
                    extra_param_counter += layer.total_param_num

                    logdet_this_manifold = logdet_this_manifold+this_log_det

                # default joint logdet
                if(-1 in sub_manifolds):
                    tot_log_det=tot_log_det+logdet_this_manifold

                ## save logdet factors for this subdimension in dict
                logdet_per_manifold[pdf_index]=logdet_this_manifold

                ## next one
                new_targets.append(this_target)

                prev_target=this_target
               
                prev_target=self.layer_list[pdf_index][-1]._embedding_conditional_return(prev_target)

                extra_conditional_input.append(prev_target)

                if(max_iter==pdf_index):
                    break

            # we want to the total logdet also
            if(-1 in sub_manifolds):
                logdet_per_manifold["total"]=tot_log_det

            if (torch.isfinite(x) == 0).sum() > 0:
                raise Exception("nonfinite samples generated .. this should never happen!")

            res=torch.cat(new_targets, dim=1)

            if(res.shape[1]<x.shape[1]):
                #artificially increase shape
                res=torch.cat([res, torch.zeros( (int(x.shape[0]), int(x.shape[1]-res.shape[1])), dtype=x.dtype, device=x.device)],dim=1)

            ## transform to desired output space 
            if(force_embedding_coordinates):

                res, embedding_log_dets=self.transform_target_space_individual_subdims(res, transform_from="default", transform_to="embedding")
                
                tot_remaining_dim=0
                for pdf_index in range(max_iter+1):

                    tot_remaining_dim+=self.target_dims_embedded[pdf_index]

                ## shorten res if necessary
                res=res[:,:tot_remaining_dim]

                ## copy logdet correction over 
                for k in logdet_per_manifold.keys():
                    logdet_per_manifold[k]=logdet_per_manifold[k]+embedding_log_dets[k]

            elif(force_intrinsic_coordinates):
                assert(x.shape[1]==self.total_target_dim_intrinsic)
               
                res, embedding_log_dets=self.transform_target_space_individual_subdims(res, transform_from="default", transform_to="intrinsic")
                
                tot_remaining_dim=0
                for pdf_index in range(max_iter+1):

                    tot_remaining_dim+=self.target_dims_intrinsic[pdf_index]
                    
                ## shorten res if necessary
                res=res[:,:tot_remaining_dim]

                ## copy logdet correction over 
                for k in logdet_per_manifold.keys():
                    logdet_per_manifold[k]=logdet_per_manifold[k]+embedding_log_dets[k]

            """
            ### 
            if(failsafe_crosscheck_tolerance):

                ## inverse and another forward

                raise Exception()
                
                with torch.no_grad():
                    new_base, bw_logdet_dict=self.all_layer_inverse_individual_subdims(res, 
                                                 data_summary, 
                                                 amortization_parameters=amortization_parameters, 
                                                 force_embedding_coordinates=force_embedding_coordinates, 
                                                 force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                 sub_manifolds=sub_manifolds)

                    ## and another forward

                   second_res, fw_logdet_dict=self.all_layer_forward_individual_subdims(
                                       new_base,
                                       data_summary,
                                       failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                       force_embedding_coordinates=force_embedding_coordinates, 
                                       force_intrinsic_coordinates=force_intrinsic_coordinates,
                                       sub_manifolds=sub_manifolds,
                                       amortization_parameters=amortization_parameters
                                       )


            """




            return res, logdet_per_manifold

    def transform_target_space_individual_subdims(self, target, transform_from="default", transform_to="embedding"):
      

        new_target=target

        if(len(target.shape)==1):
            new_target=target.unsqueeze(0)

        ## transforming only makes sense if input has correct target shape
        if(transform_from=="default"):
            assert(new_target.shape[1]==self.total_target_dim)
        elif(transform_from=="intrinsic"):
            assert(new_target.shape[1]==self.total_target_dim_intrinsic)
        elif(transform_from=="embedding"):
            assert(new_target.shape[1]==self.total_target_dim_embedded) 

        potentially_transformed_vals=[]

        index=0

        individual_logdets=dict()
        tot_logdet=0.0

        for pdf_index, pdf_type in enumerate(self.pdf_defs_list):

            if(transform_from=="default"):
                this_dim=self.target_dims[pdf_index]
            elif(transform_from=="intrinsic"):
                this_dim=self.target_dims_intrinsic[pdf_index]
            elif(transform_from=="embedding"):
                this_dim=self.target_dims_embedded[pdf_index]
            
            this_target, this_log_det=self.layer_list[pdf_index][-1].transform_target_space(new_target[:,index:index+this_dim], log_det=0, transform_from=transform_from, transform_to=transform_to)
            
            potentially_transformed_vals.append(this_target)

            index+=this_dim

            individual_logdets[pdf_index]=this_log_det

            tot_logdet=tot_logdet+this_log_det

        individual_logdets["total"]=tot_logdet

        potentially_transformed_vals=torch.cat(potentially_transformed_vals, dim=1)

        if(transform_to=="default"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim)
        elif(transform_to=="intrinsic"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_intrinsic), (potentially_transformed_vals.shape, self.total_target_dim_intrinsic)
        elif(transform_to=="embedding"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_embedded), (new_target.shape[1], self.total_target_dim_embedded)  


        if(len(target.shape)==1):
            potentially_transformed_vals=potentially_transformed_vals.squeeze(0)

        return potentially_transformed_vals, individual_logdets

    def obtain_current_dtype_n_device(self):

        ## peek into first parameter vector
        try:
            first = next(self.parameters())
        except StopIteration:
            first = None

        if(first is None):
            ## model contains no parameters
            return None, None

        else:
            return first.dtype, first.device

    def marginal_moments(self, 
                         conditional_input=None, 
                         samplesize=50, 
                         iterative_samplesize=10, 
                         max_iterative_batchsize=20,
                         mises_abs_precision=1e-7, 
                         calc_kl_diff_and_entropic_quantities=False,
                         failsafe_crosscheck_tolerance=None,
                         dtype=None,
                         device=None,
                         verbose=False,
                         s2_entropy_scanning=False,
                         s2_entropy_scan_nside=32):
        """
        Calculate the first and second central moments of the marginal distributions. For Euclidean manifolds it calculates a Gaussian approximation, for spherical distributions calculates
        a von-Mises approximation. Because these are the respective maximum entropy distributions, their entropy should always be larger than the original distribution.
        We can also calculate the kl divergence and cross entropy between the exact distribution and its approximation, by switching on the *calc_kl_diff_and_entropic_quantities* flag.

        Parameters:
            conditional_input (Tensor/list(Tensor)/None): If passed defines the input to the PDF.
            samplesize (int): Samplesize to use per event for moment approximation.
            iterative_samplesize (int): Number of target PDF samples evaluated simultaneously. Must be a divisor of samplesize.
            max_iterative_batchsize (int): The max number of batch samples evaluated simultaneously. 
            mises_abs_precision (float): The absolute precision used break out the loop for the mises distribution second moment estimation.
            calc_kl_diff_and_entropic_quantities (bool): Flag that determines if we also calculate the KL divergence between the respective marginal distribution and its respective 2nd-order approximation. Also includes the cross entropy.
            failsafe_crosscheck_tolerance (float / None): If set, is used to crosscheck forward/bakckward pass compatability and resample if necessary. Has been introduced for the v flow in particular, so it should not be necessary for other flows.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.
            verbose (bool): Some extra print statements on runtime.
            s2_entropy_scanning (bool): Use a healpix scan to determine entropy .. can be faster for certain s2 distributions.

        Returns:

            dict
                Dictionary containing moments and potentially entropies and KL-divergence between approximation and marginal distributions for each marginal distribution.
                The index indicates which marginal distribution. Means on the sphere are given in embedding coordinates and in angle coordinates. For example:

                mean_0 = mean of first marginal in embedding coordinates
                mean_0_angles = mean of first marginal in angle coordinates
                varlike_0 = covariance (Gaussian/Euclidean) or concentration parameter (Fisher-von Mises/spherical)
                entropy_0 = entropy of first marginal distribution
                kl_diff_exact_approx_0 = KL divergence between exact distribution and 2nd-order approximation
                cross_entropy_0 = cross entropy between exact distribution and 2nd-order approximation
                ...
                mean_1 = mean of second marginal
                ...
                ...
                entropy_total = total entropy (only total quantity calculated because it is essentially free)

           
        """

        def random_VMF(mu , kappa , size = None):
            """
            Von Mises - Fisher distribution sampler with
            mean direction mu and co nc en tr at io n kappa .
            Source : https :// hal . science /hal - 04004568
            """
            # parse input parameters
            n = 1 if size is None else numpy.product( size )
            shape = () if size is None else tuple( numpy.ravel( size ))
            mu = numpy.asarray( mu )
            mu = mu / numpy.linalg.norm( mu )
            (d ,) = mu.shape
            # z component : radial samples pe rp en dic ul ar to mu
            z = numpy.random.normal(0 , 1 , (n , d) )
            z /= numpy.linalg.norm(z , axis =1 , keepdims = True )
            z = z - (z @ mu[:, None ]) * mu[None , :]
            z /= numpy.linalg.norm(z , axis =1 , keepdims = True )
            # sample angles ( in cos and sin form )
            cos = _random_VMF_cos(d , kappa , n )
            sin = numpy.sqrt(1 - cos ** 2)
            # combine angles with the z component
            x = z * sin[:, None ] + cos[:, None ] * mu[None , :]
            return x.reshape((*shape , d ))

        def _random_VMF_cos(d: int , kappa : float , n: int):
            """
            Generate n iid samples t with density function given by
            p(t) = someConstant * (1-t**2) **((d-2)/2) * exp ( kappa *t)
            """
            # b = Eq. 4 of https :// doi . org / 10. 1080 / 0 3 6 1 0 9 1 9 4 0 8 8 1 3 1 6 1
            b = ( d - 1) / (2 * kappa + (4 * kappa ** 2 + ( d - 1) ** 2) ** 0.5)
            x0 = (1 - b) / ( 1 + b)
            c = kappa * x0 + ( d - 1) * numpy.log(1 - x0 ** 2)
            found = 0
            out = []
            while found < n:
                m = min(n , int((n-found)*1.5))
                z = numpy.random.beta(( d - 1 ) / 2 , (d - 1) / 2 , size =m )
                t = ( 1 - (1 + b) * z) / (1 - (1 - b) * z)
                test = kappa * t + (d - 1) * numpy.log(1 - x0 * t) - c
                accept = test >= -numpy.random.exponential( size =m)
                out.append(t[ accept ])
                found += len( out [-1])
            return numpy.concatenate( out )[:n]

        def get_rotation_matrix(target_vector):
            # Normalize the target vector
            
            if(target_vector[0]==0 and target_vector[1]==0 and target_vector[2]==1.0):
                return numpy.eye(3)
            
            target_vector = numpy.array(target_vector) / numpy.linalg.norm(target_vector)
            
            # Define the initial vector (0, 0, 1)
            initial_vector = numpy.array([0, 0, 1])
            
            # Calculate the cross product between the initial and target vectors
            cross_product = numpy.cross(initial_vector, target_vector)
            
            # Calculate the dot product between the initial and target vectors
            dot_product = numpy.dot(initial_vector, target_vector)
            
            # Construct the skew-symmetric matrix
            skew_symmetric_matrix = numpy.array([[0, -cross_product[2], cross_product[1]],
                                              [cross_product[2], 0, -cross_product[0]],
                                              [-cross_product[1], cross_product[0], 0]])
            
            # Construct the rotation matrix using Rodrigues' rotation formula
            rotation_matrix = numpy.identity(3) + skew_symmetric_matrix + numpy.dot(skew_symmetric_matrix, skew_symmetric_matrix) * (1 - dot_product) / (numpy.linalg.norm(cross_product) ** 2)
            
            return rotation_matrix

        def uniform_to_w(samples, kappa=1.0):

            w_samples=1.0+(1.0/kappa)*numpy.log(samples+(1.0-samples)*numpy.exp(-2*kappa))
            return w_samples

        def obtain_fisher_mises_samples(mu, kappa, samplesize):

            uniform_samples=numpy.random.uniform(size=(samplesize,))
            uniform_samples_2=numpy.random.uniform(size=(samplesize,))

            w_samples=uniform_to_w(uniform_samples,kappa=kappa)
            
            circle_samples_x=numpy.cos(uniform_samples_2*2*numpy.pi)
            circle_samples_y=numpy.sin(uniform_samples_2*2*numpy.pi)
            
            pre_fac=numpy.sqrt(1.0-w_samples**2)
                
            assert(type(mu)==numpy.ndarray)
            assert(len(mu)==3)
            assert(len(mu.shape)==1)
            
            rot_matrix=get_rotation_matrix(mu)
            
            rot_matrix=numpy.repeat(rot_matrix[None,:,:], samplesize, axis=0)
            
            vecs=[ (pre_fac*circle_samples_x)[:,None], (pre_fac*circle_samples_y)[:,None], w_samples[:,None]]
            vecs=numpy.concatenate(vecs, axis=1)
                
            return numpy.einsum("...ij,...j -> ...i", rot_matrix, vecs)

        def newton_iter(arg, p, normed_length_summed_pts):
            """
            p is the Euclidean embedding dimension: p=3 -> 2-d sphere, p=2 -> 1-d circle
            """
            ## p = 2 -> p/2-1 = 0, p/2 = 1
            ## p = 3 -> p/2-1 = 0.5, p/2 = 1.5

            if(p==2):

                a_p_k=i1(arg)/i0(arg)

                return arg- ((a_p_k-normed_length_summed_pts)/(1.0-a_p_k**2- (1.0/arg)*a_p_k))

               
            elif(p==3):
               
                #a_p_k=iv(1.5, arg)/iv(0.5, arg)
                # calculate ratio based on exact formulas from abramovitz & stegun (p.443)
                a_p_k=-1.0/arg+1.0/numpy.tanh(arg)
              
              
                temp=(a_p_k-normed_length_summed_pts)/(1.0-a_p_k**2- (2.0/arg)*a_p_k)
               
                return arg- ( (a_p_k-normed_length_summed_pts)/(1.0-a_p_k**2- (2.0/arg)*a_p_k) )

        

        used_dtype, used_device=self.obtain_current_dtype_n_device()

        if(device is not None):
            used_device=device
        if(dtype is not None):
            used_dtype=dtype

        initial_batch_size=1

        if(conditional_input is not None):
            if(type(conditional_input)==list):   

                initial_batch_size=conditional_input[0].shape[0]

                assert(conditional_input[0].dtype == used_dtype)
                assert(conditional_input[0].device == used_device)
                
            else:

                initial_batch_size=conditional_input.shape[0]

                assert(conditional_input[0].dtype == used_dtype)
                assert(conditional_input[0].device == used_device)

        return_dict=dict()

        if(verbose):
            tbef=time.time()

        with torch.no_grad():
        
            entropy_dict=None

            data_summary_repeated=None
            if(conditional_input is not None):
                if(type(conditional_input)==list):  
                    data_summary_repeated=[ci.repeat_interleave(samplesize, dim=0) for ci in conditional_input]
                else:
                    data_summary_repeated=conditional_input.repeat_interleave(samplesize, dim=0)

            
            if(calc_kl_diff_and_entropic_quantities):
                # also calculate total entropy [-1], because it is no extra cost 

                if(s2_entropy_scanning):
                    assert(self.pdf_defs_list[0]=="s2")

                    ent_vec=[]
                    samp_vec=[]

                    if(conditional_input is None):

                        nside=s2_entropy_scan_nside

                        tot_sum=0.0

                        diff_tol=0.001

                        is_still_bad=True

                        MAX_SIZE=100000
                        while( is_still_bad):

                            nside=nside*2

                            num_pix=healpy.nside2npix(nside)

                            area_per_pixel=4*numpy.pi/num_pix

                            theta, phi = healpy.pix2ang(nside=nside, ipix=numpy.arange(num_pix))

                            target_angles=torch.from_numpy(numpy.concatenate([theta[:,None], phi[:,None]], axis=1))
                           
                            target_xyz,_=self.transform_target_space(target_angles, transform_from="intrinsic", transform_to="embedding")

                            final_target=target_xyz.to(conditional_input)
                            
                            all_log_pdfs=[]

                            other_iter=0
                            next_target=final_target[MAX_SIZE*other_iter:MAX_SIZE*other_iter+MAX_SIZE]
                            
                            while(len(next_target)>0):
                                log_pdf,_,_=self.forward(next_target, conditional_input=None, force_embedding_coordinates=True)
                                log_pdf=log_pdf.cpu().numpy()
                                all_log_pdfs.append(log_pdf)

                                other_iter+=1

                                next_target=final_target[MAX_SIZE*other_iter:MAX_SIZE*other_iter+MAX_SIZE]
                                
                            log_pdf=numpy.concatenate(all_log_pdfs)
                            pdf_eval=numpy.exp(log_pdf)#numpy.exp(log_pdf)

                            assert(len(pdf_eval)==num_pix), (len(pdf_eval), num_pix)
                            probabilities=pdf_eval*area_per_pixel

                            tot_sum=probabilities.sum()
                           
                            sample_indices=numpy.random.choice(numpy.arange(num_pix), samplesize, p=probabilities/tot_sum)

                            num_unique_items=len(set(sample_indices))

                            is_still_bad = (numpy.fabs(tot_sum-1.0)>diff_tol) | (num_unique_items < int(samplesize/10))
                            print("nside ", nside," sidelen of pixel :", numpy.sqrt(area_per_pixel)*180.0/numpy.pi)
                            print("num unique, ",num_unique_items)
                            print("totsum ", tot_sum)
                            print("BAD ? ", is_still_bad)
                    
                    else:

                        for cur_cinput in conditional_input:

                            nside=s2_entropy_scan_nside

                            tot_sum=0.0

                            diff_tol=0.001

                            is_still_bad=True

                            MAX_SIZE=100000
                            while( is_still_bad):

                                nside=nside*2

                                num_pix=healpy.nside2npix(nside)

                                area_per_pixel=4*numpy.pi/num_pix

                                theta, phi = healpy.pix2ang(nside=nside, ipix=numpy.arange(num_pix))

                                target_angles=torch.from_numpy(numpy.concatenate([theta[:,None], phi[:,None]], axis=1))
                               
                                target_xyz,_=self.transform_target_space(target_angles, transform_from="intrinsic", transform_to="embedding")

                                final_target=target_xyz.to(conditional_input)
                                final_cinput=cur_cinput[None,:].repeat(num_pix, 1)

                                all_log_pdfs=[]

                                other_iter=0
                                next_target=final_target[MAX_SIZE*other_iter:MAX_SIZE*other_iter+MAX_SIZE]
                                next_cinput=final_cinput[MAX_SIZE*other_iter:MAX_SIZE*other_iter+MAX_SIZE]

                                while(len(next_target)>0):
                                    log_pdf,_,_=self.forward(next_target, conditional_input=next_cinput, force_embedding_coordinates=True)
                                    log_pdf=log_pdf.cpu().numpy()
                                    all_log_pdfs.append(log_pdf)

                                    other_iter+=1

                                    next_target=final_target[MAX_SIZE*other_iter:MAX_SIZE*other_iter+MAX_SIZE]
                                    next_cinput=final_cinput[MAX_SIZE*other_iter:MAX_SIZE*other_iter+MAX_SIZE]

                                    
                                
                                log_pdf=numpy.concatenate(all_log_pdfs)
                                pdf_eval=numpy.exp(log_pdf)#numpy.exp(log_pdf)

                                assert(len(pdf_eval)==num_pix), (len(pdf_eval), num_pix)
                                probabilities=pdf_eval*area_per_pixel

                                tot_sum=probabilities.sum()
                               
                                sample_indices=numpy.random.choice(numpy.arange(num_pix), samplesize, p=probabilities/tot_sum)

                                num_unique_items=len(set(sample_indices))

                                is_still_bad = (numpy.fabs(tot_sum-1.0)>diff_tol) | (num_unique_items < int(samplesize/10))
                                print("nside ", nside," sidelen of pixel :", numpy.sqrt(area_per_pixel)*180.0/numpy.pi)
                                print("num unique, ",num_unique_items)
                                print("totsum ", tot_sum)
                                print("BAD ? ", is_still_bad)
                             

                            theta, phi = healpy.pix2ang(nside=nside, ipix=sample_indices)

                            this_samples=torch.from_numpy(numpy.concatenate([theta[:,None], phi[:,None]], axis=1))
                            this_samples,_=self.transform_target_space(this_samples, transform_from="intrinsic", transform_to="embedding")
                            ent_vec.append((probabilities*(-log_pdf)).sum())
                           
                            samp_vec.append(this_samples.to(conditional_input))

                        entropy_dict=dict()
                        entropy_dict[0]=torch.from_numpy(numpy.array(ent_vec)).to(conditional_input)
                        entropy_dict["total"]=entropy_dict[0]

                        samples=torch.cat(samp_vec, dim=0)


                else:
                    entropy_dict, samples, log_pdf_dict=self.entropy_iterative(sub_manifolds=[-1]+list(range(len(self.pdf_defs_list))), 
                                                                     conditional_input=conditional_input,
                                                                     samplesize=samplesize,
                                                                     iterative_samplesize=iterative_samplesize,
                                                                     max_iterative_batchsize=max_iterative_batchsize,
                                                                     failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                                                     device=used_device,
                                                                     dtype=used_dtype,
                                                                     return_samples=True,
                                                                     verbose=verbose)

           

            
            else:
                
                # a simple sampling is typically faster than whole entropy calculation, so this might be a viable alternative

                samples,sample_logprobs,_,_=self.sample(conditional_input=data_summary_repeated, samplesize=samplesize, device=used_device, dtype=used_dtype, force_embedding_coordinates=True)

            target_dim_embedded=self.total_target_dim_embedded

            samples=samples.unsqueeze(1).reshape(initial_batch_size, -1, target_dim_embedded)

            if(verbose):
                print("1) sampling took ",time.time()-tbef)


            index_mask=None
            ## put the approximate max PDF value in there aswell .. only works if no s2 scan is performed
            if(not s2_entropy_scanning):
                if(calc_kl_diff_and_entropic_quantities):
                    ## we have log_pdf_dict
                    reshaped_log_pdfs=log_pdf_dict["total"].reshape(initial_batch_size, samplesize)

                else:
                    ## we have sample_logprobs
                    reshaped_log_pdfs=sample_logprobs.reshape(initial_batch_size, samplesize)

                index_mask=torch.argmax(reshaped_log_pdfs, dim=1)
              
            for sub_pdf_dim, sub_pdf_def in enumerate(self.pdf_defs_list):

                this_mean=None
                this_var=None
                approx_entropy=None
                cross_entropy=None
                kl_diff=None


                these_subsamples=samples[:, :, self.target_dim_indices_embedded[sub_pdf_dim][0]:self.target_dim_indices_embedded[sub_pdf_dim][1]]

                if("e" in sub_pdf_def):

                    ## this_max_value
                    if(index_mask is not None):
                        this_arg_max=these_subsamples[torch.arange(initial_batch_size), index_mask]

                    this_mean=torch.mean(these_subsamples, dim=1,keepdims=True)

                    subtracted_samples=these_subsamples-this_mean
                  
                    this_var=(torch.einsum("...ij, ...ik -> ...ijk", subtracted_samples,subtracted_samples).sum(dim=1)/(samplesize-1))

                   
                    # collapse unneded dimension
                    this_mean=this_mean.squeeze(1)

                    this_dim=this_mean.shape[1]

                    approx_entropy=0.5*(this_dim*(numpy.log(2*numpy.pi)+1)+torch.log(torch.linalg.det(this_var)) )

                    ## flatten cov matrices for storage
                    #this_var=this_var.flatten(start_dim=1)

                    if(calc_kl_diff_and_entropic_quantities):

                        log_probs = torch.distributions.MultivariateNormal(
                            this_mean.unsqueeze(1),
                            covariance_matrix=this_var.unsqueeze(1)
                        ).log_prob(these_subsamples)


                        cross_entropy=-log_probs.mean(dim=1)

                        
                        kl_diff=cross_entropy-entropy_dict[sub_pdf_dim]

                        ###
                        if(sub_pdf_dim==0):
                            mvn_samples = torch.distributions.MultivariateNormal(
                                this_mean,
                                covariance_matrix=this_var
                            ).sample(sample_shape=(samplesize,))

                            mvn_samples=mvn_samples.transpose(0,1).reshape(mvn_samples.shape[0]*mvn_samples.shape[1], -1).type_as(data_summary_repeated).to(data_summary_repeated)

                          
                            mvn_samp_logprob_exact,_,_=self.forward(mvn_samples, conditional_input=data_summary_repeated)
                
                            reverse_cross_entropy=-mvn_samp_logprob_exact.reshape(-1, samplesize).mean(dim=1)
                            reverse_kl_diff=reverse_cross_entropy-approx_entropy
                        

                      
                       
                elif("s" in sub_pdf_def):

                    ## data summation mean
                    sample_sum=torch.sum(these_subsamples, dim=1)
                    sample_sum_length=(sample_sum**2).sum(dim=1, keepdims=True).sqrt()
                    
                    ## mean vec on sphere
                    this_mean=sample_sum/(sample_sum_length)

                    angle_mean,_=self.layer_list[sub_pdf_dim][0].eucl_to_spherical_embedding(this_mean, 0.0)

                    # max PDF value
                    if(index_mask is not None):
                        this_arg_max=these_subsamples[torch.arange(initial_batch_size), index_mask]
                        this_arg_max_angles,_=self.layer_list[sub_pdf_dim][0].eucl_to_spherical_embedding(this_mean, 0.0)
                        return_dict["argmax_%d_angles"%sub_pdf_dim]=this_arg_max_angles

                    return_dict["mean_%d_angles"%sub_pdf_dim]=angle_mean
                    

                    normalized_length_R=sample_sum_length/samplesize

                    if("1" in sub_pdf_def):
                        p=2
                    elif("2" in sub_pdf_def):
                        p=3

                    last_vec=(normalized_length_R*(p-normalized_length_R**2)/(1-normalized_length_R**2))
                    
                    max_iter=20
        
                    new_diff=None

                    last_vec_cpu=last_vec.cpu()
                    cpu_length=normalized_length_R.cpu()

                    for i in range(max_iter):

                        new_vec=newton_iter(last_vec_cpu, p,cpu_length)

                        new_diff=torch.abs(new_vec-last_vec_cpu)

                        if(new_diff.max()<mises_abs_precision):
                            last_vec_cpu=new_vec
                            break

                        last_vec_cpu=new_vec
                        
                    ## set back to device before newton iters
                    this_var=last_vec_cpu

                    if(p==2):

                        c_p_k=(1.0/(2*numpy.pi*i0(this_var))).to(angle_mean)
                        a_p_k=(i1(this_var)/i0(this_var)).to(angle_mean)

                    elif(p==3):

                        a_p_k=(-1.0/this_var+1.0/numpy.tanh(this_var)).to(angle_mean)
                        
                        #c_p_k=(this_var**(0.5)/( (2*numpy.pi)**1.5 * iv(0.5, this_var))).to(angle_mean)
                        log_c_p_k=torch.zeros_like(a_p_k)



                        log_c_p_k[this_var>50]=(torch.log(this_var)-numpy.log(2*numpy.pi)-this_var)[this_var>50].to(a_p_k)
                        log_c_p_k[this_var<=50]=(torch.log(this_var)-numpy.log(2*numpy.pi)-torch.log(torch.exp(this_var)-torch.exp(-this_var)))[this_var<=50].to(a_p_k)

                    
                        #c_p_k*=1e-5
                        #a_p_k=(iv(1.5, this_var)/iv(0.5, this_var)).to(angle_mean)
                        


                    this_var=this_var.to(angle_mean)
                        
                    approx_entropy=(-log_c_p_k-this_var*a_p_k).squeeze(1)

                    #print("APPROX ENTORPY", approx_entropy)
                 
                    if(calc_kl_diff_and_entropic_quantities):

                        ## logprobs of fisher-mises
                        log_exp_facs=(these_subsamples*this_mean.unsqueeze(1)).sum(axis=2,keepdims=True)*this_var.unsqueeze(1)
                        log_probs=log_exp_facs+log_c_p_k.unsqueeze(1)

                       
                        ## cross entropy (true->approx)
                        cross_entropy=-log_probs.mean(dim=1).squeeze(1)

                        kl_diff=cross_entropy-entropy_dict[sub_pdf_dim]

                        ## reverse kl divergence

                        if(sub_pdf_dim==0):
                            mises_samples=[]
                            for cur_ind, cur_mean in enumerate(this_mean):
                                
                                #print("MEAN", cur_mean.cpu().numpy())
                                #print("KAPPA", this_var[cur_ind].cpu().numpy())
                                ms=obtain_fisher_mises_samples(cur_mean.cpu().numpy() , this_var[cur_ind].cpu().numpy() , samplesize=samplesize)
                                #print(ms)
                                lens=numpy.sqrt(numpy.sum(ms**2, axis=1))
                                #print("LENS", lens)
                                ## much slower
                                #ms=random_VMF(cur_mean.cpu().numpy() , this_var[cur_ind].cpu().numpy() , size = samplesize)
                               
                                mises_samples.append(torch.from_numpy(ms).type_as(log_probs).to(log_probs))

                            mises_samples=torch.cat(mises_samples, dim=0)

                            mises_samp_logprob_exact,_,_=self.forward(mises_samples, conditional_input=data_summary_repeated)
                            
                            
                            reverse_cross_entropy=-mises_samp_logprob_exact.reshape(-1, samplesize).mean(dim=1)

                            reverse_kl_diff=reverse_cross_entropy-approx_entropy
                        

                        
  
                else:
                    raise Exception("Unsupported sub pdf type for marginal moment calculation!", sub_pdf_def)

                return_dict["mean_%d"%sub_pdf_dim]=this_mean
                if(index_mask is not None):
                    return_dict["argmax_%d"%sub_pdf_dim]=this_arg_max

                return_dict["varlike_%d"%sub_pdf_dim]=this_var
               
                if(entropy_dict is not None):
                    return_dict["entropy_%d" % sub_pdf_dim]=entropy_dict[sub_pdf_dim]
                    return_dict["cross_entropy_%d" % sub_pdf_dim]=cross_entropy
                    return_dict["kl_diff_exact_approx_%d" % sub_pdf_dim]=kl_diff

                    if(sub_pdf_dim==0):
                        return_dict["kl_diff_approx_exact_0"]=reverse_kl_diff
                        return_dict["reverse_cross_entropy_0"]=reverse_cross_entropy

                return_dict["approx_entropy_%d" % sub_pdf_dim]=approx_entropy
                
            if(entropy_dict is not None):
                return_dict["entropy_total"]=entropy_dict["total"]

        if(verbose):
            print("total infernce took ",time.time()-tbef)

        return return_dict

    