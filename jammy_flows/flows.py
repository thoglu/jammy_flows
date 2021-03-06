import torch
from torch import nn

from .layers.euclidean.gaussianization_flow import gf_block, find_init_pars_of_chained_gf_blocks
from .layers.euclidean.polynomial_stretch_flow import psf_block

from .layers.spheres.sphere_base import sphere_base
from .layers.spheres.moebius_1d import moebius
from .layers.spheres.segmented_sphere_nd import segmented_sphere_nd
from .layers.spheres.exponential_map_s2 import exponential_map_s2
from .layers.spheres.spherical_do_nothing import spherical_do_nothing
from .layers.spheres.cnf_sphere_charts import cnf_sphere_charts

from .layers.euclidean.euclidean_base import euclidean_base
from .layers.euclidean.euclidean_do_nothing import euclidean_do_nothing

from .layers.intervals.interval_base import interval_base
from .layers.intervals.interval_do_nothing import interval_do_nothing
from .layers.intervals.rational_quadratic_spline import rational_quadratic_spline

from .layers import extra_functions
from .layers.extra_functions import list_from_str, NONLINEARITIES

import collections
import numpy
import copy
import sys

class pdf(nn.Module):

    def __init__(
        self,
        pdf_defs, ## e1
        flow_defs, ## ggggg
        flow_defs_detail=dict(),
        conditional_input_dim=None,
        data_summary_dim=None,
        input_encoder=None,
        hidden_mlp_dims_sub_pdfs="64-64",
        hidden_mlp_dims_meta="64-64",
        conditional_manifold_input_embedding_overwrite=dict(),
        predict_log_normalization=False,
        join_poisson_and_pdf_description=False,
        hidden_mlp_dims_poisson="64-64",
        rank_of_mlp_mappings_poisson=0,
        ## the following arguments are defining specifically custom MLP settings
        use_custom_low_rank_mlps=False,
        rank_of_mlp_mappings_sub_pdfs=0,
        custom_mlp_highway_mode=0
    ):  
        """
        Initializes a general flow pdf object.

        Parameters:
            pdf_defs (str): String of characters describing the joint PDF structure: Sub-space structure is spearated by "+".
            Examples: "e2+s1+s2", describes a joint PDF over a 2-dimensional euclidean space, a 1-sphere and a 2-sphere: a joint 5-dimensional PDF 
            Allowed values for sub-spaces:  "eX", euclidean space with "X" dimensions. Examples: "e5" (5-dimensional eucliden space), "e1" (1-dimensional euclidean space)
                                            "sX", spherical space with "X" dimensions: Examples: "s1" (1-sphere), "s2" (2-sphere)

            flow_defs (str): A string, that describes how each conditional subfflow defined in "pdfs_defs" is structured in terms of normalizing-flow structure.
            Example: "ggg+mmmmm+tttttt" to describe structure of the other example "e2+s1+s2". Each g is a GF (CITE GF) layer, so 3 GF layers describe the flow in e2. Each m is a convex moebius transformation layer (see CITE), so 5 m describe
                     5 convex moebius transformations to describe the flow PDF on s1. Each t is a "torus" layer on S2, so 5 t corresponds to 5 consecutive torus layers to describe the flow on the 2-sphere.

                The mapping of layer abbreviation to a specific layer is implemented for some standard layers, but can be overridden.

            flow_defs_detail (dict): Dictionary containing other flow definitions not defined as default or for overriding default flow definitions.

            conditional_dim: (None/single integer/list of integers/Nones) Conditional dimension(s)
            Example: (1) f(z) , i.e. no conditional input, would be modeled by passing None. (2) f(z;x), where x is an n-dimensional input space can be modeled by n. Another option is to split the input up and model higher-order conditionals by passing [n1,n2], or [n1,n2,n3], where sum(n_i)=n
            This models effectively a parametrization where encoding happens in layers .. first n1 input parameters are encoded to define the flow p(z;n1). Then n2-dimensional data is encoded to map to the parameters of the first flow p(z;n1), efffectively structuring the encoding of p(z;n1,n2) hierarchically.
            
            Concrete example: [5,2] (7-dimensional input, encding first 5-dim sub-space which maps directly to the flow. The mapping from 5-d to flow-params is again the output of another neural networks that encodes the 2-dimensional other space. 5 is hierarchy level 0, 2 is hierarchy level 1 .. and it could go on.
            input_encoder: (None/encoder object (str/obj) /list of encoder objects (str/obj) /Nones) The structure has to match the structure of conditional dim. If conditional dim has a None in one place, input_encoder also has to have a respective None.
            
            hidden_mlp_dims: (str or list of strings). Hidden dimensions of an mlp object that translates from encoding output (data summary dimension) to either flow params or params of the whole encoding process down the hierarchy.
            Example. ["128+128", "64+64"] could be a setting that uses 2-layer MLPs with 128 and 64 hidden dimensionals, respeectively, and works with the previuos concrete example of [5,2] for the conditional_dim encoder setting.

        Returns:
            No return value

        """
        super().__init__()

        self.use_custom_low_rank_mlps=use_custom_low_rank_mlps
        self.predict_log_normalization=predict_log_normalization
        self.join_poisson_and_pdf_description=join_poisson_and_pdf_description
        self.custom_mlp_highway_mode=custom_mlp_highway_mode

        ###################################################################
        ## CURRENT FIX .. if conditional_input_dim is not None, overwrite data_summary dim with conditional_input_dim and put input_encoder to "passthrough"
        ## Once Meta encoders are implemented, this should be changed again

        used_data_summary_dim=None
        used_input_encoder=None
        if(conditional_input_dim is not None):

            used_data_summary_dim=conditional_input_dim
            used_input_encoder="passthrough"

        ##################


        self.read_model_definition(pdf_defs, flow_defs, flow_defs_detail, conditional_input_dim, used_data_summary_dim, used_input_encoder, hidden_mlp_dims_sub_pdfs, rank_of_mlp_mappings_sub_pdfs, hidden_mlp_dims_meta, conditional_manifold_input_embedding_overwrite=conditional_manifold_input_embedding_overwrite)

        self.init_flow_structure()

        self.hidden_mlp_dims_poisson=hidden_mlp_dims_poisson
        self.rank_of_mlp_mappings_poisson=rank_of_mlp_mappings_poisson

        self.init_encoding_structure()

        ## initialize params
        self.init_params()

        ## initialize with double precision
        self.double()

    def read_model_definition(self, pdf_defs, flow_defs, flow_defs_detail,conditional_input_dim,
        data_summary_dim,
        input_encoder,
        hidden_mlp_dims_sub_pdfs,
        rank_of_mlp_mappings_sub_pdfs,
        hidden_mlp_dims_meta,
        conditional_manifold_input_embedding_overwrite=dict()):
        """
        Initializes the internal flow layer definitions with respective settings and checks settings.

        Parameters:
            pdf_defs (str): String of characters describing the joint PDF structure: Sub-space structure is spearated by "+".
            Examples: "e2+s1+s2", describes a joint PDF over a 2-dimensional euclidean space, a 1-sphere and a 2-sphere: a joint 5-dimensional PDF 
            Allowed values for sub-spaces:  "eX", euclidean space with "X" dimensions. Examples: "e5" (5-dimensional eucliden space), "e1" (1-dimensional euclidean space)
                                            "sX", spherical space with "X" dimensions: Examples: "s1" (1-sphere), "s2" (2-sphere)

            flow_defs (str): A string, that describes how each conditional subfflow defined in "pdfs_defs" is structured in terms of normalizing-flow structure.
            Example: "ggg+mmmmm+tttttt" to describe structure of the other example "e2+s1+s2". Each g is a GF () layer, so 3 GF layers describe the flow in e2. Each m is a convex moebius transformation layer, so 5 m describe
                     5 convex moebius transformations to describe the flow PDF on s1. Each t is a "torus" layer on S2, so 5 t corresponds to 5 consecutive torus layers to describe the flow on the 2-sphere.
                    

            flow_defs_detail (dict): Dictionary containing other flow definitions not defined as default or for overriding default flow definitions.
                                     The mapping of layer abbreviation to a specific layer is implemented for some standard layers, but can be overridden using this dict.

            conditional_dim: (None/single integer/list of integers/Nones) Conditional dimension(s)
            Example: (1) f(z) , i.e. no conditional input, would be modeled by passing None. (2) f(z;x), where x is an n-dimensional input space can be modeled by n. Another option is to split the input up and model higher-order conditionals by passing [n1,n2], or [n1,n2,n3], where sum(n_i)=n
            This models effectively a parametrization where encoding happens in layers .. first n1 input parameters are encoded to define the flow p(z;n1). Then n2-dimensional data is encoded to map to the parameters of the first flow p(z;n1), efffectively structuring the encoding of p(z;n1,n2) hierarchically.
            
            Concrete example: [5,2] (7-dimensional input, encding first 5-dim sub-space which maps directly to the flow. The mapping from 5-d to flow-params is again the output of another neural networks that encodes the 2-dimensional other space. 5 is hierarchy level 0, 2 is hierarchy level 1 .. and it could go on.
            input_encoder: (None/encoder object (str/obj) /list of encoder objects (str/obj) /Nones) The structure has to match the structure of conditional dim. If conditional dim has a None in one place, input_encoder also has to have a respective None.
            
            hidden_mlp_dims: (str or list of strings). Hidden dimensions of an mlp object that translates from encoding output (data summary dimension) to either flow params or params of the whole encoding process down the hierarchy.
            Example. ["128+128", "64+64"] could be a setting that uses 2-layer MLPs with 128 and 64 hidden dimensionals, respectively, and works with the previuos concrete example of [5,2] for the conditional_dim encoder setting.

        Returns:
            No return value

        """

        ## provide standard settings
        self.flow_dict = dict()

        """ 
        Euclidean flows
        """
        self.flow_dict["g"] = dict()
        self.flow_dict["g"]["module"] = gf_block
        self.flow_dict["g"]["type"] = "e"
        self.flow_dict["g"]["kwargs"] = dict()
        self.flow_dict["g"]["kwargs"]["fit_normalization"] = 1
        self.flow_dict["g"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["g"]["kwargs"]["num_householder_iter"] = -1
        self.flow_dict["g"]["kwargs"]["num_kde"] = 10
        self.flow_dict["g"]["kwargs"]["inverse_function_type"] = "isigmoid"
        self.flow_dict["g"]["kwargs"]["replace_first_sigmoid_with_icdf"]=1
        self.flow_dict["g"]["kwargs"]["skip_model_offset"]=0

        self.flow_dict["p"] = dict()
        self.flow_dict["p"]["module"] = psf_block
        self.flow_dict["p"]["type"] = "e"
        self.flow_dict["p"]["kwargs"] = dict()
        self.flow_dict["p"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["p"]["kwargs"]["num_householder_iter"] = -1
        self.flow_dict["p"]["kwargs"]["num_transforms"] = 1
        self.flow_dict["p"]["kwargs"]["exact_mode"] = True
        self.flow_dict["p"]["kwargs"]["skip_model_offset"]=0

        """
        S1 flows
        """
        self.flow_dict["m"] = dict()
        self.flow_dict["m"]["module"] = moebius
        self.flow_dict["m"]["type"] = "s"
        self.flow_dict["m"]["kwargs"] = dict()
        self.flow_dict["m"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["m"]["kwargs"]["use_extra_householder"] = 0
        self.flow_dict["m"]["kwargs"]["euclidean_to_sphere_as_first"] = 0
        self.flow_dict["m"]["kwargs"]["num_moebius"] = 5
        self.flow_dict["m"]["kwargs"]["natural_direction"] = 0

        """
        S2 flows
        """
        self.flow_dict["n"] = dict()
        self.flow_dict["n"]["module"] = segmented_sphere_nd
        self.flow_dict["n"]["type"] = "s"
        self.flow_dict["n"]["kwargs"] = dict()
        self.flow_dict["n"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["n"]["kwargs"]["use_extra_householder"] = 1
        self.flow_dict["n"]["kwargs"]["euclidean_to_sphere_as_first"] = 0
        self.flow_dict["n"]["kwargs"]["num_moebius"] = 10
        self.flow_dict["n"]["kwargs"]["higher_order_cylinder_parametrization"] = True
        self.flow_dict["n"]["kwargs"]["zenith_type_layers"] = "r"
        self.flow_dict["n"]["kwargs"]["max_rank"] = -1

        self.flow_dict["v"] = dict()
        self.flow_dict["v"]["module"] = exponential_map_s2
        self.flow_dict["v"]["type"] = "s"
        self.flow_dict["v"]["kwargs"] = dict()
        self.flow_dict["v"]["kwargs"]["use_extra_householder"] = 0
        self.flow_dict["v"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["v"]["kwargs"]["euclidean_to_sphere_as_first"] = 0
        self.flow_dict["v"]["kwargs"]["higher_order_cylinder_parametrization"] = False
        self.flow_dict["v"]["kwargs"]["exp_map_type"] = "exponential" ## supported linear  / exponential
        self.flow_dict["v"]["kwargs"]["num_components"] = 10 ## number of components in convex superposition
        self.flow_dict["v"]["kwargs"]["natural_direction"] = 0 ## natural direction corresponds to the transformation happing in the forward direction - default: 0

        self.flow_dict["c"] = dict()
        self.flow_dict["c"]["module"] = cnf_sphere_charts
        self.flow_dict["c"]["type"] = "s"
        self.flow_dict["c"]["kwargs"] = dict()
        self.flow_dict["c"]["kwargs"]["use_extra_householder"] = 0
        self.flow_dict["c"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["c"]["kwargs"]["euclidean_to_sphere_as_first"] = 0
        self.flow_dict["c"]["kwargs"]["higher_order_cylinder_parametrization"] = False
        self.flow_dict["c"]["kwargs"]["num_charts"] = 10
        self.flow_dict["c"]["kwargs"]["cnf_network_hidden_dims"] = "128" # hidden dims of cnf MLP network
        self.flow_dict["c"]["kwargs"]["cnf_network_highway_mode"] = 1 # mlp highway dim - 0-4
        self.flow_dict["c"]["kwargs"]["cnf_network_rank"] = 0 # 0 means full rank
        self.flow_dict["c"]["kwargs"]["natural_direction"] = 1 ## natural direction corresponds to the transformation happing in the forward direction - default: 0
        self.flow_dict["c"]["kwargs"]["solver"] = "rk4" ## 

        """
        Interval flows
        """
        self.flow_dict["r"] = dict()
        self.flow_dict["r"]["module"] = rational_quadratic_spline
        self.flow_dict["r"]["type"] = "i"
        self.flow_dict["r"]["kwargs"] = dict()
        self.flow_dict["r"]["kwargs"]["use_permanent_parameters"]=0
        self.flow_dict["r"]["kwargs"]["euclidean_to_interval_as_first"] = 0
        self.flow_dict["r"]["kwargs"]["num_basis_elements"] = 10


        """
        Spherical/Euclidean/Interval flows that do nothing
        """
        self.flow_dict["x"] = dict()
        self.flow_dict["x"]["module"] = euclidean_do_nothing
        self.flow_dict["x"]["type"] = "e"
        self.flow_dict["x"]["kwargs"] = dict()

        self.flow_dict["y"] = dict()
        self.flow_dict["y"]["module"] = spherical_do_nothing
        self.flow_dict["y"]["type"] = "s"
        self.flow_dict["y"]["kwargs"] = dict()

        self.flow_dict["z"] = dict()
        self.flow_dict["z"]["module"] = interval_do_nothing
        self.flow_dict["z"]["type"] = "i"
        self.flow_dict["z"]["kwargs"] = dict()
  

        for k in flow_defs_detail:
            if k not in self.flow_dict.keys():

                self.flow_dict[k] = dict()
                if "module" not in flow_defs_detail[k].keys():
                    raise Exception(
                        "Flow defs of ",
                        k,
                        " do not contain the module object .. this is a requirement!",
                    )
                    
                self.flow_dict[k]["module"] = flow_defs_detail[k]["module"]
                self.flow_dict[k]["kwargs"] = dict()

            # copy kwargs if required
            if "kwargs" in flow_defs_detail[k].keys():
                
                for kwarg in flow_defs_detail[k]["kwargs"].keys():

                    self.flow_dict[k]["kwargs"][kwarg] = flow_defs_detail[k]["kwargs"][kwarg]

                    print(" ovewrite basic option ", kwarg, " with ", flow_defs_detail[k]["kwargs"][kwarg])


    
        self.pdf_defs_list = pdf_defs.split("+")

        self.conditional_manifold_input_embedding=dict()
        self.conditional_manifold_input_embedding["e"]=0
        self.conditional_manifold_input_embedding["s"]=1
        self.conditional_manifold_input_embedding["i"]=1

        for k in conditional_manifold_input_embedding_overwrite.keys():
            self.conditional_manifold_input_embedding[k]=conditional_manifold_input_embedding_overwrite[k]

        # target dimensions of each sub-flow
        self.target_dims=[]

        # sub-flow indices for the input dimension to describe target-dim for this sub-flow
        self.target_dim_indices=[]

        ## conditonal dimension indices used as additional input for the conditional dim of this sub-flow
        #self.extra_conditional_dim_indices=[]
        #self.extra_conditional_dim_count=[]
        total_dim=0

        ## counter that tracks previous conditional inputs... if we have spherical dimensions, i.e. theta and phi for 2-d, the conditional input is the corresponding x,y,z, so 2+1 dimensional
        #prev_conditional_dim_counter=0
        self.basic_layers=[]

        for p in self.pdf_defs_list:

            ## need to separate interval from rest
            main_def=p.split("_")[0]
            self.target_dims.append(int(main_def[1:]))
            self.target_dim_indices.append((total_dim, total_dim+self.target_dims[-1]))

           
            total_dim+=self.target_dims[-1]
                
            ### place holder base layers to define embedding mappings
            if(p[0]=="e"):
                self.basic_layers.append(euclidean_base(dimension=int(main_def[1])))
            elif(p[0]=="s"):
                self.basic_layers.append(sphere_base(dimension=int(main_def[1])))
            elif(p[0]=="i"):
                self.basic_layers.append(interval_base(dimension=1))
            else:
                raise Exception("undefined base layer type :", p[0])

           
        ## the target space of the PDF
        self.total_target_dim=total_dim

        self.flow_defs_list = flow_defs.split("+")

        ## make sure all specified layers are actually defined in the internal "flow_dict" dictionary
        for flow_def in self.flow_defs_list:
            for single_layer in flow_def:
                if(single_layer not in self.flow_dict.keys()):
                    raise Exception("single layer abbreviation ", single_layer, " not defined in internal flow dict .. please give description via the flow_defs_detail kwarg")
                   

        if(len(self.pdf_defs_list)!=len(self.flow_defs_list)):
            raise Exception("PDF defs list has to be same length as flow defs list, but ... ", self.pdf_defs_list, self.flow_defs_list)
        ### now define input stuff

        # store conditional dims in a list
        self.conditional_input_dim = conditional_input_dim
        if(self.conditional_input_dim is not None):

            if(type(self.conditional_input_dim)==int):
                self.conditional_input_dim=[self.conditional_input_dim]
            elif(type(self.conditional_input_dim)!=list):
                raise Exception("Conditional dim has to be list of Nones/ints or None/int. Type is ", type(self.conditional_input_dim))
              

        ## save input encoders in list
        self.input_encoder=input_encoder
        if(self.input_encoder is not None):
            if(type(input_encoder)!=list):
                self.input_encoder=[input_encoder]

        ## make summary dims into list
        self.data_summary_dim=data_summary_dim
        if(type(self.data_summary_dim) == int):
            self.data_summary_dim=[self.data_summary_dim]

        ## check that all encoding related lists have similar length

        if(self.conditional_input_dim is None):
            if(self.input_encoder is not None):
                raise Exception("conditional input dim is None but encoder is given .. does not work")
               

            if(self.data_summary_dim is not None):
                raise Exception("data summary dim has to be None if conditional input dim is None")
               


        if(self.input_encoder is None):
            if(self.conditional_input_dim is not None):
                raise Exception("Encoder is None but conditional input dim is defined.. does not work")
              

            if(self.data_summary_dim is not None):
                raise Exception("data summary dim has to be None if encoder is None")
               

        if(self.conditional_input_dim is not None and self.input_encoder is not None):
            if(self.data_summary_dim is None):
                raise Exception("Encoder and input encoding dimension are given, but data summary dim is None!")
                

            if(len(self.conditional_input_dim)!=len(self.input_encoder)):
                raise Exception("Conditional dims and input encoder definitions do not match in length.", self.conditional_input_dim, " vs ", self.input_encoder)
            

            if(len(self.conditional_input_dim)!=len(self.data_summary_dim)):
                raise Exception("Conditional dims and data summary dims definitions do not match in length.", self.conditional_input_dim, " vs ", self.data_summary_dim)
             

        ## define internal mlp mapping dims as list
        self.hidden_mlp_dims_sub_pdfs=hidden_mlp_dims_sub_pdfs
        if(type(self.hidden_mlp_dims_sub_pdfs)==str):
            self.hidden_mlp_dims_sub_pdfs=[self.hidden_mlp_dims_sub_pdfs]*len(self.pdf_defs_list)
        elif(type(self.hidden_mlp_dims_sub_pdfs)!=list):
            raise Exception("Hidden MLP dimensions must be defined either str or list, received ", type(self.hidden_mlp_dims_sub_pdfs))
        

        self.rank_of_mlp_mappings_sub_pdfs=rank_of_mlp_mappings_sub_pdfs
        if(type(self.rank_of_mlp_mappings_sub_pdfs)==int):
            self.rank_of_mlp_mappings_sub_pdfs=[self.rank_of_mlp_mappings_sub_pdfs]*len(self.pdf_defs_list)
        elif(type(self.rank_of_mlp_mappings_sub_pdfs)==str):
            self.rank_of_mlp_mappings_sub_pdfs=[self.rank_of_mlp_mappings_sub_pdfs]*len(self.pdf_defs_list)
        elif(type(self.rank_of_mlp_mappings_sub_pdfs)!=list):
            raise Exception("Rank of MLP sub pdfs has to defined as an int or list type!")
     
        ## sub pdf hidden mlp dims must be equal to number of sub pdfs (or sub-pdfs-1 if no encoder present)
        required_hidden_mlp_dims_len=len(self.pdf_defs_list)
        if(required_hidden_mlp_dims_len>0):
            if(len(self.hidden_mlp_dims_sub_pdfs)!=required_hidden_mlp_dims_len):
                raise Exception("hidden mlp dimension definitions for sub pdfs is wrong length (%d) .. requires length (%d)" %(len(self.hidden_mlp_dims_sub_pdfs), required_hidden_mlp_dims_len))
              
        ## define internal meta mlp mapping dims as list
        self.hidden_mlp_dims_meta=hidden_mlp_dims_meta
        if(type(self.hidden_mlp_dims_meta)==str):
            self.hidden_mlp_dims_meta=[self.hidden_mlp_dims_meta]
        elif(type(self.hidden_mlp_dims_meta)!=list):
            raise Exception("hidden_mlp_dims_meta has to defined as a string or list of strings!")
           
        if(self.input_encoder is not None):
            if(len(self.hidden_mlp_dims_meta)!=len(self.input_encoder)):
                raise Exception("Every input encoder requires mlp meta dimension definition .. num encoders: %d - num meta mlp dimension definitions: %d" % (len(self.input_encoder), len(self.hidden_mlp_dims_meta)))
               
        
        ## check if we have more than 1 input encoder, meaning we have meta encoders .. otherwise mlp dims meta is ignored anyway
        if(self.input_encoder is not None):
            if(len(self.input_encoder)>1):
                # num meta encoders is len(encoders)-1 .. check that it matches meta hidden mlp dimensions
                if(len(self.hidden_mlp_dims_meta)!=len(self.input_encoder)-1):
                    print("Hidden MLP meta dimension list has different len (%d) than list of meta encoders (%d)" % (len(self.hidden_mlp_dims_meta), len(self.input_encoder)-1))
                if(len(self.hidden_mlp_dims_sub_pdfs)!=len(self.pdf_defs_list)):
                    print("Hidden MLP sub pdf list has different len (%d) than list of sub pdfs len (%d)" % (len(self.hidden_mlp_dims_sub_pdfs), len(self.pdfs_defs_list)))


        self.layer_list = nn.ModuleList()

        ## might force to use permanent parameters for flows which have no input and are not conditional
        self.force_permanent_parameters = 0

        if self.input_encoder is None:
            self.force_permanent_parameters = 1
        
        else:
            encoders=[]

            ## change str to encoder objects if necessary
            for ind, ie in enumerate(self.input_encoder):

                if ie == "passthrough":
                    if(self.data_summary_dim[ind]!=self.conditional_input_dim[ind]):
                        raise Exception("passthrough (i.e. no real encoding) does only work if summary dim and conditional dim are identical! However: cond input dim: %d / summary dim: %d" % (self.conditional_input_dim[ind], self.data_summary_dim[ind]))
                        
                    encoders.append(lambda x: x)
              
                else:
                    raise Exception("Encodings not implemented at the moment. Define Data encoding separately!")
                    
            self.input_encoder=encoders

    
    def init_flow_structure(self):

        self.num_parameter_list=[]

        total_predicted_parameters = 0

        ## loop through flows and define layers according to flow definitions
        ## also count number of parameters of each layer
        for subflow_index, subflow_description in enumerate(self.pdf_defs_list):

            ## append a collection for this subflow which will hold the number of parameters of each layer in the sub-flow
            self.num_parameter_list.append([])

            self.layer_list.append(nn.ModuleList())

            this_num_layers=len(self.flow_defs_list[subflow_index])
            ## loop thorugh the layers of this particular sub-flow
            for layer_ind, layer_type in enumerate(self.flow_defs_list[subflow_index]):
                if(self.flow_dict[layer_type]["type"]!=subflow_description[0]):
                    raise Exception("layer type ", layer_type, " is not compatible with flow type ", subflow_description)
                  
                this_kwargs = copy.deepcopy(self.flow_dict[layer_type]["kwargs"])

                ## overwrite permanent parameters if desired or necessary
                if self.force_permanent_parameters:
                    this_kwargs["use_permanent_parameters"] = 1
                ## do not use permanent parameters for all later sub flows!
                if(subflow_index>0):
                    this_kwargs["use_permanent_parameters"] = 0

                

                if("s" in subflow_description):
                    ## this flow is a spherical flow, so the first layer should also project from plane to sphere or vice versa
                    if(layer_ind==0):
                        this_kwargs["euclidean_to_sphere_as_first"]=1
                elif("i" in subflow_description):
                    ## this flow is an interval flow, so the first layer should also project from real line to interval or vice versa

                    interval_boundaries=subflow_description.split("_")[1:]

                    ## overwrite the boundary parameters
                    this_kwargs["low_boundary"]=float(interval_boundaries[0])
                    this_kwargs["high_boundary"]=float(interval_boundaries[1])
                 
                    if(layer_ind==0):
                        
                        this_kwargs["euclidean_to_interval_as_first"]=1
                    

                elif("e" in subflow_description):
                    if(layer_type!="x"):
                        if(layer_ind==(this_num_layers-1) and this_kwargs["skip_model_offset"]==0):
                            
                            this_kwargs["model_offset"]=1
                        elif(layer_ind==0):
                            if(layer_type=="g"):
                                if(this_kwargs["replace_first_sigmoid_with_icdf"]>0 and this_kwargs["inverse_function_type"]=="isigmoid"):
                                    this_kwargs["inverse_function_type"]="inormal_partly_precise"

                ## this is not a real parameter - delete it
                if("skip_model_offset" in this_kwargs):
                    del this_kwargs["skip_model_offset"]
                ## we dont want to pass this to layer
                if(layer_type=="g"):
                    del this_kwargs["replace_first_sigmoid_with_icdf"]
                    
                self.layer_list[subflow_index].append(
                    self.flow_dict[layer_type]["module"](self.target_dims[subflow_index], **this_kwargs)
                )

                self.num_parameter_list[subflow_index].append(self.layer_list[subflow_index][-1].get_total_param_num())
                
                total_predicted_parameters += self.num_parameter_list[subflow_index][-1]


        ## add log-normalization prediction
        self.log_normalization=None

        if(self.predict_log_normalization):
            total_predicted_parameters+=1

            if self.force_permanent_parameters:
                self.log_normalization=nn.Parameter(torch.randn(1).type(torch.double).unsqueeze(0))
            else:
                self.log_normalization=torch.zeros(1).type(torch.double).unsqueeze(0)

        self.total_predicted_parameters=total_predicted_parameters

    


    def init_encoding_structure(self):

        self.encoder_mlp_predictors=None
        if self.input_encoder is not None:

            ## first look at meta-encoders

            if(len(self.input_encoder)>1):

                self.encoder_mlp_predictors=nn.ModuleList()

                for encoder_index, encoder in enumerate(self.input_encoder[:-1]):

                    #num_predicted_pars=self.total_predicted_parameters

                    ## only the last encoder encodes flow parameters
                    #if(encoder_index<len(self.input_encoder)-1):
                        ## we encoder another encoder... get parameters of that one instead of flow
                    num_predicted_pars=self.num_encoder_pars(self.input_encoder[encoder_index+1])

                    mlp_in_dims = (self.data_summary_dim[encoder_index],) + list_from_str(self.hidden_mlp_dims_meta[encoder_index])
                    mlp_out_dims = list_from_str(self.hidden_mlp_dims_meta[encoder_index]) + (
                        num_predicted_pars,
                    )

                  
                    

                    nn_list = []
                    for i in range(len(mlp_in_dims)):
                        
                        l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                        nn_list.append(l)

                        if i < (len(mlp_in_dims) - 1):
                            nn_list.append(NONLINEARITIES["tanh"])

                    self.encoder_mlp_predictors.append(torch.nn.Sequential(*nn_list))




        self.mlp_predictors=nn.ModuleList()

        #num_encoder_for_other_encoders=len(self.input_encoder)-1

        ## only one MLP predictor for all ??


        prev_extra_input_num=0

        if(self.join_poisson_and_pdf_description):
            if(len(self.pdf_defs_list)>1):
                raise Exception("A common poisson log-lambda and flow parameter prediction is currently only supported for a PDF that has a single flow (no autoregressive structure) for simplicity! .. number of autoregressive parts here: ", len(self.pdf_defs_list))
            if(self.data_summary_dim is None):
                raise Exception("Flow does not depend on conditional input .. please set 'join_poisson_and_pdf_description' to False, currently True")
       
        for pdf_index, pdf in enumerate(self.pdf_defs_list):


            if(pdf_index==0 and self.data_summary_dim is None):
              
                self.mlp_predictors.append(None)
                if(self.conditional_manifold_input_embedding[pdf[0]]):
                    prev_extra_input_num+=self.target_dims[pdf_index]+1
                else:
                    prev_extra_input_num+=self.target_dims[pdf_index]

                continue

            
            if(pdf_index>0):
                tot_num_pars_this_pdf=0
                for l in self.layer_list[pdf_index]:
                    tot_num_pars_this_pdf+=l.get_total_param_num()
                
                ## if this sub pdf has no parameters, we do not need to define an MLP
                if(tot_num_pars_this_pdf==0):
                    
                    self.mlp_predictors.append(None)
                    if(self.conditional_manifold_input_embedding[pdf[0]]):
                        prev_extra_input_num+=self.target_dims[pdf_index]+1
                    else:
                        prev_extra_input_num+=self.target_dims[pdf_index]

                    continue

          
            ## only the last encoder encodes flow parameters
            #if(encoder_index<len(self.input_encoder)-1):
                ## we encoder another encoder... get parameters of that one instead of flow
            num_predicted_pars=sum(self.num_parameter_list[pdf_index])
 
            if(self.predict_log_normalization):
                if(pdf_index==0 and self.join_poisson_and_pdf_description):
                    num_predicted_pars+=1

            this_summary_dim=prev_extra_input_num
            if(self.data_summary_dim is not None):
                this_summary_dim+=self.data_summary_dim[-1]

            if(self.use_custom_low_rank_mlps):

                these_hidden_dims=list_from_str(self.hidden_mlp_dims_sub_pdfs[pdf_index])

            
                custom_mlp=extra_functions.AmortizableMLP(this_summary_dim, these_hidden_dims, num_predicted_pars, low_rank_approximations=self.rank_of_mlp_mappings_sub_pdfs[pdf_index], use_permanent_parameters=True, highway_mode=self.custom_mlp_highway_mode, svd_mode="smart")
                
                """
                if(quasi_gaussian_initialization):

               
                    bias_index=0
                    tot_bias=[]

                    for sublayer in self.layer_list[pdf_index]:
                       
                        these_params=sublayer.get_desired_init_parameters()
                       
                        if(these_params is not None):
                           tot_bias.append(these_params)
                    
                    if(len(tot_bias)==0):
                        custom_mlp.initialize_uvbs()
                    else:
                        tot_bias=torch.cat(tot_bias)
                     
                        custom_mlp.initialize_uvbs(init_b=tot_bias)
                """
                self.mlp_predictors.append(custom_mlp)

            else:

                mlp_in_dims = [this_summary_dim] + list_from_str(self.hidden_mlp_dims_sub_pdfs[pdf_index])
                mlp_out_dims = list_from_str(self.hidden_mlp_dims_sub_pdfs[pdf_index]) + [num_predicted_pars]

                nn_list = []
                for i in range(len(mlp_in_dims)):
                   
                    l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                    nn_list.append(l)
                    
                    if i < (len(mlp_in_dims) - 1):
                        nn_list.append(NONLINEARITIES["tanh"])
                    
                
                self.mlp_predictors.append(torch.nn.Sequential(*nn_list))


            ## use embedding dimension of manifold for conditional inputs?
            if(self.conditional_manifold_input_embedding[pdf[0]]):
                prev_extra_input_num+=self.target_dims[pdf_index]+1
            else:
                prev_extra_input_num+=self.target_dims[pdf_index]


        ##

        self.log_normalization_mlp=None

        if(self.predict_log_normalization):

            if(self.data_summary_dim is not None):

                ## we only have the encoding summary dim.. poisson mean does not depend on the other PDF pars

                ## only generate a Poisson MLP if poisson log-lambda and other flow parameters are to be predicted by separate MLPs
                if(self.join_poisson_and_pdf_description==False):
                    this_summary_dim=self.data_summary_dim[-1]
                    num_predicted_pars=1

                    if(self.use_custom_low_rank_mlps):
                       
                        self.log_normalization_mlp=extra_functions.AmortizableMLP(this_summary_dim, self.hidden_mlp_dims_poisson, num_predicted_pars, low_rank_approximations=self.rank_of_mlp_mappings_poisson, use_permanent_parameters=True, highway_mode=self.custom_mlp_highway_mode, svd_mode="smart")

                    else:

                        mlp_in_dims = [this_summary_dim] + list_from_str(self.hidden_mlp_dims_sub_pdfs[pdf_index])
                        mlp_out_dims = list_from_str(self.hidden_mlp_dims_sub_pdfs[pdf_index]) + [num_predicted_pars]

                        nn_list = []
                        for i in range(len(mlp_in_dims)):
                           
                            l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                            nn_list.append(l)

                            if i < (len(mlp_in_dims) - 1):
                                nn_list.append(NONLINEARITIES["tanh"])
                            else:
                                ## initialize some weights

                                nn_list[-1].weight.data/=1000.0

                                nn_list[-1].bias.data[0]=-1.0
                                
                        self.log_normalization_mlp=torch.nn.Sequential(*nn_list)

        

    def count_parameters(self, verbose=False):

        n_enc = 0
        n_enc_nograd = 0

        if(verbose):
            print("--------------------------------------------")
            print("Counting Conditional PDF Pars ... ")
            print("--------------------------------------------")

            ## Encoding part
            print("<<-- (1) encoding -->>")
        #print("--------------------------------------------")
        if self.input_encoder is not None:
            for ind, enc in enumerate(self.input_encoder):
         
                if(type(enc)==nn.Module):
                    n_enc_this=0
                    n_enc_nograd_this=0

                    for p in self.encoder.parameters():
                        if p.requires_grad:
                            n_enc += numpy.prod(p.size())
                            n_enc_this += numpy.prod(p.size())
                        else:
                            n_enc_nograd += numpy.prod(p.size())
                            n_enc_nograd_this += numpy.prod(p.size())
                    if(verbose):
                        print("  -> %d/%d pars" % (n_enc, n_enc_nograd))
                else:
                    if(verbose):
                        print("  -> 0 pars")
        else:
            if(verbose):
                print("-> no encoder")
        
        if(verbose):
            print("total encoder pars: %d / (nograd) %d" % (n_enc, n_enc_nograd))
            print("--------------------------------------------")

        
        mlp_pars = 0
        mlp_pars_nograd = 0

        ## MLP summary to flow part
        if(verbose):
            print("<<-- (2) MLP summary to flow / encoder mappings -->>")
        
        if self.encoder_mlp_predictors is not None:
            for ind, mlp in enumerate(self.encoder_mlp_predictors):
                this_mlp=0
                this_mlp_nograd=0

                if(verbose):
                    print("--------------")
                    print("Encoder LP predictor: ", ind)
                for p in mlp.parameters():
                    if p.requires_grad:
                        mlp_pars += numpy.prod(p.size())
                        this_mlp += numpy.prod(p.size())
                    else:
                        
                        this_mlp_nograd += numpy.prod(p.size())
                        mlp_pars_nograd += numpy.prod(p.size())
                if(verbose):
                    print(" %d / (nograd) %d " % (this_mlp, this_mlp_nograd))
                
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

    def _conditional_input_to_summary(self, conditional_input=None):

        if conditional_input is not None:

            if self.input_encoder is None:
                raise Exception("encoder is none but conditionl input is given ...")
               
            if self.mlp_predictors is None:
                raise Exception("mlp predictor is none but conditional input is given ...")
              
            if(type(conditional_input)!=list):
                conditional_input=[conditional_input]

            if(len(conditional_input)!=len(self.input_encoder)):
                raise Exception("conditional input given has len ", len(conditional_input), " but requires len ", len(self.input_encoder))
             
          
            num_meta_encoder=len(self.input_encoder)-1

            if(num_meta_encoder==0):

                data_summary = self.input_encoder[-1](conditional_input[-1])
            else:

                raise NotImplementedError()

                extra_params=None
                ## loop over meta encoders first
                for ind, enc in enumerate(self.input_encoder[:-1]):
              
                    if(ind==0):
                        data_summary = self.input_encoder[ind](conditional_input[ind])
                    else:
                        data_summary = self.input_encoder[ind](conditional_input[ind], encoder_params=extra_params)

                    extra_params=self.encoder_mlp_predictors[ind](data_summary)

                data_summary = self.input_encoder[-1](conditional_input[-1], encoder_params=extra_params)

            return data_summary

        else:
            return None

    def log_mean_poisson(self, conditional_input=None):

        if(self.log_normalization is None):
            raise Exception("This PDF does not predict the log-mean of a Poisson distriution. Initialize with 'predict_log_normalization'=True for this possibility.")

        data_summary = self._conditional_input_to_summary(conditional_input=conditional_input)


        if(data_summary is None):
            return self.log_normalization
        else:
            if(self.join_poisson_and_pdf_description):
                return self.mlp_predictors[0](data_summary)[:,-1:]
            else:
                return self.log_normalization_mlp(data_summary)
        
         
    def forward(self, x, conditional_input=None):

        log_det = torch.zeros(x.shape[0]).type_as(x)

        extra_params = None

       
        data_summary=self._conditional_input_to_summary(conditional_input=conditional_input)
      
        z = x
        extra_conditional_input=[]
        new_targets=[]

        for pdf_index, pdf_layers in enumerate(self.layer_list):

            extra_param_counter = 0
            this_pdf_type=self.pdf_defs_list[pdf_index]

            if(data_summary is not None):
                
                this_data_summary=data_summary
                if(len(extra_conditional_input)>0):
                    this_data_summary=torch.cat([data_summary]+extra_conditional_input, dim=1)
                
                extra_params=self.mlp_predictors[pdf_index](this_data_summary)
               

            else:

                if(self.mlp_predictors[pdf_index] is not None):
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                        
                    else:
                        raise Exception("FORWARD: extra conditional input is empty but required for encoding!")

            if(self.predict_log_normalization):
                if(pdf_index==0 and self.join_poisson_and_pdf_description):
                    extra_params=extra_params[:,:-1]

            this_target=z[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]

            
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
                   

                this_target, log_det = layer.inv_flow_mapping([this_target, log_det], extra_inputs=this_extra_params)

                extra_param_counter += layer.total_param_num

            new_targets.append(this_target)

            prev_target=z[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]
            if(self.conditional_manifold_input_embedding[this_pdf_type[0]]==1):
                ## return embedding value for next conditional inputs
                prev_target=self.basic_layers[pdf_index]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        base_pos=torch.cat(new_targets, dim=1)

        log_pdf = torch.distributions.MultivariateNormal(
            torch.zeros_like(x).to(x),
            covariance_matrix=torch.eye(self.total_target_dim).type_as(x).to(x),
        ).log_prob(base_pos)

        return log_pdf + log_det, log_pdf, base_pos

    def forward_subpdf(self, subpdf_index, this_x, extra_params=None):
        """
        Forward function of a single sub-pdf. Is used in the calculation of the total correlation.
        """
        log_det = torch.zeros(this_x.shape[0]).type_as(this_x)

        this_target=this_x

        pdf_layers=self.layer_list[subpdf_index]

        ## reverse mapping is required for pdf evaluation

        this_extra_params=None
        extra_param_counter=0

        for l, layer in reversed(list(enumerate(pdf_layers))):

            if extra_params is not None:

                if extra_param_counter == 0:
                        this_extra_params = extra_params[:, -layer.total_param_num :]
                else:

                    this_extra_params = extra_params[
                        :,
                        -extra_param_counter
                        - layer.total_param_num : -extra_param_counter,
                    ]
               

            this_target, log_det = layer.inv_flow_mapping([this_target, log_det], extra_inputs=this_extra_params)

            extra_param_counter += layer.total_param_num

        log_pdf = torch.distributions.MultivariateNormal(
            torch.zeros_like(this_x).to(this_x),
            covariance_matrix=torch.eye(self.target_dims[subpdf_index]).type_as(this_x).to(this_x),
        ).log_prob(this_target)


        return log_pdf + log_det, log_pdf, this_target

    def sample(self, conditional_input=None, samplesize=1,  seed=None, device=torch.device("cpu"), allow_gradients=False):
        """ 
        Samples from the (conditional) PDF. 
        conditional_input: None or Tensor of shape N x D wherer N is the batch size and D the input space dimension.
        """
        ###############

        if(allow_gradients):

            sample, normal_base_sample, log_pdf_target, log_pdf_base=self._obtain_sample(conditional_input=conditional_input, device=device, seed=seed, samplesize=samplesize)

            return sample, normal_base_sample, log_pdf_target, log_pdf_base

        else:   
            with torch.no_grad():
                sample, normal_base_sample, log_pdf_target, log_pdf_base=self._obtain_sample(conditional_input=conditional_input, device=device, seed=seed, samplesize=samplesize)

            return sample, normal_base_sample, log_pdf_target, log_pdf_base


    def _obtain_sample(self, conditional_input=None, predefined_target_input=None, samplesize=1, seed=None, device=torch.device("cpu")):
        
        data_type = torch.float64
        data_summary = None
        used_sample_size = samplesize
        used_device=device

        if conditional_input is not None:

            data_summary=self._conditional_input_to_summary(conditional_input=conditional_input)

            used_sample_size = data_summary.shape[0]
            data_type = data_summary.dtype
            used_device = data_summary.device

        x=None
        log_gauss_evals=0.0
        unit_gauss_samples=0.0

        if(predefined_target_input is not None):

            x=predefined_target_input

            if(conditional_input is not None):

                ## make sure inputs agree
                assert(x.shape[0]==conditional_input.shape[0])
                assert(x.dtype==data_summary.dtype)
                assert(x.device==data_summary.device)

            else:
                data_type=predefined_target_input.dtype
                used_sample_size=predefined_target_input.shape[0]
                used_device=predefined_target_input.device

            log_gauss_evals = torch.distributions.MultivariateNormal(
                torch.zeros(self.total_target_dim).type(data_type).to(device),
                covariance_matrix=torch.eye(self.total_target_dim)
                .type(data_type)
                .to(device),
            ).log_prob(predefined_target_input)

        else:

            if(seed is not None):
                numpy.random.seed(seed)

            unit_gauss = numpy.random.normal(size=(used_sample_size, self.total_target_dim))

            unit_gauss_samples = (
                torch.from_numpy(unit_gauss).type(data_type).to(device)
            )
            log_gauss_evals = torch.distributions.MultivariateNormal(
                torch.zeros(self.total_target_dim).type(data_type).to(device),
                covariance_matrix=torch.eye(self.total_target_dim)
                .type(data_type)
                .to(device),
            ).log_prob(unit_gauss_samples)

            x = unit_gauss_samples

        log_det = torch.zeros(used_sample_size).type(data_type).to(used_device)

        extra_conditional_input=[]
        new_targets=[]

        for pdf_index, pdf_layers in enumerate(self.layer_list):

         
            this_pdf_type=self.pdf_defs_list[pdf_index]

            extra_params = None
            if(data_summary is not None):
               
                this_data_summary=data_summary
                if(len(extra_conditional_input)>0):
                    this_data_summary=torch.cat([data_summary]+extra_conditional_input, dim=1)

                extra_params=self.mlp_predictors[pdf_index](this_data_summary)

            else:

                if(self.mlp_predictors[pdf_index] is not None):
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                    else:
                        raise Exception("SAMPLE: extra conditional input is empty but required for encoding!")
     
            if(self.predict_log_normalization):

                if(pdf_index==0 and self.join_poisson_and_pdf_description):
                    extra_params=extra_params[:,:-1]

            this_target=x[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]

            ## loop through all layers in each pdf and transform "this_target"
            
            extra_param_counter = 0
            for l, layer in list(enumerate(pdf_layers)):

               
                
                this_extra_params = None
                

                if extra_params is not None:
                    
                    this_extra_params = extra_params[:, extra_param_counter : extra_param_counter + layer.total_param_num]
                    

                this_target, log_det = layer.flow_mapping([this_target, log_det], extra_inputs=this_extra_params)

                extra_param_counter += layer.total_param_num

            new_targets.append(this_target)

            prev_target=this_target
            if(self.conditional_manifold_input_embedding[this_pdf_type[0]]==1):
                ## return embedding value for next conditional inputs
                prev_target=self.basic_layers[pdf_index]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        if (torch.isfinite(x) == 0).sum() > 0:
            raise Exception("nonfinite samples generated .. this should never happen!")

     
        ## -logdet because log_det in sampling is derivative of forward function d/dx(f), but log_p requires derivative of backward function d/dx(f^-1) whcih flips the sign here
        return torch.cat(new_targets, dim=1), unit_gauss_samples, -log_det + log_gauss_evals, log_gauss_evals

    def get_returnable_target_dim(self):
        """
        This function is used in the autoregressive structure and returns the total dimension of all sub PDFs. Embedded Manifold PDFs add one dimension
        to return the embedding space dimension.
        """
        tot_dim=0
        for pdf_index, pdf_type in enumerate(self.pdf_defs_list):
            tot_dim+=self.target_dims[pdf_index]
            if(self.conditional_manifold_input_embedding[pdf_type[0]]==1):
                tot_dim+=1

        return tot_dim

    def transform_target_into_returnable_params(self, target):
        """
        Transform the destimation space of the PDF into a returnable param list (i.e. transform spherical angles into x/y/z pairs, so increase the dimensionality of spherical dimensions by 1)
        """

        new_target=target

        if(len(target.shape)==1):
            new_target=target.unsqueeze(0)

        potentially_transformed_vals=[]

        index=0
        for pdf_index, pdf_type in enumerate(self.pdf_defs_list):
            this_dim=self.target_dims[pdf_index]
           
            this_target=new_target[:,index:index+this_dim]
            
            if(self.conditional_manifold_input_embedding[pdf_type[0]]==1):
                this_target=self.basic_layers[pdf_index]._embedding_conditional_return(this_target)

            potentially_transformed_vals.append(this_target)

        potentially_transformed_vals=torch.cat(potentially_transformed_vals, dim=1)

        if(len(target.shape)==1):
            potentially_transformed_vals=potentially_transformed_vals.squeeze(0)

        return potentially_transformed_vals

    ########

    def init_params(self, data=None, name="test"):

        with torch.no_grad():
            ## 0) check data
            if(data is not None):
                ## initialization data has to match pdf dimenions
                assert(data.shape[1]==self.total_target_dim)

            ## 1) Find initialization params of all layers
            params_list=[]
            ## loop through all the layers and get the initializing parameters

            this_dim_index=0

            for subflow_index, subflow_description in enumerate(self.pdf_defs_list):
                
                this_layer_list=self.layer_list[subflow_index]

                this_dim=self.target_dims[subflow_index]

                ## special initalizations
                ## check if all are gaussianization flow
                gf_init=True
                for layer_type in self.flow_defs_list[subflow_index]:
                    if(layer_type!="g"):
                        gf_init=False
                if(data is None):
                    gf_init=False

                if(gf_init):
         
                    params=find_init_pars_of_chained_gf_blocks(this_layer_list, data[:, this_dim_index:this_dim_index+this_dim],householder_inits="random",name=name)
                    params_list.append(params.type(torch.float64))

                else:
                   
                    ## get basic rough init...
                    this_list=[]
                    for l in this_layer_list:
                        this_list.append(l.get_desired_init_parameters().type(torch.float64))

                    params_list.append(torch.cat(this_list))

                this_dim_index+=this_dim

            ## 2) Depending on encoding structure, use the init params at appropriate places
            if(self.encoder_mlp_predictors is not None):

                raise Exception("encoder mlps not yet implemented")

            else:

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
                            if(self.predict_log_normalization):
                                if(self.join_poisson_and_pdf_description):
                                    if(ind==0):
                                        log_lambda_init=0.1
                                        these_params=torch.cat([these_params, torch.Tensor([log_lambda_init])])

                            ## custom low-rank MLPs - initialization is done inside the custom MLP class
                            if(self.use_custom_low_rank_mlps):
                                mlp_predictor.initialize_uvbs(init_b=these_params)
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
                                        
                                        internal_layer.weight.data/=1000.0
                                        internal_layer.bias.data/=1000.0
                                    
                                # finally overwrite bias to be equivalent to desired parameters at initialization
                                
                                mlp_predictor[-1].bias.data=these_params

                        else:
                            ## threre is no MLP - initialize parameters of flows directly
                            tot_param_index=0
                      
                            for layer_ind, layer in enumerate(self.layer_list[ind]):
                                
                                this_layer_num_params=self.layer_list[ind][layer_ind].get_total_param_num()

                               
                                self.layer_list[ind][layer_ind].init_params(these_params[tot_param_index:tot_param_index+this_layer_num_params])

                                tot_param_index+=this_layer_num_params



    
    