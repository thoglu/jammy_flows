import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.flows as f

import jammy_flows.helper_fns as helper_fns


def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)



class Test(unittest.TestCase):
    def setUp(self):

        self.flow_inits=[]

        self.flow_inits.append([ ["s2", "v"], dict()])

        self.flow_inits.append([ ["e1", "p"], dict()])
        self.flow_inits.append([ ["e1", "g"], dict()])
        self.flow_inits.append([ ["s1", "m"], dict()])
        self.flow_inits.append([ ["s1", "o"], dict()])

        ### s1 with splines

        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["m"]=dict()
    
        self.flow_inits.append([ ["s1", "m"], extra_flow_defs])

        pdf_def="e1+e2+e2+s1"
        flow_def="pp+pp+gg+m"

        ## 3d rotation

        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["g"]=dict()
        extra_flow_defs["options_overwrite"]["g"]["rotation_mode"]="angles"

        self.flow_inits.append([ ["e3", "gg"], extra_flow_defs])


  
        #self.flow_inits.append( [ [pdf_def, flow_def], {"options_overwrite":{"g":{"kwargs":{"inverse_function_type":"inormal_partly_precise"}}}}] )
        #self.flow_inits.append( [ [pdf_def, flow_def], {"options_overwrite":{"g":{"kwargs":{"inverse_function_type":"inormal_partly_crude"}}}}] )
        #self.flow_inits.append( [ [pdf_def, flow_def], {"options_overwrite":{"g":{"kwargs":{"inverse_function_type":"inormal_full_pade"}}}}] )
        #self.flow_inits.append( [ [pdf_def, flow_def], {"options_overwrite":{"g":{"kwargs":{"inverse_function_type":"isigmoid"}}}}] )

        
       
        mlp_hidden_dims=["64-30"]
        use_low_rank_option=[False, True]
        low_rank_ranks=["2-10-1000"]


        gf_settings=[{"g":{"softplus_for_width":1}},
                     {"g":{"inverse_function_type":"inormal_partly_crude"}},
                     {"g":{"inverse_function_type":"inormal_partly_precise"}},
                     {"g":{"inverse_function_type":"inormal_full_pade"}},
                     {"g":{"clamp_widths": 1}},
                     {"g":{"upper_bound_for_widths":-1, "clamp_widths": 1, "width_smooth_saturation": 0}},
                     {"g":{"add_skewness": 1}},
                     {"g":{"rotation_mode": "householder"}},
                     {"g":{"rotation_mode": "angles"}},
                     {"g":{"rotation_mode": "triangular_combination"}},
                     {"g":{"rotation_mode": "cayley"}},
                     {"g":{"nonlinear_stretch_type": "rq_splines"}}

                     ]

        
      
        for mlp_hidden in mlp_hidden_dims:
            for use_rank in use_low_rank_option:
                for lr_approx in low_rank_ranks:
                    for gf_setting in gf_settings:
                        d=dict()
                        
                        d["conditional_input_dim"]=2
                        
                       
                        d["hidden_mlp_dims_sub_pdfs"]=mlp_hidden
                        d["rank_of_mlp_mappings_sub_pdfs"]=lr_approx
                        d["use_custom_low_rank_mlps"]=use_rank
                        d["options_overwrite"]=gf_setting

                        self.flow_inits.append([[pdf_def, flow_def], d])

        # exponential map flow variations

        for nat in [0,1]:
            for setting in ["linear", "quadratic", "exponential", "splines"]:
                ### exponential map
                extra_flow_defs=dict()
                extra_flow_defs["options_overwrite"]=dict()
                extra_flow_defs["options_overwrite"]["v"]=dict()
                
                extra_flow_defs["options_overwrite"]["v"]["natural_direction"]=nat
                extra_flow_defs["options_overwrite"]["v"]["exp_map_type"]=setting


                self.flow_inits.append([ ["s2", "v"], extra_flow_defs])

        #### mvn 

        for cov_type in ["full", "diagonal_symmetric", "diagonal", "identity"]:

            extra_flow_defs=dict()
            extra_flow_defs["options_overwrite"]=dict()
            extra_flow_defs["options_overwrite"]["t"]=dict()
            extra_flow_defs["options_overwrite"]["t"]["cov_type"]=cov_type
            extra_flow_defs["conditional_input_dim"]=2
        
            self.flow_inits.append([ ["e10", "t"], extra_flow_defs])



        #######

        
        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["c"]=dict()
        extra_flow_defs["options_overwrite"]["c"]["solver"]="rk4"
        extra_flow_defs["options_overwrite"]["c"]["cnf_network_hidden_dims"]=""
        extra_flow_defs["options_overwrite"]["c"]["num_charts"]=20
        extra_flow_defs["conditional_input_dim"]=2

        self.flow_inits.append([ ["s2", "c"], extra_flow_defs])
        self.flow_inits.append([ ["s2", "c"], {}])
       
        ###################### Interval flows

        extra_flow_defs=dict()
        extra_flow_defs["conditional_input_dim"]=2

        self.flow_inits.append([ ["i1_-1.0_1.0", "r"], extra_flow_defs])

        ####
        #zenith_layers=["g", "p", "x", "z", "r"]
        zenith_layers=["g", "x", "z", "r"]

        for z_layer in zenith_layers:
            extra_flow_defs=dict()
            extra_flow_defs["options_overwrite"]=dict()
            extra_flow_defs["options_overwrite"]["n"]=dict()
         
            extra_flow_defs["options_overwrite"]["n"]["add_rotation"]=1
            #extra_flow_defs["options_overwrite"]["n"]["kwargs"]["higher_order_cylinder_parametrization"]=True
            extra_flow_defs["options_overwrite"]["n"]["zenith_type_layers"]=z_layer
            ## all n-type flows

            pdf_def="s2"
            flow_def="n"

            self.flow_inits.append([[pdf_def, flow_def], extra_flow_defs])

        # more general flow

        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["n"]=dict()
      
        extra_flow_defs["options_overwrite"]["n"]["add_rotation"]=1
        #extra_flow_defs["options_overwrite"]["n"]["kwargs"]["higher_order_cylinder_parametrization"]=True

        ## general flow
        pdf_def="s2+e2+s1"
        flow_def="n+gg+m"

        self.flow_inits.append([[pdf_def, flow_def], extra_flow_defs])

        ## general flow reversed
        pdf_def="e2+s2+s1"
        flow_def="gg+n+m"

        self.flow_inits.append([[pdf_def, flow_def], extra_flow_defs])
            
        pdf_def="a3"
        flow_def="w"

        self.flow_inits.append([[pdf_def, flow_def], {}])

        extra_flow_defs=dict()
        extra_flow_defs["conditional_input_dim"]=2
        
        self.flow_inits.append([[pdf_def, flow_def], extra_flow_defs])

        


    def test_gpu(self):

        print("-> Testing gpu-support <-")
        dev=torch.device("cuda")
        try:
            a=torch.Tensor([1.0,2.0]).to(dev)
        except:
            print("GPU not supported - skipping GPU test")
        else:
            print("GPU supported .. checking GPU support of different flows ... ")

            #check_list=[ ["e1", "g"], ["e1", "p"], ["s1", "m"], ["s2","n"] ]

            for flow_init in self.flow_inits:
                print("checking .. ", flow_init[0])

                pdf=f.pdf(*flow_init[0], **flow_init[1])
                pdf.to(dev)

                ## construct 100 samples
                c_input=None
                if("conditional_input_dim" in flow_init[1].keys()):
                    cdim=flow_init[1]["conditional_input_dim"]
                    print("conditional ....")
                    c_input=torch.ones( (100, flow_init[1]["conditional_input_dim"]), dtype=torch.float64, device=torch.device("cuda"))

                gpu_sample,_,_,_ = pdf.sample(device=dev, conditional_input=c_input, samplesize=100)

                res,_,_=pdf(gpu_sample, conditional_input=c_input)

if __name__ == '__main__':
    unittest.main()