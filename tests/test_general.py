import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.main.default as f

import jammy_flows.helper_fns as helper_fns


def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)

def check_gf_trafo(flow, crude_trafo, new_trafo, new_deriv, new_log_deriv,  plot=False, name="test.png"):

    points=torch.linspace(-2,2,1000).type(torch.float64).reshape(1000,1)
    #points=torch.linspace(-0.64669,-0.64664,5000).type(torch.float64).reshape(5000,1)
    y_crude=crude_trafo(points).detach().numpy()

    y_trafo_new=new_trafo(points).detach().numpy()
    y_deriv_new_torch=torch.autograd.functional.jacobian(new_trafo, points).sum(axis=2).squeeze().detach().numpy()
    y_deriv_new_defined=new_deriv(points).detach().numpy().squeeze()

    y_log_deriv_new_defined=new_log_deriv(points).detach().numpy().squeeze()

    log_deriv_diff=numpy.fabs((numpy.log(y_deriv_new_defined)-y_log_deriv_new_defined)).sum()
    torch_deriv_diff=numpy.fabs(y_deriv_new_torch-y_deriv_new_defined).sum()

    #y_deriv_of_deriv_torch=torch.autograd.functional.jacobian(new_deriv, points).sum(axis=2).squeeze().detach().numpy()
    y_deriv_of_deriv_torch=torch.zeros_like(points).squeeze()

    #y_deriv_new_defined=numpy.where(numpy.isfinite(y_deriv_new_defined), y_deriv_new_defined, 0)

    if(plot):
        fig=pylab.figure()

        ax0=fig.add_subplot(3,1,1)

        log_pdf,_,_=flow(points)

        pts_numpy=points.squeeze(0).detach()
        y_pts=log_pdf.exp().detach().numpy()

        ax0.plot(pts_numpy, y_pts)
        ax0.semilogy()

        ax0.set_title("log_diff %.3e / automatic/manual diff %.3e" % (log_deriv_diff, torch_deriv_diff))
        ax1=fig.add_subplot(3,1,2)

        ax1.plot(pts_numpy, y_crude, color="black")
        
        ax1.plot(pts_numpy, y_trafo_new, color="gray")
        ax1.plot(pts_numpy, y_deriv_new_torch, color="blue")
        ax1.plot(pts_numpy, y_deriv_new_defined, color="magenta")

        ax2=fig.add_subplot(3,1,3)

        ax2.plot(pts_numpy, y_deriv_new_torch-y_deriv_new_defined, color="green", label="diff automatic/defined deriv new")
        #ax2.plot(pts_numpy, y_crude-y_trafo_new, color="black", label="crude-this diff")

        ax2.legend()

        fig.tight_layout()

        pylab.savefig(name)

  
    return log_deriv_diff, torch_deriv_diff, y_deriv_of_deriv_torch.sum()

def compare_two_arrays(arr1, arr2, name1, name2, diff_value=1e-7):

    num_non_finite_1=(numpy.isfinite((arr1))==0).sum()

    if(num_non_finite_1>0):
        print(name1, " contains %d non-finite elements" % num_non_finite_1)
        print(arr1[numpy.isfinite((arr1))==0])

        raise Exception()

    num_non_finite_2=(numpy.isfinite((arr1))==0).sum()
    if(num_non_finite_2>0):
        print(name2, " contains %d non-finite elements" % num_non_finite_2)
        print(arr2[numpy.isfinite((arr2))==0])

        raise Exception()


 
    diff_too_large_mask=numpy.fabs(arr1-arr2)>diff_value

    if(diff_too_large_mask.sum()>0):
        print("%s and %s contain elements (%d/%d) that differ by value greater than %.5e" % (name1, name2,diff_too_large_mask.sum(),len(diff_too_large_mask),diff_value))
        print("selected array 1 values ...")
        print(arr1[diff_too_large_mask])
        print("selected array 2 values ...")
        print(arr2[diff_too_large_mask])
        print(".. diffs")
        diffs=numpy.fabs(arr1-arr2)[diff_too_large_mask]
        print(diffs)
        print("min / maxn diff: ", diffs.min(), "/ ", diffs.max())
        raise Exception()

    print("largest diff between ", name1, " and ", name2, " (%d items): " % len(arr1),numpy.fabs(arr1-arr2).max() )

class Test(unittest.TestCase):
    def setUp(self):

        self.flow_inits=[]

        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["v"]=dict()
        extra_flow_defs["options_overwrite"]["v"]["add_rotation"]=1

        self.flow_inits.append([ ["s2", "v"], extra_flow_defs])

        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["f"]=dict()
        extra_flow_defs["options_overwrite"]["f"]["add_vertical_rq_spline_flow"]=1
        extra_flow_defs["options_overwrite"]["f"]["add_circular_rq_spline_flow"]=1

        self.flow_inits.append([ ["s2", "f"], extra_flow_defs])

        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["f"]=dict()
        extra_flow_defs["options_overwrite"]["f"]["add_correlated_rq_spline_flow"]=1
        
        self.flow_inits.append([ ["s2", "f"], extra_flow_defs])
        
        self.flow_inits.append([ ["s2", "c"], dict()])

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

        # center mean
        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["g"]=dict()
        extra_flow_defs["options_overwrite"]["g"]["center_mean"]=1

        self.flow_inits.append([ ["e3", "ggg"], extra_flow_defs])

        ### multiple conditional inputs


        # center mean
        extra_flow_defs=dict()
     
        extra_flow_defs["conditional_input_dim"]=[3,4,5]
       
        self.flow_inits.append([ ["e1+e2+e1", "gg+g+ggg"], extra_flow_defs])


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
                        
                       
                        d["amortization_mlp_dims"]=mlp_hidden
                        d["amortization_mlp_ranks"]=lr_approx
                        d["amortization_mlp_use_custom_mode"]=use_rank
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

        """
        extra_flow_defs=dict()
        extra_flow_defs["options_overwrite"]=dict()
        extra_flow_defs["options_overwrite"]["c"]=dict()
        extra_flow_defs["options_overwrite"]["c"]["kwargs"]=dict()
        extra_flow_defs["options_overwrite"]["c"]["kwargs"]["natural_direction"]=1
        extra_flow_defs["options_overwrite"]["c"]["kwargs"]["solver"]="rk4"
        extra_flow_defs["options_overwrite"]["c"]["kwargs"]["cnf_network_hidden_dims"]=""
        extra_flow_defs["options_overwrite"]["c"]["kwargs"]["num_charts"]=20
        extra_flow_defs["conditional_input_dim"]=2

        self.flow_inits.append([ ["s2", "c"], extra_flow_defs])
        """


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
    
    
    def test_selfconsistency(self):

        print("-> Testing self consistency of sampling <-")

        samplesize=10000
        
        for ind, init in enumerate(self.flow_inits):
            ## seed everything to have consistent tests
            seed_everything(1)
            

            #seed_everything(0)
            this_flow=f.pdf(*init[0], **init[1])
            #this_flow.double()

            ## test for pure Euclidean manifold
            if( (len(init[0][0].split("+"))==1) and (init[0][0][0]=="e")):
                used_dim=int(init[0][0][1:])
                gaussian_init_data=torch.randn(size=(100,used_dim))
                
                # test init function with random data
                this_flow.init_params(data=gaussian_init_data)

               
            cinput=None
            if("conditional_input_dim" in init[1].keys()):
                if(type(init[1]["conditional_input_dim"])==int):

                    rvec=numpy.random.normal(size=(samplesize,init[1]["conditional_input_dim"]))*100.0
                    cinput=torch.from_numpy(rvec)
                else:
                    cinput=[]
                    for ci_dim in init[1]["conditional_input_dim"]:

                        rvec=numpy.random.normal(size=(samplesize,ci_dim))*100.0
                        cinput.append(torch.from_numpy(rvec))

            print("####################")
            print("INIT ", init)
            print("####################")

            for precision in [torch.float64, torch.float32]:
                

                if(precision==torch.float32):

                    if(len(init[0][0].split("+"))>1):
                        ## only test single sub pdf flows with float32
                        continue

                    ### v requires double precision
                    if("v" in init[0][1]):
                        continue

                    

                    tolerance=5e-2

                    if("o" in init[0][1]):
                        tolerance=2.0

                    if("c" in init[0][1]):
                        ## exponential map flows get a little less strict tolerance check for now
                        
                        tolerance=1e-2

                    ### n requires double precision
                    if("n" in init[0][1]):
                        tolerance=1.1

                    
                    this_flow.float()
                else:

                    tolerance=1e-6

                    

                    if("v" in init[0][1]):
                        ## exponential map flows get a little less strict tolerance check for now
                        
                        tolerance=1e-4

                    if("c" in init[0][1]):
                        ## exponential map flows get a little less strict tolerance check for now
                        
                        tolerance=1e-4

                    this_flow.double()

                print("++++ PRECISION ", precision)

                if(cinput is not None):
                    if(type(cinput)==list):
                        cinput=[ci.type(precision) for ci in cinput]
                    else:
                        cinput=cinput.type(precision)

                for use_embedding in [0,1]:
                    print("####### EMBEDDING : ", use_embedding)

                    with torch.no_grad():
                        samples, base_samples, evals, base_evals=this_flow.sample(samplesize=samplesize,conditional_input=cinput, force_embedding_coordinates=use_embedding)
                        
                        samples_bef=samples.clone()
                        ## evaluate the samples and see if the reverse direction is compatible
                        evals2, base_evals2, base_samples2=this_flow(samples, conditional_input=cinput, force_embedding_coordinates=use_embedding)

                        ## make sure there is no in place operation that changes things
                        compare_two_arrays(samples.detach().numpy(), samples_bef.detach().numpy(), "samples_before_pass", "samples_after_pass", diff_value=tolerance)


                        ## make sure log-det is not overwritten in forward/backward passes
                        test_sample=torch.rand(10,this_flow.total_target_dim, dtype=samples.dtype)
                        log_det=torch.zeros(test_sample.shape[0]).to(test_sample)
                        
                        inp=None
                        if(cinput is not None):
                            if(type(cinput)==list):
                                inp=[ci[:10,:] for ci in cinput]
                            else:
                                inp=cinput[:10,:]

                        res, log_det_new=this_flow.all_layer_forward(test_sample, log_det, inp)

                     
                        assert( (log_det==0).sum()==10)

                        #####

                        inp=None
                        if(cinput is not None):
                            if(type(cinput)==list):
                                inp=[ci[:10,:] for ci in cinput]
                            else:
                                inp=cinput[:10,:]

                        res, log_det_new=this_flow.all_layer_inverse(test_sample, log_det, inp)

                    
                        assert( (log_det==0).sum()==10)


                    #this_flow.count_parameters()
                    compare_two_arrays(base_samples.detach().numpy(), base_samples2.detach().numpy(), "base_samples", "base_samples2", diff_value=tolerance)
                    compare_two_arrays(evals.detach().numpy(), evals2.detach().numpy(), "evals", "evals2", diff_value=tolerance)
                    compare_two_arrays(base_evals.detach().numpy(), base_evals2.detach().numpy(), "base_evals", "base_evals2", diff_value=tolerance)

                    """
                    Check self consistency in flow structure.
                    """

                    ## exclude gausianization flows from param structure test because we manually add determinant in params, so difference would be detectable
                    if(not "g" in init[0][1] and not "c" in init[0][1]):
                        if(cinput is None):
                            flow_param_structure=this_flow.obtain_flow_param_structure()
                        else:

                            if(type(cinput)==list):
                                inp=[ci[:1,:] for ci in cinput]
                            else:
                                inp=cinput[:1,:]

                            flow_param_structure=this_flow.obtain_flow_param_structure(conditional_input=inp)
                        
                        fps_num=0

                        for k in flow_param_structure.keys():
                            for k2 in flow_param_structure[k].keys():
                              
                                fps_num+=flow_param_structure[k][k2].numel()
                        
                        explicit_param_num=0
                        for pdf_index, pdf_layers in enumerate(this_flow.layer_list):

                            for l in pdf_layers:
                                explicit_param_num+=l.total_param_num

                        assert(explicit_param_num==fps_num), ("explicit: ", explicit_param_num, "flow params num ", fps_num)

    
    def test_derivative(self):
        print("Testing derivatives of Gaussianization flows")

        plotting=False


        icdf_approximations=["inormal_full_pade", "inormal_partly_crude", "inormal_partly_precise", "isigmoid"]
        #seed_everything(0)

        ## h = old gf layer
        extra_flow_defs=dict()
        extra_flow_defs["h"]=dict()
       
        extra_flow_defs["h"]["inverse_function_type"]="inormal_partly_crude"
        extra_flow_defs["h"]["add_skewness"]=1
        
        crude_flow=f.pdf("e1", "h", options_overwrite=extra_flow_defs)
        crude_flow.double()
        def fn_crude(x):
            b=crude_flow.layer_list[0][0]
            this_skew_exponents=torch.exp(b.exponent_regulator(b.skew_exponents))
            return b.sigmoid_inv_error_pass(x, b.datapoints, b.log_hs, b.log_kde_weights, this_skew_exponents, b.skew_signs)

        for icdf_approx in icdf_approximations:
            #seed_everything(0)
            
            extra_flow_defs=dict()
            extra_flow_defs["h"]=dict()
       
            extra_flow_defs["h"]["inverse_function_type"]=icdf_approx
            extra_flow_defs["h"]["add_skewness"]=1

            this_flow=f.pdf("e1", "h", options_overwrite=extra_flow_defs)
            this_flow.double()

            for gf_layer in this_flow.layer_list[0]:

                gf_skew_exponents=torch.exp(gf_layer.exponent_regulator(gf_layer.skew_exponents))

                def fn_new(x):
                    return gf_layer.sigmoid_inv_error_pass(x, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, gf_skew_exponents, gf_layer.skew_signs)

                def fn_new_deriv(x):
                    return gf_layer.sigmoid_inv_error_pass_derivative(x, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, gf_skew_exponents, gf_layer.skew_signs)

                def fn_new_deriv_log(x):
                    return gf_layer.sigmoid_inv_error_pass_log_derivative(x, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, gf_skew_exponents, gf_layer.skew_signs)


                log_deriv_check, residual_sum_of_derivative_difference, deriv_of_deriv_sums=check_gf_trafo(this_flow, fn_crude, fn_new, fn_new_deriv,fn_new_deriv_log, plot=plotting, name="test_gf_pade_mode_%s.png"%(icdf_approx))
                
                if(residual_sum_of_derivative_difference>1e-4 or numpy.isfinite(residual_sum_of_derivative_difference)==False):
                    raise Exception("Sum of automatic-manual derivatives too large", icdf_approx, residual_sum_of_derivative_difference)
                if(log_deriv_check>1e-6 or numpy.isfinite(log_deriv_check)==False):
                    raise Exception("Log Derivative and log(derivative) do not match ", icdf_approx, log_deriv_check)
                if(deriv_of_deriv_sums>1e-6 or numpy.isfinite(deriv_of_deriv_sums)==False):
                    raise Exception("Derivative of derivative is not finite", icdf_approx, deriv_of_deriv_sums)
                
    
    
    

if __name__ == '__main__':
    unittest.main()