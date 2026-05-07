import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import torch.autograd
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.main.default as f
#from pytorch_lightning import seed_everything
import jammy_flows.helper_fns as helper_fns
import jammy_flows.layers.bisection_n_newton as bn

def seed_everything(seed):

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def compare_two_arrays(arr1, arr2, name1, name2):

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


    diff_value=1e-7

    print(name1, name2)
    print(arr1)
    print(arr2)
    diff_too_large_mask=numpy.fabs(arr1-arr2)>diff_value

    if(diff_too_large_mask.sum()>0):
        print("%s and %s contain elements (%d/%d) that differ by value greater than %.5e" % (name1, name2,diff_too_large_mask.sum(),len(diff_too_large_mask),diff_value))
        print("selected array 1 values ...")
        print(arr1[diff_too_large_mask])
        print("selected array 2 values ...")
        print(arr2[diff_too_large_mask])
        print(".. diffs")
        print(numpy.fabs(arr1-arr2)[diff_too_large_mask])
        raise Exception()

    print("largest diff between ", name1, " and ", name2, " (%d items): " % len(arr1),numpy.fabs(arr1-arr2).max() )


"""
This test checks the inversion by bisection and newton iterations by comparing it to the exact inverse.
"""

class Test(unittest.TestCase):
    def setUp(self):

        self.flow_inits=[]
        
       
    

    def test_compare_newton_times(self):
        """ 
        Comparing the derivatives and normal results of newton iterations with two different versions of Gaussianization flow (h - old), (g - new).
        """
        print("Testing the agreement of two different newton iteration algorithms and compares their speed")

        def get_non_none_grad(quantity, params):

            grad_res=torch.autograd.grad(quantity, params, allow_unused=True, retain_graph=False)

            return torch.cat( [res.view(-1) for res in grad_res if (res is not None)])

        samplesize=10
        seed_everything(1)
        z=torch.rand((samplesize,2), dtype=torch.double)*50

        for skew in [0,1]:

            
            #### old gaussianization flow
            
            ## we dont want normalization in the following for simplicity (in one we would need an extra transformation to apply manually)
            extra_flow_defs=dict()
            extra_flow_defs["g"]=dict()
            
            extra_flow_defs["g"]["regulate_normalization"]=0
            extra_flow_defs["g"]["add_skewness"]=skew
            extra_flow_defs["g"]["inverse_function_type"]="isigmoid"

            extra_flow_defs["h"]=dict()
            
            extra_flow_defs["h"]["regulate_normalization"]=0
            extra_flow_defs["h"]["add_skewness"]=skew
            extra_flow_defs["h"]["inverse_function_type"]="isigmoid"

            seed_everything(1)
            flow_exact=f.pdf("e2", "h", options_overwrite=extra_flow_defs)
            flow_exact.double()

            
            gf_layer=flow_exact.layer_list[0][0]
           
            skew_for_old=gf_layer.skew_exponents
            if(skew):
                skew_for_old=torch.exp(gf_layer.exponent_regulator(gf_layer.skew_exponents))

          
            ## old slow
            res_old_slow=bn.inverse_bisection_n_newton_slow(gf_layer.sigmoid_inv_error_pass, gf_layer.sigmoid_inv_error_pass_derivative, z, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, skew_for_old,gf_layer.skew_signs, min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=25, num_newton_iter=20)

            old_slow_grad=get_non_none_grad(res_old_slow.sum(),gf_layer.parameters())

            ## old fast

            seed_everything(1)
            flow_exact=f.pdf("e2", "h", options_overwrite=extra_flow_defs)
            flow_exact.double()

            
            gf_layer=flow_exact.layer_list[0][0]
           

            skew_for_old=gf_layer.skew_exponents
            if(skew):
                skew_for_old=torch.exp(gf_layer.exponent_regulator(gf_layer.skew_exponents))
          
            newton_tolerance=1e-14
            res_old_fast=bn.inverse_bisection_n_newton(gf_layer.sigmoid_inv_error_pass, gf_layer.sigmoid_inv_error_pass_derivative, z, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, skew_for_old,gf_layer.skew_signs, min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=25, num_newton_iter=20, newton_tolerance=newton_tolerance, verbose=0)
            old_fast_grad=get_non_none_grad(res_old_fast.sum(),gf_layer.parameters())

            compare_two_arrays(old_slow_grad.numpy(), old_fast_grad.numpy(), "old slow grad", "old fast grad")
           
        
            seed_everything(1)
            flow_exact=f.pdf("e2", "g", options_overwrite=extra_flow_defs)
            flow_exact.double()

            gf_layer=flow_exact.layer_list[0][0]

            flow_params,_=gf_layer._obtain_usable_flow_params(z)
        
            newton_tolerance=1e-14
            res_new_fast=bn.inverse_bisection_n_newton_joint_func_and_grad(gf_layer.sigmoid_inv_error_pass_w_params, gf_layer.sigmoid_inv_error_pass_combined_val_n_normal_derivative, z, flow_params[0], flow_params[1], flow_params[2], flow_params[3], flow_params[4], min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=25, num_newton_iter=20, newton_tolerance=newton_tolerance, verbose=0)
            new_fast_grad=get_non_none_grad(res_new_fast.sum(),gf_layer.parameters())

            compare_two_arrays(old_slow_grad.numpy(), new_fast_grad.numpy(), "old slow grad", "new fast grad")
           
      

            #assert(torch.abs(old_slow_grad-new_fast_grad).sum() < 1e-12)

            #############
            
            with torch.no_grad():
                
                    
                ### old gaussianization flow
                seed_everything(1)
                flow_exact=f.pdf("e2", "h",options_overwrite=extra_flow_defs)
                flow_exact.double()

                gf_layer=flow_exact.layer_list[0][0]

                skew_for_old=gf_layer.skew_exponents
                if(skew):
                    skew_for_old=torch.exp(gf_layer.exponent_regulator(gf_layer.skew_exponents))
              

                z=torch.rand((samplesize,1), dtype=torch.double)*50
                
                tbef=time.time()
                for i in range(100):
                    res_old_slow=bn.inverse_bisection_n_newton_slow(gf_layer.sigmoid_inv_error_pass, gf_layer.sigmoid_inv_error_pass_derivative, z, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, skew_for_old,gf_layer.skew_signs, min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=2, num_newton_iter=20)
                   
                print("Old flow / old newton iteration took ", time.time()-tbef)

                skew_for_old=gf_layer.skew_exponents
                if(skew):
                    skew_for_old=torch.exp(gf_layer.exponent_regulator(gf_layer.skew_exponents))
              
                newton_tolerance=1e-14
                tbef=time.time()
                for i in range(100):
                    res_old_fast=bn.inverse_bisection_n_newton(gf_layer.sigmoid_inv_error_pass, gf_layer.sigmoid_inv_error_pass_derivative, z, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, skew_for_old,gf_layer.skew_signs, min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=2, num_newton_iter=20, newton_tolerance=newton_tolerance, verbose=0)
                
                print("Old flow / new newton iteration with tolerance dropout took ", time.time()-tbef)
                print(torch.abs(res_old_slow-res_old_fast).max())
                assert( torch.abs(res_old_slow-res_old_fast).max() < newton_tolerance*100)

                ##### new gaussianization flow

                seed_everything(1)
                flow_exact=f.pdf("e2", "g",options_overwrite=extra_flow_defs)
                flow_exact.double()

                gf_layer=flow_exact.layer_list[0][0]

                flow_params,_=gf_layer._obtain_usable_flow_params(z)

                newton_tolerance=1e-14
                tbef=time.time()
                for i in range(100):
                    res_new_fast=bn.inverse_bisection_n_newton_joint_func_and_grad(gf_layer.sigmoid_inv_error_pass_w_params, gf_layer.sigmoid_inv_error_pass_combined_val_n_normal_derivative, z, flow_params[0], flow_params[1], flow_params[2], flow_params[3], flow_params[4], min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=2, num_newton_iter=20, newton_tolerance=newton_tolerance, verbose=0)
                
                print("new flow / new newton iteration with tolerance dropout took ", time.time()-tbef)
                print(torch.abs(res_old_slow-res_new_fast).max())
                assert( torch.abs(res_old_slow-res_new_fast).max() < newton_tolerance*100)


            ### now do the same and check gradient compatibility

        
        
                
        


if __name__ == '__main__':
    unittest.main()