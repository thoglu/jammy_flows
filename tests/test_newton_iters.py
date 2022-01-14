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


import jammy_flows.flows as f
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
        
        pdf_def="e2"
        flow_def="ppp"
        settings={ "flow_defs_detail": {"p":{"kwargs":{"exact_mode":True,"skip_model_offset": 1}}} }

        #self.flow_inits.append( [ [pdf_def, flow_def], dict()] )
        self.init_exact= [ [pdf_def, flow_def], settings] 
        
        settings={ "flow_defs_detail": {"p":{"kwargs":{"exact_mode":False,"skip_model_offset": 1}}} }

        self.init_numerical= [ [pdf_def, flow_def], settings] 

       
    
    def test_newton(self):
        """ 
        We form derivatives of various quantities, once of the exact version, and once of the verrsion with the numerical inverse.
        Parameters that can be exactly 0 (or None) are skipped (vs / mean)
        """
        print("Testing self consistency of sampling")

        samplesize=10000
        
        
        seed_everything(1)
        flow_exact=f.pdf(*self.init_exact[0], **self.init_exact[1])
        flow_exact.double()

    
        flow_exact.train()

        samples, base_samples, evals, base_evals=flow_exact.sample(samplesize=samplesize, allow_gradients=True)
        evals_again_exact,_,_=flow_exact(samples)
        evals_again_exact_detached,_,_=flow_exact(samples.detach())

        mean_evals=evals.mean()
        mean_evals_again=evals_again_exact.mean()
        mean_evals_again_detached=evals_again_exact_detached.mean()
        mean_samples=samples.mean()

       
        #print(flow_exact.count_parameters())

        eval_derivs_1=[]
        eval_derivs_again_1=[]
        eval_derivs_again_detached_1=[]
        sample_derivs_1=[]

        for name, p in flow_exact.named_parameters():
            #print("name ", name)

            if("vs" in name or "mean" in name):
                continue
            #print(torch.autograd.grad(samples[0][0], p))
            res=torch.autograd.grad(mean_evals, p, allow_unused=True, retain_graph=True)
            #print(res)
            if res[0] is not None:
                eval_derivs_1.append(res[0].flatten())

            res=torch.autograd.grad(mean_evals_again, p, allow_unused=True, retain_graph=True)
            #print(res)
            if res[0] is not None:
                eval_derivs_again_1.append(res[0].flatten())

            res=torch.autograd.grad(mean_evals_again_detached, p, allow_unused=True, retain_graph=True)

            if res[0] is not None:
                eval_derivs_again_detached_1.append(res[0].flatten())


            res=torch.autograd.grad(mean_samples, p, allow_unused=True, retain_graph=True)

            if res[0] is not None:
                sample_derivs_1.append(res[0].flatten())

        eval_derivs_1=torch.cat(eval_derivs_1)
        eval_derivs_again_1=torch.cat(eval_derivs_again_1)
        eval_derivs_again_detached_1=torch.cat(eval_derivs_again_detached_1)
        sample_derivs_1=torch.cat(sample_derivs_1)

        seed_everything(1)

        flow_numerical=f.pdf(*self.init_numerical[0], **self.init_numerical[1])
        flow_numerical.double()

   

        samples_num, base_samples_num, evals_num, base_evals_num=flow_numerical.sample(samplesize=samplesize, allow_gradients=True)
        
        ## re-evaluate samples with the function .. should be equivalent (also in deriavative structure) to evals_num   
        evals_again_num,_,_=flow_numerical(samples_num)
        evals_again_num_detached,_,_=flow_numerical(samples_num.detach())


        mean_evals_num=evals_num.mean()
        mean_evals_again_num=evals_again_num.mean()
        mean_evals_again_num_detached=evals_again_num_detached.mean()
        mean_samples_num=samples_num.mean()

        
        eval_derivs_2=[]
        eval_again_derivs_2=[]
        eval_again_derivs_detached_2=[]
        sample_derivs_2=[]

        for name, p in flow_numerical.named_parameters():
            #print(name)
            if("vs" in name or "mean" in name):
                continue

            res=torch.autograd.grad(mean_evals_num, p, allow_unused=True, retain_graph=True)
            #print(res)
            if res[0] is not None:
                eval_derivs_2.append(res[0].flatten())

            res=torch.autograd.grad(mean_evals_again_num, p, allow_unused=True, retain_graph=True)
            #print(res)
            if res[0] is not None:
                eval_again_derivs_2.append(res[0].flatten())

            res=torch.autograd.grad(mean_evals_again_num_detached, p, allow_unused=True, retain_graph=True)
            if res[0] is not None:
                eval_again_derivs_detached_2.append(res[0].flatten())

            res=torch.autograd.grad(mean_samples_num, p, allow_unused=True, retain_graph=True)
            if res[0] is not None:
                sample_derivs_2.append(res[0].flatten())

        eval_derivs_2=torch.cat(eval_derivs_2)
        eval_again_derivs_2=torch.cat(eval_again_derivs_2)
        eval_again_derivs_detached_2=torch.cat(eval_again_derivs_detached_2)
        sample_derivs_2=torch.cat(sample_derivs_2)

        ######
       
        compare_two_arrays(samples.detach().numpy(), samples_num.detach().numpy(), "samples", "samples_numerical")
        compare_two_arrays(evals.detach().numpy(), evals_num.detach().numpy(), "evals", "evals_numerical")

        compare_two_arrays(evals_num.detach().numpy(), evals_again_num.detach().numpy(), "evals_numerical", "evals_numerical_again")

        ## compare derivative of samples
        compare_two_arrays(eval_derivs_1.detach().numpy(), eval_derivs_2.detach().numpy(), "eval_derivs", "eval_derivs_numerical")
        compare_two_arrays(sample_derivs_1.detach().numpy(), sample_derivs_2.detach().numpy(), "sample_derivs", "sample_derivs_numerical")
        compare_two_arrays(eval_derivs_again_1.detach().numpy(), eval_again_derivs_2.detach().numpy(), "eval_again_derivs1", "eval_again_derivs2")
        compare_two_arrays(eval_derivs_again_detached_1.detach().numpy(), eval_again_derivs_detached_2.detach().numpy(), "eval_again_derivs1_detached", "eval_again_derivs2_detached")
        #compare_two_arrays(base_evals.detach().numpy(), base_evals2.detach().numpy(), "base_evals", "base_evals2")


    

    def test_compare_newton_times(self):
        """ 
        Comparing the derivatives and normal results of newton iterations with two different versions of Gaussianization flow (h - old), (g - new).
        """
        print("Testing the agreement of two different newton iteration algorithms and compares their speed")

        def get_non_none_grad(quantity, params):

            grad_res=torch.autograd.grad(quantity, params, allow_unused=True, retain_graph=False)

            return torch.cat( [res.view(-1) for res in grad_res if (res is not None)])

        samplesize=1000
        seed_everything(1)
        z=torch.rand((samplesize,1), dtype=torch.double)*50

        for skew in [0,1]:

            
            #### old gaussianization flow
            
            ## we dont want normalization in the following for simplicity (in one we would need an extra transformation to apply manually)
            extra_flow_defs=dict()
            extra_flow_defs["g"]=dict()
            extra_flow_defs["g"]["kwargs"]=dict()
            extra_flow_defs["g"]["kwargs"]["regulate_normalization"]=0
            extra_flow_defs["g"]["kwargs"]["add_skewness"]=skew
            extra_flow_defs["h"]=dict()
            extra_flow_defs["h"]["kwargs"]=dict()
            extra_flow_defs["h"]["kwargs"]["regulate_normalization"]=0
            extra_flow_defs["h"]["kwargs"]["add_skewness"]=skew

            seed_everything(1)
            flow_exact=f.pdf("e1", "h", flow_defs_detail=extra_flow_defs)
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
            flow_exact=f.pdf("e1", "h", flow_defs_detail=extra_flow_defs)
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
            flow_exact=f.pdf("e1", "g", flow_defs_detail=extra_flow_defs)
            flow_exact.double()

            gf_layer=flow_exact.layer_list[0][0]

            flow_params=gf_layer._obtain_usable_flow_params(z)
        
            newton_tolerance=1e-14
            res_new_fast=bn.inverse_bisection_n_newton_joint_func_and_grad(gf_layer.sigmoid_inv_error_pass_w_params, gf_layer.sigmoid_inv_error_pass_combined_val_n_normal_derivative, z, flow_params[0], flow_params[1], flow_params[2], flow_params[3], flow_params[4], min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=25, num_newton_iter=20, newton_tolerance=newton_tolerance, verbose=0)
            new_fast_grad=get_non_none_grad(res_new_fast.sum(),gf_layer.parameters())

            compare_two_arrays(old_slow_grad.numpy(), new_fast_grad.numpy(), "old slow grad", "new fast grad")
           
      

            #assert(torch.abs(old_slow_grad-new_fast_grad).sum() < 1e-12)

            #############
            
            with torch.no_grad():
                
                    
                ### old gaussianization flow
                seed_everything(1)
                flow_exact=f.pdf("e1", "h",flow_defs_detail=extra_flow_defs)
                flow_exact.double()

                gf_layer=flow_exact.layer_list[0][0]

                skew_for_old=gf_layer.skew_exponents
                if(skew):
                    skew_for_old=torch.exp(gf_layer.exponent_regulator(gf_layer.skew_exponents))
              

                z=torch.rand((samplesize,1), dtype=torch.double)*50
                
                tbef=time.time()
                for i in range(100):
                    res_old_slow=bn.inverse_bisection_n_newton_slow(gf_layer.sigmoid_inv_error_pass, gf_layer.sigmoid_inv_error_pass_derivative, z, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, skew_for_old,gf_layer.skew_signs, min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=25, num_newton_iter=20)
                   
                print("Old flow / old newton iteration took ", time.time()-tbef)

                skew_for_old=gf_layer.skew_exponents
                if(skew):
                    skew_for_old=torch.exp(gf_layer.exponent_regulator(gf_layer.skew_exponents))
              
                newton_tolerance=1e-14
                tbef=time.time()
                for i in range(100):
                    res_old_fast=bn.inverse_bisection_n_newton(gf_layer.sigmoid_inv_error_pass, gf_layer.sigmoid_inv_error_pass_derivative, z, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights, skew_for_old,gf_layer.skew_signs, min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=25, num_newton_iter=20, newton_tolerance=newton_tolerance, verbose=0)
                
                print("Old flow / new newton iteration with tolerance dropout took ", time.time()-tbef)
                print(torch.abs(res_old_slow-res_old_fast).max())
                assert( torch.abs(res_old_slow-res_old_fast).max() < newton_tolerance*100)

                ##### new gaussianization flow

                seed_everything(1)
                flow_exact=f.pdf("e1", "g",flow_defs_detail=extra_flow_defs)
                flow_exact.double()

                gf_layer=flow_exact.layer_list[0][0]

                flow_params=gf_layer._obtain_usable_flow_params(z)

                newton_tolerance=1e-14
                tbef=time.time()
                for i in range(100):
                    res_new_fast=bn.inverse_bisection_n_newton_joint_func_and_grad(gf_layer.sigmoid_inv_error_pass_w_params, gf_layer.sigmoid_inv_error_pass_combined_val_n_normal_derivative, z, flow_params[0], flow_params[1], flow_params[2], flow_params[3], flow_params[4], min_boundary=-1e5, max_boundary=1e5, num_bisection_iter=25, num_newton_iter=20, newton_tolerance=newton_tolerance, verbose=0)
                
                print("new flow / new newton iteration with tolerance dropout took ", time.time()-tbef)
                print(torch.abs(res_old_slow-res_new_fast).max())
                assert( torch.abs(res_old_slow-res_new_fast).max() < newton_tolerance*100)


            ### now do the same and check gradient compatibility

        
        
                
        


if __name__ == '__main__':
    unittest.main()