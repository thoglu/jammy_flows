import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import torch.autograd
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.flows as f
#from pytorch_lightning import seed_everything
import jammy_flows.helper_fns as helper_fns

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
        settings={ "flow_defs_detail": {"p":{"kwargs":{"exact_mode":True}}} }

        #self.flow_inits.append( [ [pdf_def, flow_def], dict()] )
        self.init_exact= [ [pdf_def, flow_def], settings] 
        
        settings={ "flow_defs_detail": {"p":{"kwargs":{"exact_mode":False}}} }

        self.init_numerical= [ [pdf_def, flow_def], settings] 

       
    
    def test_newton(self):

        print("Testing self consistency of sampling")
        samplesize=10000
        
        
        seed_everything(0)
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

       
        print(flow_exact.count_parameters())

        eval_derivs_1=[]
        eval_derivs_again_1=[]
        eval_derivs_again_detached_1=[]
        sample_derivs_1=[]

        for name, p in flow_exact.named_parameters():
            print("name ", name)
            #print(torch.autograd.grad(samples[0][0], p))
            res=torch.autograd.grad(mean_evals, p, allow_unused=True, retain_graph=True)
            print(res)
            if res[0] is not None:
                eval_derivs_1.append(res[0].flatten())

            res=torch.autograd.grad(mean_evals_again, p, allow_unused=True, retain_graph=True)
            print(res)
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

        seed_everything(0)

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
            print(name)

            res=torch.autograd.grad(mean_evals_num, p, allow_unused=True, retain_graph=True)
            print(res)
            if res[0] is not None:
                eval_derivs_2.append(res[0].flatten())

            res=torch.autograd.grad(mean_evals_again_num, p, allow_unused=True, retain_graph=True)
            print(res)
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

    

if __name__ == '__main__':
    unittest.main()