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
        

    def test_compare_newton_times(self):
        """ 
        We form derivatives of various quantities, once of the exact version, and once of the verrsion with the numerical inverse.
        Parameters that can be exactly 0 (or None) are skipped (vs / mean)
        """
        def get_non_none_grad(quantity, params):

            grad_res=torch.autograd.grad(quantity, params, allow_unused=True, retain_graph=False)

            return torch.cat( [res.view(-1) for res in grad_res if (res is not None)])


        for add_skew in [0,1]:
            print("SKEWNESS CHECK ", add_skew)
            samplesize=100
            extra_flow_defs=dict()
            extra_flow_defs["g"]=dict()
            extra_flow_defs["g"]["kwargs"]=dict()
            extra_flow_defs["g"]["kwargs"]["regulate_normalization"]=1
            extra_flow_defs["g"]["kwargs"]["add_skewness"]=add_skew
            extra_flow_defs["h"]=dict()
            extra_flow_defs["h"]["kwargs"]=dict()
            extra_flow_defs["h"]["kwargs"]["regulate_normalization"]=1
            extra_flow_defs["h"]["kwargs"]["add_skewness"]=add_skew

            seed_everything(1)
            z=torch.rand((samplesize,1), dtype=torch.double)

            seed_everything(1)
            flow_old=f.pdf("e1" , "h", flow_defs_detail=extra_flow_defs)
            flow_old.double()

            res_old,_,_=flow_old(z)

            ###

            seed_everything(1)
            flow_new=f.pdf("e1" , "g",flow_defs_detail=extra_flow_defs)
            flow_new.double()

            res_new,_,_=flow_new(z)

            compare_two_arrays(res_old.detach().numpy(), res_new.detach().numpy(), "old gf result", "new gf result")

            #### now compare gradients

            grad_old=get_non_none_grad(res_old.sum(),flow_old.parameters())
            grad_new=get_non_none_grad(res_new.sum(), flow_new.parameters())

            compare_two_arrays(grad_old.detach().numpy(), grad_old.detach().numpy(), "grad old", "grad new")

           
        


if __name__ == '__main__':
    unittest.main()