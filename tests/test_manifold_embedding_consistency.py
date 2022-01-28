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

def setup_and_test_multiple_pdfs(pdf_def, layer_def):

    seed_everything(0)

    normal_pdf=f.pdf(pdf_def, layer_def)

    normal_sample,_,_,_=normal_pdf.sample(samplesize=10000)
    normal_eval,_,_=normal_pdf(normal_sample)

    # now go to embedding space
    detail=dict()
    for ld in layer_def.split("+"):
        for l in ld:
            if(l in ["w", "u"]):

                if(l not in detail.keys()):
                    detail[l]=dict()
                    detail[l]["kwargs"]=dict()
                    detail[l]["kwargs"]["always_parametrize_in_embedding_space"]=1

    
        
    embedded_sample=normal_pdf.transform_target_into_returnable_params(normal_sample)
    

    seed_everything(0)
    embedded_pdf=f.pdf(pdf_def, layer_def, flow_defs_detail=detail)

    embedded_eval,_,_=embedded_pdf(embedded_sample)

    for ind in range(10):
        res1=torch.autograd.grad(normal_eval[ind], normal_pdf.parameters(), allow_unused=True, retain_graph=True)[0]
        res2=torch.autograd.grad(embedded_eval[ind], embedded_pdf.parameters(), allow_unused=True, retain_graph=True)[0]
       

        compare_two_arrays(res1.detach().numpy(), res2.detach().numpy(), "normal_%s" % pdf_def, "embedded_%s" % pdf_def)

class Test(unittest.TestCase):
    def setUp(self):

        self.test_cases=[]

        ## multiple simplices 
        testcase=dict()
        testcase["pdf_def"]="c3"
        testcase["layer_def"]="w"
        testcase["flow_defs_detail"]=dict()
        testcase["flow_defs_detail"]["w"]=dict()
        testcase["flow_defs_detail"]["w"]["kwargs"]=dict()

        self.test_cases.append(testcase)

    def test_manifold_embedding(self):
        """ 
        Comparing the derivatives and normal results of newton iterations with two different versions of Gaussianization flow (h - old), (g - new).
        """
        print("Testing agreement betweeen normal and embedding evaluation for manifold flows.")

        for tc in self.test_cases:
            setup_and_test_multiple_pdfs(tc["pdf_def"], tc["layer_def"])
        
           
            
if __name__ == '__main__':
    unittest.main()