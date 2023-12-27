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
    normal_pdf.double()

    seed_everything(0)
    embedded_pdf=f.pdf(pdf_def, layer_def)
    embedded_pdf.set_embedding_flags(True)
    embedded_pdf.double()

    seed_everything(0)
    intrinsic_pdf=f.pdf(pdf_def, layer_def)
    intrinsic_pdf.set_embedding_flags(False)
    intrinsic_pdf.double()

    for sub_pdf_index in [-1]+list(range(len(normal_pdf.layer_list))):
        print("sub pdf ", sub_pdf_index)
        normal_pdf.set_embedding_flags(False)
        if(sub_pdf_index>=0):
           
            normal_pdf.set_embedding_flags(True, sub_pdf_index=sub_pdf_index)

        normal_sample,_,_,_=normal_pdf.sample(samplesize=100)
        normal_eval,_,_=normal_pdf(normal_sample)

        embedded_sample,_=normal_pdf.transform_target_space(normal_sample, transform_to="embedding")
        embedded_eval,_,_=embedded_pdf(embedded_sample)

        intrinsic_sample,_=normal_pdf.transform_target_space(normal_sample, transform_to="intrinsic")
        intrinsic_eval,_,_=intrinsic_pdf(intrinsic_sample)

        normal_forced_intrinsic_eval,_,_=normal_pdf(intrinsic_sample, force_intrinsic_coordinates=True)


        embedded_forced_intrinsic_eval,_,_=embedded_pdf(intrinsic_sample, force_intrinsic_coordinates=True)

        normal_forced_embedded_eval,_,_=normal_pdf(embedded_sample, force_embedding_coordinates=True)
        intrinsic_forced_embedded_eval,_,_=intrinsic_pdf(embedded_sample, force_embedding_coordinates=True)
      
        for ind in range(1):
            
            res1=torch.autograd.grad(normal_eval[ind], normal_pdf.parameters(), allow_unused=False, retain_graph=True)[0]
            res2=torch.autograd.grad(embedded_eval[ind], embedded_pdf.parameters(), allow_unused=False, retain_graph=True)[0]

            compare_two_arrays(res1.detach().numpy(), res2.detach().numpy(), "normal_%s" % pdf_def, "embedded_%s" % pdf_def)

            res_intrinsic=torch.autograd.grad(intrinsic_eval[ind], intrinsic_pdf.parameters(), allow_unused=False, retain_graph=True)[0]

            compare_two_arrays(res1.detach().numpy(), res_intrinsic.detach().numpy(), "normal_%s" % pdf_def, "intrinsic_%s" % pdf_def)

            ### normal forced intrinsic
            res_nfi=torch.autograd.grad(normal_forced_intrinsic_eval[ind], normal_pdf.parameters(), allow_unused=False, retain_graph=True)[0]
            compare_two_arrays(res_nfi.detach().numpy(), res_intrinsic.detach().numpy(), "nfi_%s" % pdf_def, "intrinsic_%s" % pdf_def)

            ### embedded forced intrinsic
            res_efi=torch.autograd.grad(embedded_forced_intrinsic_eval[ind], embedded_pdf.parameters(), allow_unused=False, retain_graph=True)[0]
            compare_two_arrays(res_efi.detach().numpy(), res_intrinsic.detach().numpy(), "efi_%s" % pdf_def, "intrinsic_%s" % pdf_def)

            ### normal forced embedded
            res_nfe=torch.autograd.grad(normal_forced_embedded_eval[ind], normal_pdf.parameters(), allow_unused=False, retain_graph=True)[0]
            compare_two_arrays(res_nfe.detach().numpy(), res_intrinsic.detach().numpy(), "nfe_%s" % pdf_def, "intrinsic_%s" % pdf_def)

            ### intrinsic forced embedded
            res_ife=torch.autograd.grad(intrinsic_forced_embedded_eval[ind], intrinsic_pdf.parameters(), allow_unused=False, retain_graph=True)[0]
            compare_two_arrays(res_ife.detach().numpy(), res_intrinsic.detach().numpy(), "ife_%s" % pdf_def, "intrinsic_%s" % pdf_def)



class Test(unittest.TestCase):
    def setUp(self):

        self.test_cases=[]

        testcase=dict()
        testcase["pdf_def"]="e1+s2+a3+e1+s2"
        testcase["layer_def"]="g+v+w+g+v"

        self.test_cases.append(testcase)
        
        ## multiple simplices 
        testcase=dict()
        testcase["pdf_def"]="a3+e1"
        testcase["layer_def"]="w+g"

        self.test_cases.append(testcase)
        
        ### spheres

        testcase=dict()
        testcase["pdf_def"]="s2"
        testcase["layer_def"]="n"

        self.test_cases.append(testcase)

        testcase=dict()
        testcase["pdf_def"]="s2"
        testcase["layer_def"]="v"

        self.test_cases.append(testcase)

        testcase=dict()
        testcase["pdf_def"]="s1"
        testcase["layer_def"]="o"

        self.test_cases.append(testcase)

        ## mixed

        

        
        
        


        

    def test_manifold_embedding(self):
        """ 
        Testing that switch between intrinsic/embedding coordinates works correctly for manifold sub pdfs..
        """
        print("Testing agreement betweeen normal and embedding evaluation for manifold flows.")

        for tc in self.test_cases:
            print("testing ", tc)
            setup_and_test_multiple_pdfs(tc["pdf_def"], tc["layer_def"])
        
           
            
if __name__ == '__main__':
    unittest.main()