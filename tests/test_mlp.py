import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.flows as f

import jammy_flows.helper_fns as helper_fns

import jammy_flows.layers.extra_functions as extra

class Test(unittest.TestCase):
    def setUp(self):

        self.single_matrix=torch.randn(size=(1,30,15)).type(torch.float64)
    
        self.bias_vec=torch.randn(size=(30,)).type(torch.float64)

    def test_mlp(self):

        ## input dim = 40, 50-50 hidden, 30 output dim
        mlp_single=extra.AmortizableMLP(15, "",30)
        mlp_single.double()
        flattened_single=self.single_matrix.flatten()
        len_flattened=len(flattened_single)
        print("len flat", len_flattened)
        print(mlp_single.u_v_b_pars.shape)
        mlp_single.u_v_b_pars.data[0,:len_flattened]=flattened_single
        mlp_single.u_v_b_pars.data[0,len_flattened:]=self.bias_vec
        
        ## batchdim 10, input_dim 40
        batched_input_vec=torch.randn(size=(5,15)).type(torch.float64)
        
        res_mlp=mlp_single(batched_input_vec)
        res_functional=F.linear(batched_input_vec, self.single_matrix[0], bias=self.bias_vec)
        assert((res_mlp-res_functional).sum() < 1e-14)
        print("total diff dim 2", (res_mlp-res_functional).sum())
        ################

        ## batchdim 10, input_dim 40
        batched_input_vec=torch.randn(size=(5,4,15)).type(torch.float64)
        
        res_mlp=mlp_single(batched_input_vec)
        res_functional=F.linear(batched_input_vec, self.single_matrix[0], bias=self.bias_vec)
        assert((res_mlp-res_functional).sum() < 1e-14)
        print("total diff dim 3", (res_mlp-res_functional).sum())


        ## batchdim 10, input_dim 40
        batched_input_vec=torch.randn(size=(5,3,4,15)).type(torch.float64)
        
        res_mlp=mlp_single(batched_input_vec)
        res_functional=F.linear(batched_input_vec, self.single_matrix[0], bias=self.bias_vec)
        assert((res_mlp-res_functional).sum() < 1e-14)
        print("total diff dim 4", (res_mlp-res_functional).sum())



if __name__ == '__main__':
    unittest.main()