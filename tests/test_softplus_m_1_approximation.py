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

import jammy_flows.extra_functions as extra_functions

def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)

class Test(unittest.TestCase):
    
    def setUp(self, plot=None):
        self.plot=False
    def test_approximation(self):

        ## low exponent
        ## float precision

        x=torch.linspace(-50,100, 1000, dtype=torch.double)
       
        a=torch.ones_like(x)*0.01

        res=extra_functions.log_one_plus_exp_x_to_a_minus_1(x,a)
        
        assert((~torch.isfinite(res)).sum()==0), (res)

        if(self.plot):
            fig=pylab.figure()
            ax=fig.add_subplot(111)
            ax.plot(x,res)
            pylab.savefig("approximation1.png")

            fig=pylab.figure()
            ax=fig.add_subplot(111)
            ax.plot(x,torch.log(-1.0*res))
            pylab.savefig("approximation1_logdiff.png")

        a=torch.ones_like(x)*10.0

        res=extra_functions.log_one_plus_exp_x_to_a_minus_1(x,a)
        assert((~torch.isfinite(res)).sum()==0)

        if(self.plot):
            fig=pylab.figure()
            ax=fig.add_subplot(111)
            ax.plot(x,res)
            pylab.savefig("approximation2.png")
            
            fig=pylab.figure()
            ax=fig.add_subplot(111)
            ax.plot(x,torch.log(-1.0*res))
            pylab.savefig("approximation2_logdiff.png")
        
    

if __name__ == '__main__':
    unittest.main()