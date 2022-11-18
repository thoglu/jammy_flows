import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.main.default as f


def test_gf_modes():

    icdf_approximations=["inormal_full_pade", "inormal_partly_crude", "inormal_partly_precise", "isigmoid"]

    for icdf_approx in icdf_approximations:

        extra_flow_defs=dict()
        extra_flow_defs["g"]=dict()
        extra_flow_defs["g"]["kwargs"]=dict()

        extra_flow_defs["g"]["kwargs"]["inverse_function_type"]=icdf_approx

        this_flow=f.pdf("e5", "gggg", flow_defs_detail=extra_flow_defs)
        with torch.no_grad():
            tbef=time.time()
            
            samples,_,_,_=this_flow.sample(samplesize=50000)
            sample_time=time.time()-tbef

            tbef=time.time()
            this_flow(samples)
            eval_time=time.time()-tbef
        
        print("%s ... sampletime %d samples: %.2f / evaltime: %.2f" %(icdf_approx, len(samples), sample_time, eval_time))

if __name__ == '__main__':
    
    test_gf_modes()