
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

if __name__=="__main__":

    extra_flow_defs=dict()
    
    extra_flow_defs["r"]=dict()
    extra_flow_defs["r"]["kwargs"]=dict()
    extra_flow_defs["r"]["kwargs"]["num_basis_elements"]=3

    fl=f.pdf("i1_-100.5_5.5", "r", flow_defs_detail=extra_flow_defs)  

    fig=pylab.figure()
    helper_fns.visualize_pdf(fl, fig, nsamples=500000)

    if(not os.path.exists("figs")):
        os.makedirs("figs")

    pylab.savefig("interval.png")