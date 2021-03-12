
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.flows as f

import jammy_flows.helper_fns as helper_fns

if __name__=="__main__":

    fl=f.pdf("i1_-1.5_0.5", "r")  

    fig=pylab.figure()
    helper_fns.visualize_pdf(fl, fig, nsamples=500000)

    if(not os.path.exists("figs")):
        os.makedirs("figs")

    pylab.savefig("interval.png")