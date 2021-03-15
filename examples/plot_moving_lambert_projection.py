import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np
import numpy
import scipy.stats
import torch
import torch.optim as optim

import jammy_flows
from jammy_flows import helper_fns
import pylab
from matplotlib import rc
import random


def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)

############################

if __name__ == "__main__":
    seed_everything(0)
    ## define PDF

    extra_flow_defs=dict()
    extra_flow_defs["n"]=dict()
    extra_flow_defs["n"]["kwargs"]=dict()
    extra_flow_defs["n"]["kwargs"]["use_extra_householder"]=1
    extra_flow_defs["n"]["kwargs"]["higher_order_cylinder_parametrization"]=1
    extra_flow_defs["n"]["kwargs"]["zenith_type_layers"]="r"
    

    word_pdf=jammy_flows.pdf("s2", "n", flow_defs_detail=extra_flow_defs)

    #res,_,_,_=word_pdf._obtain_sample(predefined_target_input=torch.Tensor([[0.0,0.1],[0.0,0.2]]))

   
    #res,_,_,_=word_pdf.sample(samplesize=20)

    num_steps=20
    for ind in range(num_steps):

        ## generate zentih/azi steps to define some "true postion"
        max_zen=numpy.pi-0.001
        min_zen=0.001
        zen_step=(max_zen-min_zen)/num_steps
        true_zen=min_zen+ind*zen_step

        max_azi=2*numpy.pi
        min_azi=0.0
        azi_step=(max_azi-min_azi)/num_steps
        true_azi=min_azi+ind*azi_step

        ## visualize PDF for different "true positions", i.e. from different vantage points
        bounds=[ [-2.0,2.0], [-2.0,2.0]]
      
        fig=pylab.figure()

        helper_fns.visualize_pdf(word_pdf, fig, s2_norm="lambert", nsamples=500000, true_values=torch.Tensor([true_zen,true_azi]),skip_plotting_density=False, skip_plotting_samples=False, bounds=bounds, s2_rotate_to_true_value=True)


        if(not os.path.exists("figs")):
            os.makedirs("figs")
        pylab.savefig("figs/zen_%.3f_azi_%.3f.png" % (true_zen, true_azi))
