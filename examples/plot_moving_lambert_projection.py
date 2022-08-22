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

    parser = argparse.ArgumentParser('train_example')

    parser.add_argument("-layer_def", type=str, default="n")
    parser.add_argument("-num_samples", type=int, default=10)
    #parser.add_argument("-projection", type=str, default="lambert", choices=["lambert", "standard"]) 

    args=parser.parse_args()

    seed_everything(1)
    ## define PDF

    extra_flow_defs=dict()
    extra_flow_defs[0]=dict()
    extra_flow_defs[0]["n"]=dict()
    extra_flow_defs[0]["n"]["use_extra_householder"]=1
    extra_flow_defs[0]["n"]["higher_order_cylinder_parametrization"]=0
    extra_flow_defs[0]["n"]["zenith_type_layers"]="rr"

    extra_flow_defs["v"]=dict()
    #extra_flow_defs["v"]["kwargs"]["use_extra_householder"]=0
    extra_flow_defs["v"]["natural_direction"]=0
    extra_flow_defs["v"]["exp_map_type"]="nn"
    

    test_pdf=jammy_flows.pdf("s2", args.layer_def, options_overwrite=extra_flow_defs)

    #res,_,_,_=test_pdf._obtain_sample(predefined_target_input=torch.Tensor([[0.0,0.1],[0.0,0.2]]))

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
       
    
      
        fig=pylab.figure(figsize=(9,4))

        gs=fig.add_gridspec(1, 2)

        lambert_gridspec = gs[0, 0]
        lambert_bounds=[ [-2.0,2.0], [-2.0,2.0]]
        _,_,total_integral=helper_fns.visualize_pdf(test_pdf, fig, gridspec=lambert_gridspec, total_pdf_eval_pts=1600,s2_norm="lambert", nsamples=args.num_samples, true_values=torch.Tensor([true_zen,true_azi]),skip_plotting_density=False, skip_plotting_samples=False, bounds=lambert_bounds, s2_rotate_to_true_value=True)

        pylab.gca().set_title("PDF integral: %.3f" % total_integral)
        normal_gridspec = gs[0, 1]
        standard_bounds=[[0, numpy.pi],[0,2*numpy.pi]]
        _,_,total_integral=helper_fns.visualize_pdf(test_pdf, fig, gridspec=normal_gridspec, s2_norm="standard",total_pdf_eval_pts=1600, nsamples=args.num_samples, true_values=torch.Tensor([true_zen,true_azi]),skip_plotting_density=False, skip_plotting_samples=False, bounds=standard_bounds, s2_rotate_to_true_value=False)

        pylab.gca().set_title("PDF integral: %.3f" % total_integral)

        fig.tight_layout()
        if(not os.path.exists("figs")):
            os.makedirs("figs")
        pylab.savefig("figs/zen_%.3f_azi_%.3f.png" % (true_zen, true_azi))
