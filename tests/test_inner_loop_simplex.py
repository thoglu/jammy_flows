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

from scipy.stats import gumbel_r

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.flows as f
#from pytorch_lightning import seed_everything
import jammy_flows.helper_fns as helper_fns
import jammy_flows.layers.bisection_n_newton as bn

def seed_everything(seed):

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def exact_gumbel_first_coord(x, tau, probs):

    print(x.shape)
    assert( len(x.shape)==2)

    assert(x.shape[1]==(len(probs)-1))



    last_coord=1.0-numpy.sum(x, axis=1, keepdims=True)

    all_coords=numpy.concatenate( [x, last_coord], axis=1)

    sum_part=numpy.sum((probs[None,:]/(all_coords**tau)), axis=1)**(-len(probs))

    print(sum_part)

    prod_part=numpy.prod((probs[None,:]/(all_coords**(tau+1))), axis=1)

    return sum_part*prod_part*(tau**(len(probs)-1) )
if __name__ == '__main__':
    
    seed_everything(0)

    pdf=f.pdf("a1", "w")

    samp,_,_,_=pdf.sample(samplesize=100000)

    fig=pylab.figure()

    ax=fig.add_subplot(111)

    ax.hist(samp.detach().numpy(), density=True, bins=50)

    xvals1=torch.linspace(0.00001,0.99999,500,dtype=torch.float64).unsqueeze(1)
    #xvals2=torch.linspace(0.01,0.99,30,dtype=torch.float64).unsqueeze(1)
    dx=xvals1[1]-xvals1[0]

    probs1,_,base_values=pdf(xvals1)
    xvals2, _, probs2,_=pdf._obtain_sample(predefined_target_input=base_values)

    probs1=probs1.exp().detach().numpy()
    probs2=probs2.exp().detach().numpy()

    returned_1d_real_params=pdf.transform_target_into_returnable_params(xvals1)

    distances=numpy.sqrt((returned_1d_real_params[1:,0]-returned_1d_real_params[:-1,0])**2 + (returned_1d_real_params[1:,1]-returned_1d_real_params[:-1,1])**2 )
    #probs2,_,_=pdf(xvals2)
    assert( (numpy.fabs(distances[1:]-distances[:-1])>1e-12).sum()==0)

    ax.hist(xvals2.detach().numpy(), density=True, bins=50, color="red")

    #exact_probs_paper=exact_gumbel_first_coord(xvals1.numpy(), 1.5, numpy.array([1.0,1.0]) )

    #np_probs1=probs1.exp().detach().numpy()
    #np_probs2=probs2.exp().detach().numpy()

    ax.plot(xvals1, probs1, label="sum %.2f" % (dx*probs1.sum()))
    ax.plot(xvals1, probs1, ls="--", label="sum %.2f" % (dx*probs2.sum()))

    #ax.plot(xvals1, exact_probs_paper, label="sum %.2f" % (dx*exact_probs_paper.sum()))
    #ax.plot(xvals2, np_probs2, label="sum %.2f" % (dx*np_probs2.sum()))
    #ax.plot(xvals, gumbel_r.pdf(xvals), label="GUMB", ls="--")
    ax.legend()

    pylab.savefig("inner_loop_simplex_1d.png")

    ##############################################
    ##############################################

    pdf=f.pdf("a2", "w")
    
    samp,_,_,_=pdf.sample(samplesize=100000)

    fig=pylab.figure()

    ax=fig.add_subplot(111)

    ax.hist2d(samp.detach().numpy()[:,0],samp.detach().numpy()[:,1], density=True, bins=50)

    xvals1=torch.linspace(0.00001,0.99999,500,dtype=torch.float64).unsqueeze(1)
    #xvals2=torch.linspace(0.01,0.99,30,dtype=torch.float64).unsqueeze(1)
    dx=xvals1[1]-xvals1[0]

    #probs1,_,_=pdf(xvals1)
    #probs2,_,_=pdf(xvals2)

    #exact_probs_paper=exact_gumbel_first_coord(xvals1.numpy(), 1.5, numpy.array([1.0,1.0]) )

    #np_probs1=probs1.exp().detach().numpy()
    #np_probs2=probs2.exp().detach().numpy()

    #ax.plot(xvals1, np_probs1, label="sum %.2f" % (dx*np_probs1.sum()))

    #ax.plot(xvals1, exact_probs_paper, label="sum %.2f" % (dx*exact_probs_paper.sum()))
    #ax.plot(xvals2, np_probs2, label="sum %.2f" % (dx*np_probs2.sum()))
    #ax.plot(xvals, gumbel_r.pdf(xvals), label="GUMB", ls="--")
    ax.legend()
    

    pylab.savefig("inner_loop_simplex_2d.png")

    ###############

    fig=pylab.figure()

    ax=fig.add_subplot(projection="3d")

    higher_dim_coords=pdf.transform_target_into_returnable_params(samp[:100]).detach().numpy()

    print(higher_dim_coords)

    ax.scatter(higher_dim_coords[:,0],higher_dim_coords[:,1],higher_dim_coords[:,2])

    pylab.savefig("inner_loop_simplex_3d.png")

